"""Fused FusedAddRMSNorm + INT4 GEMV kernels.

INT4 per-group variants of the fused norm+GEMV kernels.
Weights packed as 2 int4 per uint8, with per-group float16 scales.

Key design: compute normed x for even/odd positions separately in registers.
This avoids scratch buffers and memory consistency issues.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_gemv_int4_kernel(
    x_ptr, residual_ptr, norm_w_ptr, w_ptr, scale_ptr, y_ptr, residual_out_ptr,
    M, N, eps: tl.constexpr,
    stride_wn,
    num_groups,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HALF_M: tl.constexpr,   # M // 2
):
    """Fused: new_res = x + residual; normed = rmsnorm(new_res); y = normed @ dequant(W_int4)^T."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    half_ids = tl.arange(0, HALF_M)

    # Load x, residual at even/odd positions (from L2, stride-2 access)
    x_even = tl.load(x_ptr + half_ids * 2).to(tl.float32)
    x_odd = tl.load(x_ptr + half_ids * 2 + 1).to(tl.float32)
    r_even = tl.load(residual_ptr + half_ids * 2).to(tl.float32)
    r_odd = tl.load(residual_ptr + half_ids * 2 + 1).to(tl.float32)

    # Fused add
    new_even = x_even + r_even
    new_odd = x_odd + r_odd

    # Block 0 writes updated residual
    if pid == 0:
        tl.store(residual_out_ptr + half_ids * 2, new_even.to(residual_out_ptr.dtype.element_ty))
        tl.store(residual_out_ptr + half_ids * 2 + 1, new_odd.to(residual_out_ptr.dtype.element_ty))

    # RMS norm (over all M elements)
    sq_sum = tl.sum(new_even * new_even) + tl.sum(new_odd * new_odd)
    rms_inv = tl.rsqrt(sq_sum / M + eps)

    nw_even = tl.load(norm_w_ptr + half_ids * 2).to(tl.float32)
    nw_odd = tl.load(norm_w_ptr + half_ids * 2 + 1).to(tl.float32)

    xn_even = new_even * rms_inv * nw_even
    xn_odd = new_odd * rms_inv * nw_odd

    # INT4 GEMV
    w_base = w_ptr + rows * stride_wn
    w_offsets = w_base[:, None] + half_ids[None, :]
    w_packed = tl.load(w_offsets, mask=mask_n[:, None], other=0)

    w_lo = (w_packed & 0x0F).to(tl.float32) - 8.0
    w_hi = (w_packed >> 4).to(tl.float32) - 8.0

    prod = w_lo * xn_even[None, :] + w_hi * xn_odd[None, :]

    pair_group_ids = (half_ids * 2) // GROUP_SIZE
    scales = tl.load(
        scale_ptr + rows[:, None] * num_groups + pair_group_ids[None, :],
        mask=mask_n[:, None], other=1.0,
    ).to(tl.float32)

    acc = tl.sum(prod * scales, axis=1)
    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _fused_add_rmsnorm_gemv_silu_int4_kernel(
    x_ptr, residual_ptr, norm_w_ptr, w_ptr, scale_ptr, y_ptr, residual_out_ptr,
    M, N_half, eps: tl.constexpr,
    stride_wn,
    num_groups,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HALF_M: tl.constexpr,
):
    """Fused add+rmsnorm+gate_up INT4 GEMV+silu."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N_half

    half_ids = tl.arange(0, HALF_M)

    x_even = tl.load(x_ptr + half_ids * 2).to(tl.float32)
    x_odd = tl.load(x_ptr + half_ids * 2 + 1).to(tl.float32)
    r_even = tl.load(residual_ptr + half_ids * 2).to(tl.float32)
    r_odd = tl.load(residual_ptr + half_ids * 2 + 1).to(tl.float32)

    new_even = x_even + r_even
    new_odd = x_odd + r_odd

    if pid == 0:
        tl.store(residual_out_ptr + half_ids * 2, new_even.to(residual_out_ptr.dtype.element_ty))
        tl.store(residual_out_ptr + half_ids * 2 + 1, new_odd.to(residual_out_ptr.dtype.element_ty))

    sq_sum = tl.sum(new_even * new_even) + tl.sum(new_odd * new_odd)
    rms_inv = tl.rsqrt(sq_sum / M + eps)

    nw_even = tl.load(norm_w_ptr + half_ids * 2).to(tl.float32)
    nw_odd = tl.load(norm_w_ptr + half_ids * 2 + 1).to(tl.float32)

    xn_even = new_even * rms_inv * nw_even
    xn_odd = new_odd * rms_inv * nw_odd

    pair_group_ids = (half_ids * 2) // GROUP_SIZE

    # Gate projection
    w_gate_base = w_ptr + rows * stride_wn
    wg_packed = tl.load(w_gate_base[:, None] + half_ids[None, :], mask=mask_n[:, None], other=0)
    wg_lo = (wg_packed & 0x0F).to(tl.float32) - 8.0
    wg_hi = (wg_packed >> 4).to(tl.float32) - 8.0

    scale_gate = tl.load(
        scale_ptr + rows[:, None] * num_groups + pair_group_ids[None, :],
        mask=mask_n[:, None], other=1.0,
    ).to(tl.float32)
    acc_gate = tl.sum((wg_lo * xn_even[None, :] + wg_hi * xn_odd[None, :]) * scale_gate, axis=1)

    # Up projection
    w_up_base = w_ptr + (rows + N_half) * stride_wn
    wu_packed = tl.load(w_up_base[:, None] + half_ids[None, :], mask=mask_n[:, None], other=0)
    wu_lo = (wu_packed & 0x0F).to(tl.float32) - 8.0
    wu_hi = (wu_packed >> 4).to(tl.float32) - 8.0

    scale_up = tl.load(
        scale_ptr + (rows + N_half)[:, None] * num_groups + pair_group_ids[None, :],
        mask=mask_n[:, None], other=1.0,
    ).to(tl.float32)
    acc_up = tl.sum((wu_lo * xn_even[None, :] + wu_hi * xn_odd[None, :]) * scale_up, axis=1)

    # silu(gate) * up
    gate_silu = acc_gate * tl.sigmoid(acc_gate)
    result = gate_silu * acc_up

    tl.store(y_ptr + rows, result.to(y_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _fused_rmsnorm_gemv_int4_kernel(
    x_ptr, norm_w_ptr, w_ptr, scale_ptr, y_ptr,
    M, N, eps: tl.constexpr,
    stride_wn,
    num_groups,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HALF_M: tl.constexpr,
):
    """Fused rmsnorm+INT4 GEMV for first layer (no residual)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rows < N

    half_ids = tl.arange(0, HALF_M)

    x_even = tl.load(x_ptr + half_ids * 2).to(tl.float32)
    x_odd = tl.load(x_ptr + half_ids * 2 + 1).to(tl.float32)

    sq_sum = tl.sum(x_even * x_even) + tl.sum(x_odd * x_odd)
    rms_inv = tl.rsqrt(sq_sum / M + eps)

    nw_even = tl.load(norm_w_ptr + half_ids * 2).to(tl.float32)
    nw_odd = tl.load(norm_w_ptr + half_ids * 2 + 1).to(tl.float32)

    xn_even = x_even * rms_inv * nw_even
    xn_odd = x_odd * rms_inv * nw_odd

    w_base = w_ptr + rows * stride_wn
    w_packed = tl.load(w_base[:, None] + half_ids[None, :], mask=mask_n[:, None], other=0)
    w_lo = (w_packed & 0x0F).to(tl.float32) - 8.0
    w_hi = (w_packed >> 4).to(tl.float32) - 8.0

    prod = w_lo * xn_even[None, :] + w_hi * xn_odd[None, :]
    pair_group_ids = (half_ids * 2) // GROUP_SIZE
    scales = tl.load(
        scale_ptr + rows[:, None] * num_groups + pair_group_ids[None, :],
        mask=mask_n[:, None], other=1.0,
    ).to(tl.float32)

    acc = tl.sum(prod * scales, axis=1)
    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


# ============= Python launchers =============

def fused_add_rmsnorm_gemv_int4(
    x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor,
    gemv_weight_int4: torch.Tensor, gemv_weight_scale: torch.Tensor,
    eps: float, residual_out: torch.Tensor, group_size: int = 128,
) -> torch.Tensor:
    N = gemv_weight_int4.shape[0]
    M = gemv_weight_int4.shape[1] * 2
    y = torch.empty(1, N, device=x.device, dtype=x.dtype)

    BLOCK_N = 2
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    _fused_add_rmsnorm_gemv_int4_kernel[grid](
        x.view(-1), residual.view(-1), norm_weight,
        gemv_weight_int4, gemv_weight_scale,
        y.view(-1), residual_out.view(-1),
        M, N, eps,
        gemv_weight_int4.stride(0),
        M // group_size,
        GROUP_SIZE=group_size,
        BLOCK_N=BLOCK_N,
        HALF_M=M // 2,
    )
    return y


def fused_add_rmsnorm_gemv_silu_int4(
    x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor,
    gate_up_weight_int4: torch.Tensor, gate_up_weight_scale: torch.Tensor,
    eps: float, residual_out: torch.Tensor, group_size: int = 128,
) -> torch.Tensor:
    N_full = gate_up_weight_int4.shape[0]
    M = gate_up_weight_int4.shape[1] * 2
    N_half = N_full // 2
    y = torch.empty(1, N_half, device=x.device, dtype=x.dtype)

    BLOCK_N = 2
    grid = ((N_half + BLOCK_N - 1) // BLOCK_N,)

    _fused_add_rmsnorm_gemv_silu_int4_kernel[grid](
        x.view(-1), residual.view(-1), norm_weight,
        gate_up_weight_int4, gate_up_weight_scale,
        y.view(-1), residual_out.view(-1),
        M, N_half, eps,
        gate_up_weight_int4.stride(0),
        M // group_size,
        GROUP_SIZE=group_size,
        BLOCK_N=BLOCK_N,
        HALF_M=M // 2,
    )
    return y


def fused_rmsnorm_gemv_int4(
    x: torch.Tensor, norm_weight: torch.Tensor,
    gemv_weight_int4: torch.Tensor, gemv_weight_scale: torch.Tensor,
    eps: float, group_size: int = 128,
) -> torch.Tensor:
    N = gemv_weight_int4.shape[0]
    M = gemv_weight_int4.shape[1] * 2
    y = torch.empty(1, N, device=x.device, dtype=x.dtype)

    BLOCK_N = 2
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    _fused_rmsnorm_gemv_int4_kernel[grid](
        x.view(-1), norm_weight,
        gemv_weight_int4, gemv_weight_scale,
        y.view(-1),
        M, N, eps,
        gemv_weight_int4.stride(0),
        M // group_size,
        GROUP_SIZE=group_size,
        BLOCK_N=BLOCK_N,
        HALF_M=M // 2,
    )
    return y
