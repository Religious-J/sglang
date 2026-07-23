"""Pointwise Triton kernels surrounding the grouped MoE GEMMs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _act_mul_quant_kernel(
    gate_up,
    scale,
    output,
    total: tl.constexpr,
    intermediate: tl.constexpr,
    USE_BF16_MUL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    element = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair = element // intermediate
    column = element - pair * intermediate
    gate = tl.load(
        gate_up + pair * (2 * intermediate) + column,
        mask=element < total,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gate_up + pair * (2 * intermediate) + intermediate + column,
        mask=element < total,
        other=0.0,
    )
    silu = gate * tl.sigmoid(gate)
    if USE_BF16_MUL:
        silu = silu.to(tl.bfloat16).to(tl.float32)
        product = (silu * up.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    else:
        product = silu * up.to(tl.float32)
    output_scale = tl.load(scale)
    tl.store(output + element, product * output_scale, mask=element < total)


@triton.jit
def _act_mul_quant_blockwise_kernel(
    gate_up,
    output,
    output_scale,
    intermediate: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    block = tl.program_id(1)
    column = block * BLOCK + tl.arange(0, BLOCK)
    mask = column < intermediate
    gate = tl.load(
        gate_up + pair * (2 * intermediate) + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gate_up + pair * (2 * intermediate) + intermediate + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    product = gate * tl.sigmoid(gate) * up
    abs_max = tl.max(tl.where(mask, tl.abs(product), 0.0), axis=0)
    scale = abs_max / 448.0
    inverse_scale = 1.0 / (scale + 1.0e-8)
    tl.store(output_scale + pair * tl.cdiv(intermediate, BLOCK) + block, scale)
    tl.store(
        output + pair * intermediate + column,
        product * inverse_scale,
        mask=mask,
    )


@triton.jit
def _act_mul_quant_blockwise_256_kernel(
    gate_up,
    output,
    output_scale,
):
    pair = tl.program_id(0)
    column = tl.arange(0, 256)
    gate = tl.load(gate_up + pair * 512 + column).to(tl.float32)
    up = tl.load(gate_up + pair * 512 + 256 + column).to(tl.float32)
    product = gate * tl.sigmoid(gate) * up
    products = product.reshape(2, 128)
    abs_max = tl.max(tl.abs(products), axis=1)
    scale = abs_max / 448.0
    inverse_scale = 1.0 / (scale + 1.0e-8)
    tl.store(output_scale + pair * 2 + tl.arange(0, 2), scale)
    quantized = products * inverse_scale[:, None]
    tl.store(output + pair * 256 + column, quantized.reshape(256))


@triton.jit
def _weighted_reduce_kernel(
    down,
    topk_ids,
    topk_scale,
    shared_output,
    output,
    hidden: tl.constexpr,
    topk: tl.constexpr,
    expert_start,
    expert_end,
    ALL_ROUTES_LOCAL: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = column < hidden
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for slot in tl.static_range(topk):
        pair = token * topk + slot
        if ALL_ROUTES_LOCAL:
            valid = True
        else:
            expert = tl.load(topk_ids + pair)
            valid = (expert >= expert_start) & (expert < expert_end)
        value = tl.load(
            down + pair * hidden + column,
            mask=mask & valid,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(topk_scale + pair)
        acc += value * weight
    if HAS_SHARED:
        acc += tl.load(shared_output + token * hidden + column, mask=mask).to(
            tl.float32
        )
    tl.store(output + token * hidden + column, acc.to(tl.bfloat16), mask=mask)


def act_mul_quant_fp8(
    gate_up: Tensor,
    scale: Tensor,
    use_bf16_mul: bool,
    out: Tensor | None = None,
) -> Tensor:
    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2:
        raise ValueError("gate_up must be BF16 [pairs, 2 * intermediate]")
    pairs, doubled = gate_up.shape
    if doubled % 2:
        raise ValueError("gate_up's last dimension must be even")
    intermediate = doubled // 2
    if out is None:
        out = torch.empty(
            (pairs, intermediate), dtype=torch.float8_e4m3fn, device=gate_up.device
        )
    total = pairs * intermediate
    block = 256
    _act_mul_quant_kernel[(triton.cdiv(total, block),)](
        gate_up,
        scale,
        out,
        total,
        intermediate,
        USE_BF16_MUL=use_bf16_mul,
        BLOCK=block,
        num_warps=4,
    )
    return out


def act_mul_quant_blockwise_fp8(
    gate_up: Tensor,
    out: Tensor | None = None,
    out_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """SiLU/multiply and dynamically quantize every 128-element block."""

    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2:
        raise ValueError("gate_up must be BF16 [pairs, 2 * intermediate]")
    pairs, doubled = gate_up.shape
    if doubled % 2:
        raise ValueError("gate_up's last dimension must be even")
    intermediate = doubled // 2
    if intermediate % 128:
        raise ValueError("blockwise activation requires intermediate multiple of 128")
    scale_shape = (pairs, intermediate // 128)
    if out is None:
        out = torch.empty(
            (pairs, intermediate), dtype=torch.float8_e4m3fn, device=gate_up.device
        )
    elif out.shape != (pairs, intermediate) or out.dtype != torch.float8_e4m3fn:
        raise ValueError(f"out must be E4M3 {(pairs, intermediate)}")
    if out_scale is None:
        out_scale = torch.empty(scale_shape, dtype=torch.float32, device=gate_up.device)
    elif out_scale.shape != scale_shape or out_scale.dtype != torch.float32:
        raise ValueError(f"out_scale must be FP32 {scale_shape}")
    if intermediate == 256 and pairs <= 1024:
        _act_mul_quant_blockwise_256_kernel[(pairs,)](
            gate_up,
            out,
            out_scale,
            num_warps=4,
        )
    else:
        _act_mul_quant_blockwise_kernel[(pairs, intermediate // 128)](
            gate_up,
            out,
            out_scale,
            intermediate,
            BLOCK=128,
            num_warps=4,
        )
    return out, out_scale


def weighted_reduce_pairs(
    down: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    experts_local: int,
    shared_output: Tensor | None,
    output: Tensor | None,
    all_routes_local: bool = False,
) -> Tensor:
    tokens, topk = topk_ids.shape
    hidden = down.shape[1]
    if output is None:
        output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=down.device)
    if tokens > 128:
        block = 1024
        num_warps = 8
    else:
        block = 256
        num_warps = 4
    placeholder = down if shared_output is None else shared_output
    _weighted_reduce_kernel[(tokens, triton.cdiv(hidden, block))](
        down,
        topk_ids,
        topk_scale,
        placeholder,
        output,
        hidden,
        topk,
        rank_ep * experts_local,
        (rank_ep + 1) * experts_local,
        ALL_ROUTES_LOCAL=all_routes_local,
        HAS_SHARED=shared_output is not None,
        BLOCK=block,
        num_warps=num_warps,
    )
    return output
