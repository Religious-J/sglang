"""Route-aware E4M3 grouped GEMM kernels for blockwise FP8 MoE.

The alignment pass produces a compact expert-major task list without a
device-to-host synchronization.  The GEMM writes results back in original
route-pair order, so the same alignment can be reused by Gate-Up and Down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import triton
import triton.language as tl
from torch import Tensor


@dataclass(frozen=True)
class RouteAlignment:
    sorted_pair_ids: Tensor
    expert_ids: Tensor
    num_pairs_post_pad: Tensor
    block_m: int
    num_pairs: int
    topk: int
    direct_topk_ids: Tensor | None = None
    expert_start: int = 0


@triton.jit
def _prepare_routes_histogram_kernel(
    topk_ids,
    cursors,
    padded_offsets,
    sorted_pair_ids,
    num_pairs_post_pad,
    num_pairs: tl.constexpr,
    expert_start,
    num_experts: tl.constexpr,
    block_m: tl.constexpr,
    pair_sentinel: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
):
    pair = tl.arange(0, PAIR_BLOCK)
    expert = tl.load(topk_ids + pair, mask=pair < num_pairs, other=-1)
    local_expert = expert - expert_start
    valid = (pair < num_pairs) & (local_expert >= 0) & (local_expert < num_experts)
    count = tl.histogram(
        tl.where(valid, local_expert, 0),
        EXPERT_BLOCK,
        mask=valid,
    )
    padded = tl.cdiv(count, block_m) * block_m
    inclusive = tl.cumsum(padded, axis=0)
    exclusive = inclusive - padded
    expert_offset = tl.arange(0, EXPERT_BLOCK)
    expert_mask = expert_offset < num_experts
    tl.store(cursors + expert_offset, exclusive, mask=expert_mask)
    tl.store(padded_offsets + expert_offset, exclusive, mask=expert_mask)
    total = tl.sum(padded, axis=0)
    tl.store(padded_offsets + num_experts, total)
    tl.store(num_pairs_post_pad, total)

    padding_count = padded - count
    for padding in tl.static_range(block_m):
        tl.store(
            sorted_pair_ids + exclusive + count + padding,
            pair_sentinel,
            mask=expert_mask & (padding < padding_count),
        )


@triton.jit
def _align_routes_histogram_single_kernel(
    topk_ids,
    cursors,
    padded_offsets,
    sorted_pair_ids,
    expert_ids,
    num_pairs_post_pad,
    num_pairs: tl.constexpr,
    expert_start,
    num_experts: tl.constexpr,
    block_m: tl.constexpr,
    pair_sentinel: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
):
    pair = tl.arange(0, PAIR_BLOCK)
    expert = tl.load(topk_ids + pair, mask=pair < num_pairs, other=-1)
    local_expert = expert - expert_start
    valid = (pair < num_pairs) & (local_expert >= 0) & (local_expert < num_experts)
    count = tl.histogram(
        tl.where(valid, local_expert, 0),
        EXPERT_BLOCK,
        mask=valid,
    )
    padded = tl.cdiv(count, block_m) * block_m
    inclusive = tl.cumsum(padded, axis=0)
    exclusive = inclusive - padded
    expert_offset = tl.arange(0, EXPERT_BLOCK)
    expert_mask = expert_offset < num_experts
    tl.store(padded_offsets + expert_offset, exclusive, mask=expert_mask)
    total = tl.sum(padded, axis=0)
    tl.store(padded_offsets + num_experts, total)
    tl.store(num_pairs_post_pad, total)

    padding_count = padded - count
    for padding in tl.static_range(block_m):
        tl.store(
            sorted_pair_ids + exclusive + count + padding,
            pair_sentinel,
            mask=expert_mask & (padding < padding_count),
        )

    tl.store(cursors + expert_offset, exclusive, mask=expert_mask)
    tl.debug_barrier()
    safe_expert = tl.where(valid, local_expert, 0)
    slot = tl.atomic_add(cursors + safe_expert, 1, mask=valid)
    base = tl.load(padded_offsets + safe_expert, mask=valid, other=0)
    tl.store(sorted_pair_ids + slot, pair, mask=valid)
    tl.store(
        expert_ids + slot // block_m,
        safe_expert,
        mask=valid & ((slot - base) % block_m == 0),
    )


@triton.jit
def _scatter_routes_absolute_kernel(
    topk_ids,
    cursors,
    sorted_pair_ids,
    expert_ids,
    num_pairs: tl.constexpr,
    expert_start,
    num_experts: tl.constexpr,
    block_m: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(topk_ids + pair, mask=pair < num_pairs, other=-1)
    local_expert = expert - expert_start
    valid = (pair < num_pairs) & (local_expert >= 0) & (local_expert < num_experts)
    safe_expert = tl.where(valid, local_expert, 0)
    slot = tl.atomic_add(cursors + safe_expert, 1, mask=valid)
    tl.store(sorted_pair_ids + slot, pair, mask=valid)
    tl.store(
        expert_ids + slot // block_m,
        safe_expert,
        mask=valid & (slot % block_m == 0),
    )


@triton.jit
def _zero_route_counts_kernel(
    counts,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(counts + offset, 0, mask=offset < num_experts)


@triton.jit
def _count_routes_kernel(
    topk_ids,
    counts,
    num_pairs: tl.constexpr,
    expert_start,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(topk_ids + pair, mask=pair < num_pairs, other=-1)
    local_expert = expert - expert_start
    valid = (pair < num_pairs) & (local_expert >= 0) & (local_expert < num_experts)
    safe_expert = tl.where(valid, local_expert, 0)
    tl.atomic_add(counts + safe_expert, 1, mask=valid)


@triton.jit
def _prefix_routes_kernel(
    counts,
    cursors,
    padded_offsets,
    sorted_pair_ids,
    num_pairs_post_pad,
    num_experts: tl.constexpr,
    block_m: tl.constexpr,
    pair_sentinel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.arange(0, BLOCK)
    expert_mask = expert < num_experts
    count = tl.load(counts + expert, mask=expert_mask, other=0)
    padded = tl.cdiv(count, block_m) * block_m
    inclusive = tl.cumsum(padded, axis=0)
    exclusive = inclusive - padded
    tl.store(padded_offsets + expert, exclusive, mask=expert_mask)
    total = tl.sum(padded, axis=0)
    tl.store(padded_offsets + num_experts, total)
    tl.store(num_pairs_post_pad, total)

    # Only the padding inside the observable prefix needs a sentinel.  Valid
    # route slots are populated by scatter, while capacity beyond ``total``
    # is rejected by GEMM before either route array is loaded.
    padding_count = padded - count
    for padding in tl.static_range(block_m):
        tl.store(
            sorted_pair_ids + exclusive + count + padding,
            pair_sentinel,
            mask=expert_mask & (padding < padding_count),
        )
    tl.store(cursors + expert, 0, mask=expert_mask)


@triton.jit
def _scatter_routes_kernel(
    topk_ids,
    padded_offsets,
    cursors,
    sorted_pair_ids,
    expert_ids,
    num_pairs: tl.constexpr,
    expert_start,
    num_experts: tl.constexpr,
    block_m: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(topk_ids + pair, mask=pair < num_pairs, other=-1)
    local_expert = expert - expert_start
    valid = (pair < num_pairs) & (local_expert >= 0) & (local_expert < num_experts)
    safe_expert = tl.where(valid, local_expert, 0)
    local_row = tl.atomic_add(cursors + safe_expert, 1, mask=valid)
    base = tl.load(padded_offsets + safe_expert, mask=valid, other=0)
    tl.store(sorted_pair_ids + base + local_row, pair, mask=valid)
    tl.store(
        expert_ids + (base + local_row) // block_m,
        safe_expert,
        mask=valid & (local_row % block_m == 0),
    )


@triton.jit
def _fp8_grouped_gemm_kernel(
    a,
    b,
    c,
    b_scale,
    sorted_pair_ids,
    expert_ids,
    num_pairs_post_pad,
    num_pairs: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    padded = tl.load(num_pairs_post_pad)
    if pid_m * BLOCK_M >= padded:
        return

    pair_slot = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_pair_ids + pair_slot)
    valid_pair = pair < num_pairs
    input_row = pair // topk
    expert = tl.load(expert_ids + pid_m)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a + input_row[:, None] * k + offs_k[None, :]
    b_ptrs = b + expert * n * k + offs_n[None, :] * k + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        a_tile = tl.load(
            a_ptrs,
            mask=valid_pair[:, None] & (offs_k[None, :] < k - k_start),
            other=0.0,
        )
        b_tile = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < n) & (offs_k[:, None] < k - k_start),
            other=0.0,
        )
        acc = tl.dot(a_tile, b_tile, acc=acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    scale = tl.load(b_scale + expert)
    acc *= scale
    c_ptrs = c + pair[:, None] * n + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(tl.bfloat16),
        mask=valid_pair[:, None] & (offs_n[None, :] < n),
    )


@triton.jit
def _fp8_blockwise_grouped_gemm_kernel(
    a,
    b,
    c,
    a_scale,
    b_scale,
    sorted_pair_ids,
    expert_ids,
    num_pairs_post_pad,
    num_pairs: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    topk: tl.constexpr,
    num_k_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    b_scale_k_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    padded = tl.load(num_pairs_post_pad)
    if pid_m * BLOCK_M >= padded:
        return

    pair_slot = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_pair_ids + pair_slot)
    valid_pair = pair < num_pairs
    input_row = pair // topk
    expert = tl.load(expert_ids + pid_m)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a + input_row[:, None] * k + offs_k[None, :]
    b_ptrs = b + expert * n * k + offs_n[None, :] * k + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for scale_k_block in range(0, num_k_blocks):
        partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, 128, BLOCK_K):
            a_tile = tl.load(
                a_ptrs,
                mask=valid_pair[:, None],
                other=0.0,
            )
            b_tile = tl.load(
                b_ptrs,
                mask=offs_n[None, :] < n,
                other=0.0,
            )
            partial = tl.dot(a_tile, b_tile, acc=partial)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        row_scale = tl.load(
            a_scale + input_row * num_k_blocks + scale_k_block,
            mask=valid_pair,
            other=0.0,
        )
        scale_n_block = pid_n * BLOCK_N // 128
        weight_scale = tl.load(
            b_scale
            + expert * num_n_blocks * b_scale_k_stride
            + scale_n_block * b_scale_k_stride
            + scale_k_block
        )
        acc += partial * row_scale[:, None] * weight_scale

    c_ptrs = c + pair[:, None] * n + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(tl.bfloat16),
        mask=valid_pair[:, None] & (offs_n[None, :] < n),
    )


@triton.jit
def _fp8_blockwise_direct_gemm_kernel(
    a,
    b,
    c,
    a_scale,
    b_scale,
    topk_ids,
    num_pairs: tl.constexpr,
    expert_start,
    num_experts: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    topk: tl.constexpr,
    num_k_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    b_scale_k_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    pair = pid // num_pid_n
    pid_n = pid - pair * num_pid_n
    row = tl.arange(0, BLOCK_M)
    valid_row = row == 0
    input_row = pair // topk
    expert = tl.load(topk_ids + pair) - expert_start
    valid_expert = (expert >= 0) & (expert < num_experts)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a + (input_row + row[:, None] * 0) * k + offs_k[None, :]
    b_ptrs = b + expert * n * k + offs_n[None, :] * k + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for scale_k_block in range(0, num_k_blocks):
        partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, 128, BLOCK_K):
            a_tile = tl.load(a_ptrs, mask=valid_row[:, None], other=0.0)
            b_tile = tl.load(
                b_ptrs,
                mask=valid_expert & (offs_n[None, :] < n),
                other=0.0,
            )
            partial = tl.dot(a_tile, b_tile, acc=partial)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        row_scale = tl.load(a_scale + input_row * num_k_blocks + scale_k_block)
        scale_n_block = pid_n * BLOCK_N // 128
        weight_scale = tl.load(
            b_scale
            + expert * num_n_blocks * b_scale_k_stride
            + scale_n_block * b_scale_k_stride
            + scale_k_block,
            mask=valid_expert,
            other=0.0,
        )
        acc += partial * row_scale * weight_scale

    c_ptrs = c + pair * n + row[:, None] * 0 + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(tl.bfloat16),
        mask=valid_row[:, None] & (offs_n[None, :] < n),
    )


def choose_block_m(num_pairs: int, num_experts: int) -> int:
    average = (num_pairs + num_experts - 1) // num_experts
    # SM120 measurements favor a very small tile only for sparse decode.  As
    # occupancy rises, wider M tiles make the short-K Down projection notably
    # more efficient even before every row in the tile is populated.
    if average <= 2:
        return 16
    if average <= 16:
        return 32
    return 64


def choose_blockwise_block_m(num_pairs: int, num_experts: int) -> int:
    """Match HPC-Ops' smaller blockwise M buckets within Triton's limits."""

    average = (num_pairs + num_experts - 1) // num_experts
    if average <= 16:
        return 16
    if average <= 32:
        return 32
    return 64


def align_routes(
    topk_ids: Tensor,
    num_experts: int,
    rank_ep: int,
    block_m: int | None = None,
    direct: bool = False,
) -> RouteAlignment:
    """Build a compact local-expert task list entirely on the current stream."""

    if (
        topk_ids.dtype != torch.int32
        or topk_ids.ndim != 2
        or not topk_ids.is_contiguous()
    ):
        raise ValueError("topk_ids must be contiguous int32 [tokens, topk]")
    num_pairs = topk_ids.numel()
    topk = topk_ids.shape[1]
    block_m = block_m or choose_block_m(num_pairs, num_experts)
    if block_m not in (16, 32, 64):
        raise ValueError(f"block_m must be 16, 32, or 64, got {block_m}")

    device = topk_ids.device
    if direct:
        placeholder = torch.empty(0, dtype=torch.int32, device=device)
        return RouteAlignment(
            placeholder,
            placeholder,
            placeholder,
            block_m,
            num_pairs,
            topk,
            topk_ids,
            rank_ep * num_experts,
        )

    cursors = torch.empty(num_experts, dtype=torch.int32, device=device)
    padded_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    num_pairs_post_pad = torch.empty(1, dtype=torch.int32, device=device)

    # Same safe capacity bound used by established Triton MoE aligners.
    if num_pairs < num_experts + 1:
        capacity = max(block_m, num_pairs * block_m)
    else:
        capacity = num_pairs + (num_experts + 1) * (block_m - 1)
    num_tile_slots = triton.cdiv(capacity, block_m)
    sorted_pair_ids = torch.empty(
        (num_tile_slots * block_m,), dtype=torch.int32, device=device
    )
    expert_ids = torch.empty((num_tile_slots,), dtype=torch.int32, device=device)

    if num_experts <= 256 and num_pairs <= 1024:
        _align_routes_histogram_single_kernel[(1,)](
            topk_ids,
            cursors,
            padded_offsets,
            sorted_pair_ids,
            expert_ids,
            num_pairs_post_pad,
            num_pairs,
            rank_ep * num_experts,
            num_experts,
            block_m,
            num_pairs,
            EXPERT_BLOCK=triton.next_power_of_2(num_experts),
            PAIR_BLOCK=max(256, triton.next_power_of_2(num_pairs)),
            num_warps=8,
        )
        return RouteAlignment(
            sorted_pair_ids,
            expert_ids,
            num_pairs_post_pad,
            block_m,
            num_pairs,
            topk,
        )

    if num_experts <= 256 and num_pairs <= 4096:
        expert_block = triton.next_power_of_2(num_experts)
        pair_block = max(256, triton.next_power_of_2(num_pairs))
        _prepare_routes_histogram_kernel[(1,)](
            topk_ids,
            cursors,
            padded_offsets,
            sorted_pair_ids,
            num_pairs_post_pad,
            num_pairs,
            rank_ep * num_experts,
            num_experts,
            block_m,
            num_pairs,
            EXPERT_BLOCK=expert_block,
            PAIR_BLOCK=pair_block,
            num_warps=8,
        )
        route_block = 256
        _scatter_routes_absolute_kernel[(triton.cdiv(num_pairs, route_block),)](
            topk_ids,
            cursors,
            sorted_pair_ids,
            expert_ids,
            num_pairs,
            rank_ep * num_experts,
            num_experts,
            block_m,
            BLOCK=route_block,
        )
        return RouteAlignment(
            sorted_pair_ids,
            expert_ids,
            num_pairs_post_pad,
            block_m,
            num_pairs,
            topk,
        )

    counts = torch.empty(num_experts, dtype=torch.int32, device=device)
    route_block = 256
    _zero_route_counts_kernel[(triton.cdiv(num_experts, route_block),)](
        counts,
        num_experts,
        BLOCK=route_block,
    )
    _count_routes_kernel[(triton.cdiv(num_pairs, route_block),)](
        topk_ids,
        counts,
        num_pairs,
        rank_ep * num_experts,
        num_experts,
        BLOCK=route_block,
    )
    prefix_block = triton.next_power_of_2(num_experts)
    _prefix_routes_kernel[(1,)](
        counts,
        cursors,
        padded_offsets,
        sorted_pair_ids,
        num_pairs_post_pad,
        num_experts,
        block_m,
        num_pairs,
        BLOCK=prefix_block,
        num_warps=1,
    )
    _scatter_routes_kernel[(triton.cdiv(num_pairs, route_block),)](
        topk_ids,
        padded_offsets,
        cursors,
        sorted_pair_ids,
        expert_ids,
        num_pairs,
        rank_ep * num_experts,
        num_experts,
        block_m,
        BLOCK=route_block,
    )
    return RouteAlignment(
        sorted_pair_ids,
        expert_ids,
        num_pairs_post_pad,
        block_m,
        num_pairs,
        topk,
    )


def triton_grouped_gemm(
    a: Tensor,
    b: Tensor,
    b_scale: Tensor,
    alignment: RouteAlignment,
    *,
    topk: int,
    out: Tensor | None = None,
) -> Tensor:
    """Run route-aware ``E4M3 @ E4M3.T`` and return pair-major BF16 rows."""

    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise TypeError("a and b must use float8_e4m3fn")
    if a.ndim != 2 or b.ndim != 3 or a.shape[1] != b.shape[2]:
        raise ValueError(f"expected a=[rows,K], b=[E,N,K], got {a.shape}, {b.shape}")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")
    experts, n, k = b.shape
    if b_scale.shape != (experts,) or b_scale.dtype != torch.float32:
        raise ValueError(f"b_scale must be FP32 [{experts}]")
    if topk not in (1, alignment.topk):
        raise ValueError(f"topk must be 1 or {alignment.topk}, got {topk}")
    if out is None:
        out = torch.empty(
            (alignment.num_pairs, n), dtype=torch.bfloat16, device=a.device
        )
    elif out.shape != (alignment.num_pairs, n) or out.dtype != torch.bfloat16:
        raise ValueError(
            f"out must be BF16 with shape {(alignment.num_pairs, n)}, got {out.shape}"
        )

    block_n = 128
    block_k = 256
    grid = (alignment.expert_ids.numel() * triton.cdiv(n, block_n),)
    _fp8_grouped_gemm_kernel[grid](
        a,
        b,
        out,
        b_scale,
        alignment.sorted_pair_ids,
        alignment.expert_ids,
        alignment.num_pairs_post_pad,
        alignment.num_pairs,
        n,
        k,
        topk,
        BLOCK_M=alignment.block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    return out


def triton_blockwise_grouped_gemm(
    a: Tensor,
    a_scale: Tensor,
    b: Tensor,
    b_scale: Tensor,
    alignment: RouteAlignment,
    *,
    topk: int,
    out: Tensor | None = None,
    tuning_config: Mapping[str, Any] | None = None,
) -> Tensor:
    """Run block-scaled E4M3 grouped GEMM with FP32 partial accumulation.

    Activation scales are per input row and K=128 block. Weight scales are
    per expert, N=128 block, and K=128 block. Each K-block dot product is
    scaled before being added to the FP32 accumulator, matching HPC-Ops'
    serialized block-wise computation.
    """

    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise TypeError("a and b must use float8_e4m3fn")
    if a.ndim != 2 or b.ndim != 3 or a.shape[1] != b.shape[2]:
        raise ValueError(f"expected a=[rows,K], b=[E,N,K], got {a.shape}, {b.shape}")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")
    experts, n, k = b.shape
    if n % 128 or k % 128:
        raise ValueError("blockwise grouped GEMM requires N and K multiples of 128")
    num_k_blocks = k // 128
    num_n_blocks = n // 128
    if (
        a_scale.shape != (a.shape[0], num_k_blocks)
        or a_scale.dtype != torch.float32
        or not a_scale.is_contiguous()
    ):
        raise ValueError(
            f"a_scale must be contiguous FP32 {(a.shape[0], num_k_blocks)}"
        )
    if (
        b_scale.shape != (experts, num_n_blocks, num_k_blocks)
        or b_scale.dtype != torch.float32
        or not b_scale.is_contiguous()
    ):
        raise ValueError(
            "b_scale must be contiguous FP32 "
            f"{(experts, num_n_blocks, num_k_blocks)}"
        )
    if topk not in (1, alignment.topk):
        raise ValueError(f"topk must be 1 or {alignment.topk}, got {topk}")
    if out is None:
        out = torch.empty(
            (alignment.num_pairs, n), dtype=torch.bfloat16, device=a.device
        )
    elif out.shape != (alignment.num_pairs, n) or out.dtype != torch.bfloat16:
        raise ValueError(
            f"out must be BF16 with shape {(alignment.num_pairs, n)}, got {out.shape}"
        )

    # SM120 favors a shorter K tile for the wide, short-K Down projection.
    # Keep the decode path unchanged: at eight route pairs its lower launch
    # and warp cost is more important than the extra occupancy.
    if alignment.direct_topk_ids is not None and k > 1024:
        block_n = 64
        block_k = 128
        num_warps = 4
        num_stages = 4
    elif alignment.direct_topk_ids is not None:
        block_n = 32
        block_k = 128
        num_warps = 4
        # The decode Down projection has only two K=128 blocks.  A deeper
        # software pipeline adds barrier/register cost without hiding another
        # load wave on the measured sparse P<64 SM120 direct path.
        num_stages = 1
    elif (
        alignment.block_m == 64
        and k == 256
        and n == 7168
        and alignment.num_pairs >= 16384
    ):
        # Large DeepSeek batches favor fewer warps and one extra pipeline
        # stage.  This keeps the wide Down projection resident without the
        # scheduling overhead of the small-batch eight-warp configuration.
        block_n = 128
        block_k = 128
        num_warps = 4
        num_stages = 3
    elif (
        alignment.block_m == 64
        and k == 7168
        and n == 512
        and alignment.num_pairs >= 32768
    ):
        # At 4096 tokens each expert has enough rows to amortize the wider N
        # tile; halving the N-grid is faster than the decode-oriented BN=64.
        block_n = 128
        block_k = 128
        num_warps = 4
        num_stages = 3
    elif k <= 256 and n >= 1024 and alignment.num_pairs > 8:
        block_n = 128
        block_k = 128
        num_warps = (
            4 if alignment.num_pairs <= 1024 or alignment.num_pairs >= 8192 else 8
        )
        num_stages = 2
    elif k > 1024 and alignment.num_pairs > 8:
        block_n = 64
        block_k = 128
        num_warps = 4
        num_stages = 4
    else:
        block_n = 128
        block_k = 128
        num_warps = 4
        num_stages = 3

    tuning_config = tuning_config if isinstance(tuning_config, Mapping) else {}
    requested_values = (
        ("block_n", (32, 64, 128)),
        ("block_k", (64, 128)),
        ("num_warps", (2, 4, 8)),
        ("num_stages", (1, 2, 3, 4, 5)),
    )
    launch_config = {
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    for name, allowed in requested_values:
        value = tuning_config.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value in allowed:
            launch_config[name] = value
    block_n = launch_config["block_n"]
    block_k = launch_config["block_k"]
    num_warps = launch_config["num_warps"]
    num_stages = launch_config["num_stages"]

    if alignment.direct_topk_ids is not None:
        grid = alignment.num_pairs * triton.cdiv(n, block_n)
        _fp8_blockwise_direct_gemm_kernel[(grid,)](
            a,
            b,
            out,
            a_scale,
            b_scale,
            alignment.direct_topk_ids,
            alignment.num_pairs,
            alignment.expert_start,
            experts,
            n,
            k,
            topk,
            num_k_blocks,
            num_n_blocks,
            b_scale.shape[2],
            BLOCK_M=16,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    grid = alignment.expert_ids.numel() * triton.cdiv(n, block_n)
    _fp8_blockwise_grouped_gemm_kernel[(grid,)](
        a,
        b,
        out,
        a_scale,
        b_scale,
        alignment.sorted_pair_ids,
        alignment.expert_ids,
        alignment.num_pairs_post_pad,
        alignment.num_pairs,
        n,
        k,
        topk,
        num_k_blocks,
        num_n_blocks,
        b_scale.shape[2],
        BLOCK_M=alignment.block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
