"""K-grouped tiny-decode Gate/Up projection for blockwise FP8 MoE."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


def choose_kgroup_gate_num_groups(
    num_pairs: int,
    n: int,
    k: int,
    sm_count: int,
) -> int:
    """Choose a legal K split that fills the device without shape pinning.

    ``1`` means that the ordinary direct grouped GEMM already exposes enough
    CTAs, or that the shape cannot be split into useful equal K128 groups.  A
    producer keeps 4--16 K blocks (target 8); shorter groups spend too much on
    the partial/reduce round trip, while longer groups under-fill tiny decode.
    """

    # The ordinary direct path uses BN=32/stages=1 for short K.  At K<=1024,
    # the smallest useful split has no more CTAs and only adds partial traffic.
    if (
        num_pairs <= 0
        # The K-split producer is only profitable for the very smallest route
        # sets. P=40 and above use the direct unsplit grouped GEMM instead.
        or num_pairs > 32
        or n <= 0
        or k <= 0
        or sm_count <= 0
        or n % 128
        or k % 128
        or k <= 1024
    ):
        return 1

    num_k_blocks = k // 128
    num_n_tiles = triton.cdiv(n, 64)
    base_ctas = num_pairs * num_n_tiles
    target_ctas = sm_count * 4
    # At 75% of the four-CTA/SM envelope, another global partial/reduce pass is
    # not worth the marginal occupancy increase.
    if base_ctas * 4 >= target_ctas * 3:
        return 1

    # Bound producer oversubscription: measured small-P shapes can benefit up to
    # roughly 2.5x the residency envelope, while larger explosions only add
    # global partial traffic and reducer work.
    divisors = [
        groups
        for groups in range(2, min(16, num_k_blocks) + 1)
        if num_k_blocks % groups == 0
        and 4 <= num_k_blocks // groups <= 16
        and base_ctas * groups * 2 <= target_ctas * 5
    ]
    if not divisors:
        return 1

    occupancy_groups = min(16, triton.cdiv(target_ctas, base_ctas))
    compute_groups = max(2, min(16, round(num_k_blocks / 8)))
    desired = max(occupancy_groups, compute_groups)
    return min(
        divisors,
        key=lambda groups: (
            abs(groups - desired),
            groups < desired,
            groups,
        ),
    )


@triton.jit
def _blockwise_fp8_kgroup_gate_partial_kernel(
    x,
    weight,
    partials,
    x_scale,
    weight_scale,
    topk_ids,
    expert_start,
    num_experts: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    topk: tl.constexpr,
    num_k_blocks: tl.constexpr,
    num_groups: tl.constexpr,
    blocks_per_group: tl.constexpr,
    num_n_blocks: tl.constexpr,
    scale_k_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one contiguous K group for a ``(route pair, N tile)``."""

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    group = pid % num_groups
    tile = pid // num_groups
    pid_n = tile % num_pid_n
    pair = tile // num_pid_n
    input_row = pair // topk
    expert = tl.load(topk_ids + pair) - expert_start
    valid_expert = (expert >= 0) & (expert < num_experts)

    row = tl.arange(0, BLOCK_M)
    valid_row = row == 0
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for local_block in range(0, blocks_per_group):
        scale_k_block = group * blocks_per_group + local_block
        offs_k = scale_k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        a_tile = tl.load(
            x + (input_row + row[:, None] * 0) * k + offs_k[None, :],
            mask=valid_row[:, None],
            other=0.0,
        )
        b_tile = tl.load(
            weight + expert * n * k + offs_n[None, :] * k + offs_k[:, None],
            mask=valid_expert & (offs_n[None, :] < n),
            other=0.0,
        )
        partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        partial = tl.dot(a_tile, b_tile, acc=partial)
        row_scale = tl.load(x_scale + input_row * num_k_blocks + scale_k_block)
        scale_n_block = pid_n * BLOCK_N // 128
        block_scale = tl.load(
            weight_scale
            + expert * num_n_blocks * scale_k_stride
            + scale_n_block * scale_k_stride
            + scale_k_block,
            mask=valid_expert,
            other=0.0,
        )
        # Keep the same left-associative scale order as the staged GEMM.
        acc += partial * row_scale * block_scale

    partial_ptrs = (
        partials
        + pair * num_groups * n
        + group * n
        + row[:, None] * 0
        + offs_n[None, :]
    )
    tl.store(
        partial_ptrs,
        acc,
        mask=valid_row[:, None] & (offs_n[None, :] < n),
    )


@triton.jit
def _blockwise_fp8_kgroup_gate_reduce_kernel(
    partials,
    output,
    n: tl.constexpr,
    num_groups: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce contiguous K-group partials and apply the BF16 checkpoint."""

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    pair = pid // num_pid_n
    pid_n = pid - pair * num_pid_n
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for group in range(0, num_groups):
        acc += tl.load(
            partials + pair * num_groups * n + group * n + offs_n,
            mask=offs_n < n,
            other=0.0,
        )
    tl.store(
        output + pair * n + offs_n,
        acc.to(tl.bfloat16),
        mask=offs_n < n,
    )


@triton.jit
def _blockwise_fp8_kgroup_gate_act_quant_kernel(
    partials,
    output,
    output_scale,
    intermediate: tl.constexpr,
    num_groups: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Finalize K groups directly into blockwise-quantized Down input."""

    pair = tl.program_id(0)
    block = tl.program_id(1)
    column = block * BLOCK + tl.arange(0, BLOCK)
    mask = column < intermediate
    doubled_intermediate = 2 * intermediate
    gate = tl.zeros((BLOCK,), dtype=tl.float32)
    up = tl.zeros((BLOCK,), dtype=tl.float32)
    for group in range(0, num_groups):
        group_base = (
            pair * num_groups * doubled_intermediate + group * doubled_intermediate
        )
        gate += tl.load(
            partials + group_base + column,
            mask=mask,
            other=0.0,
        )
        up += tl.load(
            partials + group_base + intermediate + column,
            mask=mask,
            other=0.0,
        )

    # Match the staged path: Gate/Up round to BF16 once after the complete
    # K reduction, then SwiGLU is evaluated in FP32 before K128 quantization.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    activated = gate * tl.sigmoid(gate) * up
    abs_max = tl.max(tl.where(mask, tl.abs(activated), 0.0), axis=0)
    scale = abs_max / 448.0
    inverse_scale = 1.0 / (scale + 1.0e-8)
    tl.store(output_scale + pair * tl.cdiv(intermediate, BLOCK) + block, scale)
    tl.store(
        output + pair * intermediate + column,
        activated * inverse_scale,
        mask=mask,
    )


def _prepare_kgroup_gate(
    x: Tensor,
    x_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    topk_ids: Tensor,
    *,
    expert_start: int,
    num_groups: int,
    partials: Tensor | None = None,
) -> tuple[int, int, int, int, int, int, int, Tensor]:
    """Validate a direct K-group launch and prepare its FP32 workspace."""

    if x.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        raise TypeError("x and weight must use float8_e4m3fn")
    if x.ndim != 2 or weight.ndim != 3 or x.shape[1] != weight.shape[2]:
        raise ValueError("expected x=[tokens,K] and weight=[experts,N,K]")
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [tokens, topk]")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous int32 [tokens, topk]")
    for name, value in (
        ("weight", weight),
        ("x_scale", x_scale),
        ("weight_scale", weight_scale),
        ("topk_ids", topk_ids),
    ):
        if value.device != x.device:
            raise ValueError(f"{name} must be on {x.device}")

    tokens, k = x.shape
    experts, n, _ = weight.shape
    num_pairs = topk_ids.numel()
    topk = topk_ids.shape[1]
    if topk_ids.shape[0] != tokens:
        raise ValueError("x and topk_ids must have the same token count")
    if tokens <= 0 or topk <= 0:
        raise ValueError("topk_ids must contain at least one route pair")
    if k % 128 or n % 128 or num_pairs > 32:
        raise ValueError(
            "K-group Gate requires K/N multiples of 128 and at most 32 pairs"
        )

    num_k_blocks = k // 128
    num_n_blocks = n // 128
    if not 2 <= num_groups <= min(16, num_k_blocks):
        raise ValueError("num_groups must be in [2, min(16, K / 128)]")
    blocks_per_group = num_k_blocks // num_groups
    if num_k_blocks % num_groups:
        raise ValueError("the K128 block count must be divisible by num_groups")
    if not 4 <= blocks_per_group <= 32:
        raise ValueError("each K group must contain between 4 and 32 K128 blocks")
    if (
        x_scale.shape != (tokens, num_k_blocks)
        or x_scale.dtype != torch.float32
        or not x_scale.is_contiguous()
    ):
        raise ValueError(f"x_scale must be contiguous FP32 {(tokens, num_k_blocks)}")
    expected_weight_scale = (experts, num_n_blocks, num_k_blocks)
    if (
        weight_scale.shape != expected_weight_scale
        or weight_scale.dtype != torch.float32
        or not weight_scale.is_contiguous()
    ):
        raise ValueError(
            f"weight_scale must be contiguous FP32 {expected_weight_scale}"
        )
    expected_partials = (num_pairs, num_groups, n)
    if partials is None:
        partials = torch.empty(expected_partials, dtype=torch.float32, device=x.device)
    elif (
        partials.shape != expected_partials
        or partials.dtype != torch.float32
        or partials.device != x.device
        or not partials.is_contiguous()
    ):
        raise ValueError(f"partials must be FP32 {expected_partials}")

    return (
        n,
        k,
        num_pairs,
        topk,
        num_k_blocks,
        num_n_blocks,
        blocks_per_group,
        partials,
    )


def _launch_kgroup_gate_partials(
    x: Tensor,
    x_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    topk_ids: Tensor,
    partials: Tensor,
    *,
    expert_start: int,
    num_groups: int,
    n: int,
    k: int,
    num_pairs: int,
    topk: int,
    num_k_blocks: int,
    num_n_blocks: int,
    blocks_per_group: int,
) -> None:
    """Launch the shared FP32 K-group producer."""

    block_n = 64
    num_n_tiles = triton.cdiv(n, block_n)
    _blockwise_fp8_kgroup_gate_partial_kernel[(num_pairs * num_n_tiles * num_groups,)](
        x,
        weight,
        partials,
        x_scale,
        weight_scale,
        topk_ids,
        expert_start,
        weight.shape[0],
        n,
        k,
        topk,
        num_k_blocks,
        num_groups,
        blocks_per_group,
        num_n_blocks,
        weight_scale.shape[2],
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=4,
        num_stages=4,
    )


def triton_blockwise_kgroup_gate_direct(
    x: Tensor,
    x_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    topk_ids: Tensor,
    *,
    expert_start: int,
    num_groups: int,
    out: Tensor | None = None,
    partials: Tensor | None = None,
) -> Tensor:
    """Run the shape-aware SM120 tiny-decode Gate/Up K-group schedule."""

    (
        n,
        k,
        num_pairs,
        topk,
        num_k_blocks,
        num_n_blocks,
        blocks_per_group,
        partials,
    ) = _prepare_kgroup_gate(
        x,
        x_scale,
        weight,
        weight_scale,
        topk_ids,
        expert_start=expert_start,
        num_groups=num_groups,
        partials=partials,
    )
    if out is None:
        out = torch.empty((num_pairs, n), dtype=torch.bfloat16, device=x.device)
    elif (
        out.shape != (num_pairs, n)
        or out.dtype != torch.bfloat16
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be BF16 {(num_pairs, n)}")

    _launch_kgroup_gate_partials(
        x,
        x_scale,
        weight,
        weight_scale,
        topk_ids,
        partials,
        expert_start=expert_start,
        num_groups=num_groups,
        n=n,
        k=k,
        num_pairs=num_pairs,
        topk=topk,
        num_k_blocks=num_k_blocks,
        num_n_blocks=num_n_blocks,
        blocks_per_group=blocks_per_group,
    )
    block_n = 64
    num_n_tiles = triton.cdiv(n, block_n)
    _blockwise_fp8_kgroup_gate_reduce_kernel[(num_pairs * num_n_tiles,)](
        partials,
        out,
        n,
        num_groups,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


def triton_blockwise_kgroup_gate_act_quant_direct(
    x: Tensor,
    x_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    topk_ids: Tensor,
    *,
    expert_start: int,
    num_groups: int,
    out: Tensor | None = None,
    out_scale: Tensor | None = None,
    partials: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run K-group Gate/Up and finalize directly to blockwise FP8."""

    (
        n,
        k,
        num_pairs,
        topk,
        num_k_blocks,
        num_n_blocks,
        blocks_per_group,
        partials,
    ) = _prepare_kgroup_gate(
        x,
        x_scale,
        weight,
        weight_scale,
        topk_ids,
        expert_start=expert_start,
        num_groups=num_groups,
        partials=partials,
    )
    intermediate = n // 2
    num_intermediate_blocks = triton.cdiv(intermediate, 128)
    if out is None:
        out = torch.empty(
            (num_pairs, intermediate),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
    elif (
        out.shape != (num_pairs, intermediate)
        or out.dtype != torch.float8_e4m3fn
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be E4M3 {(num_pairs, intermediate)}")
    expected_scale = (num_pairs, num_intermediate_blocks)
    if out_scale is None:
        out_scale = torch.empty(expected_scale, dtype=torch.float32, device=x.device)
    elif (
        out_scale.shape != expected_scale
        or out_scale.dtype != torch.float32
        or out_scale.device != x.device
        or not out_scale.is_contiguous()
    ):
        raise ValueError(f"out_scale must be FP32 {expected_scale}")

    _launch_kgroup_gate_partials(
        x,
        x_scale,
        weight,
        weight_scale,
        topk_ids,
        partials,
        expert_start=expert_start,
        num_groups=num_groups,
        n=n,
        k=k,
        num_pairs=num_pairs,
        topk=topk,
        num_k_blocks=num_k_blocks,
        num_n_blocks=num_n_blocks,
        blocks_per_group=blocks_per_group,
    )
    _blockwise_fp8_kgroup_gate_act_quant_kernel[(num_pairs, num_intermediate_blocks)](
        partials,
        out,
        out_scale,
        intermediate,
        num_groups,
        BLOCK=128,
        num_warps=4,
    )
    return out, out_scale
