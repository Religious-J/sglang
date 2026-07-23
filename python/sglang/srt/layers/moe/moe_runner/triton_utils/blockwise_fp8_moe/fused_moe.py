"""Triton blockwise FP8 implementation of the HPC-Ops fused MoE computation.

The public signatures intentionally mirror ``hpc/fuse_moe.py`` from HPC-Ops.
Both expert GEMMs use native E4M3 Triton Tensor Core kernels; routing,
activation/FP8 checkpoint, and weighted reduction stay on the GPU. The current
schedules are tuned for SM120.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from . import (
    act_mul_quant_blockwise_fp8,
    act_mul_quant_fp8,
    align_routes,
    choose_kgroup_gate_num_groups,
    triton_blockwise_grouped_gemm,
    triton_blockwise_kgroup_gate_act_quant_direct,
    triton_grouped_gemm,
    weighted_reduce_pairs,
)
from .triton_grouped_gemm import choose_block_m, choose_blockwise_block_m

_FP8 = torch.float8_e4m3fn


@lru_cache(maxsize=None)
def _device_sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _validate_common(
    x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
) -> tuple[int, int, int, int, int]:
    tensors = {
        "x": x,
        "gate_up_weight": gate_up_weight,
        "down_weight": down_weight,
        "topk_ids": topk_ids,
        "topk_scale": topk_scale,
    }
    for name, value in tensors.items():
        if value.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor")
        if value.device != x.device:
            raise ValueError(f"{name} must be on {x.device}, got {value.device}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if x.ndim != 2 or gate_up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError(
            "expected x=[M,H], gate_up_weight=[E,2I,H], down_weight=[E,H,I]"
        )
    if topk_ids.ndim != 2 or topk_scale.shape != topk_ids.shape:
        raise ValueError("topk_ids and topk_scale must have the same rank-2 shape")
    if topk_ids.dtype != torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if topk_scale.dtype != torch.float32:
        raise TypeError(f"topk_scale must be float32, got {topk_scale.dtype}")
    if x.shape[0] != topk_ids.shape[0]:
        raise ValueError("x and topk_ids must have the same token count")

    num_tokens, hidden = x.shape
    experts_local, gate_up_n, gate_up_k = gate_up_weight.shape
    intermediate = gate_up_n // 2
    if gate_up_n % 2 or gate_up_k != hidden:
        raise ValueError("gate_up_weight must have shape [E_local, 2*I, H]")
    if down_weight.shape != (experts_local, hidden, intermediate):
        raise ValueError(
            f"down_weight must have shape {(experts_local, hidden, intermediate)}, "
            f"got {tuple(down_weight.shape)}"
        )
    if min(hidden, intermediate) <= 0 or hidden % 64 or intermediate % 64:
        raise ValueError(
            "hidden_size and intermediate_size must be positive multiples of 64"
        )
    if num_expert_total <= 0 or num_expert_total % experts_local:
        raise ValueError("num_expert_total must be positive and divisible by E_local")
    ep_size = num_expert_total // experts_local
    if not 0 <= rank_ep < ep_size:
        raise ValueError(f"rank_ep must be in [0, {ep_size}), got {rank_ep}")
    return num_tokens, hidden, intermediate, experts_local, topk_ids.shape[1]


def _check_bf16_output(x: Tensor, value: Tensor | None, name: str) -> None:
    if value is None:
        return
    if value.device != x.device or value.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be a BF16 tensor on the same device as x")
    if value.shape != x.shape or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous with shape {tuple(x.shape)}")


def fuse_moe_pertensor_fp8(
    x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    gate_up_scale: Tensor,
    down_scale: Tensor,
    act_and_mul_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: Tensor | None = None,
    output: Tensor | None = None,
) -> Tensor:
    """HPC-Ops-compatible per-tensor E4M3 FusedMoE for SM120."""

    _, _, _, experts_local, _ = _validate_common(
        x,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
    )
    _check_bf16_output(x, shared_output, "shared_output")
    _check_bf16_output(x, output, "output")
    if x.dtype != _FP8 or gate_up_weight.dtype != _FP8 or down_weight.dtype != _FP8:
        raise TypeError("x, gate_up_weight and down_weight must use float8_e4m3fn")
    for name, scale, expected in (
        ("gate_up_scale", gate_up_scale, experts_local),
        ("down_scale", down_scale, experts_local),
    ):
        if (
            scale.device != x.device
            or scale.dtype != torch.float32
            or scale.numel() != expected
        ):
            raise ValueError(f"{name} must be FP32 [{expected}] on {x.device}")
    if (
        act_and_mul_scale.device != x.device
        or act_and_mul_scale.dtype != torch.float32
        or act_and_mul_scale.numel() != 1
    ):
        raise ValueError("act_and_mul_scale must be a one-element CUDA FP32 tensor")

    ep_size = num_expert_total // experts_local
    expected_local_pairs = (topk_ids.numel() + ep_size - 1) // ep_size
    block_m = choose_block_m(expected_local_pairs, experts_local)
    alignment = align_routes(topk_ids, experts_local, rank_ep, block_m)

    gate_up = triton_grouped_gemm(
        x,
        gate_up_weight,
        gate_up_scale.reshape(-1).contiguous(),
        alignment,
        topk=topk_ids.shape[1],
    )
    down_fp8 = act_mul_quant_fp8(gate_up, act_and_mul_scale, use_bf16_mul)
    down = triton_grouped_gemm(
        down_fp8,
        down_weight,
        down_scale.reshape(-1).contiguous(),
        alignment,
        topk=1,
    )
    return weighted_reduce_pairs(
        down,
        topk_ids,
        topk_scale,
        rank_ep,
        experts_local,
        shared_output,
        output,
        all_routes_local=num_expert_total == experts_local,
    )


def fuse_moe_blockwise_fp8(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: Tensor | None = None,
    output: Tensor | None = None,
    tuning_config: Mapping[str, Any] | None = None,
) -> Tensor:
    """HPC-Ops-compatible 128x128 block-wise E4M3 FusedMoE."""

    num_tokens, hidden, intermediate, experts_local, topk = _validate_common(
        x,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
    )
    _check_bf16_output(x, shared_output, "shared_output")
    _check_bf16_output(x, output, "output")
    if hidden % 128 or intermediate % 128:
        raise ValueError(
            "block-wise mode requires hidden_size and intermediate_size multiples of 128"
        )
    if x.dtype != _FP8 or gate_up_weight.dtype != _FP8 or down_weight.dtype != _FP8:
        raise TypeError("x, gate_up_weight and down_weight must use float8_e4m3fn")

    expected_shapes = (
        ("x_scale", x_scale, (num_tokens, hidden // 128)),
        (
            "gate_up_weight_scale",
            gate_up_weight_scale,
            (
                experts_local,
                2 * intermediate // 128,
                hidden // 128,
            ),
        ),
        (
            "down_weight_scale",
            down_weight_scale,
            (
                experts_local,
                hidden // 128,
                intermediate // 128,
            ),
        ),
    )
    for name, scale, shape in expected_shapes:
        if (
            scale.device != x.device
            or scale.dtype != torch.float32
            or not scale.is_contiguous()
            or scale.shape != shape
        ):
            raise ValueError(f"{name} must be contiguous FP32 {shape} on {x.device}")

    tuning_config = tuning_config or {}
    if not isinstance(tuning_config, Mapping):
        tuning_config = {}
    gate_tuning = tuning_config.get("gate", {})
    down_tuning = tuning_config.get("down", {})
    if not isinstance(gate_tuning, Mapping):
        gate_tuning = {}
    if not isinstance(down_tuning, Mapping):
        down_tuning = {}

    ep_size = num_expert_total // experts_local
    expected_local_pairs = (topk_ids.numel() + ep_size - 1) // ep_size
    default_block_m = choose_blockwise_block_m(expected_local_pairs, experts_local)
    block_m = tuning_config.get("block_m", default_block_m)
    if (
        not isinstance(block_m, int)
        or isinstance(block_m, bool)
        or block_m not in (16, 32, 64)
    ):
        block_m = default_block_m

    direct_default = (
        num_expert_total == experts_local
        and topk_ids.numel() < 64
        and topk_ids.numel() <= experts_local
    )
    direct = tuning_config.get("direct", direct_default)
    if not isinstance(direct, bool):
        direct = direct_default
    if direct and num_expert_total != experts_local:
        direct = False

    alignment = align_routes(
        topk_ids,
        experts_local,
        rank_ep,
        block_m,
        direct=direct,
    )

    kgroup_num_groups = 1
    if alignment.direct_topk_ids is not None:
        auto_num_groups = choose_kgroup_gate_num_groups(
            topk_ids.numel(),
            2 * intermediate,
            hidden,
            _device_sm_count(x.get_device()),
        )
        requested_num_groups = tuning_config.get("gate_num_groups", auto_num_groups)
        num_k_blocks = hidden // 128
        requested_groups_are_valid = (
            isinstance(requested_num_groups, int)
            and not isinstance(requested_num_groups, bool)
            and (
                requested_num_groups == 1
                or (
                    topk_ids.numel() <= 32
                    and 2 <= requested_num_groups <= min(16, num_k_blocks)
                    and num_k_blocks % requested_num_groups == 0
                    and 4 <= num_k_blocks // requested_num_groups <= 32
                )
            )
        )
        kgroup_num_groups = (
            requested_num_groups if requested_groups_are_valid else auto_num_groups
        )

    if kgroup_num_groups > 1:
        # Split Gate/Up K into shape-aware contiguous groups that fill the SMs.
        # Finalize the FP32 partials directly to blockwise FP8 after the
        # route-level BF16 checkpoint, avoiding a separate Gate/Up round trip.
        down_fp8, down_scale = triton_blockwise_kgroup_gate_act_quant_direct(
            x,
            x_scale,
            gate_up_weight,
            gate_up_weight_scale,
            alignment.direct_topk_ids,
            expert_start=alignment.expert_start,
            num_groups=kgroup_num_groups,
        )
    else:
        gate_up = triton_blockwise_grouped_gemm(
            x,
            x_scale,
            gate_up_weight,
            gate_up_weight_scale,
            alignment,
            topk=topk,
            tuning_config=gate_tuning,
        )
        down_fp8, down_scale = act_mul_quant_blockwise_fp8(gate_up)
    down = triton_blockwise_grouped_gemm(
        down_fp8,
        down_scale,
        down_weight,
        down_weight_scale,
        alignment,
        topk=1,
        tuning_config=down_tuning,
    )
    return weighted_reduce_pairs(
        down,
        topk_ids,
        topk_scale,
        rank_ep,
        experts_local,
        shared_output,
        output,
        all_routes_local=num_expert_total == experts_local,
    )
