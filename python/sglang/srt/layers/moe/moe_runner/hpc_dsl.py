"""SM120 BF16 MoE runner backed by the external hpc-dsl package.

Enable with ``--moe-runner-backend hpc_dsl``. The backend currently supports
the standard dispatcher, contiguous expert parallelism, gated SiLU, and BF16
activations and weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class HpcDslMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    global_num_experts: int
    rank_ep: int
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None


@lru_cache(maxsize=1)
def _load_hpc_dsl_fuse_moe() -> Callable:
    if not torch.cuda.is_available():
        raise RuntimeError("hpc_dsl requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability not in ((12, 0), (12, 1)):
        raise RuntimeError(
            f"hpc_dsl requires an SM120/SM121 GPU, got sm_{capability[0]}{capability[1]}."
        )
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
    try:
        from hpc_dsl import fuse_moe_bf16
    except ImportError as error:
        raise RuntimeError(
            "The hpc_dsl backend requires the hpc-dsl package. Install it with "
            "`pip install -e /path/to/hpc-dsl --no-deps`."
        ) from error
    return fuse_moe_bf16


def ensure_hpc_dsl_available() -> None:
    _load_hpc_dsl_fuse_moe()


def _validate_config(
    runner_config: MoeRunnerConfig, quant_info: HpcDslMoeQuantInfo
) -> None:
    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise ValueError("hpc_dsl supports only gated SiLU/SwiGLU MoE layers.")
    if runner_config.apply_router_weight_on_input:
        raise ValueError("hpc_dsl does not support apply_router_weight_on_input.")
    if runner_config.no_combine:
        raise ValueError("hpc_dsl does not support no_combine output.")
    if runner_config.num_fused_shared_experts not in (None, 0):
        raise ValueError("hpc_dsl does not support fused shared experts.")
    if runner_config.gemm1_alpha not in (None, 1.0):
        raise ValueError("hpc_dsl does not support gemm1_alpha.")
    if runner_config.gemm1_clamp_limit is not None:
        raise ValueError("hpc_dsl does not support gemm1_clamp_limit.")
    if runner_config.swiglu_limit is not None:
        raise ValueError("hpc_dsl does not support swiglu_limit.")
    if quant_info.b13 is not None or quant_info.b2 is not None:
        raise ValueError("hpc_dsl does not support expert bias tensors.")


def _validate_tensors(
    dispatch_output: StandardDispatchOutput, quant_info: HpcDslMoeQuantInfo
) -> None:
    hidden_states = dispatch_output.hidden_states
    w13_weight = quant_info.w13_weight
    w2_weight = quant_info.w2_weight
    topk_output = dispatch_output.topk_output

    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(f"hpc_dsl expects BF16 activations, got {hidden_states.dtype}.")
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise TypeError("hpc_dsl expects BF16 expert weights.")
    if hidden_states.ndim != 2 or w13_weight.ndim != 3 or w2_weight.ndim != 3:
        raise ValueError("hpc_dsl expects x=[M,H], w13=[E,2I,H], w2=[E,H,I].")
    num_local_experts, gate_up_size, hidden_size = w13_weight.shape
    if gate_up_size % 2 != 0:
        raise ValueError("hpc_dsl expects an even w13 output dimension.")
    intermediate_size = gate_up_size // 2
    if hidden_states.shape[1] != hidden_size:
        raise ValueError("hpc_dsl activation and w13 hidden dimensions do not match.")
    if w2_weight.shape != (num_local_experts, hidden_size, intermediate_size):
        raise ValueError("hpc_dsl w2 shape must be [E_local,H,I].")
    if not (
        hidden_states.device == w13_weight.device == w2_weight.device
        and topk_output.topk_ids.device == hidden_states.device
        and topk_output.topk_weights.device == hidden_states.device
    ):
        raise ValueError(
            "hpc_dsl activations, routing tensors, and weights must share a device."
        )
    if topk_output.topk_ids.shape != topk_output.topk_weights.shape:
        raise ValueError("hpc_dsl top-k ids and weights must have the same shape.")
    if topk_output.topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("hpc_dsl top-k routing must have one row per token.")
    if quant_info.global_num_experts % num_local_experts != 0:
        raise ValueError("hpc_dsl requires global experts divisible by local experts.")
    ep_size = quant_info.global_num_experts // num_local_experts
    if not 0 <= quant_info.rank_ep < ep_size:
        raise ValueError(f"hpc_dsl rank_ep must be in [0, {ep_size}).")


@register_fused_func("none", "hpc_dsl")
def fused_experts_none_to_hpc_dsl(
    dispatch_output: StandardDispatchOutput,
    quant_info: HpcDslMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if not TopKOutputChecker.format_is_standard(dispatch_output.topk_output):
        raise ValueError("hpc_dsl requires standard top-k routing output.")
    _validate_config(runner_config, quant_info)
    _validate_tensors(dispatch_output, quant_info)

    topk_output = dispatch_output.topk_output
    output = torch.empty_like(dispatch_output.hidden_states)
    output = _load_hpc_dsl_fuse_moe()(
        dispatch_output.hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_output.topk_ids.to(dtype=torch.int32).contiguous(),
        topk_output.topk_weights.to(dtype=torch.float32).contiguous(),
        quant_info.rank_ep,
        quant_info.global_num_experts,
        out=output,
    )

    if runner_config.routed_scaling_factor not in (None, 1.0):
        output.mul_(runner_config.routed_scaling_factor)
    return StandardCombineInput(hidden_states=output)
