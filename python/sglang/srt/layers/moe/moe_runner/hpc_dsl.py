"""SM120 BF16 and blockwise FP8 MoE runner backed by hpc-dsl.

Enable with ``--moe-runner-backend hpc_dsl``. The backend currently supports
the standard dispatcher, contiguous expert parallelism, gated SiLU, BF16
activations, and either BF16 or serialized blockwise FP8 expert weights.
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
    w13_weight_scale_inv: Optional[torch.Tensor] = None
    w2_weight_scale_inv: Optional[torch.Tensor] = None
    block_shape: Optional[tuple[int, int]] = None
    weight_format: Optional[str] = None


_HPC_DSL_FP8_MMA_MODE = os.getenv("HPC_DSL_FP8_MMA_MODE", "triton").lower()
if _HPC_DSL_FP8_MMA_MODE not in ("triton", "mxfp8"):
    raise ValueError("HPC_DSL_FP8_MMA_MODE must be one of: triton, mxfp8")


def hpc_dsl_mxfp8_enabled() -> bool:
    return _HPC_DSL_FP8_MMA_MODE == "mxfp8"


def _validate_hpc_dsl_device() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("hpc_dsl requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability not in ((12, 0), (12, 1)):
        raise RuntimeError(
            f"hpc_dsl requires an SM120/SM121 GPU, got sm_{capability[0]}{capability[1]}."
        )


@lru_cache(maxsize=1)
def _load_hpc_dsl_fuse_moe() -> Callable:
    _validate_hpc_dsl_device()
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
    try:
        from hpc_dsl import fuse_moe_bf16
    except ImportError as error:
        raise RuntimeError(
            "The hpc_dsl backend requires the hpc-dsl package. Install it with "
            "`pip install -e /path/to/hpc-dsl --no-deps`."
        ) from error
    return fuse_moe_bf16


@lru_cache(maxsize=1)
def _load_hpc_dsl_fuse_moe_blockwise_fp8() -> Callable:
    _validate_hpc_dsl_device()
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
    try:
        from hpc_dsl import fuse_moe_blockwise_fp8
    except ImportError as error:
        raise RuntimeError(
            "The hpc_dsl blockwise FP8 backend requires a version of hpc-dsl "
            "that exports `fuse_moe_blockwise_fp8`. Install it with "
            "`pip install -e /path/to/hpc-dsl --no-deps`."
        ) from error
    return fuse_moe_blockwise_fp8


@lru_cache(maxsize=1)
def _load_hpc_dsl_mxfp8_api() -> tuple[Callable, Callable, type]:
    _validate_hpc_dsl_device()
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
    try:
        from hpc_dsl import (
            MXFP8MoEWeights,
            convert_blockwise_fp8_moe_weights_to_mxfp8,
            fuse_moe_mxfp8,
        )
    except ImportError as error:
        raise RuntimeError(
            "HPC_DSL_FP8_MMA_MODE=mxfp8 requires an hpc-dsl build that exports "
            "`fuse_moe_mxfp8` and `convert_blockwise_fp8_moe_weights_to_mxfp8`."
        ) from error
    return (
        fuse_moe_mxfp8,
        convert_blockwise_fp8_moe_weights_to_mxfp8,
        MXFP8MoEWeights,
    )


def convert_hpc_dsl_blockwise_fp8_weights(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
):
    _, convert_weights, _ = _load_hpc_dsl_mxfp8_api()
    return convert_weights(
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        inplace=True,
        chunk_rows=512,
        backend="cute-dsl",
    )


def ensure_hpc_dsl_available(*, blockwise_fp8: bool = False) -> None:
    if blockwise_fp8:
        if hpc_dsl_mxfp8_enabled():
            _load_hpc_dsl_mxfp8_api()
        else:
            _load_hpc_dsl_fuse_moe_blockwise_fp8()
    else:
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
    uses_bf16 = (
        w13_weight.dtype == torch.bfloat16 and w2_weight.dtype == torch.bfloat16
    )
    uses_blockwise_fp8 = (
        w13_weight.dtype == torch.float8_e4m3fn
        and w2_weight.dtype == torch.float8_e4m3fn
    )
    if not uses_bf16 and not uses_blockwise_fp8:
        raise TypeError(
            "hpc_dsl expects both expert weights to use BF16 or FP8 E4M3."
        )
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

    scale_tensors = (
        quant_info.w13_weight_scale_inv,
        quant_info.w2_weight_scale_inv,
    )
    weight_format = quant_info.weight_format
    if weight_format is None:
        weight_format = "blockwise_fp8" if quant_info.block_shape is not None else "bf16"
    if weight_format not in ("bf16", "blockwise_fp8", "mxfp8"):
        raise ValueError(f"unsupported hpc_dsl weight format {weight_format!r}.")
    if uses_bf16 and weight_format != "bf16":
        raise ValueError("BF16 hpc_dsl weights require weight_format='bf16'.")
    if uses_blockwise_fp8 and weight_format == "mxfp8":
        if quant_info.block_shape is not None or any(
            scale is None for scale in scale_tensors
        ):
            raise ValueError(
                "hpc_dsl MXFP8 requires derived scales and no source block_shape."
            )
        w13_scale = quant_info.w13_weight_scale_inv
        w2_scale = quant_info.w2_weight_scale_inv
        if w13_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
            raise TypeError("hpc_dsl MXFP8 scales must use uint8 UE8M0 bytes.")
        expected_w13_scale_shape = (
            num_local_experts,
            gate_up_size * hidden_size // 32,
        )
        expected_w2_scale_shape = (
            num_local_experts,
            hidden_size * intermediate_size // 32,
        )
        if tuple(w13_scale.shape) != expected_w13_scale_shape:
            raise ValueError(
                "hpc_dsl MXFP8 w13 scale shape must be "
                f"{expected_w13_scale_shape}, got {tuple(w13_scale.shape)}."
            )
        if tuple(w2_scale.shape) != expected_w2_scale_shape:
            raise ValueError(
                "hpc_dsl MXFP8 w2 scale shape must be "
                f"{expected_w2_scale_shape}, got {tuple(w2_scale.shape)}."
            )
    elif uses_blockwise_fp8 and weight_format == "blockwise_fp8":
        if quant_info.block_shape is None or any(
            scale is None for scale in scale_tensors
        ):
            raise ValueError(
                "hpc_dsl blockwise FP8 requires w13/w2 scale_inv and block_shape."
            )
        block_shape = tuple(quant_info.block_shape)
        if block_shape != (128, 128):
            raise ValueError(
                f"hpc_dsl supports only FP8 block_shape=(128, 128), got {block_shape}."
            )
        block_n, block_k = block_shape
        if hidden_size % block_n != 0 or intermediate_size % block_k != 0:
            raise ValueError(
                "hpc_dsl blockwise FP8 requires hidden and intermediate "
                f"dimensions divisible by 128, got H={hidden_size}, "
                f"I={intermediate_size}."
            )
        w13_scale = quant_info.w13_weight_scale_inv
        w2_scale = quant_info.w2_weight_scale_inv
        if w13_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
            raise TypeError("hpc_dsl blockwise FP8 scales must use float32.")
        expected_w13_scale_shape = (
            num_local_experts,
            gate_up_size // block_n,
            hidden_size // block_k,
        )
        expected_w2_scale_shape = (
            num_local_experts,
            hidden_size // block_n,
            intermediate_size // block_k,
        )
        if tuple(w13_scale.shape) != expected_w13_scale_shape:
            raise ValueError(
                "hpc_dsl w13 scale shape must be "
                f"{expected_w13_scale_shape}, got {tuple(w13_scale.shape)}."
            )
        if tuple(w2_scale.shape) != expected_w2_scale_shape:
            raise ValueError(
                "hpc_dsl w2 scale shape must be "
                f"{expected_w2_scale_shape}, got {tuple(w2_scale.shape)}."
            )
    elif uses_blockwise_fp8:
        raise ValueError(
            f"FP8 hpc_dsl weights do not support weight_format={weight_format!r}."
        )
    elif quant_info.block_shape is not None or any(
        scale is not None for scale in scale_tensors
    ):
        raise ValueError("hpc_dsl BF16 weights must not provide FP8 scale metadata.")

    tensors = [
        w13_weight,
        w2_weight,
        topk_output.topk_ids,
        topk_output.topk_weights,
    ]
    tensors.extend(scale for scale in scale_tensors if scale is not None)
    if any(tensor.device != hidden_states.device for tensor in tensors):
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
    topk_ids = topk_output.topk_ids.to(dtype=torch.int32).contiguous()
    topk_weights = topk_output.topk_weights.to(dtype=torch.float32).contiguous()
    weight_format = quant_info.weight_format
    if weight_format is None:
        weight_format = "blockwise_fp8" if quant_info.block_shape is not None else "bf16"
    if weight_format == "mxfp8":
        fuse_moe_mxfp8, _, weights_type = _load_hpc_dsl_mxfp8_api()
        intermediate_size = quant_info.w13_weight.shape[1] // 2
        weights = weights_type(
            gate_up=quant_info.w13_weight,
            gate_up_scale=quant_info.w13_weight_scale_inv,
            down=quant_info.w2_weight,
            down_scale=quant_info.w2_weight_scale_inv,
            hidden_size=dispatch_output.hidden_states.shape[1],
            intermediate_size=intermediate_size,
        )
        workspace_slot = None
        if dispatch_output.hidden_states.is_cuda:
            workspace_slot = (
                "sglang",
                torch.cuda.current_stream(
                    dispatch_output.hidden_states.device
                ).cuda_stream,
            )
        output = fuse_moe_mxfp8(
            dispatch_output.hidden_states,
            weights,
            topk_ids,
            topk_weights,
            quant_info.rank_ep,
            quant_info.global_num_experts,
            out=output,
            workspace_slot=workspace_slot,
        )
    elif quant_info.block_shape is not None:
        output = _load_hpc_dsl_fuse_moe_blockwise_fp8()(
            dispatch_output.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            quant_info.w13_weight_scale_inv,
            quant_info.w2_weight_scale_inv,
            topk_ids,
            topk_weights,
            quant_info.rank_ep,
            quant_info.global_num_experts,
            block_shape=tuple(quant_info.block_shape),
            out=output,
        )
    else:
        output = _load_hpc_dsl_fuse_moe()(
            dispatch_output.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            topk_ids,
            topk_weights,
            quant_info.rank_ep,
            quant_info.global_num_experts,
            out=output,
        )

    if runner_config.routed_scaling_factor not in (None, 1.0):
        output.mul_(runner_config.routed_scaling_factor)
    return StandardCombineInput(hidden_states=output)
