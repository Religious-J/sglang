"""HPC fused RoPE + QK-Norm + KV-Cache-Write (+ optional FP8 Q quant).

Decoupled from HpcAttentionBackend; KV-cache write and Q quantization are
handled here, leaving the attention computation to HpcAttentionBackend.

SGLang-specific conventions:
  - Uses get_key_buffer / get_value_buffer (SGLang pool API).
  - Uses direct_register_custom_op under the torch.ops.sglang namespace.
  - Routes back via a module-level instance registry keyed by layer_name.
  - Uses get_forward_context() from sglang.srt.model_executor.forward_context.

Typical usage in a model attention layer::

    # In __init__:
    from sglang.srt.layers.hpc.rope_norm import HpcRopeNorm, QkNormPolicy
    self.hpc_rope_norm = HpcRopeNorm(
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        cos_sin_cache=self.rotary_emb.cos_sin_cache,
        use_qk_norm=self.use_qk_norm,
        fallback_qnorm=self.q_norm,
        fallback_knorm=self.k_norm,
        kv_cache_dtype=kv_cache_dtype,
        layer_name=layer_name,  # e.g. "model.layers.0.self_attn"
        qk_norm_policy=QkNormPolicy.NORM_THEN_ROPE,
    )

    # In forward():
    if self.hpc_rope_norm is not None:
        q = self.hpc_rope_norm(qkv, layer_name=self.layer_name)
        # q: [num_tokens, num_heads, head_dim] (fp8 or bf16)
        # K/V already written into paged cache by the fused kernel.
        attn_output = self.attn(q, k=None, v=None, forward_batch, save_kv_cache=False)
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from sglang.srt.model_executor.forward_context import get_forward_context
from sglang.srt.utils.common import direct_register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.attention.hpc_backend import HpcAttentionMetadata

logger = logging.getLogger(__name__)

# Global instance registry: layer_name -> HpcRopeNorm instance.
# Enables the module-level custom op to route back to the correct instance
# without passing the object through the custom op interface.
_hpc_rope_norm_instances: dict[str, "HpcRopeNorm"] = {}


class QkNormPolicy(IntEnum):
    """Order of QK-RMSNorm relative to RoPE in the fused HPC rope_norm kernel.

    The integer values are part of the HPC kernel ABI (passed as int to the
    kernel). Keep in sync with hpc-ops kernel expectations.
    """

    NONE = 0            # No QK-Norm; RoPE only.
    ROPE_THEN_NORM = 1  # RoPE first, then RMSNorm.
    NORM_THEN_ROPE = 2  # RMSNorm first, then RoPE (e.g. HunYuan V3).


# ---------------------------------------------------------------------------
# Module-level custom op
# ---------------------------------------------------------------------------

def _hpc_rope_norm_forward_impl(
    qkv: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Top-level custom op: RoPE + QK-Norm + KV-Cache-Write (+ FP8 Q quant).

    Fully opaque to torch.compile (dynamo).
    """
    forward_context = get_forward_context()
    attn_backend = forward_context.attn_backend

    # Retrieve the current forward batch from the backend.
    attn_metadata: HpcAttentionMetadata = attn_backend.forward_metadata
    if attn_metadata is None:
        output.zero_()
        return

    rope_norm = _hpc_rope_norm_instances[layer_name]
    rope_norm._forward_impl(qkv, attn_backend, attn_metadata, output)


def _hpc_rope_norm_forward_fake(
    qkv: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake impl for torch.compile abstract interpretation; output is mutated."""
    return


direct_register_custom_op(
    op_name="hpc_rope_norm_forward",
    op_func=_hpc_rope_norm_forward_impl,
    mutates_args=["output"],
    fake_impl=_hpc_rope_norm_forward_fake,
)


# ---------------------------------------------------------------------------
# HpcRopeNorm module
# ---------------------------------------------------------------------------

class HpcRopeNorm(nn.Module):
    """HPC fused RoPE + QK-Norm + KV-Cache-Write (+ optional FP8 Q quant).

    Registered as a sub-module in model attention layers.
    Norm weights are extracted from the model's existing RMSNorm modules
    via process_weights_after_loading() after the checkpoint is loaded.

    forward() dispatches through a torch custom op (compile splitting point):
    - forward_cuda(): calls torch.ops.sglang.hpc_rope_norm_forward
    - forward_native(): delegates to forward_cuda() (same behaviour)

    The custom op routes back to _forward_impl() via the global instance
    registry keyed by layer_name.

    Args (forward):
        qkv       : Packed [Q, K, V] tensor.
                    Shape [num_tokens, (nq + nk + nv) * head_dim], dtype bf16.
        layer_name: String key used to look up this instance in the registry
                    and to identify the correct KV cache buffer.

    Returns:
        q_out : Processed Q tensor.
                Shape [num_tokens, num_heads, head_dim].
                Dtype: float8_e4m3fn (FP8 path) or bfloat16 (BF16 path).
                K/V are written into the paged cache by the fused kernel.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cos_sin_cache: torch.Tensor,
        use_qk_norm: bool,
        fallback_qnorm: Optional[nn.Module],
        fallback_knorm: Optional[nn.Module],
        kv_cache_dtype: str,
        layer_name: str,
        qk_norm_policy: QkNormPolicy = QkNormPolicy.ROPE_THEN_NORM,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        self.head_per_group = num_heads // num_kv_heads

        # Register as non-persistent buffer: participates in memory management
        # but excluded from checkpoint state_dict.
        # cos_sin_cache must have shape [max_seq_len, head_dim]:
        #   columns [0 .. head_dim//2-1] = cos(freq * position)
        #   columns [head_dim//2 .. head_dim-1] = sin(freq * position)
        # where freq = 1 / (10000 ** (2i / head_dim)), i in [0, head_dim//2).
        # This is NOT the [max_seq_len, head_dim*2] cos||sin format used by
        # most HuggingFace rotary embedding implementations.
        self.register_buffer("cos_sin_cache", cos_sin_cache.float(), persistent=False)

        self.fallback_qnorm = fallback_qnorm
        self.fallback_knorm = fallback_knorm

        # Pre-allocate norm weight Parameters with stable addresses for CUDA
        # Graph replay. process_weights_after_loading() fills them inplace via
        # copy_() — no tensor pointer invalidation.
        if use_qk_norm and fallback_qnorm is not None:
            self.qnorm_weight: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(head_dim, dtype=torch.float32), requires_grad=False
            )
        else:
            self.qnorm_weight = None

        if use_qk_norm and fallback_knorm is not None:
            self.knorm_weight: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(head_dim, dtype=torch.float32), requires_grad=False
            )
        else:
            self.knorm_weight = None

        self.use_fp8 = "fp8" in kv_cache_dtype
        # When QK-Norm is disabled, force NONE regardless of caller policy.
        self.qk_norm_policy = qk_norm_policy if use_qk_norm else QkNormPolicy.NONE

        # Register in the global instance registry so the custom op can route.
        self.layer_name: str = layer_name
        self.register_layer_name(layer_name)

    # ------------------------------------------------------------------
    # Capability check
    # ------------------------------------------------------------------

    @classmethod
    def support(
        cls,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: str,
        attention_backend: str = "",
    ) -> bool:
        """Return True if HpcRopeNorm can be used for the given config."""
        if attention_backend != "hpc":
            return False

        if kv_cache_dtype not in ("fp8_e4m3", "auto"):
            logger.warning(
                "HpcRopeNorm does not support kv_cache_dtype=%s "
                "(only fp8_e4m3 / auto bf16).",
                kv_cache_dtype,
            )
            return False

        if head_dim != 128:
            logger.warning(
                "HpcRopeNorm only supports head_dim=128, got %d.", head_dim
            )
            return False

        head_per_group = num_heads // num_kv_heads
        # Note: hpc-ops 0.0.1.dev0+g50b48ac (SM90/H20) only supports the
        # specific combination (num_q_heads=64, num_kv_heads=8, head_dim=128)
        # at runtime. The check below follows the documented constraint
        # (head_per_group in {4, 8}), but in practice only hpg=8 with
        # nq=64/nkv=8 has been verified. Other combinations may raise
        # RuntimeError("unsupported config") from the underlying kernel.
        if head_per_group not in (4, 8):
            logger.warning(
                "HpcRopeNorm only supports head_per_group in {4, 8}, got %d.",
                head_per_group,
            )
            return False

        logger.info("HpcRopeNorm enabled.")
        return True

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def process_weights_after_loading(self, model=None) -> None:
        """Copy norm weights (fp32) from fallback norm modules inplace.

        Uses copy_() to preserve tensor addresses for CUDA Graph refit.
        Called once by the model loader after all weights are loaded.
        """
        if self.use_qk_norm:
            if self.fallback_qnorm is not None and self.qnorm_weight is not None:
                self.qnorm_weight.data.copy_(
                    self.fallback_qnorm.weight.data.float()
                )
            if self.fallback_knorm is not None and self.knorm_weight is not None:
                self.knorm_weight.data.copy_(
                    self.fallback_knorm.weight.data.float()
                )

    # ------------------------------------------------------------------
    # Instance registry
    # ------------------------------------------------------------------

    def register_layer_name(self, layer_name: str) -> None:
        """Register layer_name and add self to the global registry."""
        self.layer_name = layer_name
        _hpc_rope_norm_instances[layer_name] = self
        logger.debug(
            "[rope_norm] registered HpcRopeNorm for layer: %s", layer_name
        )

    # ------------------------------------------------------------------
    # Forward (CustomOp-style dispatch)
    # ------------------------------------------------------------------

    def forward(self, qkv: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Dispatch forward: call through the custom op (CUDA path)."""
        return self.forward_cuda(qkv, layer_name)

    def forward_native(self, qkv: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Native fallback: delegates to forward_cuda()."""
        return self.forward_cuda(qkv, layer_name)

    def forward_cuda(self, qkv: torch.Tensor, layer_name: str) -> torch.Tensor:
        """CUDA path: invoke the torch custom op as a compile splitting point."""
        num_tokens = qkv.shape[0]
        output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=torch.float8_e4m3fn if self.use_fp8 else qkv.dtype,
            device=qkv.device,
        )
        torch.ops.sglang.hpc_rope_norm_forward(qkv, output, layer_name)
        return output

    # ------------------------------------------------------------------
    # _forward_impl (called by the custom op)
    # ------------------------------------------------------------------

    def _forward_impl(
        self,
        qkv: torch.Tensor,
        attn_backend,
        attn_metadata: "HpcAttentionMetadata",
        output: torch.Tensor,
    ) -> None:
        """Actual kernel dispatch.

        Called by the module-level custom op _hpc_rope_norm_forward_impl().

        In SGLang, forward() is always called on a pure-prefill or pure-decode
        batch — never a mixed batch. There is no decode/prefill interleaving
        within a single call (unlike vLLM's unified forward()). Therefore no
        request-level slicing is needed; we dispatch directly to either the
        prefill or decode kernel based on attn_metadata.max_seq_len_q.
        """
        import hpc

        # Detect whether this is prefill (max_seq_len_q > 1) or decode (== 1).
        is_prefill = attn_metadata.max_seq_len_q > 1

        attn_layer = self._attn_layer
        page_size = attn_backend.page_size

        key_cache = attn_backend.token_to_kv_pool.get_key_buffer(attn_layer.layer_id)
        value_cache = attn_backend.token_to_kv_pool.get_value_buffer(attn_layer.layer_id)

        if self.use_fp8:
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(torch.float8_e4m3fn)
                value_cache = value_cache.view(torch.float8_e4m3fn)

        key_cache = key_cache.view(-1, page_size, key_cache.shape[-2], self.head_dim)
        value_cache = value_cache.view(-1, page_size, value_cache.shape[-2], self.head_dim)

        q_norm_weight = (
            self.qnorm_weight if self.qk_norm_policy != QkNormPolicy.NONE else None
        )
        k_norm_weight = (
            self.knorm_weight if self.qk_norm_policy != QkNormPolicy.NONE else None
        )

        QUANT_POLICY_DQSKV = (
            hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR.value
        )

        k_scale = attn_layer.k_scale.reshape(1) if attn_layer.k_scale is not None else None
        v_scale = attn_layer.v_scale.reshape(1) if attn_layer.v_scale is not None else None

        if is_prefill:
            if self.use_fp8:
                _, q_scale, _ = hpc.rope_norm_store_kv_fp8(
                    key_cache=key_cache,
                    value_cache=value_cache,
                    qkv=qkv,
                    cos_sin=self.cos_sin_cache,
                    num_seqlen_per_req=attn_metadata.cache_seqlens_int32,
                    q_index=attn_metadata.cu_seqlens_q,
                    kvcache_indices=attn_metadata.page_table,
                    is_prefill=True,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    quant_policy=QUANT_POLICY_DQSKV,
                    max_seqlens=attn_metadata.max_seq_len_q,
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    qk_norm_policy=int(self.qk_norm_policy),
                    out_q=output,
                )
                attn_metadata.hpc_prefill_q_scale = q_scale
            else:
                hpc.rope_norm_store_kv(
                    key_cache,
                    value_cache,
                    qkv,
                    self.cos_sin_cache,
                    attn_metadata.cache_seqlens_int32,
                    attn_metadata.cu_seqlens_q,
                    attn_metadata.page_table,
                    True,  # is_prefill
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    out_q=output,
                    qk_norm_policy=int(self.qk_norm_policy),
                )
        else:
            # Decode: q_index is a per-request prefix-sum [0, 1, ..., bs].
            bs = qkv.shape[0]
            qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device=qkv.device)

            if self.use_fp8:
                _, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
                    key_cache=key_cache,
                    value_cache=value_cache,
                    qkv=qkv,
                    cos_sin=self.cos_sin_cache,
                    num_seqlen_per_req=attn_metadata.cache_seqlens_int32,
                    q_index=qo_indptr,
                    kvcache_indices=attn_metadata.page_table,
                    is_prefill=False,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    quant_policy=QUANT_POLICY_DQSKV,
                    max_seqlens=1,
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    qk_norm_policy=int(self.qk_norm_policy),
                    out_q=output,
                )
                attn_metadata.hpc_decode_q_scale = q_scale
                if split_k_flag is not None:
                    attn_metadata.hpc_split_k_flag = split_k_flag
            else:
                hpc.rope_norm_store_kv(
                    key_cache,
                    value_cache,
                    qkv,
                    self.cos_sin_cache,
                    attn_metadata.cache_seqlens_int32,
                    qo_indptr,
                    attn_metadata.page_table,
                    False,  # is_prefill
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    out_q=output,
                    qk_norm_policy=int(self.qk_norm_policy),
                )

        attn_metadata.hpc_kv_written = True

    # ------------------------------------------------------------------
    # Helpers for model layer integration
    # ------------------------------------------------------------------

    def set_attn_layer(self, attn_layer) -> None:
        """Register the RadixAttention layer for KV cache access.

        Must be called by the model attention layer before the first forward
        (e.g. at the end of __init__ after the RadixAttention is constructed).

        Args:
            attn_layer: The RadixAttention layer corresponding to this
                        HpcRopeNorm instance. Provides layer_id, k_scale, v_scale.
        """
        self._attn_layer = attn_layer
