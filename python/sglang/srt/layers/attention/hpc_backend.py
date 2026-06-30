"""HPC Attention Backend for SGLang.

HPC-Ops (https://github.com/Tencent/hpc-ops) is a production-grade,
high-performance operator library for LLM inference developed by the
Tencent Hunyuan AI Infra team.

This backend supports BF16 and FP8 (fp8_e4m3) paged KV cache attention
with block_size 32 or 64. For the fused RoPE + QK-Norm + KV-Write path,
see sglang.srt.layers.hpc.rope_norm.HpcRopeNorm.

FP8 Q quantization has two sub-paths:
  - Fast path (HpcRopeNorm active): hpc.rope_norm_store_kv_fp8 fuses RoPE +
    KV write + Q quantization. Supported only for nq=64, nkv=8, head_dim=128.
  - Fallback path: standard SGLang RoPE/set_kv_buffer writes KV;
    sglang_per_token_group_quant_fp8 quantizes Q and pack_scale_th_triton
    reshapes the scale into [bs, H, pad] for hpc.attention_with_kvcache_prefill_fp8.
    Works for all configs supported by the hpc attention kernels, e.g.
    Qwen3-30B-A3B (nq=32, nkv=4).

Enable with: --attention-backend hpc --kv-cache-dtype fp8_e4m3 --block-size 64
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_ops.metadata import normal_decode_set_metadata
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FP8 fallback helpers
# ---------------------------------------------------------------------------


def _pad_to(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


@triton.jit
def _pack_scale_th_kernel(
    scale_ptr,
    cu_ptr,
    out_ptr,
    H: tl.constexpr,
    PAD: tl.constexpr,
    stride_s0: tl.constexpr,
    stride_s1: tl.constexpr,
    stride_o0: tl.constexpr,
    stride_o1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scatter per-token-per-head fp32 scales into [bs, H, pad] layout.

    scale_ptr : [T, H] fp32, flat token-major scale produced by
                sglang_per_token_group_quant_fp8.
    cu_ptr    : [bs+1] int32, cumulative query lengths.
    out_ptr   : [bs, H, PAD] fp32, zero-initialised before launch.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid = tl.program_id(2)

    start = tl.load(cu_ptr + b).to(tl.int32)
    end = tl.load(cu_ptr + b + 1).to(tl.int32)
    L = end - start

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask_pad = offs < PAD
    mask_len = offs < L

    tok = start + offs
    val = tl.load(
        scale_ptr + tok * stride_s0 + h * stride_s1,
        mask=mask_len,
        other=0.0,
    ).to(tl.float32)

    tl.store(
        out_ptr + b * stride_o0 + h * stride_o1 + offs,
        val,
        mask=mask_pad,
    )


@torch.no_grad()
def pack_scale_th_triton(
    scale_th: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seq_len_q: int,
    pad_multiple: int = 128,
    block: int = 256,
) -> torch.Tensor:
    """Reshape per-token-per-head scale [T, H] → [bs, H, pad].

    hpc.attention_with_kvcache_prefill_fp8 expects qscale shaped
    [bs, num_head_q, max_seqlens_q_pad] (pad to a multiple of pad_multiple).

    Args:
        scale_th      : [T, H] fp32, output of sglang_per_token_group_quant_fp8.
        cu_seqlens_q  : [bs+1] int32, cumulative query token counts.
        max_seq_len_q : maximum sequence length in the batch.
        pad_multiple  : pad the time dimension to a multiple of this value.
        block         : Triton BLOCK size (must be a power-of-2 constant).

    Returns:
        [bs, H, pad] fp32 tensor.
    """
    assert scale_th.is_cuda and cu_seqlens_q.is_cuda
    assert scale_th.dim() == 2

    T, H = scale_th.shape
    cu = cu_seqlens_q.to(torch.int32).contiguous()
    bs = cu.numel() - 1

    pad = _pad_to(int(max_seq_len_q), pad_multiple)
    out = torch.zeros(bs, H, pad, device=scale_th.device, dtype=torch.float32)

    grid = (bs, H, triton.cdiv(pad, block))
    _pack_scale_th_kernel[grid](
        scale_th,
        cu,
        out,
        H=H,
        PAD=pad,
        stride_s0=scale_th.stride(0),
        stride_s1=scale_th.stride(1),
        stride_o0=out.stride(0),
        stride_o1=out.stride(1),
        BLOCK=block,
        num_warps=4,
    )
    return out



@dataclass
class HpcAttentionMetadata:
    """Per-forward-pass metadata for HPC attention kernels.

    Fields used by the pure-attention path (HpcAttentionBackend):
      cache_seqlens_int32 : KV sequence lengths per request [bs], int32
      cu_seqlens_q        : cumulative query lengths [bs+1], int32
      max_seq_len_q       : max query length (prefill) or 1 (decode)
      page_table          : paged KV block ids [bs, max_blocks], int32
      num_prefills        : number of prefill requests in the batch
      num_decodes         : number of decode requests in the batch
      num_decode_tokens   : total decode tokens (== num_decodes for single-tok)

    Fields set by HpcRopeNorm (fused RoPE+Norm path) and consumed here:
      hpc_kv_written      : True when HpcRopeNorm already wrote K/V
      hpc_prefill_q_scale : FP8 per-token-per-head Q scale (prefill)
      hpc_decode_q_scale  : FP8 per-token-per-head Q scale (decode)
      hpc_split_k_flag    : split-K flag tensor for FP8 decode
    """

    cache_seqlens_int32: Optional[torch.Tensor] = None  # [bs] int32
    max_seq_len_q: int = 1
    max_seq_len_k: int = 0
    cu_seqlens_q: Optional[torch.Tensor] = None   # [bs+1] int32
    cu_seqlens_k: Optional[torch.Tensor] = None   # [bs+1] int32
    page_table: Optional[torch.Tensor] = None     # [bs, max_blocks] int32

    # HpcRopeNorm pass-through fields
    hpc_kv_written: bool = False
    hpc_prefill_q_scale: Optional[torch.Tensor] = None
    hpc_decode_q_scale: Optional[torch.Tensor] = None
    hpc_split_k_flag: Optional[torch.Tensor] = None


class HpcAttentionBackend(AttentionBackend):
    """HPC pure-attention backend.

    Handles only the attention computation. RoPE, QK-Norm, and KV-cache
    writes are the responsibility of the model layer (optionally via
    HpcRopeNorm for the fused path).

    KV cache layout expected by hpc kernels:
      [num_blocks, page_size, num_kv_heads, head_dim]

    SGLang MHATokenToKVPool stores:
      [num_slots, num_kv_heads, head_dim]  (slot-addressed, page_size >= 1)

    At forward time we reshape: num_blocks = num_slots // page_size.
    """

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()

        try:
            import hpc as _hpc  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "HpcAttentionBackend requires the hpc-ops package. "
                "Install it from https://github.com/Tencent/hpc-ops"
            ) from e

        self.page_size = model_runner.server_args.page_size
        assert self.page_size in (32, 64), (
            f"HPC attention backend requires page_size=32 or 64, "
            f"got {self.page_size}. Use --block-size 64."
        )

        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.forward_metadata: Optional[HpcAttentionMetadata] = None
        self.max_context_len = model_runner.model_config.context_len
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype

        # split-K decode: always enabled.
        self.splitk = True

    # ------------------------------------------------------------------
    # CUDA Graph
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Pre-allocate stable-address buffers for CUDA Graph decode."""
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size

        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs, max_num_pages, dtype=torch.int32, device=self.device
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }
        self.encoder_metadata = {}

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Initialize forward metadata for CUDA Graph capture."""
        metadata = HpcAttentionMetadata()
        device = seq_lens.device

        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
                metadata.max_seq_len_k = seq_lens.max().item()
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.cu_seqlens_q = torch.arange(
                    0, bs + 1, dtype=torch.int32, device=device
                )
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs
                ]
                self.decode_cuda_graph_metadata[bs] = metadata

        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Update in-place buffers for CUDA Graph replay."""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]
        metadata = None

        if forward_mode.is_decode_or_idle() and spec_info is None:
            metadata: HpcAttentionMetadata = self.decode_cuda_graph_metadata[bs]
            max_len = int(seq_lens_cpu.max().item()) if bs > 0 else 0
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size
            metadata.max_seq_len_k = max_len

            normal_decode_set_metadata(
                metadata.cache_seqlens_int32,
                metadata.cu_seqlens_k,
                metadata.page_table,
                self.req_to_token,
                req_pool_indices,
                self.decode_cuda_graph_metadata["strided_indices"],
                max_seq_pages,
                seq_lens,
                0,
                self.page_size,
            )

        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ------------------------------------------------------------------
    # Eager forward metadata
    # ------------------------------------------------------------------

    def _build_page_table(
        self, forward_batch: ForwardBatch, max_seq_len_k: int
    ) -> torch.Tensor:
        """Convert slot-addressed req_to_token into HPC block_ids.

        Samples one token slot per page boundary (0, page_size, 2*page_size,
        ...) and divides by page_size to obtain block ids.

        Returns:
            block_ids: [bs, max_num_blocks] int32
        """
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        max_num_blocks = (max_seq_len_k + self.page_size - 1) // self.page_size
        sample_positions = torch.arange(
            0, max_num_blocks * self.page_size, self.page_size, device=device
        )
        raw = self.req_to_token[forward_batch.req_pool_indices][:, sample_positions]
        return (raw // self.page_size).to(torch.int32)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Build HpcAttentionMetadata for the current forward batch."""
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        if not (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_extend()
        ):
            raise NotImplementedError(
                "HpcAttentionBackend does not support forward mode: "
                f"{forward_batch.forward_mode}"
            )

        max_seq_len_k = int(forward_batch.seq_lens.max().item())

        if forward_batch.forward_mode.is_decode_or_idle():
            cache_seqlens = forward_batch.seq_lens.to(torch.int32)
            cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device=device)
            max_seq_len_q = 1
            page_table = self._build_page_table(forward_batch, max_seq_len_k)
        else:
            max_seq_len_q = int(forward_batch.extend_seq_lens.max().item())
            cache_seqlens = forward_batch.seq_lens.to(torch.int32)
            cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=device)
            cu_seqlens_q[1:] = torch.cumsum(
                forward_batch.extend_seq_lens, dim=0, dtype=torch.int32
            )
            page_table = self._build_page_table(forward_batch, max_seq_len_k)

        self.forward_metadata = HpcAttentionMetadata(
            cache_seqlens_int32=cache_seqlens,
            max_seq_len_q=max_seq_len_q,
            cu_seqlens_q=cu_seqlens_q,
            page_table=page_table,
        )

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    def _get_kv_cache(
        self,
        forward_batch: ForwardBatch,
        layer: "RadixAttention",
    ):
        """Return (key_cache, value_cache) reshaped for hpc kernels.

        hpc expects: [num_blocks, page_size, num_kv_heads, head_dim]
        """
        key_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        value_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)

        key_cache = key_cache.view(
            -1, self.page_size, key_cache.shape[-2], layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, value_cache.shape[-2], layer.v_head_dim
        )
        return key_cache, value_cache

    # ------------------------------------------------------------------
    # Forward: extend (prefill)
    # ------------------------------------------------------------------

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Prefill/extend forward pass."""
        import hpc

        metadata = self.forward_metadata
        if metadata is None:
            # Safety guard: return zeros when metadata is not initialized.
            num_tokens = q.shape[0]
            tp_q_head_num = layer.tp_q_head_num
            return torch.zeros(
                num_tokens, tp_q_head_num * layer.v_head_dim,
                dtype=q.dtype, device=q.device,
            )

        # Write KV cache if not already done by HpcRopeNorm.
        if save_kv_cache and not metadata.hpc_kv_written:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        key_cache, value_cache = self._get_kv_cache(forward_batch, layer)

        tp_q_head_num = layer.tp_q_head_num
        v_head_dim = layer.v_head_dim
        # SGLang calls forward_extend with a pure-prefill batch; no decode
        # tokens are present, so no slicing is needed (unlike vLLM's unified
        # forward() which interleaves decode then prefill requests).
        output = torch.empty(
            q.shape[0], tp_q_head_num, v_head_dim,
            dtype=q.dtype, device=q.device,
        )

        if self.kv_cache_dtype_str == "fp8_e4m3":
            if metadata.hpc_prefill_q_scale is not None:
                # Fast path: HpcRopeNorm already quantized Q and populated the
                # scale in the correct [bs, H, pad] layout.
                qscale = metadata.hpc_prefill_q_scale.contiguous()
            else:
                # Fallback path: HpcRopeNorm is not active (e.g. Qwen3-30B-A3B
                # with nq=32/nkv=4 which rope_norm_store_kv_fp8 does not support).
                # Quantize Q with SGLang's per-token-per-head FP8 kernel and
                # reshape the resulting [T, H] scale into [bs, H, pad] layout
                # expected by hpc.attention_with_kvcache_prefill_fp8.
                T, H, D = q.view(-1, tp_q_head_num, layer.head_dim).shape
                x = q.view(T, H * D).contiguous()
                x_fp8, scale_th = sglang_per_token_group_quant_fp8(
                    x,
                    group_size=D,
                    column_major_scales=False,
                    scale_tma_aligned=True,
                )
                q = x_fp8.view(T, H, D)
                qscale = pack_scale_th_triton(
                    scale_th=scale_th,           # [T, H]
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    max_seq_len_q=metadata.max_seq_len_q,
                    pad_multiple=128,
                    block=256,
                )
            hpc.attention_with_kvcache_prefill_fp8(
                q.view(-1, tp_q_head_num, layer.head_dim),
                key_cache,
                value_cache,
                qscale,
                layer.k_scale.reshape(1),
                layer.v_scale.reshape(1),
                metadata.cu_seqlens_q,
                metadata.page_table,
                metadata.cache_seqlens_int32,
                metadata.max_seq_len_q,
                output=output,
            )
        else:
            hpc.attention_with_kvcache_prefill_bf16(
                q.view(-1, tp_q_head_num, layer.head_dim),
                key_cache,
                value_cache,
                metadata.cu_seqlens_q,
                metadata.page_table,
                metadata.cache_seqlens_int32,
                metadata.max_seq_len_q,
                output=output,
            )
        return output.view(-1, tp_q_head_num * v_head_dim)

    # ------------------------------------------------------------------
    # Forward: decode
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Single-token decode forward pass.

        Passes split_flag from HpcRopeNorm (FP8 path) when available.
        """
        import hpc

        metadata = self.forward_metadata
        if metadata is None:
            num_tokens = q.shape[0]
            tp_q_head_num = layer.tp_q_head_num
            return torch.zeros(
                num_tokens, tp_q_head_num * layer.v_head_dim,
                dtype=q.dtype, device=q.device,
            )

        # Write KV cache if not already done by HpcRopeNorm.
        if save_kv_cache and not metadata.hpc_kv_written:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        key_cache, value_cache = self._get_kv_cache(forward_batch, layer)

        tp_q_head_num = layer.tp_q_head_num
        v_head_dim = layer.v_head_dim
        # SGLang calls forward_decode with a pure-decode batch; no prefill
        # tokens are present, so no slicing is needed.
        bs = q.shape[0]
        output = torch.empty(bs, tp_q_head_num, v_head_dim, dtype=q.dtype, device=q.device)

        if self.kv_cache_dtype_str == "fp8_e4m3":
            if metadata.hpc_decode_q_scale is not None:
                # Fast path: HpcRopeNorm populated q_scale and (optionally)
                # split_k_flag; use them directly.
                qscale = metadata.hpc_decode_q_scale
                split_flag = metadata.hpc_split_k_flag
            else:
                # Fallback path: quantize BF16 Q with SGLang's per-token-per-head
                # FP8 kernel.  For decode, sglang_per_token_group_quant_fp8 returns
                # scale shaped [bs, H] (column_major_scales=False); hpc.attention_decode_fp8
                # accepts this directly — no pack_scale_th_triton needed.
                H, D = tp_q_head_num, layer.head_dim
                x = q.view(bs, H * D).contiguous()
                x_fp8, qscale = sglang_per_token_group_quant_fp8(
                    x,
                    group_size=D,
                    column_major_scales=False,
                    scale_tma_aligned=True,
                )
                q = x_fp8.view(bs, H, D)
                split_flag = None
            hpc.attention_decode_fp8(
                q.view(bs, tp_q_head_num, layer.head_dim),
                key_cache,
                value_cache,
                metadata.page_table,
                metadata.cache_seqlens_int32,
                qscale,
                layer.k_scale.reshape(1),
                layer.v_scale.reshape(1),
                new_kv_included=True,
                splitk=self.splitk,
                split_flag=split_flag,
                output=output,
            )
        else:
            hpc.attention_decode_bf16(
                q.view(bs, tp_q_head_num, layer.head_dim),
                key_cache,
                value_cache,
                metadata.page_table,
                metadata.cache_seqlens_int32,
                output=output,
                new_kv_included=True,
                splitk=self.splitk,
            )
        return output.view(-1, tp_q_head_num * v_head_dim)
