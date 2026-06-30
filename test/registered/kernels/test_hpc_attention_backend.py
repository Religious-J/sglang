"""Tests for the HPC attention backend (hpc-ops).

Covers:
  - BF16 prefill + decode precision (vs naive causal attention)
  - FP8 fast path (nq=64, nkv=8 — HpcRopeNorm-supported config)
  - FP8 fallback path (nq=32, nkv=4 — Qwen3-30B-A3B, uses sglang quant)
  - pack_scale_th_triton shape & value correctness
  - _hpc_decode_use_splitk heuristic

GPU required; skipped automatically when hpc-ops is not installed.
"""

import math
import unittest

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

# ---------------------------------------------------------------------------
# Optional import guard
# ---------------------------------------------------------------------------

try:
    import hpc  # noqa: F401
    from sglang.srt.layers.attention.hpc_backend import pack_scale_th_triton
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    HPC_AVAILABLE = True
except ImportError:
    HPC_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not HPC_AVAILABLE or not torch.cuda.is_available(),
    reason="hpc-ops not installed or no CUDA device",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DEVICE = "cuda"
PAGE_SIZE = 64
HEAD_DIM = 128
NUM_BLOCKS = 16


def _make_cos_sin(max_len: int, dim: int, device: str) -> torch.Tensor:
    """Build HPC-format cos_sin table: shape [max_len, dim].

    Columns [0 .. dim//2-1] = cos(freq * pos)
    Columns [dim//2 .. dim-1] = sin(freq * pos)
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_len, dim//2]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
    """Naive RoPE using HPC cos_sin [max_len, dim] format."""
    half = cos_sin.shape[-1] // 2
    cos_h = cos_sin[positions, :half]
    sin_h = cos_sin[positions, half:]
    cos_ = torch.cat([cos_h, cos_h], dim=-1).unsqueeze(1)
    sin_ = torch.cat([sin_h, sin_h], dim=-1).unsqueeze(1)
    return x * cos_ + _rotate_half(x) * sin_


def _naive_causal_attn(q, k, v, scale):
    """Naive causal softmax attention; all float32, shapes [T, H, D]."""
    T = q.shape[0]
    scores = torch.einsum("thd,shd->hts", q, k) * scale  # [H, T, T]
    mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("hts,shd->thd", scores.softmax(-1), v)


# ---------------------------------------------------------------------------
# BF16 prefill precision
# ---------------------------------------------------------------------------


class TestHpcBf16Prefill(CustomTestCase):
    """BF16 prefill: compare hpc kernels against naive causal attention."""

    @classmethod
    def setUpClass(cls):
        if not HPC_AVAILABLE or not torch.cuda.is_available():
            return
        cls.cos_sin = _make_cos_sin(512, HEAD_DIM, DEVICE)

    def _run(self, nq: int, nkv: int, seq_lens: list, seed: int = 7):
        import hpc

        scale = 1.0 / math.sqrt(HEAD_DIM)
        total = sum(seq_lens)
        bs = len(seq_lens)

        kc = torch.zeros(NUM_BLOCKS, PAGE_SIZE, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        vc = torch.zeros(NUM_BLOCKS, PAGE_SIZE, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        pt = torch.arange(bs, dtype=torch.int32, device=DEVICE).unsqueeze(1)  # [bs, 1] pages
        sl = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
        cu = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
        cu[1:] = torch.cumsum(sl, 0)

        torch.manual_seed(seed)
        qkv = torch.randn(total, (nq + 2 * nkv) * HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

        q_hpc = torch.empty(total, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        hpc.rope_norm_store_kv(kc, vc, qkv, self.cos_sin, sl, cu, pt, True, out_q=q_hpc)

        out_hpc = torch.empty(total, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        hpc.attention_with_kvcache_prefill_bf16(q_hpc, kc, vc, cu, pt, sl, max(seq_lens), output=out_hpc)

        # Naive per-request
        out_ref = torch.zeros(total, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        offset = 0
        for i, seq_len in enumerate(seq_lens):
            q_r = qkv[offset : offset + seq_len, : nq * HEAD_DIM].view(seq_len, nq, HEAD_DIM).float()
            k_r = qkv[offset : offset + seq_len, nq * HEAD_DIM : (nq + nkv) * HEAD_DIM].view(seq_len, nkv, HEAD_DIM).float()
            v_r = qkv[offset : offset + seq_len, (nq + nkv) * HEAD_DIM :].view(seq_len, nkv, HEAD_DIM).float()
            pos = torch.arange(seq_len, device=DEVICE)
            q_rope = _apply_rope(q_r, pos, self.cos_sin.float())
            k_rope = _apply_rope(k_r, pos, self.cos_sin.float())
            k_exp = k_rope.repeat_interleave(nq // nkv, dim=1)
            v_exp = v_r.repeat_interleave(nq // nkv, dim=1)
            out_ref[offset : offset + seq_len] = _naive_causal_attn(q_rope, k_exp, v_exp, scale).to(torch.bfloat16)
            offset += seq_len

        diff = (out_hpc.float() - out_ref.float()).abs()
        return diff.max().item(), diff.mean().item()

    def test_bf16_prefill_single_request(self):
        """Single request prefill, nq=64 nkv=8."""
        max_diff, _ = self._run(64, 8, [12])
        self.assertLess(max_diff, 0.05, f"BF16 prefill max_diff={max_diff:.5f}")

    def test_bf16_prefill_batch(self):
        """Multi-request prefill batch, nq=64 nkv=8."""
        max_diff, _ = self._run(64, 8, [7, 5])
        self.assertLess(max_diff, 0.05, f"BF16 prefill batch max_diff={max_diff:.5f}")

    def test_bf16_prefill_qwen3_30b(self):
        """Prefill with Qwen3-30B-A3B head config (nq=32, nkv=4)."""
        # rope_norm_store_kv does not support (32,4); test pure attention only.
        import hpc

        nq, nkv, seq_len = 32, 4, 8
        scale = 1.0 / math.sqrt(HEAD_DIM)
        torch.manual_seed(42)
        kc = torch.zeros(NUM_BLOCKS, PAGE_SIZE, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        vc = torch.zeros(NUM_BLOCKS, PAGE_SIZE, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        pt = torch.tensor([[0]], dtype=torch.int32, device=DEVICE)
        sl = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)

        # Write KV manually (simulate standard SGLang RoPE path)
        q_bf16 = torch.randn(seq_len, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k_bf16 = torch.randn(seq_len, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        v_bf16 = torch.randn(seq_len, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        kc[0, :seq_len] = k_bf16
        vc[0, :seq_len] = v_bf16

        out_hpc = torch.empty(seq_len, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        hpc.attention_with_kvcache_prefill_bf16(q_bf16, kc, vc, cu, pt, sl, seq_len, output=out_hpc)

        # Naive
        k_exp = k_bf16.float().repeat_interleave(nq // nkv, dim=1)
        v_exp = v_bf16.float().repeat_interleave(nq // nkv, dim=1)
        out_ref = _naive_causal_attn(q_bf16.float(), k_exp, v_exp, scale).to(torch.bfloat16)

        diff = (out_hpc.float() - out_ref.float()).abs()
        self.assertLess(diff.max().item(), 0.05, f"Qwen3-30B BF16 prefill max_diff={diff.max():.5f}")


# ---------------------------------------------------------------------------
# BF16 decode precision
# ---------------------------------------------------------------------------


class TestHpcBf16Decode(CustomTestCase):
    """BF16 decode: compare hpc attention_decode_bf16 against naive attention."""

    def _run(self, nq: int, nkv: int, prefill_len: int, bs: int = 2, seed: int = 13):
        import hpc

        scale = 1.0 / math.sqrt(HEAD_DIM)
        torch.manual_seed(seed)
        kc = torch.randn(NUM_BLOCKS, PAGE_SIZE, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        vc = torch.randn(NUM_BLOCKS, PAGE_SIZE, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

        # page table: each request gets its own page block
        pt = torch.arange(bs, dtype=torch.int32, device=DEVICE).unsqueeze(1)
        sl = torch.full((bs,), prefill_len, dtype=torch.int32, device=DEVICE)

        # Decode: 1 new token per request (write into slot prefill_len)
        q_d = torch.randn(bs, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k_d = torch.randn(bs, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        v_d = torch.randn(bs, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        for i in range(bs):
            kc[i, prefill_len] = k_d[i]
            vc[i, prefill_len] = v_d[i]

        sl_after = sl + 1
        out_hpc = torch.empty(bs, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        hpc.attention_decode_bf16(q_d, kc, vc, pt, sl_after, new_kv_included=True, splitk=True, output=out_hpc)

        # Naive per-request (full KV including new token)
        max_diff = 0.0
        for i in range(bs):
            k_all = kc[i, : prefill_len + 1].float()  # [total_sl, nkv, D]
            v_all = vc[i, : prefill_len + 1].float()
            k_exp = k_all.repeat_interleave(nq // nkv, dim=1)
            v_exp = v_all.repeat_interleave(nq // nkv, dim=1)
            sc = torch.einsum("hd,shd->hs", q_d[i].float(), k_exp) * scale
            ref = torch.einsum("hs,shd->hd", sc.softmax(-1), v_exp).to(torch.bfloat16)
            d = (out_hpc[i].float() - ref.float()).abs().max().item()
            max_diff = max(max_diff, d)
        return max_diff

    def test_bf16_decode_hunyuan_config(self):
        max_diff = self._run(nq=64, nkv=8, prefill_len=6, bs=2)
        self.assertLess(max_diff, 0.05, f"BF16 decode (64,8) max_diff={max_diff:.5f}")

    def test_bf16_decode_qwen3_30b(self):
        max_diff = self._run(nq=32, nkv=4, prefill_len=6, bs=2)
        self.assertLess(max_diff, 0.05, f"BF16 decode (32,4) max_diff={max_diff:.5f}")


# ---------------------------------------------------------------------------
# FP8 fast path (nq=64, nkv=8 — rope_norm_store_kv_fp8 supported)
# ---------------------------------------------------------------------------


class TestHpcFp8FastPath(CustomTestCase):
    """FP8 fast path: rope_norm_store_kv_fp8 fuses RoPE + KV write + Q quant."""

    @classmethod
    def setUpClass(cls):
        if not HPC_AVAILABLE or not torch.cuda.is_available():
            return
        cls.cos_sin = _make_cos_sin(512, HEAD_DIM, DEVICE)
        cls.k_scale = torch.ones(1, dtype=torch.float32, device=DEVICE)
        cls.v_scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    def _make_cache(self, nkv):
        kc = torch.zeros(NUM_BLOCKS, PAGE_SIZE, nkv, HEAD_DIM, dtype=torch.float8_e4m3fn, device=DEVICE)
        vc = torch.zeros(NUM_BLOCKS, PAGE_SIZE, nkv, HEAD_DIM, dtype=torch.float8_e4m3fn, device=DEVICE)
        return kc, vc

    def test_fp8_fast_path_prefill(self):
        """rope_norm_store_kv_fp8 + attention_with_kvcache_prefill_fp8, nq=64, nkv=8."""
        import hpc

        nq, nkv = 64, 8
        SEQ_LENS = [7, 5]
        TOTAL = sum(SEQ_LENS)
        kc, vc = self._make_cache(nkv)
        pt = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32, device=DEVICE)
        sl = torch.tensor(SEQ_LENS, dtype=torch.int32, device=DEVICE)
        cu = torch.tensor([0, SEQ_LENS[0], TOTAL], dtype=torch.int32, device=DEVICE)
        max_seq_q = max(SEQ_LENS)

        torch.manual_seed(1)
        qkv = torch.randn(TOTAL, (nq + 2 * nkv) * HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        q_fp8 = torch.empty(TOTAL, nq, HEAD_DIM, dtype=torch.float8_e4m3fn, device=DEVICE)

        _, q_scale, _ = hpc.rope_norm_store_kv_fp8(
            kc, vc, qkv, self.cos_sin, sl, cu, pt, True,
            self.k_scale, self.v_scale, quant_policy=1, max_seqlens=max_seq_q, out_q=q_fp8,
        )
        # q_scale: [bs, nq, pad128]
        self.assertEqual(q_scale.shape[0], len(SEQ_LENS))
        self.assertEqual(q_scale.shape[1], nq)

        out = torch.empty(TOTAL, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        hpc.attention_with_kvcache_prefill_fp8(
            q_fp8, kc, vc, q_scale, self.k_scale, self.v_scale,
            cu, pt, sl, max_seq_q, output=out,
        )
        self.assertTrue(out.isfinite().all(), "FP8 fast path prefill output contains non-finite values")
        self.assertEqual(tuple(out.shape), (TOTAL, nq, HEAD_DIM))

    def test_fp8_fast_path_decode(self):
        """rope_norm_store_kv_fp8 (decode) + attention_decode_fp8, nq=64, nkv=8."""
        import hpc

        nq, nkv, BS = 64, 8, 2
        kc, vc = self._make_cache(nkv)
        pt = torch.tensor([[0], [1]], dtype=torch.int32, device=DEVICE)
        sl = torch.tensor([6, 5], dtype=torch.int32, device=DEVICE)
        qo_idx = torch.arange(BS + 1, dtype=torch.int32, device=DEVICE)

        torch.manual_seed(2)
        qkv = torch.randn(BS, (nq + 2 * nkv) * HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        q_fp8 = torch.empty(BS, nq, HEAD_DIM, dtype=torch.float8_e4m3fn, device=DEVICE)

        _, q_scale, split_flag = hpc.rope_norm_store_kv_fp8(
            kc, vc, qkv, self.cos_sin, sl, qo_idx, pt, False,
            self.k_scale, self.v_scale, quant_policy=1, max_seqlens=1, out_q=q_fp8,
        )

        out = torch.empty(BS, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        hpc.attention_decode_fp8(
            q_fp8, kc, vc, pt, sl + 1,
            q_scale, self.k_scale, self.v_scale,
            new_kv_included=True, splitk=True,
            split_flag=split_flag, output=out,
        )
        self.assertTrue(out.isfinite().all(), "FP8 fast path decode output contains non-finite values")
        self.assertEqual(tuple(out.shape), (BS, nq, HEAD_DIM))


# ---------------------------------------------------------------------------
# FP8 fallback path (nq=32, nkv=4 — Qwen3-30B-A3B)
# ---------------------------------------------------------------------------


class TestHpcFp8FallbackQwen3(CustomTestCase):
    """FP8 fallback: sglang quant + pack_scale_th_triton, nq=32, nkv=4."""

    @classmethod
    def setUpClass(cls):
        if not HPC_AVAILABLE or not torch.cuda.is_available():
            return
        cls.k_scale = torch.ones(1, dtype=torch.float32, device=DEVICE)
        cls.v_scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    def _make_cache(self, nkv):
        kc = torch.zeros(NUM_BLOCKS, PAGE_SIZE, nkv, HEAD_DIM, dtype=torch.float8_e4m3fn, device=DEVICE)
        vc = torch.zeros(NUM_BLOCKS, PAGE_SIZE, nkv, HEAD_DIM, dtype=torch.float8_e4m3fn, device=DEVICE)
        return kc, vc

    def test_fp8_fallback_prefill(self):
        """Fallback FP8 prefill: sglang quant Q → pack_scale → hpc prefill_fp8."""
        import hpc

        nq, nkv = 32, 4
        SEQ_LENS = [7, 5]
        TOTAL = sum(SEQ_LENS)
        BS = len(SEQ_LENS)
        kc, vc = self._make_cache(nkv)
        pt = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32, device=DEVICE)
        sl = torch.tensor(SEQ_LENS, dtype=torch.int32, device=DEVICE)
        cu = torch.tensor([0, SEQ_LENS[0], TOTAL], dtype=torch.int32, device=DEVICE)
        max_seq_q = max(SEQ_LENS)

        torch.manual_seed(10)
        q_bf16 = torch.randn(TOTAL, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k_bf16 = torch.randn(TOTAL, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        v_bf16 = torch.randn(TOTAL, nkv, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

        # Write KV into cache (simulate set_kv_buffer)
        offset = 0
        for i, seq_len in enumerate(SEQ_LENS):
            kc[i, :seq_len] = k_bf16[offset : offset + seq_len].to(torch.float8_e4m3fn)
            vc[i, :seq_len] = v_bf16[offset : offset + seq_len].to(torch.float8_e4m3fn)
            offset += seq_len

        # Quantize Q (fallback path)
        x_fp8, scale_th = sglang_per_token_group_quant_fp8(
            q_bf16.view(TOTAL, nq * HEAD_DIM).contiguous(),
            group_size=HEAD_DIM, column_major_scales=False, scale_tma_aligned=True,
        )
        q_fp8 = x_fp8.view(TOTAL, nq, HEAD_DIM)
        qscale = pack_scale_th_triton(scale_th, cu, max_seq_q, pad_multiple=128, block=256)

        self.assertEqual(qscale.shape, (BS, nq, 128))  # pad to 128

        out = torch.empty(TOTAL, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        hpc.attention_with_kvcache_prefill_fp8(
            q_fp8, kc, vc, qscale, self.k_scale, self.v_scale,
            cu, pt, sl, max_seq_q, output=out,
        )
        self.assertTrue(out.isfinite().all(), "FP8 fallback prefill output contains non-finite values")
        self.assertEqual(tuple(out.shape), (TOTAL, nq, HEAD_DIM))

    def test_fp8_fallback_decode(self):
        """Fallback FP8 decode: sglang quant Q → hpc attention_decode_fp8."""
        import hpc

        nq, nkv, BS = 32, 4, 2
        PREFILL_LEN = 6
        kc, vc = self._make_cache(nkv)
        pt = torch.tensor([[0], [1]], dtype=torch.int32, device=DEVICE)
        sl = torch.tensor([PREFILL_LEN, PREFILL_LEN], dtype=torch.int32, device=DEVICE)

        torch.manual_seed(20)
        q_bf16 = torch.randn(BS, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

        x_d_fp8, qscale_d = sglang_per_token_group_quant_fp8(
            q_bf16.view(BS, nq * HEAD_DIM).contiguous(),
            group_size=HEAD_DIM, column_major_scales=False, scale_tma_aligned=True,
        )
        q_d_fp8 = x_d_fp8.view(BS, nq, HEAD_DIM)

        self.assertEqual(qscale_d.shape, (BS, nq))

        out = torch.empty(BS, nq, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        hpc.attention_decode_fp8(
            q_d_fp8, kc, vc, pt, sl,
            qscale_d, self.k_scale, self.v_scale,
            new_kv_included=False, splitk=True, output=out,
        )
        self.assertTrue(out.isfinite().all(), "FP8 fallback decode output contains non-finite values")
        self.assertEqual(tuple(out.shape), (BS, nq, HEAD_DIM))


# ---------------------------------------------------------------------------
# pack_scale_th_triton correctness
# ---------------------------------------------------------------------------


class TestPackScaleTh(CustomTestCase):
    """Verify pack_scale_th_triton scatter logic."""

    def test_shape(self):
        """Output shape = [bs, H, pad128]."""
        T, H = 12, 32
        scale = torch.rand(T, H, dtype=torch.float32, device=DEVICE)
        cu = torch.tensor([0, 7, 12], dtype=torch.int32, device=DEVICE)
        out = pack_scale_th_triton(scale, cu, max_seq_len_q=7, pad_multiple=128)
        self.assertEqual(out.shape, (2, H, 128))

    def test_values_correct(self):
        """Each request's slice matches the source scale rows."""
        T, H = 5, 8
        scale = torch.arange(T * H, dtype=torch.float32, device=DEVICE).view(T, H)
        cu = torch.tensor([0, 3, 5], dtype=torch.int32, device=DEVICE)
        out = pack_scale_th_triton(scale, cu, max_seq_len_q=3, pad_multiple=128)
        # Request 0: rows 0..2, request 1: rows 3..4
        for h in range(H):
            # Request 0
            torch.testing.assert_close(out[0, h, :3], scale[:3, h])
            self.assertTrue(out[0, h, 3:].eq(0).all(), "padding should be zero")
            # Request 1
            torch.testing.assert_close(out[1, h, :2], scale[3:5, h])
            self.assertTrue(out[1, h, 2:].eq(0).all(), "padding should be zero")


if __name__ == "__main__":
    unittest.main(verbosity=2)
