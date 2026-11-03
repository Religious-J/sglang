# Qwen3.8-27B MXFP4 with native Humming on H20

This recipe targets upstream SGLang main at
`af98fcd545f90eaaf4b26d20bfb42e3ec1d91020` (2026-11-03), with a small fix
that preserves compressed-tensors KV-scale metadata in `HummingConfig` and
registers the checkpoint's calibrated K/V scales on `RadixAttention`.

The runtime uses the upstream native `--quantization humming` implementation.
MXFP4 packing, Qwen weight mapping, and Humming's legacy `actorder="static"`
support come from current upstream dependencies.

## Environment

Validated on one NVIDIA H20 (SM90), TP=1, with:

- PyTorch 2.14.1+cu130 and Triton 3.8.0
- humming-kernels 0.1.18
- flashinfer-python 0.7.2
- compressed-tensors 0.18.0
- Transformers 5.19.0
- sglang-kernel 0.4.10+cu130

Install from this checkout in a CUDA 13-compatible environment. The official
SGLang wheel index supplies the CUDA-specific dependencies:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
SGLANG_BUILD_RUST_EXTS=none uv pip install \
  --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  --extra-index-url https://download.pytorch.org/whl/cu130/ \
  -e ./python
uv pip install --no-deps 'flashinfer-python===0.7.2'
```

`SGLANG_BUILD_RUST_EXTS=none` matches the tested HTTP-serving setup, using the
Python TreeCore fallback. Use the repository's standard build process when you
need its Rust extensions. In a shallow checkout, the development package
version may start with `0.0.0.dev`; record `git rev-parse HEAD` to identify the
source revision.

## Precision

The validated configuration executes **W4A8**: packed E2M1 weights with E8M0
scales in groups of 32, dynamic E4M3 FP8 activations in groups of 128 with FP32
scales, BF16 outputs, and FP32 accumulation. It explicitly replaces the
checkpoint's MXFP4 activation quantization through
`SGLANG_HUMMING_INPUT_QUANT_CONFIG`.

KV cache remains **BF16**. The loader preserves calibrated checkpoint scales,
but this recipe does not validate FP8 KV execution. Text inference is the
validation scope.

## Launch and query

```bash
MODEL_PATH=/path/to/Qwen3.8-27B-MXFP4 bash examples/mxfp4_humming/serve.sh
```

Omit `MODEL_PATH` to load `RedHatAI/Qwen3.8-27B-MXFP4` from Hugging Face.
The script uses explicit KV and Mamba budgets to support eight requests with
the tested lengths, CUDA graphs for decode batch sizes 1/2/4/8, eager prefill,
FA3 attention, Triton GDN decode, and FlashInfer GDN prefill.

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-mxfp4","messages":[{"role":"user","content":"Why is the sky blue? Answer in two sentences."}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Regression tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  test/registered/unit/layers/quantization/test_humming_compressed_tensors_kv_cache.py \
  test/registered/unit/layers/quantization/test_humming_w4afp8_schemas.py \
  test/registered/unit/layers/quantization/test_humming_utils.py \
  test/registered/unit/models/test_qwen3_5_modelopt_fp4.py
```

The new tests cover independent K/V-scale loading, absent metadata, unsupported
scale schemes, and preservation of checkpoint metadata. Existing Humming and
Qwen tests cover the surrounding paths.

## Serving benchmark

Use the same tokenizer for all runs. This samples ShareGPT text, repeating or
truncating it to the requested random lengths:

```bash
python -m sglang.benchmark.serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /path/to/Qwen3.8-27B-MXFP4 \
  --tokenizer /path/to/Qwen3.8-27B-MXFP4 \
  --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --random-input-len 5120 --random-output-len 640 \
  --random-range-ratio 0.8 --num-prompts 30 \
  --request-rate 1 --max-concurrency 8 \
  --temperature 0 --seed 42 \
  --warmup-requests 1 --flush-cache \
  --output-details --output-file result.jsonl
```

Run an excluded full-workload warmup before measuring; the client's short
warmup alone is insufficient for a cold kernel cache. The rate is a target
arrival rate and concurrency eight is a limit. Native client request latency
excludes waiting for the concurrency semaphore. A finite 30-request run is
not a sustained-capacity guarantee.

## H20 validation results

The four CPU test modules above passed 35 tests. Eight simultaneous arithmetic
chat requests all returned the expected answers. A full-workload warmup was
excluded before one measured 30-request serving run:

| Metric | Humming MXFP4 |
| --- | ---: |
| Successful requests | 30/30 |
| Observed concurrency peak | 8 |
| Output throughput | 277.74 token/s |
| P50 TTFT | 0.985 s |
| P50 TPOT | 24.44 ms |
| P50 request completion latency | 15.03 s |
| Whole run duration | 61.29 s |

A post-timing trace contained 4096 Humming GEMM events, including E4M3 FP8
activation and E2M1 FP4 weight types. The trace is dispatch validation, not a
timing result.

A separate 32-question GSM8K smoke, zero-shot, thinking disabled, temperature
zero, concurrency four, and a 1024-output-token budget, scored 29/32 for
Humming MXFP4 (two truncated outputs) and 30/32 for FP8 (no truncations).
Both used this same source revision and tokenizer, with BF16 KV cache. The
sample is too small for an accuracy-equivalence claim. Extending the two
truncated MXFP4 cases to 2048 tokens corrected one; the other was still
truncated. That targeted retry is separate from the original score.
