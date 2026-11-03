# Compressed-tensors MXFP4 dense inference on H20

This branch backports MXFP4 dense-linear loading to SGLang at commit
`85fd90072d1a9f2432842b03588f63b745e524e4` (0.5.13.post1), the revision used
for the H20 validation. It is a deployment backport, with validation scoped to
single-GPU, TP=1 inference of `RedHatAI/Qwen3.8-27B-MXFP4`.

## Precision and backend selection

Set `SGLANG_CT_MXFP4_BACKEND=humming` to opt into Humming. The adapter keeps
packed MXFP4 E2M1 weights and E8M0 group scales (group size 32), transforms the
loaded tensors into Humming's layout, and uses dynamic per-token E4M3 FP8
activations with BF16 outputs and FP32 accumulation. Stream-K is disabled for
CUDA graph capture/replay. On H20 this is **W4A8**, replacing the checkpoint's
MXFP4 activation quantization; it is not native W4A4 execution.

The unset/default backend is Marlin W4A16. Both loaders require BF16 activation
dtype. Marlin additionally requires shard dimensions N divisible by 64 and K
by 128. The Humming schema accepts the legacy compressed-tensors
`actorder="static"` spelling by normalizing it to `"weight"`.

The patch also registers compressed-tensors KV-scale parameters and maps their
Qwen checkpoint names. The validated serving configuration explicitly uses
**BF16 KV cache**. Loading calibrated FP8 KV scales does not establish
correctness of FP8 KV execution for this model/backend combination.

## Environment

Validated versions: PyTorch 2.11.0+cu130, Triton 3.6.0, humming-kernels 0.1.4,
compressed-tensors 0.17.0, and Transformers 5.8.1 on NVIDIA H20 (SM90).

Install SGLang from this checkout in a compatible CUDA environment and add the
optional Humming dependency:

```bash
pip install -e ./python
pip install humming-kernels==0.1.4
```

## Launch

Replace the Hugging Face model ID with a downloaded checkpoint directory if
running offline. The explicit cache budgets below allow eight requests of the
benchmark length; `--max-running-requests` alone does not guarantee that the
Mamba cache has enough capacity.

```bash
export SGLANG_CT_MXFP4_BACKEND=humming
python3 -m sglang.launch_server \
  --model-path RedHatAI/Qwen3.8-27B-MXFP4 \
  --tokenizer-path RedHatAI/Qwen3.8-27B-MXFP4 \
  --served-model-name qwen3.8-27b-mxfp4 \
  --random-seed 24380794 \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --kv-cache-dtype bf16 \
  --attention-backend fa3 \
  --linear-attn-backend triton \
  --context-length 8192 \
  --max-total-tokens 65536 \
  --max-running-requests 8 \
  --max-mamba-cache-size 32 \
  --mem-fraction-static 0.65 \
  --cuda-graph-bs 1 2 4 8 \
  --disable-radix-cache \
  --trust-remote-code \
  --host 127.0.0.1 --port 30000
```

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-mxfp4","messages":[{"role":"user","content":"Why is the sky blue? Answer in two sentences."}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Validation evidence

Before publishing this branch, the implementation was exercised on H20 with
CUDA graph replay and input lengths up to 4096, output lengths up to 1024, and
concurrency 1/4/8/16. Post-timing GPU traces confirmed Humming dense GEMMs; the
FP8 comparison used SM90 DeepGEMM. A separate 5120/640 upper-bound workload,
range ratio 0.8, request rate 1, and concurrency limit 8 was repeated three
times after a full warmup, with identical sampled prompts, requested output
lengths, arrival intervals, tokenizer, and BF16 KV cache on both backends.

| Median of three runs | Humming MXFP4 | FP8 |
| --- | ---: | ---: |
| Output throughput (token/s) | 251.71 | 202.05 |
| P50 TTFT (s) | 1.05 | 1.09 |
| P50 TPOT (ms) | 26.98 | 33.94 |
| P50 request completion latency (s) | 16.40 | 20.53 |

Each measured run completed all 30 requests and reached concurrency eight.
These are finite-load measurements including prefill, arrivals, and drain,
not a sustained-capacity guarantee. Native client latency excludes waiting for
its concurrency semaphore. The repeats reuse the same sample; they measure
run-to-run variation, not independent dataset coverage. These performance
results do not establish improved model accuracy.

For the FP8 comparison, use the matching FP8 checkpoint with the same tokenizer
and launch flags, `--quantization fp8`, and unset `SGLANG_CT_MXFP4_BACKEND`.
`SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` disables exhaustive startup precompilation;
DeepGEMM itself remains enabled. Warm the actual workload before timing.

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --tokenizer RedHatAI/Qwen3.8-27B-MXFP4 \
  --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --random-input-len 5120 \
  --random-output-len 640 \
  --random-range-ratio 0.8 \
  --num-prompts 30 \
  --request-rate 1 \
  --max-concurrency 8 \
  --temperature 0 --seed 42 \
  --warmup-requests 1 --flush-cache \
  --output-details --output-file result.jsonl
```

The native client's short warmup does not replace the excluded full-workload
warmup used for the numbers above. Use one shared tokenizer for comparison:
the evaluated MXFP4 and FP8 repositories had different tokenizer serialization.
