#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-RedHatAI/Qwen3.8-27B-MXFP4}"
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"a_dtype":"float8e4m3","input_scale_group_size":128,"input_scale_dtype":"float32","input_quant_mode":"dynamic_group"}'
export SGLANG_HUMMING_USE_F16_ACCUM=0

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --served-model-name qwen3.8-27b-mxfp4 \
  --quantization humming --dtype bfloat16 --kv-cache-dtype bf16 \
  --attention-backend fa3 --linear-attn-backend triton \
  --linear-attn-prefill-backend flashinfer \
  --context-length 8192 --max-total-tokens 65536 \
  --max-running-requests 8 --max-mamba-cache-size 32 \
  --mem-fraction-static 0.65 --mamba-ssm-dtype float32 \
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 4 8 \
  --cuda-graph-backend-prefill disabled \
  --disable-radix-cache --trust-remote-code --random-seed 42 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-30000}" "$@"
