# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, List, Tuple

import ray
import torch
import triton
from common_utils import (
    BenchmarkConfig,
    BlockwiseFp8MoeConfig,
    get_config_filename,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    save_configs,
    sort_config,
)
from ray.experimental.tqdm_ray import tqdm

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_dtype_str,
    get_default_config,
    get_moe_configs,
    is_blockwise_fp8_moe_batch_size_supported,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_device, is_hip, is_sm120_supported, is_xpu

_is_hip = is_hip()
_is_sm120 = is_sm120_supported()
_is_xpu = is_xpu()
_BLOCKWISE_FP8_MOE_MIN_SPEEDUP = 1.01


def _make_benchmark_moe_runner_config(
    num_experts: int,
    ep_size: int,
    architecture: str,
) -> MoeRunnerConfig:
    """Build one production-equivalent runner config for every candidate."""
    all_experts_local = ep_size == 1
    return MoeRunnerConfig(
        inplace=True,
        swiglu_limit=10.0 if architecture == "DeepseekV4ForCausalLM" else None,
        num_experts=num_experts if all_experts_local else None,
        num_local_experts=num_experts if all_experts_local else None,
    )


def _can_tune_blockwise_fp8_moe(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    per_channel_quant: bool,
    block_shape: List[int],
    ep_size: int,
    architecture: str,
) -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.cuda is not None
        and _is_sm120
        and ep_size == 1
        and architecture != "DeepseekV4ForCausalLM"
        and use_fp8_w8a8
        and dtype == torch.bfloat16
        and not per_channel_quant
        and block_shape == [128, 128]
        and hidden_size % 128 == 0
        and (shard_intermediate_size // 2) % 128 == 0
        and is_blockwise_fp8_moe_batch_size_supported(num_tokens)
        and num_tokens * topk <= num_experts
    )


def _get_blockwise_fp8_moe_profiles(
    hidden_size: int,
    intermediate_size: int,
    num_pairs: int,
) -> List[Tuple[str, BlockwiseFp8MoeConfig]]:
    """Return at most eight legal end-to-end fast-path profiles."""
    down_latency = {
        "block_n": 32 if intermediate_size <= 1024 else 64,
        "block_k": 128,
        "num_warps": 4,
        "num_stages": 1 if intermediate_size <= 1024 else 4,
    }
    profiles: List[Tuple[str, BlockwiseFp8MoeConfig]] = [("auto", {})]

    # Direct routing stops paying off once the route set is large enough to
    # amortize alignment. Keep these profiles below the measured 64-pair
    # boundary; larger batches only compare the automatic/aligned schedules.
    if num_pairs < 64:
        profiles.append(
            (
                "direct_unsplit",
                {
                    "direct": True,
                    "gate_num_groups": 1,
                    "gate": {
                        "block_n": 64,
                        "block_k": 128,
                        "num_warps": 4,
                        "num_stages": 4,
                    },
                    "down": down_latency,
                },
            )
        )

    num_k_blocks = hidden_size // 128
    if num_pairs <= 32:
        for groups in (2, 4, 8):
            if num_k_blocks % groups == 0 and 4 <= num_k_blocks // groups <= 32:
                profiles.append(
                    (
                        f"direct_split_{groups}",
                        {
                            "direct": True,
                            "gate_num_groups": groups,
                            "down": down_latency,
                        },
                    )
                )

    if num_pairs < 64:
        profiles.append(
            (
                "direct_balanced",
                {
                    "direct": True,
                    "gate": {
                        "block_n": 128,
                        "block_k": 128,
                        "num_warps": 4,
                        "num_stages": 3,
                    },
                    "down": {
                        "block_n": 64,
                        "block_k": 128,
                        "num_warps": 4,
                        "num_stages": 2,
                    },
                },
            )
        )

    profiles.extend(
        [
            (
                "aligned_m16",
                {
                    "direct": False,
                    "block_m": 16,
                    "gate": {
                        "block_n": 64,
                        "block_k": 128,
                        "num_warps": 4,
                        "num_stages": 4,
                    },
                    "down": {
                        "block_n": 128,
                        "block_k": 128,
                        "num_warps": 4,
                        "num_stages": 2,
                    },
                },
            ),
            (
                "aligned_m32_wide",
                {
                    "direct": False,
                    "block_m": 32,
                    "gate": {
                        "block_n": 128,
                        "block_k": 128,
                        "num_warps": 8,
                        "num_stages": 2,
                    },
                    "down": {
                        "block_n": 128,
                        "block_k": 128,
                        "num_warps": 8,
                        "num_stages": 2,
                    },
                },
            ),
        ]
    )
    return profiles[:8]


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: List[int] = None,
    num_iters: int = 100,
    ep_size: int = 1,
    architecture: str = "",
) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    elif use_int4_w4a16:
        w1 = torch.randint(
            0,
            255,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size // 2,
            ),
            dtype=torch.uint8,
        )
        w2 = torch.randint(
            0,
            255,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 4,
            ),
            dtype=torch.uint8,
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_int4_w4a16:
        block_n = 1 if (block_shape[0] == 0) else block_shape[0]
        block_k = block_shape[1]
        n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
        n_tiles_w2 = (hidden_size + block_n - 1) // block_n
        k_tiles_w1 = (hidden_size + block_k - 1) // block_k
        k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
        w1_scale = torch.randn(
            (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.bfloat16
        )
        w2_scale = torch.randn(
            (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.bfloat16
        )
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(
                num_experts, shard_intermediate_size, dtype=torch.float32
            )
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32
            )

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_config = TopKConfig(
        top_k=topk,
        renormalize=True,
    )
    topk_output = select_experts(x, input_gating, topk_config)

    def prepare(i: int):
        input_gating = gating_output[i]
        new_topk_output = select_experts(x, input_gating, topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        topk_output.topk_ids.copy_(new_topk_output.topk_ids)
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def run():
        moe_runner_config = _make_benchmark_moe_runner_config(
            num_experts,
            ep_size,
            architecture,
        )

        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                topk_output,
                moe_runner_config=moe_runner_config,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    # Flush L2 cache with 256 MB data
    cache_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache_flush.zero_()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        prepare(i)
        start_events[i].record()
        graph.replay()
        end_events[i].record()
    torch.cuda.synchronize()

    latencies: List[float] = []
    for i in range(num_iters):
        latencies.append(start_events[i].elapsed_time(end_events[i]))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self, seed: int, server_args: ServerArgs, architecture: str) -> None:
        torch.set_default_device(get_device())
        torch.get_device_module().manual_seed_all(0)
        self.seed = seed
        self.ep_size = server_args.ep_size
        self.architecture = architecture
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU. Ray isolates each worker to a single visible
        # GPU via CUDA_VISIBLE_DEVICES, so the local ordinal is always 0. On
        # ROCm using the global ray gpu id here raises "invalid device ordinal".
        self.device_id = 0 if is_hip() else int(ray.get_gpu_ids()[0])
        set_global_server_args_for_scheduler(server_args)

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
    ) -> Tuple[BenchmarkConfig, float]:
        torch.cuda.manual_seed_all(0)
        dtype_str = get_config_dtype_str(
            dtype,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int4_w4a16=use_int4_w4a16,
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        N = shard_intermediate_size // 2
        if use_int4_w4a16:
            N = N // 2
        op_config = get_moe_configs(
            num_experts,
            N,
            dtype_str,
            block_n,
            block_k,
            per_channel_quant,
        )
        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_str,
                False,
                block_shape,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            kernel_time = benchmark_config(
                config,
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                per_channel_quant,
                block_shape,
                ep_size=self.ep_size,
                architecture=self.architecture,
            )
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
        search_space: List[BenchmarkConfig],
        architecture: str = "",
    ) -> BenchmarkConfig:
        best_config = None
        best_time = float("inf")
        with (
            torch.get_device_module().device(self.device_id)
            if _is_xpu or _is_hip
            else nullcontext()
        ):
            for config in tqdm(search_space):
                try:
                    kernel_time = benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        use_int4_w4a16,
                        per_channel_quant,
                        block_shape,
                        num_iters=10,
                        ep_size=self.ep_size,
                        architecture=architecture,
                    )
                except (triton.runtime.autotuner.OutOfResources, RuntimeError):
                    # Some configurations may be invalid and fail to compile.
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config

        assert best_config is not None
        if _can_tune_blockwise_fp8_moe(
            num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            per_channel_quant,
            block_shape,
            self.ep_size,
            architecture,
        ):
            candidate_seed = self.seed + num_tokens
            best_candidate_config = None
            best_candidate_time = float("inf")
            best_profile_name = None
            tuning_errors = (
                triton.runtime.autotuner.OutOfResources,
                triton.errors.TritonError,
                RuntimeError,
            )
            for profile_name, profile in _get_blockwise_fp8_moe_profiles(
                hidden_size,
                shard_intermediate_size // 2,
                num_tokens * topk,
            ):
                candidate_config = {
                    **best_config,
                    "USE_BLOCKWISE_FP8_MOE": True,
                    "BLOCKWISE_FP8_MOE_CONFIG": {
                        "profile": profile_name,
                        **profile,
                    },
                }
                try:
                    torch.cuda.manual_seed_all(candidate_seed)
                    candidate_time = benchmark_config(
                        candidate_config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        use_int4_w4a16,
                        per_channel_quant,
                        block_shape,
                        num_iters=10,
                        ep_size=self.ep_size,
                        architecture=architecture,
                    )
                except tuning_errors as error:
                    print(
                        "Skipping blockwise FP8 MoE profile "
                        f"{profile_name!r} for batch_size={num_tokens}: {error}"
                    )
                    continue

                print(
                    "Blockwise FP8 MoE profile "
                    f"{profile_name!r}, batch_size={num_tokens}: "
                    f"{candidate_time:.2f} us"
                )
                if candidate_time < best_candidate_time:
                    best_candidate_time = candidate_time
                    best_candidate_config = candidate_config
                    best_profile_name = profile_name

            if best_candidate_config is not None:
                try:
                    torch.cuda.manual_seed_all(candidate_seed)
                    generic_time = benchmark_config(
                        best_config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        use_int4_w4a16,
                        per_channel_quant,
                        block_shape,
                        num_iters=100,
                        ep_size=self.ep_size,
                        architecture=architecture,
                    )
                    torch.cuda.manual_seed_all(candidate_seed)
                    candidate_time = benchmark_config(
                        best_candidate_config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        use_int4_w4a16,
                        per_channel_quant,
                        block_shape,
                        num_iters=100,
                        ep_size=self.ep_size,
                        architecture=architecture,
                    )
                    speedup = generic_time / candidate_time
                    if speedup >= _BLOCKWISE_FP8_MOE_MIN_SPEEDUP:
                        best_config = best_candidate_config
                        best_time = candidate_time
                        print(
                            "Selected blockwise FP8 MoE profile "
                            f"{best_profile_name!r} for batch_size={num_tokens}: "
                            f"{generic_time:.2f} us -> {candidate_time:.2f} us "
                            f"({speedup:.3f}x)"
                        )
                    else:
                        best_time = generic_time
                        print(
                            "Kept generic Triton MoE for "
                            f"batch_size={num_tokens}: generic={generic_time:.2f} us, "
                            f"best blockwise FP8 profile={candidate_time:.2f} us "
                            f"({speedup:.3f}x; need "
                            f"{_BLOCKWISE_FP8_MOE_MIN_SPEEDUP:.3f}x)"
                        )
                except tuning_errors as error:
                    print(
                        "Skipping final blockwise FP8 MoE comparison for "
                        f"batch_size={num_tokens}: {error}"
                    )
        now = datetime.now()
        print(
            f"{now.ctime()}] Completed tuning for batch_size={num_tokens}, "
            f"best_time={best_time:.2f} us"
        )
        return best_config


def main(args: argparse.Namespace):
    server_args = ServerArgs(
        model_path=args.model, tp_size=args.tp_size, ep_size=args.ep_size
    )

    model_config = get_model_config(
        args.model, args.tp_size, args.ep_size, args.disable_shared_experts_fusion
    )

    E = model_config["num_experts"]
    topk = model_config["topk"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]

    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    use_int4_w4a16 = args.dtype == "int4_w4a16"
    per_channel_quant = args.per_channel_quant

    if args.batch_sizes is not None:
        batch_sizes = args.batch_sizes
    elif args.batch_size is not None:
        batch_sizes = [args.batch_size]
    else:
        batch_sizes = get_default_batch_sizes()

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [
        BenchmarkWorker.remote(
            args.seed,
            server_args,
            model_config["architecture"],
        )
        for _ in range(num_gpus)
    ]

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:
        if args.search_space_file:
            with open(args.search_space_file) as f:
                search_space = json.load(f)
            if not isinstance(search_space, list) or not all(
                isinstance(config, dict) for config in search_space
            ):
                raise ValueError(
                    "--search-space-file must contain a JSON list of configs"
                )
        else:
            search_space = get_configs_compute_bound()
        if block_shape is not None:
            block_k = block_shape[1]
            search_space = [
                config
                for config in search_space
                if block_k % config["BLOCK_SIZE_K"] == 0
            ]

        filename = get_config_filename(
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
        )
        print(
            f"Start tuning over {len(search_space)} configurations to create {filename}..."
        )

        start = time.perf_counter()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                    search_space,
                    model_config["architecture"],
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            filename,
        )
        end = time.perf_counter()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                )
                for batch_size in batch_sizes
            ],
        )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "int4_w4a16"],
        default="auto",
    )
    parser.add_argument(
        "--per-channel-quant",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Tune or benchmark an explicit set of token counts in parallel.",
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--search-space-file",
        type=str,
        help="JSON file containing an explicit list of Triton configs to evaluate with --tune.",
    )
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    args = parser.parse_args()

    main(args)
