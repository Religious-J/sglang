import importlib
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe,
    fused_moe_triton_config,
    override_config,
)
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


def _load_tuning_module():
    tuning_dir = (
        Path(__file__).resolve().parents[5]
        / "benchmark"
        / "kernels"
        / "fused_moe_triton"
    )
    sys.path.insert(0, str(tuning_dir))
    try:
        return importlib.import_module("tuning_fused_moe_triton")
    finally:
        sys.path.pop(0)


class TestBlockwiseFp8MoeTuningConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuning = _load_tuning_module()

    def test_fast_path_tuning_is_limited_to_16_tokens(self):
        common_args = {
            "num_experts": 256,
            "shard_intermediate_size": 512,
            "hidden_size": 7168,
            "topk": 8,
            "dtype": torch.bfloat16,
            "use_fp8_w8a8": True,
            "per_channel_quant": False,
            "block_shape": [128, 128],
            "ep_size": 1,
            "architecture": "DeepseekV3ForCausalLM",
        }
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.version, "cuda", "13.0"),
            patch.object(self.tuning, "_is_sm120", True),
        ):
            self.assertTrue(
                self.tuning._can_tune_blockwise_fp8_moe(num_tokens=16, **common_args)
            )
            self.assertFalse(
                self.tuning._can_tune_blockwise_fp8_moe(num_tokens=17, **common_args)
            )
            self.assertFalse(
                self.tuning._can_tune_blockwise_fp8_moe(
                    num_tokens=16,
                    **{**common_args, "ep_size": 2},
                )
            )
            self.assertFalse(
                self.tuning._can_tune_blockwise_fp8_moe(
                    num_tokens=16,
                    **{
                        **common_args,
                        "architecture": "DeepseekV4ForCausalLM",
                    },
                )
            )
        with patch.object(self.tuning, "_is_sm120", False):
            self.assertFalse(
                self.tuning._can_tune_blockwise_fp8_moe(num_tokens=16, **common_args)
            )

    def test_benchmark_runner_config_matches_ep_semantics(self):
        ep1_config = self.tuning._make_benchmark_moe_runner_config(
            num_experts=256,
            ep_size=1,
            architecture="DeepseekV3ForCausalLM",
        )
        self.assertEqual(ep1_config.num_experts, 256)
        self.assertEqual(ep1_config.num_local_experts, 256)

        ep2_config = self.tuning._make_benchmark_moe_runner_config(
            num_experts=128,
            ep_size=2,
            architecture="DeepseekV3ForCausalLM",
        )
        self.assertIsNone(ep2_config.num_experts)
        self.assertIsNone(ep2_config.num_local_experts)

        dsv4_config = self.tuning._make_benchmark_moe_runner_config(
            num_experts=256,
            ep_size=1,
            architecture="DeepseekV4ForCausalLM",
        )
        self.assertEqual(dsv4_config.swiglu_limit, 10.0)

    def test_benchmark_worker_forwards_architecture(self):
        worker_cls = self.tuning.BenchmarkWorker.__ray_metadata__.modified_class
        worker = object.__new__(worker_cls)
        worker.device_id = 0
        worker.ep_size = 1
        worker.architecture = "Qwen3MoeForCausalLM"
        generic_config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }

        with (
            patch.object(torch.cuda, "manual_seed_all"),
            patch.object(
                self.tuning,
                "get_moe_configs",
                return_value={1: generic_config},
            ),
            patch.object(
                self.tuning,
                "benchmark_config",
                return_value=1.0,
            ) as benchmark_config,
        ):
            config, latency = worker.benchmark(
                num_tokens=1,
                num_experts=128,
                shard_intermediate_size=1536,
                hidden_size=2048,
                topk=8,
                dtype=torch.bfloat16,
                use_fp8_w8a8=True,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=[128, 128],
            )

        self.assertEqual(config, generic_config)
        self.assertEqual(latency, 1.0)
        self.assertEqual(
            benchmark_config.call_args.kwargs["architecture"],
            "Qwen3MoeForCausalLM",
        )

    def test_profiles_are_pruned_by_route_pair_count(self):
        small = {
            name
            for name, _ in self.tuning._get_blockwise_fp8_moe_profiles(
                hidden_size=7168,
                intermediate_size=256,
                num_pairs=32,
            )
        }
        medium = {
            name
            for name, _ in self.tuning._get_blockwise_fp8_moe_profiles(
                hidden_size=7168,
                intermediate_size=256,
                num_pairs=48,
            )
        }
        large = {
            name
            for name, _ in self.tuning._get_blockwise_fp8_moe_profiles(
                hidden_size=7168,
                intermediate_size=256,
                num_pairs=64,
            )
        }

        self.assertIn("direct_split_2", small)
        self.assertIn("direct_unsplit", medium)
        self.assertFalse(any(name.startswith("direct_split_") for name in medium))
        self.assertFalse(any(name.startswith("direct") for name in large))
        self.assertEqual(large, {"auto", "aligned_m16", "aligned_m32_wide"})

    def test_algorithm_metadata_is_selected_but_not_forwarded_to_kernel(self):
        profile = {
            "profile": "direct_unsplit",
            "direct": True,
            "gate_num_groups": 1,
            "down": {
                "block_n": 32,
                "block_k": 128,
                "num_warps": 4,
                "num_stages": 1,
            },
        }
        config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
            "USE_BLOCKWISE_FP8_MOE": True,
            "BLOCKWISE_FP8_MOE_CONFIG": profile,
        }
        with override_config(config):
            self.assertTrue(
                fused_moe_triton_config.use_tuned_blockwise_fp8_moe(
                    (8, 1280, 128),
                    "fp8_w8a8",
                    2,
                    block_shape=[128, 128],
                )
            )
            self.assertEqual(
                fused_moe_triton_config.get_tuned_blockwise_fp8_moe_config(
                    (8, 1280, 128),
                    "fp8_w8a8",
                    2,
                    block_shape=[128, 128],
                ),
                profile,
            )
            kernel_config = fused_moe_triton_config.try_get_optimal_moe_config(
                (8, 256, 1280),
                (8, 1280, 128),
                2,
                "fp8_w8a8",
                2,
                block_shape=[128, 128],
            )

        self.assertNotIn("USE_BLOCKWISE_FP8_MOE", kernel_config)
        self.assertNotIn(
            "BLOCKWISE_FP8_MOE_CONFIG",
            kernel_config,
        )
        self.assertEqual(kernel_config["BLOCK_SIZE_M"], 32)

    def test_algorithm_metadata_requires_exact_batch_size(self):
        configs = {
            1: {"BLOCK_SIZE_M": 16},
            4: {
                "BLOCK_SIZE_M": 32,
                "USE_BLOCKWISE_FP8_MOE": True,
                "BLOCKWISE_FP8_MOE_CONFIG": {"profile": "auto"},
            },
        }
        with patch.object(
            fused_moe_triton_config, "get_moe_configs", return_value=configs
        ):
            self.assertFalse(
                fused_moe_triton_config.use_tuned_blockwise_fp8_moe(
                    (8, 1280, 128),
                    "fp8_w8a8",
                    3,
                    block_shape=[128, 128],
                )
            )
            self.assertEqual(
                fused_moe_triton_config.get_tuned_blockwise_fp8_moe_config(
                    (8, 1280, 128),
                    "fp8_w8a8",
                    3,
                    block_shape=[128, 128],
                ),
                None,
            )
            self.assertEqual(
                fused_moe_triton_config.get_tuned_blockwise_fp8_moe_config(
                    (8, 1280, 128),
                    "fp8_w8a8",
                    4,
                    block_shape=[128, 128],
                ),
                {"profile": "auto"},
            )

    def test_algorithm_metadata_fails_closed(self):
        invalid_config = {
            "USE_BLOCKWISE_FP8_MOE": True,
            "BLOCKWISE_FP8_MOE_CONFIG": "invalid",
        }
        with override_config(invalid_config):
            self.assertIsNone(
                fused_moe_triton_config.get_tuned_blockwise_fp8_moe_config(
                    (8, 1280, 128),
                    "fp8_w8a8",
                    4,
                    block_shape=[128, 128],
                )
            )

        with patch.object(
            fused_moe_triton_config,
            "get_moe_configs",
            return_value={4: "invalid"},
        ):
            self.assertIsNone(
                fused_moe_triton_config.get_tuned_blockwise_fp8_moe_config(
                    (8, 1280, 128),
                    "fp8_w8a8",
                    4,
                    block_shape=[128, 128],
                )
            )

        persisted_config = {
            "BLOCK_SIZE_M": 16,
            "USE_BLOCKWISE_FP8_MOE": True,
            "BLOCKWISE_FP8_MOE_CONFIG": {"profile": "auto"},
        }
        self.assertEqual(
            fused_moe_triton_config._kernel_config(persisted_config),
            {"BLOCK_SIZE_M": 16},
        )

    def test_cross_triton_fallback_strips_algorithm_metadata(self):
        persisted_config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
            "USE_BLOCKWISE_FP8_MOE": True,
            "BLOCKWISE_FP8_MOE_CONFIG": {"profile": "auto"},
        }
        with tempfile.TemporaryDirectory() as config_dir:
            config_path = (
                Path(config_dir)
                / "configs"
                / "triton_0_0_0"
                / fused_moe_triton_config.get_config_file_name(
                    8,
                    128,
                    "fp8_w8a8",
                    [128, 128],
                )
            )
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({"4": persisted_config}))
            server_args = SimpleNamespace(enable_deterministic_inference=False)

            fused_moe_triton_config.get_moe_configs.cache_clear()
            with (
                patch.dict(os.environ, {"SGLANG_MOE_CONFIG_DIR": config_dir}),
                patch.object(
                    fused_moe_triton_config,
                    "get_server_args",
                    return_value=server_args,
                ),
            ):
                configs = fused_moe_triton_config.get_moe_configs(
                    8,
                    128,
                    "fp8_w8a8",
                    128,
                    128,
                )
            fused_moe_triton_config.get_moe_configs.cache_clear()

        self.assertIsNotNone(configs)
        self.assertNotIn("USE_BLOCKWISE_FP8_MOE", configs[4])
        self.assertNotIn("BLOCKWISE_FP8_MOE_CONFIG", configs[4])
        self.assertEqual(configs[4]["BLOCK_SIZE_M"], 16)

    def test_runtime_policy_fails_closed(self):
        server_args = SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
        with (
            patch.object(fused_moe, "_is_cuda", True),
            patch.object(fused_moe, "_is_sm120", True),
            patch.object(fused_moe, "_use_blockwise_fp8_moe", True),
            patch.object(fused_moe, "get_server_args", return_value=server_args),
        ):
            self.assertTrue(
                fused_moe._is_blockwise_fp8_moe_runtime_policy_enabled(16, 8)
            )
            self.assertFalse(
                fused_moe._is_blockwise_fp8_moe_runtime_policy_enabled(17, 8)
            )
            server_args.enable_fused_moe_sum_all_reduce = True
            self.assertFalse(
                fused_moe._is_blockwise_fp8_moe_runtime_policy_enabled(16, 8)
            )
            self.assertTrue(
                fused_moe._is_blockwise_fp8_moe_runtime_policy_enabled(16, 2)
            )

        with patch.object(fused_moe, "_is_sm120", False):
            self.assertFalse(
                fused_moe._is_blockwise_fp8_moe_runtime_policy_enabled(16, 8)
            )


@unittest.skipUnless(
    is_sm120_supported(),
    "This numerical regression test currently runs on a validated SM120/SM121 GPU",
)
class TestBlockwiseFp8MoeFastPath(unittest.TestCase):
    def test_matches_generic_path_and_supports_inplace(self):
        torch.manual_seed(52001)
        fp8 = torch.float8_e4m3fn
        experts, hidden, intermediate, topk = 8, 1280, 128, 2
        tokens = 2

        hidden_states = torch.randn(
            (tokens, hidden), device="cuda", dtype=torch.bfloat16
        )
        w1 = torch.randint(
            -8,
            9,
            (experts, 2 * intermediate, hidden),
            device="cuda",
            dtype=torch.int8,
        ).to(fp8)
        w2 = torch.randint(
            -8,
            9,
            (experts, hidden, intermediate),
            device="cuda",
            dtype=torch.int8,
        ).to(fp8)
        w1_scale = (
            torch.rand(
                (experts, 2 * intermediate // 128, hidden // 128),
                device="cuda",
            )
            * 0.001
            + 0.0015
        )
        w2_scale = (
            torch.rand(
                (experts, hidden // 128, intermediate // 128),
                device="cuda",
            )
            * 0.001
            + 0.0015
        )
        topk_ids = (
            torch.rand((tokens, experts), device="cuda")
            .topk(topk, dim=1)
            .indices.to(torch.int32)
            .contiguous()
        )
        topk_weights = torch.rand((tokens, topk), device="cuda", dtype=torch.float32)
        topk_weights /= topk_weights.sum(dim=1, keepdim=True)

        server_args = SimpleNamespace(
            enable_fused_moe_sum_all_reduce=False,
            enable_deterministic_inference=False,
        )

        def run(enabled, value, *, inplace=False, tuning_config=None):
            with (
                patch.object(fused_moe, "_use_blockwise_fp8_moe", enabled),
                patch.object(fused_moe, "_is_sm120", True),
                patch.object(fused_moe, "get_server_args", return_value=server_args),
                patch.object(
                    fused_moe_triton_config,
                    "get_server_args",
                    return_value=server_args,
                ),
                patch.object(
                    fused_moe,
                    "get_tuned_blockwise_fp8_moe_config",
                    return_value=(tuning_config or {}) if enabled else None,
                ),
                patch.object(fused_moe, "get_tp_group", return_value=None),
                patch.object(
                    fused_moe, "use_symmetric_memory", return_value=nullcontext()
                ),
            ):
                return fused_moe.fused_experts_impl(
                    value,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    inplace=inplace,
                    use_fp8_w8a8=True,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    block_shape=[128, 128],
                    filter_expert=False,
                )

        fast = run(True, hidden_states).clone()
        fallback = run(False, hidden_states).clone()
        rel_l2 = (fast.float() - fallback.float()).norm() / fallback.float().norm()
        cosine = torch.nn.functional.cosine_similarity(
            fast.float().flatten(), fallback.float().flatten(), dim=0
        )
        self.assertLess(float(rel_l2), 0.02)
        self.assertGreater(float(cosine), 0.999)

        tuned_profiles = [
            {
                "direct": True,
                "gate_num_groups": 1,
                "gate": {
                    "block_n": 64,
                    "block_k": 128,
                    "num_warps": 4,
                    "num_stages": 4,
                },
                "down": {
                    "block_n": 32,
                    "block_k": 128,
                    "num_warps": 4,
                    "num_stages": 1,
                },
            },
            {
                "direct": True,
                "gate_num_groups": 2,
                "down": {
                    "block_n": 32,
                    "block_k": 128,
                    "num_warps": 4,
                    "num_stages": 1,
                },
            },
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
            # A stale profile may carry a K split that is illegal for the
            # current shape; runtime must safely fall back to the auto chooser.
            {"direct": True, "gate_num_groups": 8},
        ]
        for tuned_profile in tuned_profiles:
            tuned = run(
                True,
                hidden_states,
                tuning_config=tuned_profile,
            ).clone()
            tuned_rel_l2 = (
                tuned.float() - fallback.float()
            ).norm() / fallback.float().norm()
            tuned_cosine = torch.nn.functional.cosine_similarity(
                tuned.float().flatten(), fallback.float().flatten(), dim=0
            )
            self.assertLess(float(tuned_rel_l2), 0.02)
            self.assertGreater(float(tuned_cosine), 0.999)

        inplace_input = hidden_states.clone()
        inplace_output = run(True, inplace_input, inplace=True)
        self.assertEqual(inplace_output.data_ptr(), inplace_input.data_ptr())
        torch.testing.assert_close(inplace_output, fast, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
