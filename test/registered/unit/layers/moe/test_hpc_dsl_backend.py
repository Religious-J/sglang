import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.hpc_dsl import (
    HpcDslMoeQuantInfo,
    fused_experts_none_to_hpc_dsl,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHpcDslBackend(unittest.TestCase):
    def setUp(self):
        self.hidden_states = torch.randn(3, 8, dtype=torch.bfloat16)
        self.w13_weight = torch.randn(4, 12, 8, dtype=torch.bfloat16)
        self.w2_weight = torch.randn(4, 8, 6, dtype=torch.bfloat16)
        self.topk_ids = torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.int64)
        self.topk_weights = torch.rand(3, 2, dtype=torch.bfloat16)
        self.dispatch_output = StandardDispatchOutput(
            hidden_states=self.hidden_states,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=self.topk_weights,
                topk_ids=self.topk_ids,
                router_logits=torch.empty(0),
            ),
        )
        self.quant_info = HpcDslMoeQuantInfo(
            w13_weight=self.w13_weight,
            w2_weight=self.w2_weight,
            global_num_experts=4,
            rank_ep=0,
        )

    def test_backend_enum(self):
        self.assertTrue(MoeRunnerBackend("hpc_dsl").is_hpc_dsl())

    def test_fused_adapter_converts_routing_and_provides_output(self):
        workspace_output = torch.ones(3, 8, dtype=torch.bfloat16)

        def fake_fuse(
            x, w13, w2, topk_ids, topk_weights, rank_ep, num_experts, out=None
        ):
            self.assertIs(x, self.hidden_states)
            self.assertIs(w13, self.w13_weight)
            self.assertIs(w2, self.w2_weight)
            self.assertEqual(topk_ids.dtype, torch.int32)
            self.assertEqual(topk_weights.dtype, torch.float32)
            self.assertEqual(rank_ep, 0)
            self.assertEqual(num_experts, 4)
            self.assertIsNotNone(out)
            out.copy_(workspace_output)
            return out

        config = MoeRunnerConfig(routed_scaling_factor=2.0)
        with patch(
            "sglang.srt.layers.moe.moe_runner.hpc_dsl._load_hpc_dsl_fuse_moe",
            return_value=fake_fuse,
        ):
            result = fused_experts_none_to_hpc_dsl(
                self.dispatch_output, self.quant_info, config
            )

        self.assertNotEqual(
            result.hidden_states.data_ptr(), workspace_output.data_ptr()
        )
        torch.testing.assert_close(
            result.hidden_states,
            torch.full_like(result.hidden_states, 2.0),
        )

    def test_rejects_unsupported_config(self):
        config = MoeRunnerConfig(apply_router_weight_on_input=True)
        with self.assertRaisesRegex(ValueError, "apply_router_weight_on_input"):
            fused_experts_none_to_hpc_dsl(self.dispatch_output, self.quant_info, config)

    def test_fused_adapter_supports_blockwise_fp8(self):
        hidden_states = torch.randn(3, 128, dtype=torch.bfloat16)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=self.dispatch_output.topk_output,
        )
        w13_weight = torch.zeros(4, 256, 128, dtype=torch.float8_e4m3fn)
        w2_weight = torch.zeros(4, 128, 128, dtype=torch.float8_e4m3fn)
        w13_scale = torch.ones(4, 2, 1, dtype=torch.float32)
        w2_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        quant_info = HpcDslMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_weight_scale_inv=w13_scale,
            w2_weight_scale_inv=w2_scale,
            block_shape=(128, 128),
            global_num_experts=4,
            rank_ep=0,
        )

        def fake_fuse(
            x,
            w13,
            w2,
            w13_scale_inv,
            w2_scale_inv,
            topk_ids,
            topk_weights,
            rank_ep,
            num_experts,
            *,
            block_shape,
            out,
        ):
            self.assertIs(x, hidden_states)
            self.assertIs(w13, w13_weight)
            self.assertIs(w2, w2_weight)
            self.assertIs(w13_scale_inv, w13_scale)
            self.assertIs(w2_scale_inv, w2_scale)
            self.assertEqual(topk_ids.dtype, torch.int32)
            self.assertEqual(topk_weights.dtype, torch.float32)
            self.assertEqual(rank_ep, 0)
            self.assertEqual(num_experts, 4)
            self.assertEqual(block_shape, (128, 128))
            out.fill_(3)
            return out

        with patch(
            "sglang.srt.layers.moe.moe_runner.hpc_dsl."
            "_load_hpc_dsl_fuse_moe_blockwise_fp8",
            return_value=fake_fuse,
        ):
            result = fused_experts_none_to_hpc_dsl(
                dispatch_output, quant_info, MoeRunnerConfig()
            )

        torch.testing.assert_close(
            result.hidden_states,
            torch.full_like(result.hidden_states, 3.0),
        )

    def test_rejects_invalid_blockwise_fp8_scale_shape(self):
        hidden_states = torch.randn(3, 128, dtype=torch.bfloat16)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=self.dispatch_output.topk_output,
        )
        quant_info = HpcDslMoeQuantInfo(
            w13_weight=torch.zeros(4, 256, 128, dtype=torch.float8_e4m3fn),
            w2_weight=torch.zeros(4, 128, 128, dtype=torch.float8_e4m3fn),
            w13_weight_scale_inv=torch.ones(4, 3, 1, dtype=torch.float32),
            w2_weight_scale_inv=torch.ones(4, 1, 1, dtype=torch.float32),
            block_shape=(128, 128),
            global_num_experts=4,
            rank_ep=0,
        )

        with self.assertRaisesRegex(ValueError, "w13 scale shape"):
            fused_experts_none_to_hpc_dsl(
                dispatch_output, quant_info, MoeRunnerConfig()
            )

    def test_rejects_unaligned_blockwise_fp8_dimensions(self):
        quant_info = HpcDslMoeQuantInfo(
            w13_weight=torch.zeros(4, 12, 8, dtype=torch.float8_e4m3fn),
            w2_weight=torch.zeros(4, 8, 6, dtype=torch.float8_e4m3fn),
            w13_weight_scale_inv=torch.ones(4, 1, 1, dtype=torch.float32),
            w2_weight_scale_inv=torch.ones(4, 1, 1, dtype=torch.float32),
            block_shape=(128, 128),
            global_num_experts=4,
            rank_ep=0,
        )

        with self.assertRaisesRegex(ValueError, "dimensions divisible by 128"):
            fused_experts_none_to_hpc_dsl(
                self.dispatch_output, quant_info, MoeRunnerConfig()
            )

    def test_fused_adapter_supports_native_mxfp8(self):
        hidden_states = torch.randn(3, 128, dtype=torch.bfloat16)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=self.dispatch_output.topk_output,
        )
        w13_weight = torch.zeros(4, 256, 128, dtype=torch.float8_e4m3fn)
        w2_weight = torch.zeros(4, 128, 128, dtype=torch.float8_e4m3fn)
        w13_scale = torch.zeros(4, 1024, dtype=torch.uint8)
        w2_scale = torch.zeros(4, 512, dtype=torch.uint8)
        quant_info = HpcDslMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_weight_scale_inv=w13_scale,
            w2_weight_scale_inv=w2_scale,
            weight_format="mxfp8",
            global_num_experts=4,
            rank_ep=0,
        )

        class FakeWeights:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        def fake_fuse(
            x,
            weights,
            topk_ids,
            topk_weights,
            rank_ep,
            num_experts,
            *,
            out,
            workspace_slot,
        ):
            self.assertIs(x, hidden_states)
            self.assertIs(weights.gate_up, w13_weight)
            self.assertIs(weights.gate_up_scale, w13_scale)
            self.assertIs(weights.down, w2_weight)
            self.assertIs(weights.down_scale, w2_scale)
            self.assertEqual(weights.hidden_size, 128)
            self.assertEqual(weights.intermediate_size, 128)
            self.assertEqual(topk_ids.dtype, torch.int32)
            self.assertEqual(topk_weights.dtype, torch.float32)
            self.assertEqual(rank_ep, 0)
            self.assertEqual(num_experts, 4)
            self.assertIsNone(workspace_slot)
            out.fill_(5)
            return out

        with patch(
            "sglang.srt.layers.moe.moe_runner.hpc_dsl._load_hpc_dsl_mxfp8_api",
            return_value=(fake_fuse, None, FakeWeights),
        ):
            result = fused_experts_none_to_hpc_dsl(
                dispatch_output, quant_info, MoeRunnerConfig()
            )

        torch.testing.assert_close(
            result.hidden_states,
            torch.full_like(result.hidden_states, 5.0),
        )

    def test_rejects_invalid_mxfp8_scale_shape(self):
        hidden_states = torch.randn(3, 128, dtype=torch.bfloat16)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=self.dispatch_output.topk_output,
        )
        quant_info = HpcDslMoeQuantInfo(
            w13_weight=torch.zeros(4, 256, 128, dtype=torch.float8_e4m3fn),
            w2_weight=torch.zeros(4, 128, 128, dtype=torch.float8_e4m3fn),
            w13_weight_scale_inv=torch.zeros(4, 1023, dtype=torch.uint8),
            w2_weight_scale_inv=torch.zeros(4, 512, dtype=torch.uint8),
            weight_format="mxfp8",
            global_num_experts=4,
            rank_ep=0,
        )

        with self.assertRaisesRegex(ValueError, "MXFP8 w13 scale shape"):
            fused_experts_none_to_hpc_dsl(
                dispatch_output, quant_info, MoeRunnerConfig()
            )


if __name__ == "__main__":
    unittest.main()
