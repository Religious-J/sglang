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


if __name__ == "__main__":
    unittest.main()
