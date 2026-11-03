"""Regression coverage for calibrated KV scales with native Humming."""

import copy
import unittest

import torch

from sglang.srt.layers.quantization.humming import HummingConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def checkpoint_config(kv_scheme):
    return {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "type": "float",
                    "num_bits": 4,
                    "strategy": "group",
                    "group_size": 32,
                    "symmetric": True,
                    "dynamic": False,
                    "actorder": "static",
                },
                "input_activations": None,
            }
        },
        "kv_cache_scheme": kv_scheme,
    }


def static_fp8_scheme():
    return {
        "type": "float",
        "num_bits": 8,
        "strategy": "tensor",
        "symmetric": True,
        "dynamic": False,
    }


class TestHummingCompressedTensorsKVCache(CustomTestCase):
    def make_attention(self, config):
        return RadixAttention(
            num_heads=2,
            head_dim=8,
            scaling=1.0,
            num_kv_heads=2,
            layer_id=0,
            quant_config=config,
            prefix="model.layers.0.attn",
        )

    def test_checkpoint_scales_register_and_load_independently(self):
        checkpoint = checkpoint_config(static_fp8_scheme())
        original = copy.deepcopy(checkpoint)
        config = HummingConfig.from_config(checkpoint)
        attention = self.make_attention(config)
        self.assertIsInstance(attention.k_scale, torch.nn.Parameter)
        self.assertIsInstance(attention.v_scale, torch.nn.Parameter)
        self.assertEqual(config.kv_cache_quant_algo, "FP8")
        self.assertEqual(attention.k_scale.item(), -1.0)
        self.assertEqual(attention.v_scale.item(), -1.0)
        default_weight_loader(attention.k_scale, torch.tensor([0.03125]))
        default_weight_loader(attention.v_scale, torch.tensor([0.0625]))
        attention.quant_method.process_weights_after_loading(attention)
        self.assertEqual(attention.k_scale_float, 0.03125)
        self.assertEqual(attention.v_scale_float, 0.0625)
        self.assertEqual(checkpoint, original)

    def test_missing_scheme_leaves_kv_unquantized(self):
        config = HummingConfig.from_config(checkpoint_config(None))
        attention = self.make_attention(config)
        self.assertIsNone(attention.k_scale)
        self.assertIsNone(attention.v_scale)
        self.assertIsNone(config.kv_cache_quant_algo)

    def test_unsupported_schemes_do_not_register_misleading_scales(self):
        for override in (
            {"dynamic": True},
            {"strategy": "channel"},
            {"symmetric": False},
            {"num_bits": 4},
        ):
            with self.subTest(override=override):
                scheme = {**static_fp8_scheme(), **override}
                config = HummingConfig.from_config(checkpoint_config(scheme))
                attention = self.make_attention(config)
                self.assertIsNone(attention.k_scale)
                self.assertIsNone(attention.v_scale)
                self.assertIsNone(config.kv_cache_quant_algo)


if __name__ == "__main__":
    unittest.main()
