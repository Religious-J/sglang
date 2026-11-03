# SPDX-License-Identifier: Apache-2.0
"""Compressed-tensors MXFP4 dense linears on Hopper using Humming W4A8."""

import json
from typing import Optional

import torch

from .compressed_tensors_w4a16_mxfp4 import CompressedTensorsW4A16Mxfp4


class CompressedTensorsW4A8Mxfp4Humming(CompressedTensorsW4A16Mxfp4):
    """Reuse the checkpoint parameter layout, then transform it for Humming.

    The checkpoint's dynamic MXFP4 activation quantization is replaced with
    Humming's per-token E4M3 FP8 activation quantization on SM90.
    """

    def __init__(self, has_input_activations: bool = False, actorder=None):
        super().__init__(has_input_activations=has_input_activations)
        actorder = getattr(actorder, "value", actorder)
        self.actorder = "weight" if actorder == "static" else actorder
        if self.actorder not in (None, "weight"):
            raise NotImplementedError(
                f"Humming MXFP4 does not support activation ordering {actorder}."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from humming.config import GemmType
        from humming.layer import HummingLayer, HummingMethod
        from humming.schema.compressed_tensors import CompressedTensorsWeightSchema

        # compressed-tensors calls activation-ordering "weight" by its legacy
        # alias "static" in this checkpoint. Export has restored column order;
        # Humming's schema accepts the canonical spelling.
        schema = CompressedTensorsWeightSchema(
            format="mxfp4-pack-quantized",
            type="float",
            num_bits=4,
            strategy="group",
            group_size=32,
            actorder=self.actorder,
        )
        with torch.device(layer.weight_packed.device):
            humming_layer = HummingLayer(
                shape_n=layer.output_size_per_partition,
                shape_k=layer.input_size_per_partition,
                weight_config=schema,
                input_config={"dtype": "float8e4m3"},
                torch_dtype=layer.params_dtype,
            )
        # Bind the already loaded tensors; avoid an extra full weight copy.
        humming_layer.weight_packed = torch.nn.Parameter(
            layer.weight_packed.data, requires_grad=False
        )
        humming_layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )
        humming_layer.transform()
        configs = HummingMethod.get_default_tuning_configs(
            layer=humming_layer,
            gemm_type=GemmType.DENSE,
            use_f16_accum=False,
        )
        # Disable Stream-K so capture/replay does not rely on lock state.
        configs = [
            [lo, hi, {**config, "use_stream_k": False}] for lo, hi, config in configs
        ]
        layer.humming_tuning_config = json.dumps(configs)
        layer.humming_compute_config = json.dumps(
            {"gemm_type": "dense", "use_f16_accum": False}
        )
        layer.humming_linear = humming_layer
        del layer.weight_packed
        del layer.weight_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.output_size_per_partition,)
        output = layer.humming_linear(
            x.reshape(-1, x.shape[-1]),
            compute_config=layer.humming_compute_config,
            tuning_config=layer.humming_tuning_config,
        )
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)
