from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .native_planes import (
    NativePlaneLayer,
    NativePlanePack,
    NativePlanePrerequisiteError,
    _fail,
)

try:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
except ImportError as exc:  # pragma: no cover - wheel dependency is mandatory
    raise RuntimeError("banana-smasher-plugin requires stock vLLM") from exc


_DEEPSEEK_V4_EXPERT_PREFIX = re.compile(
    r"(?:^|\.)model\.layers\.(?P<layer>[0-9]+)\.ffn\.experts$"
)


class BananaSmasherMoEMethod(FusedMoEMethodBase):
    """Stock-vLLM routed-expert method backed only by P1016 native planes."""

    def __init__(
        self,
        quant_config: "BananaSmasherQuantizationConfig",
        moe: Any,
        layer_index: int,
        prefix: str,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.layer_index = int(layer_index)
        self.prefix = prefix
        self.native_layer: NativePlaneLayer | None = None

    @property
    def supports_internal_mk(self) -> bool:
        # vLLM 0.24 uses this migration flag to leave a quant method's native
        # apply() hook intact instead of wrapping it in a stock GEMM method.
        return True

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        return False

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del extra_weight_attrs
        expected = (256, 4096, 2048, torch.bfloat16)
        actual = (
            int(num_experts),
            int(hidden_size),
            int(intermediate_size_per_partition),
            params_dtype,
        )
        if actual != expected:
            raise _fail(
                f"stock DeepSeek-V4 MoE shape prerequisite mismatch for {self.prefix}: "
                f"actual={actual} expected={expected}"
            )
        pack = self.quant_config.pack
        if pack is None:
            raise _fail("quantization config was not bound to the stock model root")
        device = next(layer.buffers(), torch.empty(0)).device
        if device.type != "cuda":
            raise _fail(
                f"accelerated native-plane layer requires CUDA, got device={device}"
            )
        self.native_layer = NativePlaneLayer(
            pack,
            self.layer_index,
            device=device,
        )
        # No dense routed-expert parameters are registered. The complete V5
        # checkpoint intentionally carries only dense/non-expert shards; the
        # immutable expert planes above are the sole routed weight source.
        layer.bs_native_plane_layer = self.native_layer

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        del layer
        return FusedMoEQuantConfig.make()

    def maybe_make_prepare_finalize(self, routing_tables=None):
        del routing_tables
        raise _fail("stock modular-GEMM wrapping is forbidden for native planes")

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del layer, shared_experts, shared_experts_input
        native_layer = self.native_layer
        if native_layer is None:
            raise _fail(f"native planes were not loaded for {self.prefix}")
        original_shape = x.shape
        flat = x.reshape(-1, x.shape[-1])
        weights = topk_weights.reshape(flat.shape[0], -1)
        ids = topk_ids.to(device=flat.device, dtype=torch.int64).reshape(
            flat.shape[0], -1
        )
        if tuple(weights.shape) != tuple(ids.shape) or ids.shape[1] != 6:
            raise _fail(
                f"stock DeepSeek-V4 top-k route shape mismatch for {self.prefix}: "
                f"weights={tuple(weights.shape)} ids={tuple(ids.shape)}"
            )
        routed_ids = ids.reshape(-1)
        expanded = flat[:, None, :].expand(
            flat.shape[0], ids.shape[1], flat.shape[1]
        ).reshape(-1, flat.shape[1])
        fused = native_layer.forward(expanded, routed_ids, "fused13")
        gate, up = fused.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        down = native_layer.forward(activated, routed_ids, "down")
        result = (
            down.reshape(flat.shape[0], ids.shape[1], flat.shape[1])
            * weights[..., None].to(down.dtype)
        ).sum(dim=1)
        return result.reshape(original_shape)


class BananaSmasherQuantizationConfig(QuantizationConfig):
    def __init__(self, raw: dict[str, Any]):
        super().__init__()
        self.raw = dict(raw)
        self.model_root: Path | None = None
        self.pack: NativePlanePack | None = None

    @classmethod
    def get_name(cls) -> str:
        return "bs-mixed-tier"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BananaSmasherQuantizationConfig":
        if config.get("quant_method") != "bs-mixed-tier":
            raise ValueError("banana-smasher-plugin refuses non-bs-mixed-tier config")
        if config.get("format") != "bs-pack" or config.get("format_version") != 1:
            raise ValueError("banana-smasher-plugin refuses unsupported pack format")
        return cls(config)

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: Any | None = None,
        revision: str | None = None,
    ) -> None:
        del hf_config, revision
        root = Path(model_name).expanduser()
        if not root.is_dir():
            raise _fail(
                "native-plane stock adapter requires a local complete model directory: "
                f"{model_name}"
            )
        self.model_root = root.resolve()
        self.pack = NativePlanePack.from_model_root(self.model_root)
        if self.pack.architecture not in {"sm_120", "sm_121", "sm_121a"}:
            raise _fail(
                f"native-plane architecture prerequisite mismatch: {self.pack.architecture}"
            )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Any | None:
        if not isinstance(layer, RoutedExperts):
            return None
        match = _DEEPSEEK_V4_EXPERT_PREFIX.search(prefix)
        if match is None:
            raise NativePlanePrerequisiteError(
                "BANANA_SMASHER_FAST_PATH_PREREQUISITE_MISSING: "
                f"stock DeepSeek-V4 routed-expert prefix mismatch: {prefix}"
            )
        layer_index = int(match.group("layer"))
        if self.pack is None or layer_index not in self.pack.layers:
            raise _fail(
                f"stock DeepSeek-V4 layer {layer_index} is not bound by the native-plane pack"
            )
        return BananaSmasherMoEMethod(
            self,
            layer.moe_config,
            layer_index,
            prefix,
        )
