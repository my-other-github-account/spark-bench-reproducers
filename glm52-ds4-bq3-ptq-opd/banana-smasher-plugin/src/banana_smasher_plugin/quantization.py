from __future__ import annotations

from typing import Any

import torch


class BananaSmasherQuantizationConfig:
    """Constructed as a real vLLM QuantizationConfig at plugin import time.

    The class is rebound below so importing this module without vLLM remains a
    focused packaging error rather than silently selecting a dense fallback.
    """


try:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
except ImportError as exc:  # pragma: no cover - wheel dependency is mandatory
    raise RuntimeError("banana-smasher-plugin requires stock vLLM") from exc


class BananaSmasherQuantizationConfig(QuantizationConfig):
    def __init__(self, raw: dict[str, Any]):
        super().__init__()
        self.raw = raw

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

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        raise RuntimeError(
            "BANANA_SMASHER_FAST_PATH_PREREQUISITE_MISSING: stock vLLM has no "
            "P1016 native-plane MoE layer bound for prefix=" + prefix
        )
