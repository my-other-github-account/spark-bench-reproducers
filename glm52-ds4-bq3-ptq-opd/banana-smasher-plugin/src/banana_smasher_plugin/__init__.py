from __future__ import annotations

import logging

_LOG = logging.getLogger("banana_smasher_plugin")
_REGISTERED = False


def register() -> None:
    """Register the bs-mixed-tier quantization config in every vLLM process."""
    global _REGISTERED
    if _REGISTERED:
        return
    from .quantization import BananaSmasherQuantizationConfig
    from vllm.model_executor.layers.quantization import register_quantization_config

    register_quantization_config("bs-mixed-tier")(BananaSmasherQuantizationConfig)
    _REGISTERED = True
    _LOG.warning("BANANA_SMASHER_PLUGIN_REGISTERED quant_method=bs-mixed-tier fast_path_or_fail=true")


__all__ = ["register"]
