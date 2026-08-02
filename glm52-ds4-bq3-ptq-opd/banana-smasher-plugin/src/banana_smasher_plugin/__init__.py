from __future__ import annotations

import logging
import os
import sys

_LOG = logging.getLogger("banana_smasher_plugin")
_REGISTERED = False


def configure_stock_mhc_backend() -> bool:
    """Route MHC away from DeepGEMM where its hyperconnection API is unsupported."""
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or (capability.major, capability.minor) != (12, 1):
        return False
    previous = os.environ.get("VLLM_USE_DEEP_GEMM")
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    deep_gemm = sys.modules.get("vllm.utils.deep_gemm")
    if deep_gemm is not None:
        deep_gemm.is_deep_gemm_supported.cache_clear()
    _LOG.warning(
        "BANANA_SMASHER_MHC_BACKEND_OVERRIDE compute_capability=12.1 "
        "VLLM_USE_DEEP_GEMM=%r->'0' reason=unsupported_hyperconnection_api",
        previous,
    )
    return True


def register() -> None:
    """Register the canonical product quantization config in every vLLM process."""
    global _REGISTERED
    if _REGISTERED:
        return
    configure_stock_mhc_backend()
    from .quantization import (
        BananaSmasherQuantizationConfig,
        QUANT_METHOD,
        install_deepseek_v4_dense_preflight,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config

    register_quantization_config(QUANT_METHOD)(BananaSmasherQuantizationConfig)
    install_deepseek_v4_dense_preflight()
    _REGISTERED = True
    _LOG.warning(
        "BANANA_SMASHER_PLUGIN_REGISTERED quant_method=%s fast_path_or_fail=true",
        QUANT_METHOD,
    )


__all__ = ["configure_stock_mhc_backend", "register"]
