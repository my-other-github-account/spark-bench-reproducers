from __future__ import annotations

import functools
import importlib
import inspect
import logging
import os
import sys

import torch

_LOG = logging.getLogger("banana_smasher_plugin")
_REGISTERED = False


def configure_flashinfer_sparse_mla_signature_compat() -> bool:
    """Normalize optional FlashInfer sparse-MLA kwargs used by vLLM 0.24."""
    api_name = "trtllm_batch_decode_sparse_mla_dsv4"
    decode = importlib.import_module("flashinfer.decode")
    target = getattr(decode, api_name)
    utils = importlib.import_module("vllm.utils.flashinfer")
    wrapper_name = "flashinfer_trtllm_batch_decode_sparse_mla_dsv4"
    original = getattr(utils, wrapper_name)
    if getattr(original, "_banana_smasher_sparse_mla_signature_compat", False):
        return True

    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot inspect FlashInfer {api_name} signature for compatibility"
        ) from exc
    accepts_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    optional_compat_kwargs = (
        "extra_sparse_indices",
        "swa_topk_lens",
    )
    unsupported_optional_kwargs = (
        ()
        if accepts_var_keyword
        else tuple(
            name for name in optional_compat_kwargs if name not in parameters
        )
    )
    variant = (
        "all_optional_kwargs"
        if not unsupported_optional_kwargs
        else "without_" + "+".join(unsupported_optional_kwargs)
    )

    @functools.wraps(original)
    def compatible_sparse_decode(*args, **kwargs):
        if any(name in kwargs for name in unsupported_optional_kwargs):
            kwargs = kwargs.copy()
            for name in unsupported_optional_kwargs:
                kwargs.pop(name, None)
        return original(*args, **kwargs)

    compatible_sparse_decode._banana_smasher_sparse_mla_signature_compat = True  # type: ignore[attr-defined]
    compatible_sparse_decode._banana_smasher_sparse_mla_signature_variant = variant  # type: ignore[attr-defined]
    setattr(utils, wrapper_name, compatible_sparse_decode)
    sparse_module = sys.modules.get(
        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse"
    )
    if (
        sparse_module is not None
        and getattr(sparse_module, wrapper_name, None) is original
    ):
        setattr(sparse_module, wrapper_name, compatible_sparse_decode)
    _LOG.warning(
        "BANANA_SMASHER_FLASHINFER_SPARSE_MLA_API_VARIANT "
        "api=%s variant=%s inspect_once=true",
        api_name,
        variant,
    )
    return True


def _fused_inv_rope_fp8_quant(*args, **kwargs):
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    return fused_inv_rope_fp8_quant(*args, **kwargs)


def _triton_block_scaled_mm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    # Importing the public backend module registers the custom op.
    importlib.import_module("vllm.model_executor.kernels.linear.scaled_mm.triton")
    return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
        activation,
        weight,
        activation_scale,
        weight_scale,
        [128, 128],
        torch.bfloat16,
    )


def configure_stock_deepseek_v4_o_proj() -> bool:
    """Route SM12x output projection through stock vLLM's Triton FP8 kernel."""
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 12:
        return False

    module = importlib.import_module("vllm.models.deepseek_v4.nvidia.ops.o_proj")
    original = module.deep_gemm_fp8_o_proj
    if getattr(original, "_banana_smasher_sm12x_layout", False):
        return True

    @functools.wraps(original)
    def triton_fp8_o_proj(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        wo_b,
        *,
        n_groups: int,
        heads_per_group: int,
        nope_dim: int,
        rope_dim: int,
        o_lora_rank: int,
        einsum_recipe: tuple[int, int, int],
        tma_aligned_scales: bool,
    ):
        current = current_platform.get_device_capability()
        if current is None or current.major != 12:
            return original(
                o,
                positions,
                cos_sin_cache,
                wo_a,
                wo_b,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                o_lora_rank=o_lora_rank,
                einsum_recipe=einsum_recipe,
                tma_aligned_scales=tma_aligned_scales,
            )

        input_width = heads_per_group * (nope_dim + rope_dim)
        expected_weight_shape = (n_groups, o_lora_rank, input_width)
        weight = wo_a.weight
        if weight.dim() == 2:
            flat_weight_shape = (n_groups * o_lora_rank, input_width)
            if tuple(weight.shape) != flat_weight_shape:
                raise RuntimeError(
                    "stock DeepSeek-V4 o_proj flattened FP8 weight shape mismatch: "
                    f"actual={tuple(weight.shape)} expected={flat_weight_shape}"
                )
            weight = weight.view(expected_weight_shape)
        elif tuple(weight.shape) != expected_weight_shape:
            raise RuntimeError(
                "stock DeepSeek-V4 o_proj grouped FP8 weight shape mismatch: "
                f"actual={tuple(weight.shape)} expected={expected_weight_shape}"
            )

        expected_scale_shape = (
            n_groups,
            o_lora_rank // 128,
            input_width // 128,
        )
        scale = wo_a.weight_scale_inv
        if scale.dim() == 2:
            flat_scale_shape = (
                n_groups * expected_scale_shape[1],
                expected_scale_shape[2],
            )
            if tuple(scale.shape) != flat_scale_shape:
                raise RuntimeError(
                    "stock DeepSeek-V4 o_proj flattened FP8 scale shape mismatch: "
                    f"actual={tuple(scale.shape)} expected={flat_scale_shape}"
                )
            scale = scale.view(expected_scale_shape)
        elif tuple(scale.shape) != expected_scale_shape:
            raise RuntimeError(
                "stock DeepSeek-V4 o_proj grouped FP8 scale shape mismatch: "
                f"actual={tuple(scale.shape)} expected={expected_scale_shape}"
            )
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is not None and scale.dtype == e8m0_dtype:
            scale = scale.to(torch.float32)
        elif scale.dtype != torch.float32:
            raise RuntimeError(
                "stock DeepSeek-V4 SM12x Triton o_proj requires FP32 or E8M0 block scales, "
                f"got dtype={scale.dtype}"
            )

        o_fp8, o_scale = _fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            tma_aligned_scales=False,
        )
        z = torch.stack(
            [
                _triton_block_scaled_mm(
                    o_fp8[:, group, :].contiguous(),
                    weight[group],
                    o_scale[:, group, :].contiguous(),
                    scale[group],
                )
                for group in range(n_groups)
            ],
            dim=1,
        )
        return wo_b(z.flatten(1))

    triton_fp8_o_proj._banana_smasher_sm12x_layout = True  # type: ignore[attr-defined]
    module.deep_gemm_fp8_o_proj = triton_fp8_o_proj
    for name in (
        "vllm.models.deepseek_v4.nvidia.flashmla",
        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse",
    ):
        loaded = sys.modules.get(name)
        if loaded is not None and getattr(loaded, "deep_gemm_fp8_o_proj", None) is original:
            loaded.deep_gemm_fp8_o_proj = triton_fp8_o_proj
    _LOG.warning(
        "BANANA_SMASHER_DSV4_O_PROJ_LAYOUT_OVERRIDE "
        "compute_capability=%d.%d backend=stock_triton_fp8 grouped_bmm=true",
        capability.major,
        capability.minor,
    )
    return True


def configure_stock_deepseek_v4_attention_backend() -> bool:
    """Route explicit stock FlashMLA DSv4 requests to FlashInfer on SM12x."""
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 12:
        return False

    module = importlib.import_module("vllm.models.deepseek_v4.nvidia.model")
    original = module._select_dsv4_attn_cls
    if getattr(original, "_banana_smasher_sm12x_attention", False):
        return True
    warned = False

    @functools.wraps(original)
    def select_supported_attention(vllm_config):
        nonlocal warned
        capability = module.current_platform.get_device_capability()
        backend = vllm_config.attention_config.backend
        if (
            capability is not None
            and capability.major == 12
            and getattr(backend, "name", None) == "FLASHMLA_SPARSE_DSV4"
        ):
            flashinfer_utils = importlib.import_module("vllm.utils.flashinfer")
            has_sparse_mla = getattr(
                flashinfer_utils,
                "has_flashinfer_sparse_mla_sm120",
                None,
            )
            if not callable(has_sparse_mla) or not has_sparse_mla():
                raise RuntimeError(
                    "No physically supported stock SM12x sparse MLA attention route: "
                    "FlashInfer's sparse MLA decode API is physically unavailable. "
                    "Install a flashinfer-python build exposing "
                    "flashinfer.decode.trtllm_batch_decode_sparse_mla_dsv4 and "
                    "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla."
                )
            if not warned:
                _LOG.warning(
                    "BANANA_SMASHER_DSV4_ATTENTION_BACKEND_OVERRIDE "
                    "compute_capability=%d.%d requested=FLASHMLA_SPARSE_DSV4 "
                    "selected=FLASHINFER_MLA_SPARSE_DSV4",
                    capability.major,
                    capability.minor,
                )
                warned = True
            return module.DeepseekV4FlashInferSM120Attention
        return original(vllm_config)

    select_supported_attention._banana_smasher_sm12x_attention = True  # type: ignore[attr-defined]
    module._select_dsv4_attn_cls = select_supported_attention
    return True


def configure_sparse_indexer_deep_gemm_backend() -> bool:
    """Select the public SM12x DeepGEMM indexer implementation once at boot."""
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 12:
        return False

    utils = importlib.import_module("vllm.utils.deep_gemm")
    original_import = utils._import_deep_gemm
    if getattr(original_import, "_banana_smasher_sm12x_indexer", False):
        return True

    external = importlib.import_module("deep_gemm")
    required = {
        "_get_paged_mqa_logits_metadata_impl": "get_paged_mqa_logits_metadata",
        "_fp8_fp4_paged_mqa_logits_impl": "fp8_fp4_paged_mqa_logits",
    }
    selected: dict[str, object] = {}
    for slot, name in required.items():
        implementation = getattr(external, name, None)
        if not callable(implementation):
            raise RuntimeError(
                "Public SM12x DeepGEMM sparse-indexer backend is incomplete: "
                f"deep_gemm.{name} is not callable. Rebuild the pinned public "
                "DeepGEMM source wheel."
            )
        selected[slot] = implementation

    @functools.wraps(original_import)
    def import_external_deep_gemm():
        return external

    import_external_deep_gemm._banana_smasher_sm12x_indexer = True  # type: ignore[attr-defined]
    utils._import_deep_gemm = import_external_deep_gemm
    for slot, implementation in selected.items():
        setattr(utils, slot, implementation)
    _LOG.warning(
        "BANANA_SMASHER_INDEXER_DEEPGEMM_BACKEND "
        "compute_capability=%d.%d module=%s source=public_external",
        capability.major,
        capability.minor,
        getattr(external, "__name__", "deep_gemm"),
    )
    return True


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
    configure_flashinfer_sparse_mla_signature_compat()
    configure_stock_deepseek_v4_attention_backend()
    configure_sparse_indexer_deep_gemm_backend()
    configure_stock_mhc_backend()
    configure_stock_deepseek_v4_o_proj()
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


__all__ = [
    "configure_flashinfer_sparse_mla_signature_compat",
    "configure_sparse_indexer_deep_gemm_backend",
    "configure_stock_deepseek_v4_attention_backend",
    "configure_stock_deepseek_v4_o_proj",
    "configure_stock_mhc_backend",
    "register",
]
