from __future__ import annotations

import functools
import importlib
import inspect
import logging
import sys

import torch

_LOG = logging.getLogger("banana_smasher_plugin")
_REGISTERED = False


def configure_flashinfer_sparse_mla_signature_compat() -> bool:
    """Bridge vLLM 0.24 and installed FlashInfer sparse-MLA API variants."""
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
    legacy_combined_sparse_api = (
        not accepts_var_keyword
        and "sparse_topk_lens" in parameters
        and "seq_lens" in parameters
        and "swa_topk_lens" not in parameters
    )
    optional_compat_kwargs = (
        "extra_sparse_indices",
        "extra_sparse_topk_lens",
        "swa_topk_lens",
    )
    unsupported_optional_kwargs = (
        ()
        if accepts_var_keyword or legacy_combined_sparse_api
        else tuple(
            name for name in optional_compat_kwargs if name not in parameters
        )
    )
    variant = (
        "legacy_combined_sparse_api"
        if legacy_combined_sparse_api
        else (
            "all_optional_kwargs"
            if not unsupported_optional_kwargs
            else "without_" + "+".join(unsupported_optional_kwargs)
        )
    )

    @functools.wraps(original)
    def compatible_sparse_decode(*args, **kwargs):
        if legacy_combined_sparse_api:
            kwargs = kwargs.copy()
            query = kwargs["query"]
            swa_kv_cache = kwargs["swa_kv_cache"]
            if swa_kv_cache.dtype == torch.uint8:
                swa_kv_cache = swa_kv_cache.view(torch.float8_e4m3fn)
                kwargs["swa_kv_cache"] = swa_kv_cache
                query = query.to(dtype=torch.float8_e4m3fn)
                kwargs["query"] = query
            swa_indices = kwargs["sparse_indices"]
            swa_topk_lens = kwargs.pop("swa_topk_lens")
            extra_indices = kwargs.pop("extra_sparse_indices", None)
            extra_topk_lens = kwargs.pop("extra_sparse_topk_lens", None)
            combined_indices = swa_indices
            if extra_indices is not None:
                combined_indices = torch.cat((swa_indices, extra_indices), dim=-1)
            token_count = query.shape[0]
            kwargs["sparse_indices"] = combined_indices.reshape(
                token_count, -1
            ).contiguous()
            compressed_kv_cache = kwargs.get("compressed_kv_cache")
            if compressed_kv_cache is None:
                kwargs["compressed_kv_cache"] = swa_kv_cache
            elif compressed_kv_cache.dtype == torch.uint8:
                kwargs["compressed_kv_cache"] = compressed_kv_cache.view(
                    torch.float8_e4m3fn
                )
            total_topk_lens = torch.full_like(swa_topk_lens, 128)
            if extra_topk_lens is not None:
                total_topk_lens = total_topk_lens + extra_topk_lens
            kwargs["sparse_topk_lens"] = total_topk_lens
            kwargs["seq_lens"] = swa_topk_lens.reshape(-1).to(dtype=torch.int32)
            kwargs["cum_seq_lens_q"] = torch.arange(
                token_count + 1, dtype=torch.int32, device=query.device
            )
            kwargs["max_q_len"] = 1
        elif any(name in kwargs for name in unsupported_optional_kwargs):
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


def configure_stock_deepseek_v4_o_proj() -> bool:
    """Require vLLM's stock SM100+ packed-scale DeepGEMM O-projection."""
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 12:
        return False

    module = importlib.import_module("vllm.models.deepseek_v4.nvidia.ops.o_proj")
    recipe, tma_aligned_scales = module.compute_fp8_einsum_recipe()
    if recipe != (1, 1, 128) or tma_aligned_scales is not True:
        raise RuntimeError(
            "SM12x requires vLLM's stock SM100+ packed-scale DeepGEMM "
            "O-projection contract: recipe=(1, 1, 128), "
            f"tma_aligned_scales=True; observed recipe={recipe}, "
            f"tma_aligned_scales={tma_aligned_scales}"
        )
    _LOG.warning(
        "BANANA_SMASHER_DSV4_O_PROJ_LAYOUT "
        "compute_capability=%d.%d backend=deep_gemm_e8m0 "
        "einsum_recipe=1x1x128 tma_aligned_scales=true stock_sm100_plus=true",
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
        "_fp8_fp4_mqa_logits_impl": "fp8_fp4_mqa_logits",
        "_get_mk_alignment_for_contiguous_layout_impl": (
            "get_mk_alignment_for_contiguous_layout"
        ),
        "_fp8_gemm_nt_impl": "fp8_gemm_nt",
    }
    for name in required.values():
        implementation = getattr(external, name, None)
        if not callable(implementation):
            raise RuntimeError(
                "Public SM12x DeepGEMM backend is incomplete: "
                f"deep_gemm.{name} is not callable. Rebuild the pinned public "
                "DeepGEMM source wheel."
            )

    @functools.wraps(original_import)
    def import_external_deep_gemm():
        return external

    import_external_deep_gemm._banana_smasher_sm12x_indexer = True  # type: ignore[attr-defined]
    utils._import_deep_gemm = import_external_deep_gemm

    # Do not populate only the sparse-indexer slots: vLLM's lazy initializer
    # treats any populated implementation as evidence that *all* DeepGEMM
    # symbols were resolved.  A partial table makes the later stock warmup fail
    # on a missing alignment function.  Clear the table and let stock vLLM
    # resolve the complete public module in one transaction.
    implementation_slots = (
        "_cublaslt_gemm_nt_impl",
        "_fp8_gemm_nt_impl",
        "_fp8_einsum_impl",
        "_grouped_impl",
        "_grouped_masked_impl",
        "_grouped_fp4_impl",
        "_fp8_fp4_mqa_logits_impl",
        "_fp8_fp4_paged_mqa_logits_impl",
        "_get_paged_mqa_logits_metadata_impl",
        "_tf32_hc_prenorm_gemm_impl",
        "_get_mn_major_tma_aligned_tensor_impl",
        "_get_mk_alignment_for_contiguous_layout_impl",
        "_get_theoretical_mk_alignment_for_contiguous_layout_impl",
        "_transform_sf_into_required_layout_impl",
        "_pack_ue8m0_to_int_impl",
        "_get_mn_major_tma_aligned_packed_ue8m0_tensor_impl",
        "_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl",
    )
    for slot in implementation_slots:
        setattr(utils, slot, None)
    utils._lazy_init()
    unresolved = [slot for slot in required if not callable(getattr(utils, slot, None))]
    if unresolved:
        raise RuntimeError(
            "Public SM12x DeepGEMM registration did not initialize required "
            f"vLLM slots: {unresolved}"
        )

    _LOG.warning(
        "BANANA_SMASHER_INDEXER_DEEPGEMM_BACKEND "
        "compute_capability=%d.%d module=%s source=public_external "
        "dense_fp8_recipe=enabled_e8m0",
        capability.major,
        capability.minor,
        getattr(external, "__name__", "deep_gemm"),
    )
    return True


def configure_deep_gemm_ue8m0_warmup_contract(*, warmup_module=None) -> bool:
    """Give DeepGEMM 2.6.x valid UE8M0 dummy scales during warmup.

    vLLM 0.24 creates its dummy FP32 activation-scale buffers with
    ``torch.empty``.  DeepGEMM 2.6.x packs every element to UE8M0 and traps on
    uninitialized sign or mantissa bits, including rows that are only padding.
    Exact powers of two are valid, so initialize every dummy scale to one.
    """
    if warmup_module is None:
        from vllm.platforms import current_platform

        capability = current_platform.get_device_capability()
        if capability is None or capability.major != 12:
            return False
        warmup_module = importlib.import_module(
            "vllm.model_executor.warmup.deep_gemm_warmup"
        )
    original_dense = warmup_module._deepgemm_fp8_gemm_nt_warmup
    original_grouped = (
        warmup_module._deepgemm_grouped_fp8_gemm_nt_contiguous_warmup
    )
    if (
        getattr(original_dense, "_banana_smasher_ue8m0_scales", False)
        and getattr(original_grouped, "_banana_smasher_ue8m0_scales", False)
    ):
        return True

    @functools.wraps(original_dense)
    def ue8m0_safe_dense_warmup(w, ws, max_tokens, pbar=None):
        if w.size() in warmup_module.FP8_GEMM_NT_WARMUP_CACHE:
            return

        n, k = w.size()
        block_m = warmup_module.get_mk_alignment_for_contiguous_layout()[0]
        device = w.device
        a1q = torch.empty(
            (max_tokens, k), device=device, dtype=torch.float8_e4m3fn
        )
        a1q_scales = torch.ones(
            (max_tokens, k // block_m), device=device, dtype=torch.float32
        )
        out = torch.empty((max_tokens, n), device=device, dtype=torch.bfloat16)

        for num_tokens in warmup_module._get_fp8_gemm_nt_m_values(w, max_tokens):
            warmup_module.fp8_gemm_nt(
                (a1q[:num_tokens], a1q_scales[:num_tokens]),
                (w, ws),
                out[:num_tokens],
            )
            if pbar is not None:
                pbar.update(1)

        warmup_module.FP8_GEMM_NT_WARMUP_CACHE.add(w.size())

    @functools.wraps(original_grouped)
    def ue8m0_safe_grouped_warmup(
        w1, w2, w1_scale, w2_scale, num_topk, max_tokens, pbar=None
    ):
        cache = warmup_module.GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
        if w1.size() in cache and w2.size() in cache:
            return

        max_m, block_m, warmup_cases = warmup_module._get_grouped_gemm_params(
            w1, w2, num_topk, max_tokens
        )
        if not warmup_cases:
            return
        device = w1.device

        def warm_one(w, w_scale):
            _, n, k = w.size()
            a1q = torch.empty((max_m, k), device=device, dtype=torch.float8_e4m3fn)
            a1q_scales = torch.ones(
                (max_m, k // block_m), device=device, dtype=torch.float32
            )
            out = torch.empty((max_m, n), device=device, dtype=torch.bfloat16)
            for num_tokens, align_used, expert_ids in warmup_cases:
                with warmup_module.mk_alignment_scope(align_used):
                    warmup_module.m_grouped_fp8_gemm_nt_contiguous(
                        (a1q[:num_tokens], a1q_scales[:num_tokens]),
                        (w, w_scale),
                        out[:num_tokens],
                        expert_ids,
                    )
                if pbar is not None:
                    pbar.update(1)

        for w, ws in ((w1, w1_scale), (w2, w2_scale)):
            if w.size() not in cache:
                warm_one(w, ws)
                cache.add(w.size())

    ue8m0_safe_dense_warmup._banana_smasher_ue8m0_scales = True  # type: ignore[attr-defined]
    ue8m0_safe_grouped_warmup._banana_smasher_ue8m0_scales = True  # type: ignore[attr-defined]
    warmup_module._deepgemm_fp8_gemm_nt_warmup = ue8m0_safe_dense_warmup
    warmup_module._deepgemm_grouped_fp8_gemm_nt_contiguous_warmup = (
        ue8m0_safe_grouped_warmup
    )
    _LOG.warning(
        "BANANA_SMASHER_DEEPGEMM_UE8M0_WARMUP "
        "dense_dummy_scales=ones grouped_dummy_scales=ones padding=initialized"
    )
    return True


def configure_sparse_indexer_topk_backend() -> bool:
    """Route cluster-launch TopK to persistent TopK on SM12x devices.

    vLLM 0.24 selects ``cooperative_topk`` for every CUDA capability >= 9.0,
    but its cooperative cluster launch is rejected with ``cudaErrorInvalidValue``
    on the SM120 capability family (including SM121).  The production
    ``persistent_topk`` kernel accepts the same contract and remains exact, so
    replace only this unsupported op on SM12x while preserving stock dispatch
    everywhere else.
    """
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 12:
        return False

    namespace = torch.ops._C
    cooperative = getattr(namespace, "cooperative_topk", None)
    persistent = getattr(namespace, "persistent_topk", None)
    if not callable(cooperative):
        raise RuntimeError("vLLM cooperative_topk op is unavailable")
    if getattr(cooperative, "_banana_smasher_sm12x_persistent_topk", False):
        return True
    if not callable(persistent):
        raise RuntimeError(
            "SM12x sparse-indexer correction requires vLLM persistent_topk"
        )

    @functools.wraps(cooperative)
    def sm12x_persistent_topk(*args, **kwargs):
        return persistent(*args, **kwargs)

    sm12x_persistent_topk._banana_smasher_sm12x_persistent_topk = True  # type: ignore[attr-defined]
    sm12x_persistent_topk._banana_smasher_original_cooperative_topk = cooperative  # type: ignore[attr-defined]
    setattr(namespace, "cooperative_topk", sm12x_persistent_topk)
    _LOG.warning(
        "BANANA_SMASHER_SPARSE_INDEXER_TOPK_OVERRIDE "
        "compute_capability=%d.%d requested=cooperative_topk "
        "selected=persistent_topk reason=sm12x_cluster_launch_unsupported",
        capability.major,
        capability.minor,
    )
    return True


def configure_stock_mhc_backend() -> bool:
    """Deprecated compatibility hook; stock public DeepGEMM owns MHC on SM12x."""
    return False


def register() -> None:
    """Register the canonical product quantization config in every vLLM process."""
    global _REGISTERED
    if _REGISTERED:
        return
    configure_flashinfer_sparse_mla_signature_compat()
    configure_stock_deepseek_v4_attention_backend()
    configure_sparse_indexer_deep_gemm_backend()
    configure_deep_gemm_ue8m0_warmup_contract()
    configure_sparse_indexer_topk_backend()
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
    "configure_deep_gemm_ue8m0_warmup_contract",
    "configure_flashinfer_sparse_mla_signature_compat",
    "configure_sparse_indexer_deep_gemm_backend",
    "configure_stock_deepseek_v4_attention_backend",
    "configure_stock_deepseek_v4_o_proj",
    "configure_stock_mhc_backend",
    "register",
]
