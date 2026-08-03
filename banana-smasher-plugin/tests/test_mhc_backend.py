from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import banana_smasher_plugin


def _install_indexer_deep_gemm_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    major: int,
    minor: int,
) -> tuple[ModuleType, ModuleType, ModuleType]:
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=major, minor=minor)
    )
    vendored = ModuleType("vllm.third_party.deep_gemm")
    external = ModuleType("deep_gemm")
    utils = ModuleType("vllm.utils.deep_gemm")
    warmup = ModuleType("vllm.model_executor.warmup.deep_gemm_warmup")

    def vendored_metadata(*args):
        raise RuntimeError("Unsupported architecture")

    def external_metadata(*args):
        return ("external-metadata", args)

    def external_logits(*args, **kwargs):
        return ("external-logits", args, kwargs)

    def external_dense_logits(*args, **kwargs):
        return ("external-dense-logits", args, kwargs)

    def external_fp8_gemm_nt(*args, **kwargs):
        return ("external-fp8-gemm-nt", args, kwargs)

    def external_alignment():
        return 128

    def lazy_init():
        selected = utils._import_deep_gemm()
        utils._get_paged_mqa_logits_metadata_impl = (
            selected.get_paged_mqa_logits_metadata
        )
        utils._fp8_fp4_paged_mqa_logits_impl = selected.fp8_fp4_paged_mqa_logits
        utils._fp8_fp4_mqa_logits_impl = selected.fp8_fp4_mqa_logits
        utils._get_mk_alignment_for_contiguous_layout_impl = (
            selected.get_mk_alignment_for_contiguous_layout
        )
        utils._fp8_gemm_nt_impl = selected.fp8_gemm_nt

    vendored.get_paged_mqa_logits_metadata = vendored_metadata
    external.get_paged_mqa_logits_metadata = external_metadata
    external.fp8_fp4_paged_mqa_logits = external_logits
    setattr(external, "fp8_fp4_mqa_logits", external_dense_logits)
    external.fp8_gemm_nt = external_fp8_gemm_nt
    external.get_mk_alignment_for_contiguous_layout = external_alignment
    utils._import_deep_gemm = lambda: vendored
    utils._lazy_init = lazy_init
    utils.is_deep_gemm_supported = lambda: True
    utils._get_paged_mqa_logits_metadata_impl = vendored_metadata
    utils._fp8_fp4_paged_mqa_logits_impl = None
    setattr(utils, "_fp8_fp4_mqa_logits_impl", None)
    warmup._fp8_linear_may_use_deep_gemm = lambda _module: True

    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "vllm.utils.deep_gemm", utils)
    monkeypatch.setitem(sys.modules, "vllm.third_party.deep_gemm", vendored)
    monkeypatch.setitem(sys.modules, "deep_gemm", external)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.warmup.deep_gemm_warmup",
        warmup,
    )
    return utils, vendored, external


def test_public_indexer_backend_selects_external_sm12x_deepgemm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utils, vendored, external = _install_indexer_deep_gemm_modules(
        monkeypatch,
        major=12,
        minor=1,
    )

    selector = getattr(
        banana_smasher_plugin,
        "configure_sparse_indexer_deep_gemm_backend",
        None,
    )
    assert callable(selector), "public sparse-indexer DeepGEMM selector is missing"
    assert selector() is True
    assert utils._import_deep_gemm() is external
    assert utils._get_paged_mqa_logits_metadata_impl(2, 64, 20) == (
        "external-metadata",
        (2, 64, 20),
    )
    assert utils._fp8_fp4_paged_mqa_logits_impl is external.fp8_fp4_paged_mqa_logits
    assert getattr(utils, "_fp8_fp4_mqa_logits_impl") is getattr(
        external, "fp8_fp4_mqa_logits"
    )
    assert utils._get_mk_alignment_for_contiguous_layout_impl() == 128
    assert utils._fp8_gemm_nt_impl is external.fp8_gemm_nt
    assert utils._get_paged_mqa_logits_metadata_impl is not (
        vendored.get_paged_mqa_logits_metadata
    )


def test_public_indexer_backend_preserves_supported_pre_sm12x_deepgemm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utils, vendored, external = _install_indexer_deep_gemm_modules(
        monkeypatch,
        major=9,
        minor=0,
    )
    original_import = utils._import_deep_gemm
    original_metadata = utils._get_paged_mqa_logits_metadata_impl

    assert banana_smasher_plugin.configure_sparse_indexer_deep_gemm_backend() is False
    assert utils._import_deep_gemm is original_import
    assert utils._import_deep_gemm() is vendored
    assert utils._get_paged_mqa_logits_metadata_impl is original_metadata
    assert utils._fp8_fp4_paged_mqa_logits_impl is None
    assert external is sys.modules["deep_gemm"]


def test_deprecated_stock_mhc_compatibility_hook_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=12, minor=1)
    )
    deep_gemm = ModuleType("vllm.utils.deep_gemm")

    def original_mhc(*args, **kwargs):
        return None

    deep_gemm.is_deep_gemm_supported = lambda: False
    deep_gemm.tf32_hc_prenorm_gemm = original_mhc
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "vllm.utils.deep_gemm", deep_gemm)

    monkeypatch.delenv("BANANA_SMASHER_MHC_BACKEND_OVERRIDE", raising=False)

    assert banana_smasher_plugin.configure_stock_mhc_backend() is False
    assert deep_gemm.tf32_hc_prenorm_gemm is original_mhc
    assert "BANANA_SMASHER_MHC_BACKEND_OVERRIDE" not in os.environ


def test_public_mhc_selector_routes_only_prenorm_to_tilelang_on_sm121(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.skip("superseded by stock public DeepGEMM MHC preservation coverage")
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=12, minor=1)
    )
    deep_gemm = ModuleType("vllm.utils.deep_gemm")
    tilelang = ModuleType("vllm.model_executor.kernels.mhc.tilelang")
    original_calls: list[object] = []
    tilelang_calls: list[dict[str, int]] = []

    def original_prenorm(*args, **kwargs):
        original_calls.append((args, kwargs))
        raise AssertionError("SM121 MHC must not call DeepGEMM prenorm")

    def tilelang_prenorm(
        x,
        fn,
        out,
        sqrsum,
        hidden_size,
        hc_mult,
        *,
        n_splits,
    ):
        del fn
        assert x.shape[1] % n_splits == 0
        tilelang_calls.append(
            {
                "hidden_size": hidden_size,
                "hc_mult": hc_mult,
                "n_splits": n_splits,
                "out_splits": out.shape[0],
                "sqrsum_splits": sqrsum.shape[0],
            }
        )
        out.fill_(7.0)
        sqrsum.fill_(11.0)

    setattr(deep_gemm, "tf32_hc_prenorm_gemm", original_prenorm)
    setattr(tilelang, "_tilelang_hc_prenorm_gemm", tilelang_prenorm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "vllm.utils.deep_gemm", deep_gemm)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.kernels.mhc.tilelang",
        tilelang,
    )
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")

    selector = getattr(banana_smasher_plugin, "configure_stock_mhc_backend", None)
    assert callable(selector), "public stock MHC backend selector is not installed"
    assert selector() is True
    assert os.environ["VLLM_USE_DEEP_GEMM"] == "1"
    assert deep_gemm.tf32_hc_prenorm_gemm is not original_prenorm

    x = torch.zeros((2, 8192))
    fn = torch.zeros((24, 8192))
    out = torch.ones((3, 2, 24))
    sqrsum = torch.ones((3, 2))
    deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 3)
    assert original_calls == []
    assert tilelang_calls == [
        {
            "hidden_size": 2048,
            "hc_mult": 4,
            "n_splits": 1,
            "out_splits": 1,
            "sqrsum_splits": 1,
        }
    ]
    assert torch.all(out[0] == 7.0)
    assert torch.all(out[1:] == 0.0)
    assert torch.all(sqrsum[0] == 11.0)
    assert torch.all(sqrsum[1:] == 0.0)


def test_public_deepseek_v4_o_proj_routes_stock_fp8_weights_to_triton_on_sm121(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.skip("superseded by SM12x DeepGEMM E8M0 grouped-layout coverage")
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=12, minor=1)
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    o_proj = ModuleType("vllm.models.deepseek_v4.nvidia.ops.o_proj")

    def stock_compute_fp8_einsum_recipe():
        capability = platforms.current_platform.get_device_capability()
        return ((1, 128, 128), False) if capability.major <= 9 else ((1, 1, 128), True)

    def stock_deep_gemm_fp8_o_proj(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        wo_b,
        **kwargs,
    ):
        del o, positions, cos_sin_cache, wo_b, kwargs
        if wo_a.weight.dim() != 3:
            raise RuntimeError(
                "Assertion error (.../utils/layout.hpp:39): t.dim() == N"
            )
        raise AssertionError("SM121 must not call stock DeepGEMM fp8_einsum")

    o_proj.compute_fp8_einsum_recipe = stock_compute_fp8_einsum_recipe
    o_proj.deep_gemm_fp8_o_proj = stock_deep_gemm_fp8_o_proj
    monkeypatch.setitem(
        sys.modules,
        "vllm.models.deepseek_v4.nvidia.ops.o_proj",
        o_proj,
    )

    wo_a = SimpleNamespace(
        weight=torch.zeros((256, 512)),
        weight_scale_inv=torch.ones((2, 4)),
    )
    recipe, tma_aligned_scales = o_proj.compute_fp8_einsum_recipe()
    assert (recipe, tma_aligned_scales) == ((1, 1, 128), True)
    with pytest.raises(RuntimeError, match=r"t\.dim\(\) == N"):
        o_proj.deep_gemm_fp8_o_proj(
            torch.zeros((1, 2, 512)),
            None,
            None,
            wo_a,
            None,
            n_groups=2,
            heads_per_group=1,
            nope_dim=448,
            rope_dim=64,
            o_lora_rank=128,
            einsum_recipe=recipe,
            tma_aligned_scales=tma_aligned_scales,
        )

    fused_calls: list[dict[str, object]] = []
    mm_calls: list[dict[str, object]] = []

    def fake_fused(o, *args, **kwargs):
        del args
        fused_calls.append(kwargs)
        return (
            torch.zeros((o.shape[0], 2, 512)),
            torch.ones((o.shape[0], 2, 4)),
        )

    def fake_mm(activation, weight, activation_scale, weight_scale):
        mm_calls.append(
            {
                "activation_shape": tuple(activation.shape),
                "weight_shape": tuple(weight.shape),
                "activation_scale_shape": tuple(activation_scale.shape),
                "weight_scale_shape": tuple(weight_scale.shape),
            }
        )
        return torch.zeros((activation.shape[0], weight.shape[0]))

    monkeypatch.setattr(banana_smasher_plugin, "_fused_inv_rope_fp8_quant", fake_fused)
    monkeypatch.setattr(banana_smasher_plugin, "_triton_block_scaled_mm", fake_mm)

    selector = getattr(
        banana_smasher_plugin,
        "configure_stock_deepseek_v4_o_proj",
        None,
    )
    assert callable(selector), "public stock DeepSeek-V4 o_proj selector is not installed"
    assert selector() is True
    recipe, tma_aligned_scales = o_proj.compute_fp8_einsum_recipe()
    assert (recipe, tma_aligned_scales) == ((1, 1, 128), True)
    output = o_proj.deep_gemm_fp8_o_proj(
        torch.zeros((1, 2, 512)),
        None,
        None,
        wo_a,
        lambda z: z,
        n_groups=2,
        heads_per_group=1,
        nope_dim=448,
        rope_dim=64,
        o_lora_rank=128,
        einsum_recipe=recipe,
        tma_aligned_scales=tma_aligned_scales,
    )
    assert output.shape == (1, 256)
    assert fused_calls == [
        {
            "n_groups": 2,
            "heads_per_group": 1,
            "nope_dim": 448,
            "rope_dim": 64,
            "tma_aligned_scales": False,
        }
    ]
    assert mm_calls == [
        {
            "activation_shape": (1, 512),
            "weight_shape": (128, 512),
            "activation_scale_shape": (1, 4),
            "weight_scale_shape": (1, 4),
        },
        {
            "activation_shape": (1, 512),
            "weight_shape": (128, 512),
            "activation_scale_shape": (1, 4),
            "weight_scale_shape": (1, 4),
        },
    ]


def test_real_deepseek_v4_o_proj_routes_sm121_to_triton_fp8() -> None:
    pytest.skip("superseded by real SM121 DeepGEMM E8M0 dispatch coverage")
    o_proj = pytest.importorskip("vllm.models.deepseek_v4.nvidia.ops.o_proj")
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or (capability.major, capability.minor) != (12, 1):
        pytest.skip("real DeepSeek-V4 o_proj regression requires SM121")
    if not torch.cuda.is_available():
        pytest.skip("real DeepSeek-V4 o_proj regression requires CUDA")

    recipe, tma_aligned_scales = o_proj.compute_fp8_einsum_recipe()
    assert (recipe, tma_aligned_scales) == ((1, 1, 128), True)
    o = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="cuda")
    positions = torch.zeros((1,), dtype=torch.int64, device="cuda")
    cos_sin_cache = torch.zeros((1, 64), dtype=torch.float32, device="cuda")
    wo_a = SimpleNamespace(
        weight=torch.zeros(
            (8192, 4096), dtype=torch.float8_e4m3fn, device="cuda"
        ),
        weight_scale_inv=torch.ones(
            (64, 32), dtype=torch.float8_e8m0fnu, device="cuda"
        ),
    )
    kwargs = {
        "n_groups": 8,
        "heads_per_group": 8,
        "nope_dim": 448,
        "rope_dim": 64,
        "o_lora_rank": 1024,
        "einsum_recipe": recipe,
        "tma_aligned_scales": tma_aligned_scales,
    }
    with pytest.raises(RuntimeError, match=r"t\.dim\(\) == N"):
        o_proj.deep_gemm_fp8_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            lambda z: z,
            **kwargs,
        )

    selector = getattr(
        banana_smasher_plugin,
        "configure_stock_deepseek_v4_o_proj",
        None,
    )
    assert callable(selector), "public stock DeepSeek-V4 o_proj selector is not installed"
    assert selector() is True
    output = o_proj.deep_gemm_fp8_o_proj(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        lambda z: z,
        **kwargs,
    )
    torch.cuda.synchronize()
    assert output.shape == (1, 8192)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_real_deepseek_v4_mhc_preop_routes_sm121_to_tilelang_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.skip("stock public DeepGEMM now owns SM121 MHC")
    tilelang = pytest.importorskip("vllm.model_executor.kernels.mhc.tilelang")
    deep_gemm = pytest.importorskip("vllm.utils.deep_gemm")
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or (capability.major, capability.minor) != (12, 1):
        pytest.skip("real DeepSeek-V4 MHC regression requires SM121")
    if not torch.cuda.is_available():
        pytest.skip("real DeepSeek-V4 MHC regression requires CUDA")

    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    deep_gemm.is_deep_gemm_supported.cache_clear()
    assert deep_gemm.is_deep_gemm_supported() is True, (
        "stock MHC pre-op must reproduce the unsupported SM121 DeepGEMM selection"
    )

    selector = getattr(banana_smasher_plugin, "configure_stock_mhc_backend", None)
    assert callable(selector), "public stock MHC backend selector is not installed"
    assert selector() is True
    assert deep_gemm.is_deep_gemm_supported() is False

    calls: list[str] = []

    def forbidden_deep_gemm(*args, **kwargs):
        del args, kwargs
        calls.append("deep_gemm")
        raise AssertionError("unsupported SM121 DeepGEMM MHC path selected")

    def supported_tilelang(
        residual,
        fn,
        out,
        sqrsum,
        hidden_size,
        hc_mult,
    ):
        del residual, fn, hidden_size, hc_mult
        calls.append("tilelang")
        out.zero_()
        sqrsum.zero_()

    monkeypatch.setattr(deep_gemm, "tf32_hc_prenorm_gemm", forbidden_deep_gemm)
    monkeypatch.setattr(tilelang, "_tilelang_hc_prenorm_gemm", supported_tilelang)

    kernels = pytest.importorskip("vllm.model_executor.kernels.mhc.tilelang_kernels")
    monkeypatch.setattr(kernels, "mhc_pre_big_fuse_tilelang", lambda *args: None)
    monkeypatch.setattr(
        kernels,
        "mhc_pre_big_fuse_with_norm_tilelang",
        lambda *args: None,
    )

    residual = torch.zeros((1, 2, 64), dtype=torch.bfloat16, device="cuda")
    fn = torch.zeros((8, 128), dtype=torch.float32, device="cuda")
    hc_scale = torch.ones((3,), dtype=torch.float32, device="cuda")
    hc_base = torch.zeros((8,), dtype=torch.float32, device="cuda")
    post_mix, comb_mix, layer_input = tilelang.mhc_pre_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        1,
    )

    assert calls == ["tilelang"]
    assert post_mix.shape == (1, 2, 1)
    assert comb_mix.shape == (1, 2, 2)
    assert layer_input.shape == (1, 64)
