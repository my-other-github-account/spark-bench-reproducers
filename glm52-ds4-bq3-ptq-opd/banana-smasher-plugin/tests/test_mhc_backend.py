from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import banana_smasher_plugin


def test_public_mhc_selector_disables_deepgemm_on_sm121(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=12, minor=1)
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")

    selector = getattr(banana_smasher_plugin, "configure_stock_mhc_backend", None)
    assert callable(selector), "public stock MHC backend selector is not installed"
    assert selector() is True
    assert os.environ["VLLM_USE_DEEP_GEMM"] == "0"


def test_real_deepseek_v4_mhc_preop_routes_sm121_to_tilelang_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
