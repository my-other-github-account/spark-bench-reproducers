from __future__ import annotations

import importlib
import logging
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest
import torch

import banana_smasher_plugin


class _Backend(Enum):
    FLASHMLA_SPARSE_DSV4 = "flashmla"
    FLASHINFER_MLA_SPARSE_DSV4 = "flashinfer"


class _UnsupportedFlashMLA:
    pass


class _SupportedFlashInferSM120:
    pass


def _install_stock_dsv4_route(
    monkeypatch: pytest.MonkeyPatch,
    *,
    major: int,
    minor: int,
    flashinfer_sparse_mla_available: bool = True,
) -> tuple[ModuleType, SimpleNamespace]:
    model = ModuleType("vllm.models.deepseek_v4.nvidia.model")
    current_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=major, minor=minor)
    )
    platforms = ModuleType("vllm.platforms")
    setattr(platforms, "current_platform", current_platform)
    setattr(model, "current_platform", current_platform)
    setattr(model, "AttentionBackendEnum", _Backend)
    setattr(model, "DeepseekV4FlashMLAAttention", _UnsupportedFlashMLA)
    setattr(model, "DeepseekV4FlashInferSM120Attention", _SupportedFlashInferSM120)

    def stock_select(vllm_config):
        backend = vllm_config.attention_config.backend
        capability = model.current_platform.get_device_capability()
        if backend == _Backend.FLASHINFER_MLA_SPARSE_DSV4:
            if capability is not None and capability.major == 12:
                return _SupportedFlashInferSM120
        if backend == _Backend.FLASHMLA_SPARSE_DSV4:
            return _UnsupportedFlashMLA
        if capability is not None and capability.major == 12:
            return _SupportedFlashInferSM120
        return _UnsupportedFlashMLA

    setattr(model, "_select_dsv4_attn_cls", stock_select)
    flashinfer_utils = ModuleType("vllm.utils.flashinfer")
    setattr(
        flashinfer_utils,
        "has_flashinfer_sparse_mla_sm120",
        lambda: flashinfer_sparse_mla_available,
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "vllm.utils.flashinfer", flashinfer_utils)
    monkeypatch.setitem(sys.modules, model.__name__, model)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=_Backend.FLASHMLA_SPARSE_DSV4)
    )
    return model, config


def test_public_dsv4_attention_selector_routes_explicit_flashmla_off_sm121(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model, config = _install_stock_dsv4_route(monkeypatch, major=12, minor=1)

    assert model._select_dsv4_attn_cls(config) is _UnsupportedFlashMLA, (
        "stock DeepSeek-V4 explicit FLASHMLA_SPARSE_DSV4 selects the unsupported "
        "sparse-forward kernel on SM121"
    )
    selector = getattr(
        banana_smasher_plugin,
        "configure_stock_deepseek_v4_attention_backend",
        None,
    )
    assert callable(selector), (
        "public stock DeepSeek-V4 attention backend selector is not installed"
    )
    caplog.set_level(logging.WARNING, logger="banana_smasher_plugin")
    assert selector() is True
    wrapped = model._select_dsv4_attn_cls
    assert wrapped(config) is _SupportedFlashInferSM120
    assert wrapped(config) is _SupportedFlashInferSM120
    assert selector() is True
    assert model._select_dsv4_attn_cls is wrapped
    assert sum(
        "BANANA_SMASHER_DSV4_ATTENTION_BACKEND_OVERRIDE" in record.message
        for record in caplog.records
    ) == 1


def test_public_dsv4_attention_selector_refuses_unavailable_flashinfer_sparse_mla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, config = _install_stock_dsv4_route(
        monkeypatch,
        major=12,
        minor=1,
        flashinfer_sparse_mla_available=False,
    )

    assert banana_smasher_plugin.configure_stock_deepseek_v4_attention_backend()
    with pytest.raises(
        RuntimeError,
        match="FlashInfer.*sparse MLA decode API.*physically unavailable",
    ):
        model._select_dsv4_attn_cls(config)


def test_public_dsv4_attention_selector_skips_nvidia_model_import_off_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platforms = ModuleType("vllm.platforms")
    setattr(
        platforms,
        "current_platform",
        SimpleNamespace(
            get_device_capability=lambda: SimpleNamespace(major=10, minor=0)
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    imports: list[str] = []

    def forbidden_import(name: str):
        imports.append(name)
        raise AssertionError(f"non-SM12 startup imported {name}")

    monkeypatch.setattr(banana_smasher_plugin.importlib, "import_module", forbidden_import)

    assert banana_smasher_plugin.configure_stock_deepseek_v4_attention_backend() is False
    assert imports == []


@pytest.mark.parametrize(("major", "minor"), [(9, 0), (10, 0)])
def test_public_dsv4_attention_selector_preserves_stock_flashmla_architectures(
    monkeypatch: pytest.MonkeyPatch,
    major: int,
    minor: int,
) -> None:
    model, config = _install_stock_dsv4_route(
        monkeypatch,
        major=major,
        minor=minor,
    )

    original = model._select_dsv4_attn_cls
    assert (
        banana_smasher_plugin.configure_stock_deepseek_v4_attention_backend()
        is False
    )
    assert model._select_dsv4_attn_cls is original
    assert original(config) is _UnsupportedFlashMLA


def test_real_stock_dsv4_attention_route_selects_flashinfer_on_sm121(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = pytest.importorskip("vllm.models.deepseek_v4.nvidia.model")
    capability = model.current_platform.get_device_capability()
    if capability is None or (capability.major, capability.minor) != (12, 1):
        pytest.skip("real DeepSeek-V4 attention regression requires SM121")
    if not torch.cuda.is_available():
        pytest.skip("real DeepSeek-V4 attention regression requires CUDA")

    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=model.AttentionBackendEnum.FLASHMLA_SPARSE_DSV4
        )
    )
    assert model._select_dsv4_attn_cls(config) is model.DeepseekV4FlashMLAAttention

    assert banana_smasher_plugin.configure_stock_deepseek_v4_attention_backend()
    selected = model._select_dsv4_attn_cls(config)
    assert selected is model.DeepseekV4FlashInferSM120Attention
    backend_cls = selected.backend_cls
    assert backend_cls.get_name() == "FLASHINFER_MLA_SPARSE_DSV4"
    assert backend_cls.supports_compute_capability(capability)
    assert (
        backend_cls.supports_combination(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=256,
            use_mla=True,
            has_sink=True,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=capability,
        )
        is None
    )

    flashinfer_utils = importlib.import_module("vllm.utils.flashinfer")
    assert flashinfer_utils.has_flashinfer_sparse_mla_sm120()
    flashinfer_sparse = importlib.import_module(
        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse"
    )
    monkeypatch.setattr(
        flashinfer_sparse,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    attention = object.__new__(selected)
    q = torch.zeros((1, 16, 512), dtype=torch.bfloat16, device="cuda")
    output = torch.full_like(q, 1)
    selected.forward_mqa(
        attention,
        q,
        torch.empty(0, dtype=torch.uint8, device="cuda"),
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        output,
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
