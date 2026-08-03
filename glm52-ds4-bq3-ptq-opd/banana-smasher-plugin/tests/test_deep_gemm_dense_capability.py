from __future__ import annotations

import contextlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import banana_smasher_plugin


def _install_modules(
    monkeypatch: pytest.MonkeyPatch, *, major: int, minor: int
) -> tuple[ModuleType, ModuleType]:
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=major, minor=minor)
    )
    utils = ModuleType("vllm.utils.deep_gemm")
    external = ModuleType("deep_gemm")
    warmup = ModuleType("vllm.model_executor.warmup.deep_gemm_warmup")

    def implementation(*args, **kwargs):
        return args, kwargs

    def lazy_init():
        utils._get_paged_mqa_logits_metadata_impl = (
            external.get_paged_mqa_logits_metadata
        )
        utils._fp8_fp4_paged_mqa_logits_impl = external.fp8_fp4_paged_mqa_logits
        utils._fp8_fp4_mqa_logits_impl = external.fp8_fp4_mqa_logits
        utils._get_mk_alignment_for_contiguous_layout_impl = (
            external.get_mk_alignment_for_contiguous_layout
        )
        utils._fp8_gemm_nt_impl = external.fp8_gemm_nt

    external.get_paged_mqa_logits_metadata = implementation
    external.fp8_fp4_paged_mqa_logits = implementation
    external.fp8_fp4_mqa_logits = implementation
    external.get_mk_alignment_for_contiguous_layout = lambda: (128, 128)
    external.fp8_gemm_nt = implementation
    utils._import_deep_gemm = lambda: external
    utils._lazy_init = lazy_init
    utils.is_deep_gemm_supported = lambda: True
    warmup._fp8_linear_may_use_deep_gemm = lambda _module: True

    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "vllm.utils.deep_gemm", utils)
    monkeypatch.setitem(sys.modules, "deep_gemm", external)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.warmup.deep_gemm_warmup",
        warmup,
    )
    return utils, warmup


@pytest.mark.parametrize(("minor"), [0, 1])
def test_sm12x_keeps_sparse_and_dense_deepgemm_recipe_warmup(
    monkeypatch: pytest.MonkeyPatch, minor: int
) -> None:
    utils, warmup = _install_modules(monkeypatch, major=12, minor=minor)

    assert banana_smasher_plugin.configure_sparse_indexer_deep_gemm_backend()
    assert callable(utils._get_paged_mqa_logits_metadata_impl)
    assert callable(utils._fp8_fp4_paged_mqa_logits_impl)
    assert callable(utils._fp8_fp4_mqa_logits_impl)
    assert callable(utils._get_mk_alignment_for_contiguous_layout_impl)
    assert callable(utils._fp8_gemm_nt_impl)
    assert utils.is_deep_gemm_supported() is True
    assert warmup._fp8_linear_may_use_deep_gemm(object()) is True


def test_pre_sm12x_preserves_stock_dense_deepgemm_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utils, warmup = _install_modules(monkeypatch, major=9, minor=0)
    support = utils.is_deep_gemm_supported
    predicate = warmup._fp8_linear_may_use_deep_gemm

    assert banana_smasher_plugin.configure_sparse_indexer_deep_gemm_backend() is False
    assert utils.is_deep_gemm_supported is support
    assert warmup._fp8_linear_may_use_deep_gemm is predicate
    assert utils.is_deep_gemm_supported() is True
    assert warmup._fp8_linear_may_use_deep_gemm(object()) is True


def test_sm12x_dense_warmup_initializes_every_ue8m0_scale() -> None:
    warmup = ModuleType("vllm.model_executor.warmup.deep_gemm_warmup")
    warmup.FP8_GEMM_NT_WARMUP_CACHE = set()
    warmup.get_mk_alignment_for_contiguous_layout = lambda: (128, 128)
    warmup._get_fp8_gemm_nt_m_values = lambda _w, _max_tokens: [1, 2]
    observed_scales: list[torch.Tensor] = []

    def fp8_gemm_nt(a, _b, _out):
        scales = a[1].clone()
        observed_scales.append(scales)
        assert torch.all(scales == 1)

    warmup.fp8_gemm_nt = fp8_gemm_nt
    original = lambda *_args, **_kwargs: None
    warmup._deepgemm_fp8_gemm_nt_warmup = original
    warmup._deepgemm_grouped_fp8_gemm_nt_contiguous_warmup = original

    assert banana_smasher_plugin.configure_deep_gemm_ue8m0_warmup_contract(
        warmup_module=warmup
    )
    assert warmup._deepgemm_fp8_gemm_nt_warmup is not original

    weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones((1, 1), dtype=torch.float32)
    warmup._deepgemm_fp8_gemm_nt_warmup(weight, weight_scale, 2)

    assert len(observed_scales) == 2
    assert all(torch.all(scales == 1) for scales in observed_scales)


def test_sm12x_grouped_warmup_initializes_every_ue8m0_scale() -> None:
    warmup = ModuleType("vllm.model_executor.warmup.deep_gemm_warmup")
    warmup.FP8_GEMM_NT_WARMUP_CACHE = set()
    warmup.GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE = set()
    warmup.get_mk_alignment_for_contiguous_layout = lambda: (128, 128)
    warmup._get_fp8_gemm_nt_m_values = lambda _w, _max_tokens: []
    expert_ids = torch.zeros((2,), dtype=torch.int32)
    warmup._get_grouped_gemm_params = (
        lambda _w1, _w2, _num_topk, _max_tokens: (2, 128, [(2, 128, expert_ids)])
    )
    warmup.mk_alignment_scope = lambda _alignment: contextlib.nullcontext()
    observed_scales: list[torch.Tensor] = []

    def grouped(a, _b, _out, _expert_ids):
        scales = a[1].clone()
        observed_scales.append(scales)
        assert torch.all(scales == 1)

    warmup.m_grouped_fp8_gemm_nt_contiguous = grouped
    warmup._deepgemm_fp8_gemm_nt_warmup = lambda *_args, **_kwargs: None
    original_grouped = lambda *_args, **_kwargs: None
    warmup._deepgemm_grouped_fp8_gemm_nt_contiguous_warmup = original_grouped

    assert banana_smasher_plugin.configure_deep_gemm_ue8m0_warmup_contract(
        warmup_module=warmup
    )
    assert (
        warmup._deepgemm_grouped_fp8_gemm_nt_contiguous_warmup
        is not original_grouped
    )

    weight1 = torch.empty((2, 128, 128), dtype=torch.float8_e4m3fn)
    weight2 = torch.empty((2, 256, 128), dtype=torch.float8_e4m3fn)
    scale1 = torch.ones((2, 1, 1), dtype=torch.float32)
    scale2 = torch.ones((2, 1, 1), dtype=torch.float32)
    warmup._deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(
        weight1, weight2, scale1, scale2, 1, 2
    )

    assert len(observed_scales) == 2
    assert all(torch.all(scales == 1) for scales in observed_scales)
