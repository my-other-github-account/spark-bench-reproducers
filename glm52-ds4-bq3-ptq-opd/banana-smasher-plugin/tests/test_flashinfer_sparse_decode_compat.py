from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import banana_smasher_plugin


@pytest.fixture(autouse=True)
def _reset_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(banana_smasher_plugin, "_REGISTERED", False)


def _install_api(monkeypatch: pytest.MonkeyPatch, target):
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "sentinel"

    decode = ModuleType("flashinfer.decode")
    setattr(decode, "trtllm_batch_decode_sparse_mla_dsv4", target)
    utils = ModuleType("vllm.utils.flashinfer")
    setattr(utils, "flashinfer_trtllm_batch_decode_sparse_mla_dsv4", original)
    sparse = ModuleType("vllm.models.deepseek_v4.nvidia.flashinfer_sparse")
    setattr(sparse, "flashinfer_trtllm_batch_decode_sparse_mla_dsv4", original)

    modules = {
        "flashinfer": ModuleType("flashinfer"),
        "flashinfer.decode": decode,
        "vllm": ModuleType("vllm"),
        "vllm.utils": ModuleType("vllm.utils"),
        "vllm.utils.flashinfer": utils,
        "vllm.models": ModuleType("vllm.models"),
        "vllm.models.deepseek_v4": ModuleType("vllm.models.deepseek_v4"),
        "vllm.models.deepseek_v4.nvidia": ModuleType("vllm.models.deepseek_v4.nvidia"),
        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse": sparse,
    }
    setattr(modules["flashinfer"], "decode", decode)
    setattr(modules["vllm"], "utils", modules["vllm.utils"])
    setattr(modules["vllm.utils"], "flashinfer", utils)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(calls=calls, original=original, utils=utils, sparse=sparse)


def test_legacy_flashinfer_omits_only_known_optional_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def target(query, key, *, scale):
        return query, key, scale

    api = _install_api(monkeypatch, target)
    assert banana_smasher_plugin.configure_flashinfer_sparse_mla_signature_compat()
    wrapped = api.utils.flashinfer_trtllm_batch_decode_sparse_mla_dsv4
    result = wrapped(
        "q",
        "k",
        scale=3,
        swa_topk_lens="swa",
        extra_sparse_indices="extra",
        passthrough="kept",
    )
    assert result == "sentinel"
    assert api.calls == [(('q', 'k'), {"scale": 3, "passthrough": "kept"})]
    assert api.sparse.flashinfer_trtllm_batch_decode_sparse_mla_dsv4 is wrapped
    assert wrapped._banana_smasher_sparse_mla_signature_variant == (
        "without_extra_sparse_indices+swa_topk_lens"
    )


def test_current_flashinfer_forwards_both_optional_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def target(
        query,
        key,
        *,
        scale,
        swa_topk_lens=None,
        extra_sparse_indices=None,
    ):
        return query, key, scale, swa_topk_lens, extra_sparse_indices

    api = _install_api(monkeypatch, target)
    assert banana_smasher_plugin.configure_flashinfer_sparse_mla_signature_compat()
    wrapped = api.utils.flashinfer_trtllm_batch_decode_sparse_mla_dsv4
    result = wrapped(
        "q",
        "k",
        scale=3,
        swa_topk_lens="swa",
        extra_sparse_indices="extra",
    )
    assert result == "sentinel"
    assert api.calls == [
        (
            ("q", "k"),
            {
                "scale": 3,
                "swa_topk_lens": "swa",
                "extra_sparse_indices": "extra",
            },
        )
    ]
    assert wrapped._banana_smasher_sparse_mla_signature_variant == "all_optional_kwargs"


def test_var_keyword_flashinfer_forwards_both_optional_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def target(query, key, **kwargs):
        return query, key, kwargs

    api = _install_api(monkeypatch, target)
    assert banana_smasher_plugin.configure_flashinfer_sparse_mla_signature_compat()
    wrapped = api.utils.flashinfer_trtllm_batch_decode_sparse_mla_dsv4
    wrapped("q", "k", swa_topk_lens="swa", extra_sparse_indices="extra")
    assert api.calls[0][1] == {
        "swa_topk_lens": "swa",
        "extra_sparse_indices": "extra",
    }
