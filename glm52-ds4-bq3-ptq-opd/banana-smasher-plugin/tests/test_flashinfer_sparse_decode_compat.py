from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

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


def test_legacy_flashinfer_combines_separate_sparse_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def target(
        query,
        swa_kv_cache,
        workspace_buffer,
        sparse_indices,
        compressed_kv_cache,
        sparse_topk_lens,
        seq_lens,
        *,
        out=None,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        sinks=None,
        kv_layout="HND",
        cum_seq_lens_q=None,
        max_q_len=None,
        enable_pdl=None,
    ):
        return query, sparse_indices, sparse_topk_lens, seq_lens

    api = _install_api(monkeypatch, target)
    assert banana_smasher_plugin.configure_flashinfer_sparse_mla_signature_compat()
    wrapped = api.utils.flashinfer_trtllm_batch_decode_sparse_mla_dsv4
    query = torch.zeros((2, 64, 512), dtype=torch.bfloat16)
    swa_cache = torch.zeros((1, 1, 1, 512), dtype=torch.bfloat16)
    extra_cache = torch.ones((1, 1, 1, 512), dtype=torch.bfloat16)
    swa_indices = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    extra_indices = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    swa_lens = torch.tensor([7, 9], dtype=torch.int32)
    extra_lens = torch.tensor([3, 2], dtype=torch.int32)
    result = wrapped(
        query=query,
        swa_kv_cache=swa_cache,
        workspace_buffer=torch.zeros(8, dtype=torch.uint8),
        sparse_indices=swa_indices,
        compressed_kv_cache=extra_cache,
        out=torch.empty_like(query),
        bmm1_scale=3,
        kv_layout="NHD",
        swa_topk_lens=swa_lens,
        extra_sparse_indices=extra_indices,
        extra_sparse_topk_lens=extra_lens,
    )
    assert result == "sentinel"
    args, kwargs = api.calls[0]
    assert args == ()
    assert torch.equal(
        kwargs["sparse_indices"], torch.cat((swa_indices, extra_indices), dim=-1)
    )
    assert torch.equal(kwargs["sparse_topk_lens"], torch.tensor([131, 130]))
    assert torch.equal(kwargs["seq_lens"], swa_lens)
    assert torch.equal(kwargs["cum_seq_lens_q"], torch.tensor([0, 1, 2]))
    assert kwargs["max_q_len"] == 1
    assert kwargs["compressed_kv_cache"] is extra_cache
    assert "swa_topk_lens" not in kwargs
    assert "extra_sparse_indices" not in kwargs
    assert "extra_sparse_topk_lens" not in kwargs
    assert api.sparse.flashinfer_trtllm_batch_decode_sparse_mla_dsv4 is wrapped
    assert (
        wrapped._banana_smasher_sparse_mla_signature_variant
        == "legacy_combined_sparse_api"
    )


def test_current_flashinfer_forwards_all_optional_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def target(
        query,
        key,
        *,
        scale,
        swa_topk_lens=None,
        extra_sparse_indices=None,
        extra_sparse_topk_lens=None,
    ):
        return (
            query,
            key,
            scale,
            swa_topk_lens,
            extra_sparse_indices,
            extra_sparse_topk_lens,
        )

    api = _install_api(monkeypatch, target)
    assert banana_smasher_plugin.configure_flashinfer_sparse_mla_signature_compat()
    wrapped = api.utils.flashinfer_trtllm_batch_decode_sparse_mla_dsv4
    result = wrapped(
        "q",
        "k",
        scale=3,
        swa_topk_lens="swa",
        extra_sparse_indices="extra",
        extra_sparse_topk_lens="extra_lens",
    )
    assert result == "sentinel"
    assert api.calls == [
        (
            ("q", "k"),
            {
                "scale": 3,
                "swa_topk_lens": "swa",
                "extra_sparse_indices": "extra",
                "extra_sparse_topk_lens": "extra_lens",
            },
        )
    ]
    assert wrapped._banana_smasher_sparse_mla_signature_variant == "all_optional_kwargs"


def test_var_keyword_flashinfer_forwards_all_optional_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def target(query, key, **kwargs):
        return query, key, kwargs

    api = _install_api(monkeypatch, target)
    assert banana_smasher_plugin.configure_flashinfer_sparse_mla_signature_compat()
    wrapped = api.utils.flashinfer_trtllm_batch_decode_sparse_mla_dsv4
    wrapped(
        "q",
        "k",
        swa_topk_lens="swa",
        extra_sparse_indices="extra",
        extra_sparse_topk_lens="extra_lens",
    )
    assert api.calls[0][1] == {
        "swa_topk_lens": "swa",
        "extra_sparse_indices": "extra",
        "extra_sparse_topk_lens": "extra_lens",
    }
