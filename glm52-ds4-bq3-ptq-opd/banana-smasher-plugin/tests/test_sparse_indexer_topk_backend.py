from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import banana_smasher_plugin


def _install_platform(
    monkeypatch: pytest.MonkeyPatch, *, major: int, minor: int
) -> None:
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=major, minor=minor)
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)


def _install_topk_ops(
    monkeypatch: pytest.MonkeyPatch, *, persistent: object | None = None
) -> tuple[SimpleNamespace, list[tuple[str, tuple, dict]]]:
    calls: list[tuple[str, tuple, dict]] = []

    def cooperative(*args, **kwargs):
        calls.append(("cooperative", args, kwargs))
        return "cooperative-result"

    def default_persistent(*args, **kwargs):
        calls.append(("persistent", args, kwargs))
        return "persistent-result"

    namespace = SimpleNamespace(
        cooperative_topk=cooperative,
        persistent_topk=default_persistent if persistent is None else persistent,
    )
    monkeypatch.setattr(banana_smasher_plugin.torch.ops, "_C", namespace)
    return namespace, calls


@pytest.mark.parametrize(("major", "minor"), [(12, 0), (12, 1)])
def test_sm12x_routes_cooperative_topk_to_persistent_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    major: int,
    minor: int,
) -> None:
    _install_platform(monkeypatch, major=major, minor=minor)
    namespace, calls = _install_topk_ops(monkeypatch)
    original = namespace.cooperative_topk
    caplog.set_level(logging.WARNING, logger="banana_smasher_plugin")

    assert banana_smasher_plugin.configure_sparse_indexer_topk_backend() is True
    corrected = namespace.cooperative_topk
    assert corrected is not original
    assert corrected("logits", k=2048) == "persistent-result"
    assert calls == [("persistent", ("logits",), {"k": 2048})]

    assert banana_smasher_plugin.configure_sparse_indexer_topk_backend() is True
    assert namespace.cooperative_topk is corrected
    assert sum(
        "BANANA_SMASHER_SPARSE_INDEXER_TOPK_OVERRIDE" in record.message
        for record in caplog.records
    ) == 1


@pytest.mark.parametrize(("major", "minor"), [(9, 0), (10, 0)])
def test_supported_stock_architectures_preserve_cooperative_topk(
    monkeypatch: pytest.MonkeyPatch, major: int, minor: int
) -> None:
    _install_platform(monkeypatch, major=major, minor=minor)
    namespace, _ = _install_topk_ops(monkeypatch)
    original = namespace.cooperative_topk

    assert banana_smasher_plugin.configure_sparse_indexer_topk_backend() is False
    assert namespace.cooperative_topk is original


def test_sm12x_refuses_when_persistent_topk_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_platform(monkeypatch, major=12, minor=1)
    _install_topk_ops(monkeypatch, persistent=object())

    with pytest.raises(RuntimeError, match="requires vLLM persistent_topk"):
        banana_smasher_plugin.configure_sparse_indexer_topk_backend()


def test_real_sm12x_corrected_topk_matches_torch_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("real sparse-indexer TopK regression requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 12:
        pytest.skip("real sparse-indexer TopK regression requires SM12x")

    from vllm.model_executor.layers.sparse_attn_indexer import (
        RADIX_TOPK_WORKSPACE_SIZE,
    )

    assert banana_smasher_plugin.configure_sparse_indexer_topk_backend()
    rows, columns, k = 1, 8192, 2048
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1234)
    logits = torch.randn(
        rows,
        columns,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    lengths = torch.full(
        (rows, 1), columns, device="cuda", dtype=torch.int32
    )
    output = torch.full((rows, k), -1, device="cuda", dtype=torch.int32)
    workspace = torch.empty(
        RADIX_TOPK_WORKSPACE_SIZE, device="cuda", dtype=torch.uint8
    )

    torch.ops._C.cooperative_topk(
        logits, lengths, output, workspace, k, columns
    )
    torch.cuda.synchronize()
    expected = torch.topk(logits, k, dim=-1).indices.sort(dim=-1).values
    observed = output.sort(dim=-1).values
    assert torch.equal(observed, expected)
