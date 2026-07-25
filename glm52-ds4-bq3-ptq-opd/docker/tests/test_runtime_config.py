from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime" / "mixed_prefill_server.py"


def _load_runtime():
    import importlib.util

    spec = importlib.util.spec_from_file_location("runtime_server_for_test", RUNTIME)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_reads_product_identity_from_validated_pack_environment() -> None:
    code = (
        "import importlib.util,json; "
        f"p={str(RUNTIME)!r}; "
        "s=importlib.util.spec_from_file_location('runtime_server',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "print(json.dumps({'bytes':m.PRODUCT_BYTES,'files':m.PRODUCT_FILES,'model':m.MODEL_ID,'inventory':m.PRODUCT_INVENTORY_SHA256}))"
    )
    environment = dict(os.environ)
    environment.update({
        "GENESIS_PRODUCT_BYTES": "1234",
        "GENESIS_PRODUCT_FILES": "7",
        "GENESIS_MODEL_ID": "mounted-export",
        "GENESIS_PRODUCT_INVENTORY_SHA256": "a" * 64,
    })

    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "bytes": 1234,
        "files": 7,
        "model": "mounted-export",
        "inventory": "a" * 64,
    }


def test_runtime_allows_zero_setup_without_an_internal_host_claim(tmp_path: Path) -> None:
    stop_receipt = tmp_path / "stop.json"
    code = (
        "import importlib.util,json,pathlib; "
        f"p={str(RUNTIME)!r}; "
        "s=importlib.util.spec_from_file_location('runtime_server',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"g=m.CotenantGuard(None,pathlib.Path({str(stop_receipt)!r})); "
        "print(json.dumps(g.assert_wait(),sort_keys=True))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "allowed": True,
        "mode": "standalone-container",
        "status_path": None,
    }


def test_file_backed_residency_uses_hash_warmed_mincore_without_page_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _load_runtime()
    monkeypatch.setattr(runtime, "mem_available", lambda: 1 << 40)
    monkeypatch.setattr(runtime, "proc_memory", lambda: {})
    planes = tmp_path / "planes"
    planes.mkdir()
    payloads = (b"a" * 8192, b"b" * 4097)
    for index, payload in enumerate(payloads):
        (planes / f"plane-{index:03d}.bin").write_bytes(payload)
    runtime.PRODUCT_FILES = len(payloads)
    runtime.PRODUCT_BYTES = sum(map(len, payloads))
    guard = runtime.CotenantGuard(None, tmp_path / "stop.json")

    residency, receipt = runtime.map_local_tree_file_backed(
        str(planes), runtime.PRODUCT_BYTES, tmp_path / "progress.json", "test", guard
    )

    assert residency.resident_logical_bytes() == runtime.PRODUCT_BYTES
    assert receipt["bytes_mapped"] == runtime.PRODUCT_BYTES
    assert receipt["resident_logical_bytes_mincore"] == runtime.PRODUCT_BYTES
    assert "bytes_faulted" not in receipt


def test_non_stream_response_is_openai_compatible() -> None:
    runtime = _load_runtime()
    events = [
        {
            "event": "first_token",
            "id": "cmpl-test",
            "created": 123,
            "model": "mounted-export",
            "mixed_tier": {"ttft_seconds": 1.25},
        },
        {
            "event": "done",
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 123,
            "model": "mounted-export",
            "choices": [{"index": 0, "text": "ok", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            "mixed_tier": {"decode_tok_s": 17.0},
        },
    ]

    response = runtime.openai_completion(events)

    assert response["id"] == "cmpl-test"
    assert response["object"] == "text_completion"
    assert response["choices"][0]["text"] == "ok"
    assert response["usage"]["total_tokens"] == 4
    assert response["mixed_tier"]["ttft_seconds"] == 1.25
    assert response["mixed_tier"]["decode_tok_s"] == 17.0
    assert "event" not in response
