from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from entrypoint import (
    build_server_command,
    build_startup_receipt,
    pack_hash_workers,
    prepare_pack,
)
from pack_contract import canonical_inventory_sha256, validate_pack


def _write_pack(root: Path) -> Path:
    payloads = {
        "planes/layer.bin": b"plane",
        "overlay/mixed_tier_compact.pt": b"overlay",
        "tokenizer/tokenizer.json": b'{"version":"1.0"}',
    }
    rows = []
    for relative, data in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        role = "plane" if relative.startswith("planes/") else (
            "mixed_tier_overlay" if relative.startswith("overlay/") else "tokenizer"
        )
        rows.append({
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "role": role,
        })
    planes = [row for row in rows if row["role"] == "plane"]
    (root / "MANIFEST.json").write_text(json.dumps({
        "schema": "banana_smasher-pack",
        "schema_version": 1,
        "model_id": "fixture",
        "validation_scope": "systems-serving-only",
        "quality_validated": False,
        "files": rows,
        "resident_envelope": {
            "root": "planes",
            "files": len(planes),
            "bytes": sum(row["bytes"] for row in planes),
            "inventory_sha256": canonical_inventory_sha256(planes),
            "sealed_source_inventory_sha256": "a" * 64,
        },
        "serving": {
            "artifact": "overlay/mixed_tier_compact.pt",
            "tokenizer": "tokenizer/tokenizer.json",
            "layers": 43,
            "experts": 256,
            "topk": 6,
            "tier_names": ["qtip", "truevq_d4", "truevq_d8", "native_mxfp4"],
            "prefill_mode": "dense_all",
            "dense_threshold": 64,
            "layer_stride": 43,
        },
    }))
    return root


def test_entrypoint_verify_accepts_one_pack_argument_without_environment(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    entrypoint = SCRIPTS / "entrypoint.py"

    completed = subprocess.run(
        [sys.executable, str(entrypoint), "verify", str(pack)],
        text=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["status"] == "PASS"
    assert result["pack"]["model_id"] == "fixture"
    assert result["container_schema_version"] == 1


def test_server_command_uses_only_the_mounted_pack(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    validation = validate_pack(pack)
    mission = tmp_path / "run"

    command, environment = build_server_command(pack, validation, mission, port=8000)

    assert command[command.index("--source-host") + 1] == "local"
    assert command[command.index("--source-root") + 1] == str(pack.resolve() / "planes")
    assert command[command.index("--artifact") + 1] == str(pack.resolve() / "overlay/mixed_tier_compact.pt")
    assert command[command.index("--tokenizer-json") + 1] == str(pack.resolve() / "tokenizer/tokenizer.json")
    assert command[command.index("--port") + 1] == "8000"
    assert environment["BANANA_SMASHER_PRODUCT_BYTES"] == str(len(b"plane"))
    assert environment["BANANA_SMASHER_PRODUCT_FILES"] == "1"
    assert environment["BANANA_SMASHER_PRODUCT_INVENTORY_SHA256"] == "a" * 64
    assert environment["BANANA_SMASHER_MODEL_ID"] == "fixture"
    assert environment["BANANA_SMASHER_MANIFEST_PATH"] == str(pack.resolve() / "MANIFEST.json")
    assert environment["P530_FILE_BACKED_ENVELOPE"] == "1"
    assert "BANANA_SMASHER_SOURCE_HOST" not in environment


def test_prepare_pack_keeps_a_local_read_only_mount_in_place(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")

    resolved = prepare_pack(str(pack), tmp_path / "downloads")

    assert resolved == pack.resolve()
    assert (resolved / "MANIFEST.json").is_file()


def test_pack_hashing_defaults_to_accelerated_parallel_read(monkeypatch) -> None:
    monkeypatch.delenv("BANANA_SMASHER_PACK_HASH_WORKERS", raising=False)

    assert pack_hash_workers() == 32


def test_startup_receipt_reports_bind_first_token_and_speed() -> None:
    response = {
        "id": "cmpl-fixture",
        "model": "fixture",
        "mixed_tier": {
            "ttft_seconds": 1.5,
            "prefill_tok_s_server": 1200.0,
            "decode_tok_s": 17.0,
            "resident_product_bytes": 101_346_700_411,
        },
    }

    receipt = build_startup_receipt(
        started_at=100.0,
        models_ready_at=110.0,
        request_started_at=111.0,
        request_finished_at=115.0,
        response=response,
    )

    assert receipt["status"] == "PASS"
    assert receipt["bind_seconds"] == 10.0
    assert receipt["first_token_seconds_from_container_start"] == 12.5
    assert receipt["smoke_response_seconds_from_container_start"] == 15.0
    assert receipt["prefill_tok_s"] == 1200.0
    assert receipt["decode_tok_s"] == 17.0
    assert receipt["resident_product_bytes"] == 101_346_700_411
