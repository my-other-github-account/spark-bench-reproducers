from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from pack_contract import PackValidationError, canonical_inventory_sha256, validate_pack


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_valid_pack(root: Path) -> Path:
    payloads = {
        "planes/layer-000.bin": b"plane-data",
        "overlay/mixed_tier_compact.pt": b"overlay-data",
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
        rows.append({"path": relative, "bytes": len(data), "sha256": _sha(data), "role": role})
    plane_rows = [row for row in rows if row["role"] == "plane"]
    manifest = {
        "schema": "banana_smasher-pack",
        "schema_version": 1,
        "model_id": "fixture-model",
        "validation_scope": "systems-serving-only",
        "quality_validated": False,
        "files": rows,
        "resident_envelope": {
            "root": "planes",
            "files": len(plane_rows),
            "bytes": sum(row["bytes"] for row in plane_rows),
            "inventory_sha256": canonical_inventory_sha256(plane_rows),
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
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    return root


def test_validate_pack_accepts_exact_schema_layout_and_hashes(tmp_path: Path) -> None:
    result = validate_pack(_write_valid_pack(tmp_path / "pack"), expected_schema_version=1)

    assert result["status"] == "PASS"
    assert result["schema"] == "banana_smasher-pack-validation-v1"
    assert result["model_id"] == "fixture-model"
    assert result["resident_envelope"]["bytes"] == len(b"plane-data")
    assert result["serving"]["artifact"] == "overlay/mixed_tier_compact.pt"


def test_canonical_inventory_binds_pack_relative_paths() -> None:
    row = {
        "path": "planes/layer-000.bin",
        "bytes": 10,
        "sha256": "a" * 64,
    }
    expected = hashlib.sha256(
        b"planes/layer-000.bin\0" + b"10\0" + b"a" * 64 + b"\n"
    ).hexdigest()

    assert canonical_inventory_sha256([row]) == expected
    assert canonical_inventory_sha256([row]) != canonical_inventory_sha256(
        [{**row, "path": "layer-000.bin"}]
    )


def test_validate_pack_rejects_payload_missing_from_manifest(tmp_path: Path) -> None:
    pack = _write_valid_pack(tmp_path / "pack")
    (pack / "planes" / "unlisted.bin").write_bytes(b"not declared")

    with pytest.raises(PackValidationError, match="unlisted payload"):
        validate_pack(pack, expected_schema_version=1)


def test_validate_pack_rejects_missing_or_malformed_manifest(tmp_path: Path) -> None:
    with pytest.raises(PackValidationError, match="missing or unsafe"):
        validate_pack(tmp_path / "missing", expected_schema_version=1)

    pack = tmp_path / "malformed"
    pack.mkdir()
    (pack / "MANIFEST.json").write_text("{not-json")
    with pytest.raises(PackValidationError, match="unreadable"):
        validate_pack(pack, expected_schema_version=1)


def test_validate_pack_rejects_symlink_payload(tmp_path: Path) -> None:
    pack = _write_valid_pack(tmp_path / "pack")
    payload = pack / "planes/layer-000.bin"
    backing = pack / "planes/backing.bin"
    payload.rename(backing)
    payload.symlink_to(backing.name)

    with pytest.raises(PackValidationError, match="payload missing"):
        validate_pack(pack, expected_schema_version=1)


def test_validate_pack_rejects_runtime_contract_drift(tmp_path: Path) -> None:
    pack = _write_valid_pack(tmp_path / "pack")
    manifest_path = pack / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["serving"]["layers"] = 42
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(PackValidationError, match="serving.layers mismatch"):
        validate_pack(pack, expected_schema_version=1)


def test_validate_pack_rejects_malformed_source_inventory_provenance(tmp_path: Path) -> None:
    pack = _write_valid_pack(tmp_path / "pack")
    manifest_path = pack / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resident_envelope"]["sealed_source_inventory_sha256"] = "not-a-sha"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(PackValidationError, match="sealed source inventory"):
        validate_pack(pack, expected_schema_version=1)
