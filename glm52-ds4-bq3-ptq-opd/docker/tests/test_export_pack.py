from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from export_pack import export_pack
from pack_contract import validate_pack


def test_export_pack_is_self_describing_and_validates(tmp_path: Path) -> None:
    planes = tmp_path / "source"
    planes.mkdir()
    (planes / "plane-000.bin").write_bytes(b"plane-a")
    (planes / "plane-001.bin").write_bytes(b"plane-b")
    overlay = tmp_path / "mixed_tier_compact.pt"
    overlay.write_bytes(b"overlay")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(json.dumps({"version": "1.0"}))
    output = tmp_path / "export"

    manifest = export_pack(
        source_root=planes,
        overlay=overlay,
        tokenizer=tokenizer,
        output=output,
        model_id="fixture",
        expected_bytes=14,
        expected_files=2,
        expected_inventory_sha256=None,
        sealed_source_inventory_sha256="a" * 64,
    )
    validation = validate_pack(output)

    assert manifest["schema"] == "banana_smasher-pack"
    assert manifest["container_schema_version"] == 1
    assert manifest["validation_scope"] == "systems-serving-only"
    assert manifest["quality_validated"] is False
    assert validation["resident_envelope"]["bytes"] == 14
    assert validation["resident_envelope"]["files"] == 2
    assert validation["resident_envelope"]["sealed_source_inventory_sha256"] == "a" * 64
    assert (output / "planes/plane-000.bin").read_bytes() == b"plane-a"
    assert (output / "overlay/mixed_tier_compact.pt").read_bytes() == b"overlay"
