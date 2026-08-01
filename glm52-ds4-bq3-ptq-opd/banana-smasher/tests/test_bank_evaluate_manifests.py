from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
BANK_EVALUATE_PAYLOADS = (
    "schema/bs-pack-v1.schema.json",
    "schema/bs-paired-real-axis-evaluation-v1.schema.json",
    "schema/bs-real-axis-instrument-v1.schema.json",
    "schema/bs-real-axis-runtime-v1.schema.json",
    "schema/bs-real-axis-windows-v1.schema.json",
    "schema/bs-teacher-bank-v1.schema.json",
    "src/banana_smasher/bank.py",
    "src/banana_smasher/durability.py",
    "src/banana_smasher/evaluate.py",
    "src/banana_smasher/metrics.py",
    "src/banana_smasher/profiles/real-axis-v1.json",
    "src/banana_smasher/real_axis.py",
    "tests/real_axis_fixtures.py",
    "tests/test_bank.py",
    "tests/test_bank_evaluate_cli.py",
    "tests/test_bank_evaluate_manifests.py",
    "tests/test_evaluate.py",
    "tests/test_real_axis_metrics.py",
)


def test_source_manifest_binds_bank_evaluate_payloads() -> None:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for relative in BANK_EVALUATE_PAYLOADS:
        payload = (ROOT / relative).read_bytes()
        assert manifest["files"][relative] == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


def test_publication_transform_records_sanitized_s6_lineage() -> None:
    transform_path = ROOT / "PUBLICATION_TRANSFORM.json"
    transform = json.loads(transform_path.read_text())
    integration = transform["post_publication_integrations"]["bank_evaluate"]
    assert integration["design_sha256"] == (
        "884f5442da0f0ce827ae17aae7fd6d3c08a060ea3388c6bc3b9d08fcbcb83dcf"
    )
    assert integration["status"] == "PUBLIC_PORT_WITH_FULL_MIXED_TIER_ACCEPTANCE_OPEN"
    source = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    assert source["publication"]["transform_receipt_sha256"] == hashlib.sha256(
        transform_path.read_bytes()
    ).hexdigest()


def test_profiles_and_schemas_are_valid_json_objects() -> None:
    for relative in BANK_EVALUATE_PAYLOADS:
        if not relative.endswith(".json"):
            continue
        assert isinstance(json.loads((ROOT / relative).read_text()), dict)


def test_public_bank_and_evaluation_schemas_are_nested_contracts() -> None:
    bank = json.loads((ROOT / "schema/bs-teacher-bank-v1.schema.json").read_text())
    evaluation = json.loads(
        (ROOT / "schema/bs-paired-real-axis-evaluation-v1.schema.json").read_text()
    )

    for name in ("model", "corpus", "instrument", "population"):
        assert bank["properties"][name]["additionalProperties"] is False
        assert bank["properties"][name]["required"]
    member = bank["properties"]["members"]["items"]
    assert member["additionalProperties"] is False
    assert {"path", "sidecar", "sha256", "positions", "tensors"} <= set(
        member["required"]
    )

    for name in (
        "bank",
        "population",
        "instrument",
        "topology",
        "paired",
        "performance",
        "resume",
    ):
        assert evaluation["properties"][name]["additionalProperties"] is False
        assert evaluation["properties"][name]["required"]
    performance = evaluation["properties"]["performance"]
    assert set(performance["required"]) == {
        "tokens_per_second",
        "wall_seconds",
        "peak_vram_bytes",
        "quality_result",
        "kernel",
        "fallback_used",
        "fallback_status",
        "window_batch_size",
        "layer_forwards_per_arm",
        "head_forwards_per_arm",
    }
    assert performance["properties"]["fallback_used"] == {"const": False}
    assert performance["properties"]["fallback_status"] == {"const": "none"}
    arms = evaluation["properties"]["arms"]
    assert arms["additionalProperties"] is False
    assert arms["properties"]["candidate"]["required"]
