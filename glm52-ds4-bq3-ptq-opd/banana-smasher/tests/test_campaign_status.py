from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.cli import main


TIERS = ["qtip3", "qtip2", "d4_k2048", "d4_k4096", "mxfp4"]
NOW = 1_000.0


def _write_json(path: Path, value: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _direct_anchor(
    root: Path,
    tier: str,
    *,
    layers: list[int],
    units_per_layer: int,
    created_unix: float,
) -> dict[str, object]:
    receipt_rows = []
    receipt_schema = (
        "banana-smasher-mxfp4-anchor-layer-receipt-v1"
        if tier == "mxfp4"
        else "banana-smasher-fixed-anchor-layer-receipt-v1"
    )
    for layer in layers:
        receipt_path = root / "anchors" / tier / "layers" / f"L{layer:03d}_RECEIPT.json"
        receipt_rows.append(
            {
                "layer": layer,
                **_write_json(
                    receipt_path,
                    {
                        "schema": receipt_schema,
                        "status": "PASS",
                        "tier": tier,
                        "layer": layer,
                        "selected_cells": units_per_layer,
                        "created_unix": created_unix + layer,
                    },
                ),
            }
        )
    manifest_schema = (
        "banana-smasher-mxfp4-reference-manifest-v1"
        if tier == "mxfp4"
        else "banana-smasher-fixed-anchor-manifest-v1"
    )
    manifest_path = root / "anchors" / f"ANCHOR_{tier}_MANIFEST.json"
    manifest_record = _write_json(
        manifest_path,
        {
            "schema": manifest_schema,
            "status": "PASS",
            "tier": tier,
            "layers": layers,
            "selected_cells": len(layers) * units_per_layer,
            "layer_receipts": receipt_rows,
            "created_unix": created_unix,
        },
    )
    anchor_path = root / "anchors" / tier / "ANCHOR.json"
    anchor_record = _write_json(
        anchor_path,
        {
            "schema": "banana-smasher-tier-anchor-v1",
            "status": "PASS",
            "tier": tier,
            "layers": layers,
            "fixed_anchor_manifest": manifest_record,
            "created_unix": created_unix,
        },
    )
    return {"tier": tier, **anchor_record, "fixed_anchor_manifest": manifest_record}


def _write_campaign_root(
    root: Path,
    *,
    layers: list[int] | None = None,
    units_per_layer: int = 4,
    direct_tiers: tuple[str, ...] = ("d4_k2048", "d4_k4096", "mxfp4"),
) -> None:
    layers = list(range(43)) if layers is None else layers
    solve_record = _write_json(
        root / "solve" / "MANIFEST.json",
        {
            "schema": "banana-smasher-vq-solve-manifest-v1",
            "status": "PASS",
            "layers": layers,
            "created_unix": NOW - 200,
        },
    )
    anchors = [
        _direct_anchor(
            root,
            tier,
            layers=layers,
            units_per_layer=units_per_layer,
            created_unix=NOW - 100,
        )
        for tier in direct_tiers
    ]
    anchor_record = _write_json(
        root / "anchors" / "MANIFEST.json",
        {
            "schema": "banana-smasher-anchor-manifest-v1",
            "status": "PASS",
            "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "input_solve_manifest": solve_record,
            "anchors": anchors,
            "created_unix": NOW - 90,
        },
    )
    _write_json(
        root / "WORKFLOW_CHAIN.json",
        {
            "schema": "banana-smasher-workflow-chain-v1",
            "status": "PASS",
            "run_root": str(root.resolve()),
            "solve_manifest": solve_record,
            "anchor_manifest": anchor_record,
            "updated_unix": NOW - 90,
        },
    )


def _write_sealed_shard(
    root: Path,
    tier: str,
    *,
    layers: list[int],
    units_per_layer: int = 4,
) -> dict[str, object]:
    receipt_rows = []
    for layer in layers:
        receipt_rows.append(
            {
                "layer": layer,
                **_write_json(
                    root
                    / "external"
                    / tier
                    / "layers"
                    / f"L{layer:03d}_RECEIPT.json",
                    {
                        "schema": f"banana-smasher-{tier}-anchor-layer-receipt-v1",
                        "status": "PASS",
                        "tier": tier,
                        "layer": layer,
                        "units": units_per_layer,
                        "created_unix": NOW - 40 + layer,
                    },
                ),
            }
        )
    shard_record = _write_json(
        root / "external" / tier / "SHARD_MANIFEST.json",
        {
            "schema": f"banana-smasher-{tier}-anchor-shard-manifest-v1",
            "status": "PASS",
            "tier": tier,
            "layers": layers,
            "cell_count": len(layers) * units_per_layer,
            "layer_receipts": receipt_rows,
            "created_unix": NOW - 35,
        },
    )
    return {
        "host": "spark-fixture",
        "layers": layers,
        "cell_count": len(layers) * units_per_layer,
        "manifest": shard_record,
    }


def _write_shard_index(
    root: Path,
    tier: str,
    shards: list[dict[str, object]],
) -> None:
    _write_json(
        root / "anchors" / tier / "SHARDS.json",
        {
            "schema": f"banana-smasher-{tier}-shard-index-v1",
            "status": "PASS",
            "tier": tier,
            "shards": shards,
            "updated_unix": NOW - 30,
        },
    )


def _write_active_run(
    root: Path,
    tier: str,
    *,
    layer: int,
    completed_units: int,
    active_units: int,
    updated_unix: float = NOW - 5,
    stale_after_seconds: float = 60,
) -> None:
    receipt = _write_json(
        root / "external" / tier / "LATEST_PROGRESS_RECEIPT.json",
        {
            "schema": "banana-smasher-anchor-progress-receipt-v1",
            "status": "PASS",
            "tier": tier,
            "layer": layer,
            "batch": 3,
            "unit": 2,
            "created_unix": updated_unix,
        },
    )
    run_manifest = _write_json(
        root / "external" / tier / "RUN_MANIFEST.json",
        {
            "schema": "banana-smasher-anchor-run-v1",
            "status": "RUNNING",
            "tier": tier,
            "host": "spark-active",
            "updated_unix": updated_unix,
            "stale_after_seconds": stale_after_seconds,
            "progress": [
                {
                    "layer": layer,
                    "completed_units": completed_units,
                    "active_units": active_units,
                }
            ],
            "current": {"layer": layer, "batch": 3, "unit": 2},
            "newest_receipt": receipt,
        },
    )
    _write_json(
        root / "anchors" / tier / "RUNS.json",
        {
            "schema": "banana-smasher-anchor-run-index-v1",
            "status": "PASS",
            "tier": tier,
            "runs": [{"host": "spark-active", "manifest": run_manifest}],
            "updated_unix": updated_unix,
        },
    )


def test_campaign_status_is_exhaustive_for_complete_active_and_missing_coverage(
    tmp_path: Path,
) -> None:
    from banana_smasher.campaign_status import inspect_anchor_campaign, render_anchor_campaign

    root = tmp_path / "run"
    _write_campaign_root(root)
    _write_shard_index(root, "qtip3", [_write_sealed_shard(root, "qtip3", layers=[2])])
    _write_active_run(
        root,
        "qtip3",
        layer=1,
        completed_units=2,
        active_units=1,
    )

    status = inspect_anchor_campaign(root, now=NOW)

    assert status["schema"] == "banana-smasher-anchor-campaign-status-v1"
    assert status["status"] == "RUNNING"
    assert [tier["tier"] for tier in status["tiers"]] == TIERS
    qtip3 = status["tiers"][0]
    assert qtip3["coverage"] == {
        "expected": [f"L{layer:03d}" for layer in range(43)],
        "completed": ["L002"],
        "active": ["L001"],
        "missing": ["L000", *[f"L{layer:03d}" for layer in range(3, 43)]],
        "counts": {"expected": 43, "completed": 1, "active": 1, "missing": 41},
    }
    assert qtip3["units"] == {
        "expected": 172,
        "completed": 6,
        "active": 1,
        "missing": 165,
        "percent_completed": 3.488372,
    }
    assert qtip3["current"] == [
        {
            "host": "spark-active",
            "layer": "L001",
            "batch": 3,
            "unit": 2,
            "manifest": str((root / "external/qtip3/RUN_MANIFEST.json").resolve()),
        }
    ]
    assert qtip3["newest_receipt"]["path"].endswith("LATEST_PROGRESS_RECEIPT.json")
    assert qtip3["newest_receipt"]["age_seconds"] == 5.0
    assert qtip3["mergeable"] is True
    assert qtip3["ready"] is False

    qtip2 = status["tiers"][1]
    assert qtip2["coverage"]["missing"] == [f"L{layer:03d}" for layer in range(43)]
    assert qtip2["units"]["missing"] == 172
    assert qtip2["mergeable"] is False
    assert "anchors/qtip2/SHARDS.json" in qtip2["blockers"][0]
    assert "smash merge" in qtip2["blockers"][0]

    d4 = status["tiers"][2]
    assert d4["units"]["completed"] == 172
    assert d4["coverage"]["completed"] == [f"L{layer:03d}" for layer in range(43)]
    assert d4["ready"] is True
    assert len(qtip3["layers"]) == 43
    assert qtip3["layers"][1] == {
        "layer": "L001",
        "expected_units": 4,
        "completed_units": 2,
        "active_units": 1,
        "missing_units": 1,
        "state": "active",
    }

    text = render_anchor_campaign(status)
    for tier in TIERS:
        assert tier in text
    for layer in (f"L{value:03d}" for value in range(43)):
        assert text.count(layer) >= len(TIERS)
    assert "3.49%" in text
    assert "L001/B003/U002" in text
    assert "READY" in text and "MERGEABLE" in text


def test_complete_campaign_is_ready(tmp_path: Path) -> None:
    from banana_smasher.campaign_status import inspect_anchor_campaign

    root = tmp_path / "run"
    _write_campaign_root(root, direct_tiers=tuple(TIERS))

    status = inspect_anchor_campaign(root, now=NOW)

    assert status["status"] == "READY"
    assert status["campaign"]["ready"] is True
    assert status["campaign"]["units"] == {
        "expected": 860,
        "completed": 860,
        "active": 0,
        "missing": 0,
        "percent_completed": 100.0,
    }
    assert all(tier["ready"] for tier in status["tiers"])


def test_truncated_d4_baseline_is_not_a_complete_flash_full_campaign(
    tmp_path: Path,
) -> None:
    from banana_smasher.campaign_status import StatusContractError, inspect_anchor_campaign

    root = tmp_path / "run"
    _write_campaign_root(root, layers=[0, 1, 2])

    with pytest.raises(StatusContractError, match="exactly L000-L042") as raised:
        inspect_anchor_campaign(root, now=NOW)

    assert "smash anchor --run-root" in str(raised.value)


def test_missing_referenced_manifest_fails_with_artifact_and_public_producer(
    tmp_path: Path,
) -> None:
    from banana_smasher.campaign_status import StatusContractError, inspect_anchor_campaign

    root = tmp_path / "run"
    _write_campaign_root(root)
    manifest = root / "anchors" / "ANCHOR_d4_k2048_MANIFEST.json"
    manifest.unlink()

    with pytest.raises(StatusContractError) as raised:
        inspect_anchor_campaign(root, now=NOW)

    message = str(raised.value)
    assert str(manifest.resolve()) in message
    assert "smash anchor --run-root" in message


def test_stale_referenced_manifest_sha_fails_loudly(tmp_path: Path) -> None:
    from banana_smasher.campaign_status import StatusContractError, inspect_anchor_campaign

    root = tmp_path / "run"
    _write_campaign_root(root)
    receipt = root / "anchors" / "d4_k4096" / "layers" / "L001_RECEIPT.json"
    receipt.write_text(receipt.read_text().replace('"PASS"', '"POSS"'))

    with pytest.raises(StatusContractError, match="SHA256 mismatch") as raised:
        inspect_anchor_campaign(root, now=NOW)

    assert str(receipt.resolve()) in str(raised.value)
    assert "smash anchor --run-root" in str(raised.value)


def test_artifact_path_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    from banana_smasher.campaign_status import StatusContractError, _artifact_path

    real_parent = tmp_path / "real"
    record = _write_json(real_parent / "receipt.json", {"status": "PASS"})
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    record["path"] = str(alias / "receipt.json")

    with pytest.raises(StatusContractError, match="symlink component") as raised:
        _artifact_path(record, base=tmp_path, label="test receipt", producer="smash merge")

    assert str(alias) in str(raised.value)
    assert "smash merge" in str(raised.value)


def test_malformed_manifest_fails_loudly_without_directory_fallback(tmp_path: Path) -> None:
    from banana_smasher.campaign_status import StatusContractError, inspect_anchor_campaign

    root = tmp_path / "run"
    _write_campaign_root(root)
    index = root / "anchors" / "qtip3" / "SHARDS.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("{not-json")

    with pytest.raises(StatusContractError) as raised:
        inspect_anchor_campaign(root, now=NOW)

    assert str(index.resolve()) in str(raised.value)
    assert "smash merge" in str(raised.value)


def test_stale_active_manifest_fails_loudly(tmp_path: Path) -> None:
    from banana_smasher.campaign_status import StatusContractError, inspect_anchor_campaign

    root = tmp_path / "run"
    _write_campaign_root(root)
    _write_active_run(
        root,
        "qtip3",
        layer=1,
        completed_units=0,
        active_units=1,
        updated_unix=NOW - 100,
        stale_after_seconds=10,
    )

    with pytest.raises(StatusContractError, match="stale active run manifest") as raised:
        inspect_anchor_campaign(root, now=NOW)

    assert "smash solve" in str(raised.value)


def test_status_cli_defaults_to_human_and_json_is_machine_readable(
    tmp_path: Path, capsys
) -> None:
    root = tmp_path / "run"
    _write_campaign_root(root, direct_tiers=tuple(TIERS))

    assert main(["status", "--run-root", str(root)]) == 0
    human = capsys.readouterr().out
    assert human.startswith("RUN ROOT:")
    assert "d4_k2048" in human

    assert main(["status", "--run-root", str(root), "--json"]) == 0
    machine = json.loads(capsys.readouterr().out)
    assert machine["schema"] == "banana-smasher-anchor-campaign-status-v1"
    assert [row["tier"] for row in machine["tiers"]] == TIERS
