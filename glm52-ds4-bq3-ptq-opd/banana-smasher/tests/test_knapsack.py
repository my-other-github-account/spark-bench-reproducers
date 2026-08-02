from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.cli import main
from banana_smasher.knapsack import KnapsackValidationError, run_knapsack


BASIS = "a" * 64


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _fixture_run(
    root: Path,
    *,
    tiers: tuple[str, ...] = ("qtip_0.75", "future_1.25"),
) -> dict[str, object]:
    cells = ("L000/fused13/E000", "L000/fused13/E001", "L001/down/E000")
    costs = {
        tiers[0]: (4, 4, 4),
        tiers[1]: (8, 8, 8),
    }
    damages = {
        tiers[0]: (10.0, 9.0, 8.0),
        tiers[1]: (0.0, 1.0, 2.0),
    }
    anchors: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for tier in tiers:
        relative = Path("anchors") / tier / "MANIFEST.json"
        digest = _write_json(
            root / relative,
            {
                "schema": "banana-smasher-anchor-v1",
                "status": "PASS",
                "tier": tier,
                "basis_sha256": BASIS,
                "cells": [
                    {"cell_id": cell, "bytes": size}
                    for cell, size in zip(cells, costs[tier], strict=True)
                ],
            },
        )
        anchors[tier] = {
            "path": relative.as_posix(),
            "sha256": digest,
            "producer_command": f"smash anchor --run-root {root} --tier {tier}",
        }
        rows.extend(
            {"cell_id": cell, "tier": tier, "damage": damage}
            for cell, damage in zip(cells, damages[tier], strict=True)
        )
    damage_path = Path("damage") / "ROWS.json"
    damage_sha = _write_json(
        root / damage_path,
        {
            "schema": "banana-smasher-damage-rows-v1",
            "basis_sha256": BASIS,
            "rows": rows,
        },
    )
    manifest = {
        "schema": "banana-smasher-run-v1",
        "intended_basis_sha256": BASIS,
        "intended_tiers": list(tiers),
        "anchor_manifests": anchors,
        "damage_rows": {
            "path": damage_path.as_posix(),
            "sha256": damage_sha,
            "producer_command": f"smash damage --run-root {root}",
        },
    }
    _write_json(root / "MANIFEST.json", manifest)
    return manifest


def test_cli_knapsack_discovers_open_tier_set_and_solves_exact_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "run"
    tiers = ("qtip_0.75", "future_1.25")
    _fixture_run(root, tiers=tiers)

    assert main(["knapsack", "--run-root", str(root), "--envelope-bytes", "16"]) == 0
    result = json.loads(capsys.readouterr().out)
    assignment = json.loads((root / "knapsack" / "ASSIGNMENT.json").read_text())
    receipt = json.loads((root / "knapsack" / "RECEIPT.json").read_text())

    assert result["status"] == "PASS"
    assert result["tiers"] == list(tiers)
    assert assignment["byte_accounting"] == {
        "assigned_bytes": 16,
        "envelope_bytes": 16,
        "slack_bytes": 0,
    }
    assert assignment["assignments"] == [
        {"bytes": 8, "cell_id": "L000/fused13/E000", "damage": 0.0, "tier": tiers[1]},
        {"bytes": 4, "cell_id": "L000/fused13/E001", "damage": 9.0, "tier": tiers[0]},
        {"bytes": 4, "cell_id": "L001/down/E000", "damage": 8.0, "tier": tiers[0]},
    ]
    assert receipt["status"] == "PASS"
    assert receipt["assignment"]["sha256"] == hashlib.sha256(
        (root / "knapsack" / "ASSIGNMENT.json").read_bytes()
    ).hexdigest()
    assert {source["tier"] for source in receipt["anchor_manifests"]} == set(tiers)


def test_knapsack_preflights_every_intended_anchor_and_names_missing_producer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    manifest = _fixture_run(root)
    missing_tier = "later_2.00"
    manifest["intended_tiers"].append(missing_tier)  # type: ignore[union-attr]
    manifest["anchor_manifests"][missing_tier] = {  # type: ignore[index]
        "path": f"anchors/{missing_tier}/MANIFEST.json",
        "sha256": "0" * 64,
        "producer_command": f"smash anchor --run-root {root} --tier {missing_tier}",
    }
    _write_json(root / "MANIFEST.json", manifest)

    with pytest.raises(KnapsackValidationError, match="missing intended anchor manifest") as caught:
        run_knapsack(run_root=root, envelope_bytes=16)

    message = str(caught.value)
    assert missing_tier in message
    assert f"smash anchor --run-root {root} --tier {missing_tier}" in message
    assert not (root / "knapsack").exists()


def test_knapsack_rejects_anchor_sha_mismatch_before_solve(tmp_path: Path) -> None:
    root = tmp_path / "run"
    manifest = _fixture_run(root)
    tier = manifest["intended_tiers"][0]  # type: ignore[index]
    anchor = manifest["anchor_manifests"][tier]  # type: ignore[index]
    anchor["sha256"] = "f" * 64
    _write_json(root / "MANIFEST.json", manifest)

    with pytest.raises(
        KnapsackValidationError, match=r"anchor manifest .*SHA-256 mismatch"
    ) as caught:
        run_knapsack(run_root=root, envelope_bytes=16)

    assert tier in str(caught.value)
    assert not (root / "knapsack").exists()


def test_knapsack_rejects_basis_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "run"
    manifest = _fixture_run(root)
    tier = manifest["intended_tiers"][1]  # type: ignore[index]
    anchor_path = root / manifest["anchor_manifests"][tier]["path"]  # type: ignore[index]
    anchor = json.loads(anchor_path.read_text())
    anchor["basis_sha256"] = "b" * 64
    digest = _write_json(anchor_path, anchor)
    manifest["anchor_manifests"][tier]["sha256"] = digest  # type: ignore[index]
    _write_json(root / "MANIFEST.json", manifest)

    with pytest.raises(KnapsackValidationError, match="basis mismatch"):
        run_knapsack(run_root=root, envelope_bytes=16)

    assert not (root / "knapsack").exists()
