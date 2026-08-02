from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Sequence
import uuid

_TIER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def validate_open_tiers(value: object) -> list[str]:
    """Validate manifest tier identifiers without a package-global tier menu."""
    if not isinstance(value, list) or not value or any(
        not isinstance(tier, str) or _TIER_NAME.fullmatch(tier) is None for tier in value
    ):
        raise ValueError(f"invalid open tier population: {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"invalid open tier population: {value!r}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def artifact_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    """Describe the exact immutable byte snapshot that was parsed."""
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _require_contained(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} outside {resolved_root}: {resolved}")
    return resolved


def _validate_artifact_record(
    record: dict[str, Any],
    *,
    root: Path,
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"invalid {label} artifact record")
    path = _require_contained(Path(str(record.get("path", ""))), root, label=label)
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    expected_size = record.get("bytes")
    expected_sha = record.get("sha256")
    if expected_size != path.stat().st_size or expected_sha != sha256_file(path):
        raise ValueError(f"{label} hash/size drift: {path}")
    return path


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items or len(set(items)) != len(items):
        raise ValueError(f"invalid comma-separated list: {value!r}")
    return items


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in parse_csv(value):
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first, last = int(first_text), int(last_text)
            if last < first:
                raise ValueError(f"descending layer range: {part}")
            layers.extend(range(first, last + 1))
        else:
            layers.append(int(part))
    if len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise ValueError(f"invalid layer selection: {value!r}")
    return layers


def solver_profile_main(argv: list[str], *, emit_summary: bool = False) -> dict[str, Any]:
    from .solver_profile import main

    return main(argv, emit_summary=emit_summary)


def _validated_summary(
    path: Path,
    *,
    layer: int,
    tier: str,
    windows: int,
    audit_codeword_assignments: bool,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    exact = {
        "schema": "banana-smasher-solver-profile-v1",
        "status": "PASS",
        "layer": layer,
        "tiers": [tier],
        "windows": windows,
        "audit_codeword_assignments": audit_codeword_assignments,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        return None
    scientific = Path(str(value.get("scientific_rows", "")))
    if not scientific.is_file():
        return None
    expected_scientific_sha = value.get("scientific_rows_sha256")
    if (
        not isinstance(expected_scientific_sha, str)
        or len(expected_scientific_sha) != 64
        or sha256_file(scientific) != expected_scientific_sha
    ):
        return None
    objective = value.get("objective")
    if not isinstance(objective, dict):
        return None
    if (
        audit_codeword_assignments
        and objective.get("assignment_scope")
        != "full-codeword-assignment-by-cell"
    ):
        return None
    assignment_sha = objective.get("assignment_sha256")
    if not isinstance(assignment_sha, str) or len(assignment_sha) != 64:
        return None
    return value


def _adopt_pricing_summary(
    *,
    prices_root: Path,
    output: Path,
    layer: int,
    tier: str,
    windows: int,
) -> dict[str, Any]:
    """Validate and adopt one fixed tier from a sealed SOLVER_PRICING_V2 table."""
    layer_root = prices_root / f"L{layer:03d}"
    complete_path = layer_root / "COMPLETE.json"
    rows_path = layer_root / "prices.jsonl"
    complete_raw = complete_path.read_bytes()
    rows_raw = rows_path.read_bytes()
    complete = json.loads(complete_raw)
    expected_complete = {
        "schema": "solver-pricing-v2-layer-complete-v1",
        "layer": layer,
        "windows": windows,
    }
    if any(complete.get(key) != expected for key, expected in expected_complete.items()):
        raise ValueError(f"pricing completion binding mismatch: {complete_path}")
    if complete.get("status") not in (None, "PASS"):
        raise ValueError(f"pricing completion is not PASS: {complete_path}")

    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(rows_raw.splitlines(), start=1):
        if not raw.strip():
            raise ValueError(f"blank pricing row at {rows_path}:{line_number}")
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"non-object pricing row at {rows_path}:{line_number}")
        rows.append(row)
    declared_rows = complete.get("rows")
    expected_rows = complete.get("expected_rows")
    if not isinstance(declared_rows, int) or declared_rows != len(rows):
        raise ValueError(f"pricing row count mismatch: {complete_path}")
    if not isinstance(expected_rows, int) or expected_rows != len(rows):
        raise ValueError(f"incomplete pricing table: {complete_path}")

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    cells: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if row.get("schema") != "solver-pricing-v2-cell-tier-v1":
            raise ValueError(f"pricing row schema mismatch: {rows_path}:{row_number}")
        if row.get("layer") != layer or row.get("n_windows") != windows:
            raise ValueError(f"pricing row layer/window mismatch: {rows_path}:{row_number}")
        if row.get("tier") != tier:
            continue
        expert = row.get("expert")
        projection = row.get("projection")
        variant = row.get("variant")
        if not isinstance(expert, int) or not 0 <= expert < 256:
            raise ValueError(f"pricing expert out of range: {rows_path}:{row_number}")
        if projection not in ("13", "2") or not isinstance(variant, str) or not variant:
            raise ValueError(f"pricing projection/variant mismatch: {rows_path}:{row_number}")
        cell = f"L{layer:03d}.E{expert:03d}.P{projection}"
        if row.get("cell") != cell:
            raise ValueError(f"pricing cell identity mismatch: {rows_path}:{row_number}")
        relative_error = row.get("relative_weighted_error")
        if not isinstance(relative_error, (int, float)) or not math.isfinite(relative_error):
            raise ValueError(f"non-finite pricing objective: {rows_path}:{row_number}")
        weighted_sse = row.get("weighted_sse", 0.0)
        if not isinstance(weighted_sse, (int, float)) or not math.isfinite(weighted_sse):
            raise ValueError(f"non-finite pricing SSE: {rows_path}:{row_number}")
        key = (cell, variant)
        if key in seen:
            raise ValueError(f"duplicate pricing cell/variant: {key}")
        seen.add(key)
        cells.add(cell)
        selected.append(row)

    expected_cells = {
        f"L{layer:03d}.E{expert:03d}.P{projection}"
        for expert in range(256)
        for projection in ("13", "2")
    }
    if cells != expected_cells:
        missing = sorted(expected_cells - cells)
        extra = sorted(cells - expected_cells)
        raise ValueError(
            f"pricing tier population mismatch for {tier}: missing={missing[:3]} extra={extra[:3]}"
        )

    assignments: dict[str, dict[str, Any]] = {}
    for row in selected:
        cell = str(row["cell"])
        if cell not in assignments or float(row["relative_weighted_error"]) < float(
            assignments[cell]["relative_weighted_error"]
        ):
            assignments[cell] = row
    assignment_payload = {
        cell: row for cell, row in sorted(assignments.items())
    }
    assignment_sha = hashlib.sha256(
        json.dumps(assignment_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    objective = {
        "selected_cells": len(assignments),
        "assignment_sha256": assignment_sha,
        "assignment_scope": "sealed-pricing-row-by-cell",
        "sum_relative_weighted_error": math.fsum(
            float(row["relative_weighted_error"]) for row in assignments.values()
        ),
        "sum_weighted_sse": math.fsum(
            float(row.get("weighted_sse", 0.0)) for row in assignments.values()
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    scientific_path = output / "SCIENTIFIC_ROWS.json"
    atomic_json(scientific_path, selected)
    summary = {
        "schema": "banana-smasher-solver-profile-v1",
        "status": "PASS",
        "implementation": "sealed-pricing-v2-adoption",
        "source_kind": "sealed-pricing-v2",
        "layer": layer,
        "tiers": [tier],
        "windows": windows,
        "audit_codeword_assignments": False,
        "experts": 256,
        "cells": 512,
        "solver_rows": len(selected),
        "expected_solver_rows": len(selected),
        "objective": objective,
        "scientific_rows": str(scientific_path.resolve()),
        "scientific_rows_sha256": sha256_file(scientific_path),
        "source_pricing_manifest": artifact_bytes(complete_path, complete_raw),
        "source_pricing_rows": artifact_bytes(rows_path, rows_raw),
        "created_unix": time.time(),
    }
    atomic_json(output / "PROFILE_SUMMARY.json", summary)
    return summary


def _write_chain(run_root: Path, **updates: dict[str, Any]) -> Path:
    path = run_root / "WORKFLOW_CHAIN.json"
    if path.is_file():
        value = json.loads(path.read_text())
        if value.get("schema") != "banana-smasher-workflow-chain-v1":
            raise ValueError(f"workflow chain schema drift: {path}")
        if value.get("run_root") != str(run_root.resolve()):
            raise ValueError(f"workflow chain run-root drift: {path}")
        for key in ("hessian_manifest", "solve_manifest", "anchor_manifest"):
            if key in value and key not in updates:
                _validate_artifact_record(
                    value[key], root=run_root, label=key.replace("_", " ")
                )
    else:
        value: dict[str, Any] = {
            "schema": "banana-smasher-workflow-chain-v1",
            "status": "IN_PROGRESS",
            "run_root": str(run_root.resolve()),
        }
    previous_hessian = value.get("hessian_manifest")
    previous_solve = value.get("solve_manifest")
    next_hessian = updates.get("hessian_manifest")
    next_solve = updates.get("solve_manifest")
    if next_hessian is not None and previous_hessian != next_hessian:
        value.pop("solve_manifest", None)
        value.pop("anchor_manifest", None)
    if next_solve is not None and previous_solve != next_solve:
        value.pop("anchor_manifest", None)
    value.update(updates)
    for key in ("hessian_manifest", "solve_manifest", "anchor_manifest"):
        if key in value:
            _validate_artifact_record(
                value[key], root=run_root, label=key.replace("_", " ")
            )
    value["updated_unix"] = time.time()
    value["status"] = (
        "PASS"
        if all(key in value for key in ("hessian_manifest", "solve_manifest", "anchor_manifest"))
        else "IN_PROGRESS"
    )
    atomic_json(path, value)
    return path


def _validated_hessian_layer_manifest(
    path: Path, *, run_root: Path, layer: int, windows: int
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
        if any(
            value.get(key) != expected
            for key, expected in {
                "schema": "banana-smasher-hessian-layer-manifest-v1",
                "status": "PASS",
                "layer": layer,
                "windows": windows,
            }.items()
        ):
            return None
        members = value.get("members")
        if not isinstance(members, list) or len(members) != windows:
            return None
        for member in members:
            for key in ("capture", "capture_done", "capture_stage_receipt", "done_stage_receipt"):
                _validate_artifact_record(
                    member[key], root=run_root, label=f"Hessian L{layer:03d} {key}"
                )
        return value
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def run_hessian(*, run_root: Path, layers: list[int], windows: int) -> dict[str, Any]:
    """Prefetch complete sealed capture banks and publish a resumable manifest chain."""
    from . import solver_core as core

    run_root = run_root.resolve()
    if not layers:
        raise ValueError("at least one layer is required")
    if windows not in (32, 64):
        raise ValueError("Hessian windows must be exactly 32 or 64")
    hessian_root = run_root / "hessians"
    hessian_root.mkdir(parents=True, exist_ok=True)
    layer_records: list[dict[str, Any]] = []
    for layer in layers:
        layer_manifest_path = hessian_root / f"L{layer:03d}" / "MANIFEST.json"
        existing = _validated_hessian_layer_manifest(
            layer_manifest_path,
            run_root=run_root,
            layer=layer,
            windows=windows,
        )
        if existing is None:
            capture_root = core.capture_dir(
                run_root,
                layer,
                windows,
                staging_root=hessian_root,
            )
            public_root = (run_root / "captures").resolve()
            public_capture = capture_root.resolve().is_relative_to(public_root)
            source_root: str | None = None
            if public_capture:
                source_label = f"banana-smasher-public-capture:{capture_root.resolve()}"
            else:
                source_root = core._capture_source(run_root, layer)
                source_label = source_root
            members = []
            for window in range(windows):
                filename = f"xmoe_L{layer:03d}_win{window:04d}.pt"
                capture = capture_root / filename
                done = capture_root / f"{filename}.DONE.json"
                if public_capture:
                    capture_stage = json.loads(
                        core.staged_input_receipt_path(capture).read_text()
                    )
                    done_stage = json.loads(
                        core.staged_input_receipt_path(done).read_text()
                    )
                    capture_source = str(capture_stage.get("source", ""))
                    done_source = str(done_stage.get("source", ""))
                    if not capture_source or not done_source:
                        raise RuntimeError(
                            f"public capture source receipt is incomplete: {capture}"
                        )
                else:
                    assert source_root is not None
                    capture_source = f"{source_root.rstrip('/')}/{filename}"
                    done_source = f"{source_root.rstrip('/')}/{filename}.DONE.json"
                core.validate_staged_input(capture, capture_source, min_size=1)
                core.validate_staged_input(done, done_source, min_size=1)
                members.append(
                    {
                        "window": window,
                        "capture": artifact(capture),
                        "capture_done": artifact(done),
                        "capture_stage_receipt": artifact(core.staged_input_receipt_path(capture)),
                        "done_stage_receipt": artifact(core.staged_input_receipt_path(done)),
                    }
                )
            existing = {
                "schema": "banana-smasher-hessian-layer-manifest-v1",
                "status": "PASS",
                "layer": layer,
                "windows": windows,
                "capture_root": str(capture_root.resolve()),
                "source": source_label,
                "members": members,
                "created_unix": time.time(),
            }
            atomic_json(layer_manifest_path, existing)
        layer_records.append({"layer": layer, **artifact(layer_manifest_path)})

    manifest_path = hessian_root / "MANIFEST.json"
    manifest = {
        "schema": "banana-smasher-hessian-manifest-v1",
        "status": "PASS",
        "layers": layers,
        "windows": windows,
        "layer_manifests": layer_records,
        "created_unix": time.time(),
    }
    atomic_json(manifest_path, manifest)
    chain = _write_chain(run_root, hessian_manifest=artifact(manifest_path))
    return {
        "schema": "banana-smasher-hessian-receipt-v1",
        "status": "PASS",
        "command": "hessian",
        "run_root": str(run_root),
        "layers": layers,
        "windows": windows,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "workflow_chain": str(chain),
    }


def run_fresh_solve(
    *,
    run_root: Path,
    source_root: Path,
    model_root: Path | None = None,
    layers: list[int],
    tiers: list[str],
    windows: int,
    staging_root: Path | None,
    reference_search: bool,
    hessian_manifest: Path | None,
    prices_root: Path | None = None,
    audit_codeword_assignments: bool = False,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if windows not in (32, 64):
        raise ValueError("fresh-model solve windows must be exactly 32 or 64")
    tiers = validate_open_tiers(tiers)
    if not layers:
        raise ValueError("at least one layer is required")
    if prices_root is not None:
        prices_root = prices_root.resolve()
        if not prices_root.is_dir():
            raise FileNotFoundError(prices_root)
        if audit_codeword_assignments:
            raise ValueError(
                "--audit-codeword-assignments requires fresh exact search, not --prices-root"
            )

    hessian_input: dict[str, Any] | None = None
    hessian_capture_roots: dict[int, Path] = {}
    if hessian_manifest is not None:
        hessian_manifest = hessian_manifest.resolve()
        hessian = json.loads(hessian_manifest.read_text())
        if hessian.get("schema") != "banana-smasher-hessian-manifest-v1" or hessian.get("status") != "PASS":
            raise ValueError(f"invalid Hessian manifest: {hessian_manifest}")
        layer_manifests = hessian.get("layer_manifests")
        if not isinstance(layer_manifests, list):
            raise ValueError(f"invalid Hessian layer population: {hessian_manifest}")
        for row in layer_manifests:
            if not isinstance(row, dict) or not isinstance(row.get("layer"), int):
                raise ValueError(f"invalid Hessian layer row: {hessian_manifest}")
            layer_manifest_path = _validate_artifact_record(
                row,
                root=run_root,
                label=f"Hessian L{row['layer']:03d} manifest",
            )
            layer_manifest = json.loads(layer_manifest_path.read_text())
            capture_root = _require_contained(
                Path(str(layer_manifest.get("capture_root", ""))),
                run_root,
                label=f"Hessian L{row['layer']:03d} capture root",
            )
            if not capture_root.is_dir():
                raise ValueError(f"missing Hessian capture root: {capture_root}")
            hessian_capture_roots[int(row["layer"])] = capture_root
        missing_hessian_layers = sorted(set(layers) - set(hessian_capture_roots))
        if missing_hessian_layers:
            raise ValueError(
                f"Hessian manifest missing solve layers: {missing_hessian_layers}"
            )
        hessian_input = artifact(hessian_manifest)

    tier_manifest_rows: list[dict[str, Any]] = []
    implementation = "serial" if reference_search else "exact-gemm"
    for tier in tiers:
        tier_root = run_root / "solve" / tier
        tier_root.mkdir(parents=True, exist_ok=True)
        layer_rows = []
        for layer in layers:
            summary_path = tier_root / "profile" / f"L{layer:03d}" / "PROFILE_SUMMARY.json"
            summary = _validated_summary(
                summary_path,
                layer=layer,
                tier=tier,
                windows=windows,
                audit_codeword_assignments=audit_codeword_assignments,
            )
            resumed = summary is not None
            if summary is None:
                pricing_complete = (
                    prices_root / f"L{layer:03d}" / "COMPLETE.json"
                    if prices_root is not None
                    else None
                )
                pricing_rows = (
                    prices_root / f"L{layer:03d}" / "prices.jsonl"
                    if prices_root is not None
                    else None
                )
                has_pricing_complete = (
                    pricing_complete is not None and pricing_complete.is_file()
                )
                has_pricing_rows = pricing_rows is not None and pricing_rows.is_file()
                if has_pricing_complete != has_pricing_rows:
                    raise ValueError(
                        "partial sealed pricing layer: "
                        f"complete={has_pricing_complete} rows={has_pricing_rows} "
                        f"layer=L{layer:03d}"
                    )
                if has_pricing_complete and has_pricing_rows:
                    assert prices_root is not None
                    summary = _adopt_pricing_summary(
                        prices_root=prices_root,
                        output=summary_path.parent,
                        layer=layer,
                        tier=tier,
                        windows=windows,
                    )
                else:
                    argv = [
                        "--root",
                        str(tier_root),
                        "--source-root",
                        str(source_root),
                        "--layer",
                        str(layer),
                        "--windows",
                        str(windows),
                        "--implementation",
                        implementation,
                        "--tiers",
                        tier,
                    ]
                    if model_root is not None:
                        argv.extend(["--model-root", str(model_root.resolve())])
                    if hessian_manifest is not None:
                        argv.extend(
                            ["--capture-root", str(hessian_capture_roots[layer])]
                        )
                    if staging_root is not None:
                        argv.extend(["--staging-root", str(staging_root.resolve())])
                    if audit_codeword_assignments:
                        argv.append("--audit-codeword-assignments")
                    try:
                        summary = solver_profile_main(argv, emit_summary=False)
                    except (FileNotFoundError, RuntimeError) as exc:
                        if "capture" not in str(exc).lower():
                            raise
                        raise RuntimeError(f"{exc}; run smash hessian first") from exc
            if summary.get("status") != "PASS" or summary.get("tiers") != [tier]:
                raise RuntimeError(f"solver summary failed tier binding: {summary_path}")
            if (
                audit_codeword_assignments
                and summary.get("objective", {}).get("assignment_scope")
                != "full-codeword-assignment-by-cell"
            ):
                raise RuntimeError(
                    f"solver summary lacks full codeword assignment audit: {summary_path}"
                )
            scientific_path = Path(str(summary["scientific_rows"]))
            summary["audit_codeword_assignments"] = audit_codeword_assignments
            summary["scientific_rows_sha256"] = sha256_file(scientific_path)
            atomic_json(summary_path, summary)
            layer_rows.append(
                {
                    "layer": layer,
                    "resumed": resumed,
                    "objective": summary["objective"],
                    "summary": artifact(summary_path),
                    "scientific_rows": artifact(scientific_path),
                }
            )
        tier_manifest_path = tier_root / "MANIFEST.json"
        tier_manifest = {
            "schema": "banana-smasher-vq-tier-solve-manifest-v1",
            "status": "PASS",
            "tier": tier,
            "layers": layer_rows,
            "windows": windows,
            "implementation": implementation,
            "fresh_model": True,
            "warm_start": False,
            "audit_codeword_assignments": audit_codeword_assignments,
            "source_root": str(source_root),
            "prices_root": str(prices_root) if prices_root is not None else None,
            "hessian_manifest": hessian_input,
            "measurement_label": "MEASURED_CAPTURE_WEIGHTED_ERROR_NOT_MODEL_KLD",
            "created_unix": time.time(),
        }
        atomic_json(tier_manifest_path, tier_manifest)
        tier_manifest_rows.append({"tier": tier, **artifact(tier_manifest_path)})

    aggregate_path = run_root / "solve" / "MANIFEST.json"
    aggregate = {
        "schema": "banana-smasher-vq-solve-manifest-v1",
        "status": "PASS",
        "layers": layers,
        "tiers": tiers,
        "windows": windows,
        "implementation": implementation,
        "fresh_model": True,
        "warm_start": False,
        "audit_codeword_assignments": audit_codeword_assignments,
        "hessian_manifest": hessian_input,
        "tier_manifests": tier_manifest_rows,
        "created_unix": time.time(),
    }
    atomic_json(aggregate_path, aggregate)
    chain = _write_chain(run_root, solve_manifest=artifact(aggregate_path))
    return {
        "schema": "banana-smasher-vq-solve-receipt-v1",
        "status": "PASS",
        "command": "solve",
        "run_root": str(run_root),
        "layers": layers,
        "tiers": tiers,
        "manifest": str(aggregate_path),
        "manifest_sha256": sha256_file(aggregate_path),
        "workflow_chain": str(chain),
    }


def run_anchor(*, run_root: Path) -> dict[str, Any]:
    started_unix = time.time()
    run_root = run_root.resolve()
    solve_path = run_root / "solve" / "MANIFEST.json"
    solve = json.loads(solve_path.read_text())
    if (
        solve.get("schema") != "banana-smasher-vq-solve-manifest-v1"
        or solve.get("status") != "PASS"
    ):
        raise ValueError(f"invalid solve manifest: {solve_path}")
    windows = solve.get("windows")
    if windows not in (32, 64):
        raise ValueError(f"invalid solve windows: {windows}")
    tiers = validate_open_tiers(solve.get("tiers"))
    tier_rows = solve.get("tier_manifests")
    if (
        not isinstance(tier_rows, list)
        or len(tiers) != len(tier_rows)
    ):
        raise ValueError("invalid solve tier population")

    chain_path = run_root / "WORKFLOW_CHAIN.json"
    if chain_path.is_file():
        chain_value = json.loads(chain_path.read_text())
        if "solve_manifest" in chain_value:
            chained_solve = _validate_artifact_record(
                chain_value["solve_manifest"], root=run_root, label="solve manifest"
            )
            if chained_solve != solve_path:
                raise ValueError("workflow chain points at a different solve manifest")

    anchors = []
    solve_root = (run_root / "solve").resolve()
    for expected_tier, tier_row in zip(tiers, tier_rows, strict=True):
        if not isinstance(tier_row, dict) or tier_row.get("tier") != expected_tier:
            raise ValueError("tier row/order mismatch")
        tier_candidate = Path(str(tier_row.get("path", ""))).resolve()
        if not tier_candidate.is_relative_to(solve_root):
            raise ValueError(f"tier manifest outside solve root: {tier_candidate}")
        tier_path = _validate_artifact_record(
            tier_row, root=solve_root, label="tier manifest"
        )
        expected_path = (solve_root / expected_tier / "MANIFEST.json").resolve()
        if tier_path != expected_path:
            raise ValueError(f"tier manifest path binding mismatch: {tier_path}")
        tier_manifest = json.loads(tier_path.read_text())
        tier = str(tier_manifest.get("tier", ""))
        if (
            tier != expected_tier
            or tier_manifest.get("schema")
            != "banana-smasher-vq-tier-solve-manifest-v1"
            or tier_manifest.get("status") != "PASS"
            or tier_manifest.get("windows") != windows
        ):
            raise ValueError(f"invalid tier solve manifest: {tier_path}")
        layer_rows = tier_manifest.get("layers")
        if not isinstance(layer_rows, list):
            raise ValueError(f"invalid tier layer rows: {tier_path}")
        for row in layer_rows:
            if not isinstance(row, dict):
                raise ValueError(f"invalid tier layer row: {tier_path}")
            _validate_artifact_record(
                row["summary"], root=tier_path.parent, label="layer summary"
            )
            _validate_artifact_record(
                row["scientific_rows"],
                root=tier_path.parent,
                label="scientific rows",
            )
        layer_objectives = [
            {
                "layer": int(row["layer"]),
                "selected_cells": int(row["objective"].get("selected_cells", 0)),
                "sum_relative_weighted_error": float(
                    row["objective"]["sum_relative_weighted_error"]
                ),
                "sum_weighted_sse": float(row["objective"]["sum_weighted_sse"]),
                "assignment_sha256": row["objective"]["assignment_sha256"],
            }
            for row in layer_rows
        ]
        totals = {
            "selected_cells": sum(row["selected_cells"] for row in layer_objectives),
            "sum_relative_weighted_error": math.fsum(
                row["sum_relative_weighted_error"] for row in layer_objectives
            ),
            "sum_weighted_sse": math.fsum(
                row["sum_weighted_sse"] for row in layer_objectives
            ),
        }
        layer_receipts = []
        for source_row, objective_row in zip(layer_rows, layer_objectives, strict=True):
            summary_path = _validate_artifact_record(
                source_row["summary"], root=tier_path.parent, label="layer summary"
            )
            summary_value = json.loads(summary_path.read_text())
            layer_receipt_path = (
                run_root
                / "anchors"
                / tier
                / "layers"
                / f"L{objective_row['layer']:03d}_RECEIPT.json"
            )
            layer_receipt = {
                "schema": "banana-smasher-fixed-anchor-layer-receipt-v1",
                "status": "PASS",
                "tier": tier,
                "fixed_tier": True,
                "warm_start": False,
                **objective_row,
                "input_solver_summary": artifact(summary_path),
                "input_scientific_rows": source_row["scientific_rows"],
                "solver_timing": {
                    "outer_wall_s": summary_value.get("outer_wall_s"),
                    "bucket_seconds": summary_value.get("bucket_seconds"),
                },
                "created_unix": time.time(),
            }
            atomic_json(layer_receipt_path, layer_receipt)
            layer_receipts.append(
                {"layer": objective_row["layer"], **artifact(layer_receipt_path)}
            )
        named_manifest_path = run_root / "anchors" / f"ANCHOR_{tier}_MANIFEST.json"
        named_manifest = {
            "schema": "banana-smasher-fixed-anchor-manifest-v1",
            "status": "PASS",
            "tier": tier,
            "fixed_tier": True,
            "warm_start": False,
            "layers": [row["layer"] for row in layer_objectives],
            "selected_cells": totals["selected_cells"],
            "layer_receipts": layer_receipts,
            "input_tier_solve_manifest": artifact(tier_path),
            "measurement_label": (
                f"MEASURED_{windows}_WINDOW_WEIGHTED_ERROR_NOT_MODEL_KLD"
            ),
            "created_unix": time.time(),
        }
        atomic_json(named_manifest_path, named_manifest)
        anchor_path = run_root / "anchors" / tier / "ANCHOR.json"
        tier_ended_unix = time.time()
        anchor = {
            "schema": "banana-smasher-tier-anchor-v1",
            "status": "PASS",
            "tier": tier,
            "layers": [row["layer"] for row in layer_objectives],
            "layer_objectives": layer_objectives,
            "totals": totals,
            "input_tier_solve_manifest": artifact(tier_path),
            "fixed_anchor_manifest": artifact(named_manifest_path),
            "measurement_label": (
                f"MEASURED_{windows}_WINDOW_WEIGHTED_ERROR_NOT_MODEL_KLD"
            ),
            "rebalance_scope": "PER_TIER_CALIBRATION_SURFACE",
            "timing": {
                "stage_started_unix": started_unix,
                "tier_ended_unix": tier_ended_unix,
                "elapsed_s": tier_ended_unix - started_unix,
            },
            "created_unix": tier_ended_unix,
        }
        atomic_json(anchor_path, anchor)
        anchors.append(
            {
                "tier": tier,
                **artifact(anchor_path),
                "fixed_anchor_manifest": artifact(named_manifest_path),
            }
        )

    manifest_path = run_root / "anchors" / "MANIFEST.json"
    ended_unix = time.time()
    manifest = {
        "schema": "banana-smasher-anchor-manifest-v1",
        "status": "PASS",
        "input_solve_manifest": artifact(solve_path),
        "anchors": anchors,
        "windows": windows,
        "timing": {
            "stage_started_unix": started_unix,
            "stage_ended_unix": ended_unix,
            "elapsed_s": ended_unix - started_unix,
        },
        "created_unix": ended_unix,
    }
    atomic_json(manifest_path, manifest)
    chain = _write_chain(run_root, anchor_manifest=artifact(manifest_path))
    return {
        "schema": "banana-smasher-anchor-receipt-v1",
        "status": "PASS",
        "command": "anchor",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "workflow_chain": str(chain),
    }


def process_startticks(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/stat")
    if not path.is_file():
        return None
    raw = path.read_text().rstrip()
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    return int(fields[19]) if len(fields) > 19 else None


def workflow_status(*, run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    launches = []
    for path in sorted((run_root / "run").glob("*.launch.json")):
        value = json.loads(path.read_text())
        pid = int(value["pid"])
        current = process_startticks(pid)
        expected = value.get("startticks")
        identity_matches = current is not None and current == expected
        launches.append(
            {
                **value,
                "launch_receipt": str(path),
                "current_startticks": current,
                "identity_matches": identity_matches,
                "live": identity_matches,
            }
        )
    manifests = []
    for relative in (
        "captures/MANIFEST.json",
        "hessians/MANIFEST.json",
        "solve/MANIFEST.json",
        "anchors/MANIFEST.json",
    ):
        path = run_root / relative
        if path.is_file():
            value = json.loads(path.read_text())
            manifests.append(
                {
                    "path": str(path),
                    "schema": value.get("schema"),
                    "status": value.get("status"),
                    "sha256": sha256_file(path),
                }
            )
    manifest_by_name = {Path(row["path"]).parts[-2]: row for row in manifests}
    manifest_stage_by_verb = {
        "capture": "captures",
        "hessian": "hessians",
        "solve": "solve",
        "anchor": "anchors",
    }
    dead_running_launch = any(
        row.get("status") == "RUNNING"
        and not row["live"]
        and manifest_by_name.get(
            manifest_stage_by_verb.get(str(row.get("verb", "")), ""), {}
        ).get("status")
        != "PASS"
        for row in launches
    )
    chain_path = run_root / "WORKFLOW_CHAIN.json"
    chain = json.loads(chain_path.read_text()) if chain_path.is_file() else None
    required_stages = {"hessians", "solve", "anchors"}
    complete = (
        required_stages.issubset(manifest_by_name)
        and all(row["status"] == "PASS" for row in manifests)
        and isinstance(chain, dict)
        and chain.get("status") == "PASS"
    )
    if any(row["live"] for row in launches):
        status = "RUNNING"
    elif (
        dead_running_launch
        or any(str(row.get("status", "")).startswith("FAIL") for row in launches)
        or any(str(row.get("status", "")).startswith("FAIL") for row in manifests)
    ):
        status = "FAIL"
    elif complete:
        status = "PASS"
    elif launches or manifests or chain is not None:
        status = "IN_PROGRESS"
    else:
        status = "IDLE"
    return {
        "schema": "banana-smasher-status-v1",
        "status": status,
        "run_root": str(run_root),
        "launches": launches,
        "manifests": manifests,
        "workflow_chain": chain,
        "checked_unix": time.time(),
    }


def launch_detached(*, run_root: Path, verb: str, argv: Sequence[str]) -> dict[str, Any]:
    run_root = run_root.resolve()
    run_dir = run_root / "run"
    log_dir = run_root / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / f"{verb}.launch.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        current = process_startticks(int(existing["pid"]))
        if current is not None and current == existing.get("startticks"):
            raise RuntimeError(f"{verb} is already running with PID {existing['pid']}")
    command = [sys.executable, "-m", "banana_smasher.cli", *argv]
    log_path = log_dir / f"{verb}.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    startticks = process_startticks(process.pid)
    if startticks is None:
        process.terminate()
        raise RuntimeError("detached launch process identity unavailable")
    value = {
        "schema": "banana-smasher-launch-v1",
        "status": "RUNNING",
        "verb": verb,
        "pid": process.pid,
        "startticks": startticks,
        "command": command,
        "log": str(log_path),
        "created_unix": time.time(),
    }
    atomic_json(receipt_path, value)
    return value
