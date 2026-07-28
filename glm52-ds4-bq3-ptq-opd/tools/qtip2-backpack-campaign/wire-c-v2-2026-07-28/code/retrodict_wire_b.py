#!/usr/bin/env python3
"""Binding corrected-grid Wire-B retrodiction gate.

This is a backward evaluation only.  It freezes Wire B's exact 22,016-cell
assignment and the P760/P693 menu, validates the corrected P860 grid plus its
pack=1.0 L000 authority, re-pins the complete dependency closure before loading
the solver harness, reprices QTIP2/QTIP3, and evaluates the unchanged assignment.
It never solves, reassigns, resamples, or adds/removes menu choices.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
EXPECTED_WIRE_B_MAP_SHA = "c0eef631f07fd6c136182f83d40d02e5faf8323ca38207f4e474b396b16f80ba"
EXPECTED_WIRE_B_FILE_SHA = "41842156c9abec1e023783dddf8272bc31ba56b72a6ee01b0fa8e23f63c214d8"
EXPECTED_WIRE_B_RESULT_SHA = "70ee0d05b05d1eac3d848adee481a7c670b7584c10e7ec8c6d6218430ecc7548"
EXPECTED_WIRE_B_DONE_SHA = "c04aeccfca77931b9e19016d11e38604a6bf12e623ff0277920588c7f3f9c4e3"
EXPECTED_BALANCED64_SHA = "7f756b898aea80cb4dd9320da4cd0c855f258d055f62ef6c37151d27857fa0ad"
EXPECTED_GRID_SHA = "74869b5f8e3ef4eb43dc98c6ee060c2d9ad048bb215cadd308fb2c9983933dda"
EXPECTED_GRID_MANIFEST_SHA = "2ef3cfea36e3d21198e2f1762b3a4f16e5d54f1b0fd7f0b2ab08d6dc738b9214"
EXPECTED_Q2_SHA = "96e09515e61e87669e5a378b714262184173b625844898a20f210838a3ed0b5b"
EXPECTED_Q3_SHA = "d79a79653f66067aee9255d95e0212013abae128df5c0ac2c7727ab899e44315"
EXPECTED_Q2_L000_SHA = "a11d983a1324b34b0ecdd8cafbc0f38ce6484ceaab7ef9db45ddef1025389d9c"
EXPECTED_ASSIGNED_CELLS = 43 * 256 * 2
Q2_TIER = "qtip2_2.0117"
Q3_TIER = "qtip3_3.0117"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return sha256(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_assignment(path: Path, gs: Any) -> tuple[dict[tuple[int, int, str], str], dict[str, Any]]:
    doc = json.loads(path.read_text())
    amap = doc["assignment"]
    if doc.get("assignment_map_sha256") != EXPECTED_WIRE_B_MAP_SHA:
        raise RuntimeError("Wire-B assignment-map field drift")
    observed_map_sha = canonical(amap)
    if observed_map_sha != EXPECTED_WIRE_B_MAP_SHA:
        raise RuntimeError(f"Wire-B canonical assignment-map drift: {observed_map_sha}")
    selected = {
        (layer, expert, projection): str(amap[str(layer)][str(expert)][projection])
        for layer in range(gs.LAYERS)
        for expert in range(gs.EXPERTS)
        for projection in gs.PROJECTIONS
    }
    return selected, doc


def anchor_delta(receipt: dict[str, Any], label: str) -> float:
    row = receipt["matched_delta_vs_measured_pre_repair"]
    selected = row["global"] if label == "global" else row["six_classes"][label]
    if "candidate_minus_pre_repair_mean" in selected:
        value = float(selected["candidate_minus_pre_repair_mean"])
    elif "mean" in selected:
        value = float(selected["mean"])
    else:
        raise RuntimeError(f"anchor delta field missing for {label}")
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite anchor delta for {label}")
    return value


def anchor_ratio(receipt: dict[str, Any], label: str) -> float:
    candidate_row = receipt["global"] if label == "global" else receipt["six_classes"][label]
    candidate = float(candidate_row["mean"])
    pre_repair = candidate - anchor_delta(receipt, label)
    if not math.isfinite(candidate) or candidate < 0 or not math.isfinite(pre_repair) or pre_repair <= 0:
        raise RuntimeError(f"invalid anchor candidate/pre-repair pair for {label}")
    return candidate / pre_repair


def receipt_layers(receipt: dict[str, Any]) -> list[int]:
    coverage = receipt.get("coverage", {})
    raw = coverage.get("coverage_layers") or coverage.get("sealed_layers") or receipt.get("sealed_layers")
    if not isinstance(raw, list):
        raise RuntimeError("anchor receipt has no explicit coverage layer list")
    layers = sorted(int(x) for x in raw)
    if not layers or len(layers) != len(set(layers)):
        raise RuntimeError("anchor coverage layer closure drift")
    return layers


def validate_receipt(
    name: str,
    receipt: dict[str, Any],
    expected_anchor: str,
    windows: list[int],
    required_selected_layers: set[int],
    expected_window_sha: str | None = None,
) -> list[int]:
    if receipt.get("status") != "PASS_BALANCED64_V1" or receipt.get("measurement_label") != "balanced64_v1":
        raise RuntimeError(f"{name} receipt is not a sealed BALANCED64_V1 anchor")
    if receipt.get("anchor") != expected_anchor:
        raise RuntimeError(f"{name} anchor kind drift")
    if [int(x) for x in receipt.get("window_ids", [])] != windows:
        raise RuntimeError(f"{name} window-set drift")
    if expected_window_sha is not None and receipt.get("window_manifest", {}).get("sha256") != expected_window_sha:
        raise RuntimeError(f"{name} BALANCED64 manifest hash drift")
    layers = receipt_layers(receipt)
    coverage = str(receipt.get("coverage_status", ""))
    if expected_anchor == "qtip3":
        if coverage != "UNIFORM_40_OF_40" or len(layers) != 40:
            raise RuntimeError("q3 must be honest UNIFORM_40_OF_40")
    else:
        if "PARTIAL" not in coverage or not coverage.endswith(f"{len(layers)}_OF_40") or len(layers) > 40:
            raise RuntimeError(f"{name} partial-vertical coverage drift: {coverage}")
    missing = sorted(required_selected_layers - set(layers))
    if missing:
        raise RuntimeError(f"{name} anchor omits selected layer(s): {missing}")
    return layers


def _bounded_weighted_allocate(
    target: float,
    weights: dict[tuple[int, int, str], float],
    lower: dict[tuple[int, int, str], float],
) -> dict[tuple[int, int, str], float]:
    result = {key: 0.0 for key in weights}
    active = set(weights)
    remaining = target
    while active:
        denom = math.fsum(weights[key] for key in active)
        if denom <= 0:
            local_weights = {key: 1.0 for key in active}
            denom = float(len(active))
        else:
            local_weights = weights
        proposed = {key: remaining * local_weights[key] / denom for key in active}
        saturated = [key for key in active if proposed[key] < lower[key] - 1e-18]
        if not saturated:
            for key in active:
                result[key] = proposed[key]
            remaining = 0.0
            break
        for key in saturated:
            result[key] = lower[key]
            remaining -= lower[key]
            active.remove(key)
    if abs(math.fsum(result.values()) - target) > 1e-12:
        raise RuntimeError("bounded anchor allocation cannot close without negative cell costs")
    return result


def anchor_apportionment(
    opts: dict[tuple[int, int, str], dict[str, dict[str, Any]]],
    original: dict[tuple[int, int, str], str],
    coverage_layers: list[int],
    receipt: dict[str, Any],
    extra_exact_keys: set[tuple[int, int, str]] | None = None,
) -> tuple[dict[tuple[int, int, str], dict[str, float]], dict[str, float]]:
    """Apportion a measured family ratio over its sealed pricing domain.

    ``extra_exact_keys`` is reserved for exact pack=1.0 cells outside a partial
    vertical anchor (Wire-B's 17 L000 QTIP2 cells).  It must contain only exact
    manifest-bound cells; callers validate that authority before reaching here.
    """
    layer_set = set(coverage_layers)
    extra = set(extra_exact_keys or ())
    unknown = extra - set(opts)
    if unknown:
        raise RuntimeError(f"exact pricing keys absent from solver universe: {sorted(unknown)}")
    keys = [key for key in opts if key[0] in layer_set or key in extra]
    if not keys:
        raise RuntimeError("anchor coverage has no cells in the solver universe")
    increments = {key: {} for key in keys}
    closure: dict[str, float] = {}
    for label in CLASSES:
        masses = {}
        lower = {}
        for key in keys:
            reference = opts[key].get("d4_k4096")
            if reference is None:
                raise RuntimeError(f"d4_k4096 reference missing for {key}")
            mass = float(reference["costs"][label])
            base = float(opts[key][original[key]]["costs"][label])
            if not math.isfinite(mass) or mass < 0 or not math.isfinite(base) or base < 0:
                raise RuntimeError(f"invalid reference/base mass for {key}/{label}")
            masses[key] = mass
            lower[key] = -base
        ratio = anchor_ratio(receipt, label)
        baseline_total = math.fsum(float(opts[key][original[key]]["costs"][label]) for key in opts)
        target_model_delta = baseline_total * (ratio - 1.0)
        allocated = _bounded_weighted_allocate(target_model_delta, masses, lower)
        for key in keys:
            increments[key][label] = allocated[key]
        closure[label] = math.fsum(increments[key][label] for key in keys)
        if abs(closure[label] - target_model_delta) > 1e-12:
            raise RuntimeError(f"anchor apportionment closure drift for {label}")
    return increments, closure


def summarize(opts: dict, selected: dict) -> dict[str, float]:
    pred = {label: 0.0 for label in CLASSES}
    for key, tier in selected.items():
        option = opts[key][tier]
        for label in CLASSES:
            pred[label] += float(option["costs"][label])
    return pred


def weighted_global(pred: dict[str, float], p693: Any) -> float:
    return float(p693.weighted512(pred))


def percent_delta(candidate: float, baseline: float) -> float:
    if baseline <= 0 or not math.isfinite(baseline):
        raise RuntimeError(f"invalid percent baseline {baseline}")
    return 100.0 * (candidate / baseline - 1.0)


def direction(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def parse_identity(text: str) -> tuple[int, int, str]:
    try:
        layer, expert, projection = text.split("/")
        return int(layer[1:]), int(expert[1:]), projection
    except Exception as exc:
        raise RuntimeError(f"bad identity {text!r}") from exc


def validate_l000_authority(
    authority: dict[str, Any],
    wire_b: dict[tuple[int, int, str], str],
    grid_manifest: dict[str, Any],
) -> set[tuple[int, int, str]]:
    selected = {key for key, tier in wire_b.items() if key[0] == 0 and tier == Q2_TIER}
    mapped = {parse_identity(str(x)) for x in authority.get("inclusion_identity_keys", [])}
    required = {
        "status": authority.get("status") == "PASS_INCLUDE",
        "wire_tier": authority.get("wire_tier") == Q2_TIER,
        "layer": int(authority.get("layer", -1)) == 0,
        "pack_fraction": float(authority.get("pack_fraction", -1)) == 1.0,
        "units": int(authority.get("units", -1)) == 17,
        "unique_identities": int(authority.get("unique_identities", -1)) == 17,
        "no_extras": int(authority.get("extras", -1)) == 0,
        "no_gaps": int(authority.get("gaps", -1)) == 0,
        "no_duplicates": int(authority.get("duplicates", -1)) == 0,
        "no_omissions": int(authority.get("omissions", -1)) == 0,
        "no_quarantine": int(authority.get("quarantine", -1)) == 0,
        "external_p760_pass": authority.get("all_external_p760_fp16_bf16_pass") is True,
        "runtime_reread_exact": authority.get("all_runtime_reread_exact") is True,
        "pre_post_pack_parity": authority.get("all_true_pre_post_pack_parity") is True,
        "identity_set_exact": mapped == selected and len(mapped) == 17,
    }
    rows = authority.get("machine_inclusion_mapping", [])
    row_keys = {
        (int(row["identity"]["layer"]), int(row["identity"]["expert"]), str(row["identity"]["projection"]))
        for row in rows
    }
    required["mapping_count"] = len(rows) == 17 and row_keys == selected
    required["mapping_pack1"] = all(float(row.get("pack_fraction", -1)) == 1.0 for row in rows)
    required["mapping_exact"] = all(
        row.get("runtime_reread_exact") is True and row.get("true_pre_post_pack_parity") is True
        for row in rows
    )
    manifested = grid_manifest.get("retrodiction_authority", {}).get("qtip2_l000", {})
    required["grid_manifest_authority"] = (
        manifested.get("authority") == "authoritative_exact_pack1_subset"
        and float(manifested.get("pack_fraction", -1)) == 1.0
        and int(manifested.get("units", -1)) == 17
        and grid_manifest.get("retrodiction_authority", {}).get("selected_l000_closed_exactly") is True
    )
    failures = sorted(key for key, ok in required.items() if not ok)
    if failures:
        raise RuntimeError(f"L000 pack=1.0 authority failed: {failures}")
    return selected


def load_measurement_source(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    doc = json.loads(path.read_text())
    row = doc.get("wire_b", {})
    measured_raw = row.get("measured_delta_pct") or row.get("measured_percent")
    if not isinstance(measured_raw, dict):
        raise RuntimeError("Wire-B measured-percent source missing")
    measured = {label: float(measured_raw[label]) for label in ("global", *CLASSES)}
    expected = {
        "global": -10.2,
        "agentic": -12.7,
        "chat": -22.0,
        "code": 23.3,
        "multilingual": -11.6,
        "prose": -13.9,
        "reasoning": -5.1,
    }
    if measured != expected:
        raise RuntimeError(f"Wire-B measured calibration drift: {measured}")
    if row.get("assignment_map_sha256") != EXPECTED_WIRE_B_MAP_SHA:
        raise RuntimeError("measurement-source assignment map drift")
    assignment_sha = row.get("assignment_receipt_sha256") or row.get("assignment_file_sha256")
    if assignment_sha != EXPECTED_WIRE_B_FILE_SHA:
        raise RuntimeError("measurement-source assignment file drift")
    if row.get("result_sha256") not in (None, EXPECTED_WIRE_B_RESULT_SHA):
        raise RuntimeError("measurement-source Wire-B result drift")
    return measured, doc


def load_grid(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 14:
        raise RuntimeError(f"corrected grid row-count drift: {len(rows)}")
    by_tier = {str(row["tier"]): row for row in rows}
    if len(by_tier) != len(rows) or Q2_TIER not in by_tier or Q3_TIER not in by_tier:
        raise RuntimeError("corrected grid tier closure drift")
    return by_tier


def validate_grid_contract(
    grid_path: Path,
    manifest_path: Path,
    q2_path: Path,
    q3_path: Path,
    l000_path: Path,
    assignment_path: Path,
    q2: dict[str, Any],
    q3: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    observed = {
        "grid": sha256(grid_path),
        "manifest": sha256(manifest_path),
        "q2": sha256(q2_path),
        "q3": sha256(q3_path),
        "q2_l000": sha256(l000_path),
        "assignment": sha256(assignment_path),
    }
    expected = {
        "grid": EXPECTED_GRID_SHA,
        "manifest": EXPECTED_GRID_MANIFEST_SHA,
        "q2": EXPECTED_Q2_SHA,
        "q3": EXPECTED_Q3_SHA,
        "q2_l000": EXPECTED_Q2_L000_SHA,
        "assignment": EXPECTED_WIRE_B_FILE_SHA,
    }
    if observed != expected:
        raise RuntimeError(f"corrected-grid contract SHA drift: observed={observed} expected={expected}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PASS_CORRECTED_GRID" or manifest.get("schema") != "p860-corrected-vertical-grid-v1":
        raise RuntimeError("corrected-grid manifest status/schema drift")
    checks = {
        "grid_manifest_hash": manifest.get("corrected_grid", {}).get("sha256") == observed["grid"],
        "q2_hash": manifest.get("anchors", {}).get("qtip2", {}).get("sha256") == observed["q2"],
        "q3_hash": manifest.get("anchors", {}).get("qtip3", {}).get("sha256") == observed["q3"],
        "l000_hash": manifest.get("retrodiction_authority", {}).get("qtip2_l000", {}).get("sha256") == observed["q2_l000"],
        "assignment_hash": manifest.get("retrodiction_authority", {}).get("qtip2_l000", {}).get("wire_b_assignment", {}).get("sha256") == observed["assignment"],
        "pack1": float(manifest.get("safety", {}).get("q2_l000_pack_fraction_validated", -1)) == 1.0,
        "wire_b_l000_closed": manifest.get("validation", {}).get("wire_b_l000_exact_authority_closed") is True,
        "no_inference": manifest.get("validation", {}).get("inference_or_backfill_used") is False,
        "k1_absent": manifest.get("anchors", {}).get("qtip15", {}).get("authority") == "unavailable_not_included",
    }
    failures = sorted(key for key, ok in checks.items() if not ok)
    if failures:
        raise RuntimeError(f"corrected-grid manifest binding failed: {failures}")
    grid = load_grid(grid_path)
    for tier, receipt in ((Q2_TIER, q2), (Q3_TIER, q3)):
        row = grid[tier]
        values = {"global": float(row["global_mean_kld"]), **{label: float(row[label]) for label in CLASSES}}
        receipt_values = {
            "global": float(receipt["global"]["mean"]),
            **{label: float(receipt["six_classes"][label]["mean"]) for label in CLASSES},
        }
        error = max(abs(values[label] - receipt_values[label]) for label in values)
        if error > 5.1e-13:
            raise RuntimeError(f"corrected-grid {tier} row/receipt drift: max_error={error}")
    return manifest, grid


def pin_row(path: Path, role: str, root: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required contract file missing: {path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(root.resolve())) if root is not None else None,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_preexec_pin_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = args.solver_root.resolve()
    paths: list[tuple[Path, str, Path | None]] = [
        (Path(__file__).resolve(), "retrodiction_harness", None),
        (root / "code/solve_p760.py", "p760_code", root),
        (root / "code/solve_p693.py", "p693_code", root),
        (root / "code/solve_actual.py", "p637_surface_code", root),
        (root / "lineage/original_genesis_code_solve.py", "genesis_surface_code", root),
        (root / "inputs/INPUT_MANIFEST.json", "p760_input_manifest", root),
        (root / "lineage/NOMINATED_ASSIGNMENT.json", "frozen_incumbent_assignment", root),
        (root / "lineage/FRONTIER.json", "frozen_frontier_config", root),
        (root / "lineage/P620_WITH_QTIP2_RESULT.json", "p620_lineage_result", root),
        (root / "receipts/STEP1_WITHOUT_REPRO.json", "p629_reproduction_receipt", root),
        (root / "source_p637_out/FINAL_TABLE.json", "p637_final_table", root),
        (root / "source_p637_out/ASSIGNMENT_RESPENT.json", "p637_assignment", root),
        (root / "inputs/MENU_SNAPSHOT.json", "p760_menu_snapshot", root),
        (root / "inputs/qtip3/P696_QTIP3_ARCHIVE_AUDIT.json", "qtip3_archive_audit", root),
        (root / "inputs/qtip3/QTIP_MENU_EXTENSION.json", "qtip3_price_extension", root),
        (root / "inputs/qtip3/QTIP_TIER_PRICE_ROWS.csv", "qtip3_price_rows", root),
        (args.wire_b_assignment, "wire_b_exact_assignment", None),
        (args.wire_b_result, "wire_b_solver_result", None),
        (args.wire_b_done, "wire_b_solver_done", None),
        (args.wire_b_measurement, "wire_b_measured_calibration", None),
        (args.q2_receipt, "qtip2_balanced64_anchor", None),
        (args.q3_receipt, "qtip3_balanced64_anchor", None),
        (args.q2_l000_authority, "qtip2_l000_exact_pack1_authority", None),
        (args.balanced64, "balanced64_window_manifest", None),
        (args.corrected_grid, "corrected_vertical_grid", None),
        (args.corrected_grid_manifest, "corrected_vertical_grid_manifest", None),
    ]
    input_manifest = json.loads((root / "inputs/INPUT_MANIFEST.json").read_text())
    for row in input_manifest.get("files", []):
        paths.append((root / row["path"], "p760_manifest_bound_input", root))
    for path in sorted((root / "anchors").glob("QTIP2_ANCHOR_L*.json")):
        paths.append((path, "p760_qtip2_anchor", root))
    for path in sorted((root / "inputs/qtip15").glob("*.json")):
        paths.append((path, "p760_qtip15_config", root))
    unique: dict[str, tuple[Path, set[str], Path | None]] = {}
    for path, role, relative_root in paths:
        key = str(path.resolve())
        if key not in unique:
            unique[key] = (path, {role}, relative_root)
        else:
            unique[key][1].add(role)
    files = []
    for key in sorted(unique):
        path, roles, relative_root = unique[key]
        row = pin_row(path, ",".join(sorted(roles)), relative_root)
        files.append(row)
    expected_manifest_rows = {str((root / row["path"]).resolve()): row for row in input_manifest.get("files", [])}
    manifest_errors = []
    for row in files:
        expected = expected_manifest_rows.get(row["path"])
        if expected and (row["bytes"] != int(expected["bytes"]) or row["sha256"] != expected["sha256"]):
            manifest_errors.append(row["path"])
    if manifest_errors:
        raise RuntimeError(f"P760 input-manifest dependency drift: {manifest_errors}")
    try:
        ortools_version = importlib.metadata.version("ortools")
    except importlib.metadata.PackageNotFoundError:
        ortools_version = "NOT_INSTALLED_AT_PIN_TIME"
    contract_digest = canonical(
        [{"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"], "role": row["role"]} for row in files]
    )
    return {
        "schema": "p860-wire-b-retrodiction-preexec-pins-v1",
        "status": "PASS_PREEXEC_PINNED",
        "task_id": args.task_id,
        "host": platform.node(),
        "created_unix": time.time(),
        "solver_root": str(root),
        "canonical_commit": input_manifest.get("canonical_commit"),
        "repository_pin_status": "CANONICAL_COMMIT_FROM_SEALED_P760_INPUT_MANIFEST_PLUS_EXACT_CONTENT_SHA256",
        "execution": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "ortools_version": ortools_version,
            "max_global_error_pp": args.max_global_error_pp,
            "max_class_error_pp": args.max_class_error_pp,
            "min_code_retrodicted_pp": args.min_code_retrodicted_pp,
            "expected_assigned_cells": EXPECTED_ASSIGNED_CELLS,
            "assignment_policy": "frozen exact map; no reassignment/resampling/menu change",
        },
        "files": files,
        "file_count": len(files),
        "contract_content_sha256": contract_digest,
    }


def write_table(path: Path, result: dict[str, Any]) -> None:
    rows = result["retrodiction_table"]
    lines = [
        "# Binding Wire B corrected-grid retrodiction gate",
        "",
        f"Decision: **{result['signed_gate_decision']['decision']}**",
        f"Status: **{result['status']}**",
        f"Assignment cells evaluated: **{result['wire_b']['assigned_cells_evaluated']:,}** / **{EXPECTED_ASSIGNED_CELLS:,}**",
        "",
        "| label | measured % | retrodicted % | measured dir | retro dir | direction | signed error pp | |error| pp | error direction | error gate | material-positive gate |",
        "|---|---:|---:|---|---|:---:|---:|---:|---|:---:|:---:|",
    ]
    for label in ("global", *CLASSES):
        row = rows[label]
        material = "PASS" if row["code_materially_positive_ok"] else "FAIL" if label == "code" else "N/A"
        lines.append(
            f"| {label} | {row['measured_percent']:+.6f} | {row['anchor_corrected_retrodicted_percent']:+.6f} | "
            f"{row['measured_direction']} | {row['retrodicted_direction']} | {'PASS' if row['direction_ok'] else 'FAIL'} | "
            f"{row['error_percentage_points']:+.6f} | {row['absolute_error_percentage_points']:.6f} | "
            f"{row['error_direction']} | {'PASS' if row['error_within_limit'] else 'FAIL'} | {material} |"
        )
    lines += [
        "",
        f"- Missing classes: `{result['diagnostics']['missing_classes']}`",
        f"- Sign reversals: `{result['diagnostics']['sign_reversals']}`",
        f"- Assignment-count discrepancy: `{result['diagnostics']['assignment_count_discrepancy']}`",
        f"- QTIP2 anchor missing layers: `{result['diagnostics']['qtip2_anchor_missing_layers']}`; L000 closed by exact pack=1.0 identities: `{result['diagnostics']['qtip2_l000_exact_pack1_cells']}`.",
        f"- Wire C solve/build permitted: `{result['gates']['wire_c_solve_or_build_permitted']}`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-root", type=Path, required=True, help="exact complete P760 solver mission root")
    parser.add_argument("--wire-b-assignment", type=Path, required=True)
    parser.add_argument("--wire-b-result", type=Path, required=True)
    parser.add_argument("--wire-b-done", type=Path, required=True)
    parser.add_argument("--wire-b-measurement", type=Path, required=True)
    parser.add_argument("--q2-receipt", type=Path, required=True)
    parser.add_argument("--q3-receipt", type=Path, required=True)
    parser.add_argument("--q2-l000-authority", type=Path, required=True)
    parser.add_argument("--balanced64", type=Path, required=True)
    parser.add_argument("--corrected-grid", type=Path, required=True)
    parser.add_argument("--corrected-grid-manifest", type=Path, required=True)
    parser.add_argument("--pin-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table-out", type=Path, required=True)
    parser.add_argument("--task-id", default="PUBLIC_TASK")
    parser.add_argument("--max-global-error-pp", type=float, default=10.0)
    parser.add_argument("--max-class-error-pp", type=float, default=10.0)
    parser.add_argument("--min-code-retrodicted-pp", type=float, default=5.0)
    args = parser.parse_args()

    args.solver_root = args.solver_root.resolve()
    for name in (
        "wire_b_assignment", "wire_b_result", "wire_b_done", "wire_b_measurement", "q2_receipt",
        "q3_receipt", "q2_l000_authority", "balanced64", "corrected_grid", "corrected_grid_manifest",
        "pin_manifest", "out", "table_out",
    ):
        setattr(args, name, getattr(args, name).resolve())

    # Binding requirement: pin every code/data/config dependency before importing
    # or executing the P760/P693 harness.
    pins = build_preexec_pin_manifest(args)
    pin_sha = atomic_json(args.pin_manifest, pins)

    if sha256(args.wire_b_assignment) != EXPECTED_WIRE_B_FILE_SHA:
        raise RuntimeError("Wire-B assignment file SHA drift")
    if sha256(args.wire_b_result) != EXPECTED_WIRE_B_RESULT_SHA:
        raise RuntimeError("Wire-B result SHA drift")
    if sha256(args.wire_b_done) != EXPECTED_WIRE_B_DONE_SHA:
        raise RuntimeError("Wire-B DONE SHA drift")
    if sha256(args.balanced64) != EXPECTED_BALANCED64_SHA:
        raise RuntimeError("BALANCED64_V1 SHA drift")

    measured, measurement_doc = load_measurement_source(args.wire_b_measurement)
    windows = [int(x) for x in json.loads(args.balanced64.read_text())["windows"]]
    if len(windows) != 64 or len(set(windows)) != 64:
        raise RuntimeError("BALANCED64_V1 window closure drift")
    q2 = json.loads(args.q2_receipt.read_text())
    q3 = json.loads(args.q3_receipt.read_text())
    l000_authority = json.loads(args.q2_l000_authority.read_text())
    grid_manifest, grid = validate_grid_contract(
        args.corrected_grid,
        args.corrected_grid_manifest,
        args.q2_receipt,
        args.q3_receipt,
        args.q2_l000_authority,
        args.wire_b_assignment,
        q2,
        q3,
    )

    solver = args.solver_root / "code/solve_p760.py"
    p760 = load_module("wire_c_binding_p760_surface", solver)
    surface = p760.p693.build_surface()
    p693, gs = p760.p693, surface["gs"]
    opts, original = surface["opts"], surface["original"]
    wire_b, assignment_doc = load_assignment(args.wire_b_assignment, gs)
    if set(wire_b) != set(original):
        raise RuntimeError("Wire-B assignment key closure drift")
    assigned_cells = len(wire_b)
    if assigned_cells != EXPECTED_ASSIGNED_CELLS:
        raise RuntimeError(f"Wire-B assignment count drift: {assigned_cells}")
    assignment_tier_counts = dict(sorted(Counter(wire_b.values()).items()))

    selected_layers = {
        Q2_TIER: {key[0] for key, tier in wire_b.items() if tier == Q2_TIER},
        Q3_TIER: {key[0] for key, tier in wire_b.items() if tier == Q3_TIER},
    }
    l000_exact_keys = validate_l000_authority(l000_authority, wire_b, grid_manifest)
    q2_non_l000_layers = selected_layers[Q2_TIER] - {0}
    q2_layers = validate_receipt("q2", q2, "qtip2", windows, q2_non_l000_layers, EXPECTED_BALANCED64_SHA)
    q3_layers = validate_receipt("q3", q3, "qtip3", windows, selected_layers[Q3_TIER], EXPECTED_BALANCED64_SHA)
    for key, tier in wire_b.items():
        if tier in (Q2_TIER, Q3_TIER) and tier not in opts[key]:
            raise RuntimeError(f"Wire-B selected tier absent from frozen surface: {key}/{tier}")

    baseline_pred = summarize(opts, original)
    raw_wire_b_pred = summarize(opts, wire_b)
    raw_uniform = {}
    apportionment: dict[str, Any] = {}
    corrected_opts = {
        key: {tier: {**option, "costs": dict(option["costs"])} for tier, option in local.items()}
        for key, local in opts.items()
    }

    for tier, receipt, layers, exact_extra in (
        (Q2_TIER, q2, q2_layers, l000_exact_keys),
        (Q3_TIER, q3, q3_layers, set()),
    ):
        eligible = [key for key, local in opts.items() if tier in local]
        uniform = dict(original)
        for key in eligible:
            uniform[key] = tier
        raw_pred = summarize(opts, uniform)
        raw_uniform[tier] = {
            "legacy_surface_eligible_cells": len(eligible),
            "legacy_surface_prediction_by_class": raw_pred,
            "not_used_for_correction": True,
        }
        increments, closure = anchor_apportionment(opts, original, layers, receipt, exact_extra)
        applied = 0
        selected_applied = 0
        for key in eligible:
            if key not in increments:
                continue
            applied += 1
            if wire_b.get(key) == tier:
                selected_applied += 1
            for label in CLASSES:
                base_cost = float(opts[key][original[key]]["costs"][label])
                corrected = base_cost + increments[key][label]
                if not math.isfinite(corrected) or corrected < -1e-12:
                    raise RuntimeError(f"negative/nonfinite corrected cost for {key}/{tier}/{label}: {corrected}")
                corrected_opts[key][tier]["costs"][label] = max(0.0, corrected)
        selected_total = sum(1 for selected_tier in wire_b.values() if selected_tier == tier)
        if selected_applied != selected_total:
            raise RuntimeError(f"selected {tier} repricing closure drift: {selected_applied}/{selected_total}")
        pricing_domain_cells = len(increments)
        apportionment[tier] = {
            "coverage_layers": layers,
            "coverage_layer_count": len(layers),
            "exact_extra_keys": [f"L{k[0]:03d}/E{k[1]:03d}/{k[2]}" for k in sorted(exact_extra)],
            "exact_extra_key_count": len(exact_extra),
            "cells_in_corrected_pricing_domain": pricing_domain_cells,
            "legacy_surface_options_corrected": applied,
            "selected_wire_b_cells_corrected": selected_applied,
            "selected_wire_b_cells_total": selected_total,
            "selected_wire_b_layers": sorted(selected_layers[tier]),
            "measured_physical_delta_by_class": {label: anchor_delta(receipt, label) for label in CLASSES},
            "measured_candidate_to_pre_repair_ratio_by_class": {label: anchor_ratio(receipt, label) for label in CLASSES},
            "corrected_grid_row": {
                "global": float(grid[tier]["global_mean_kld"]),
                **{label: float(grid[tier][label]) for label in CLASSES},
            },
            "transported_model_delta_closure_by_class": closure,
            "mass_reference_tier": "d4_k4096",
        }

    corrected_wire_b_pred = summarize(corrected_opts, wire_b)
    baseline_global = weighted_global(baseline_pred, p693)
    raw_global = weighted_global(raw_wire_b_pred, p693)
    corrected_global = weighted_global(corrected_wire_b_pred, p693)
    raw_percent = {label: percent_delta(raw_wire_b_pred[label], baseline_pred[label]) for label in CLASSES}
    corrected_percent = {label: percent_delta(corrected_wire_b_pred[label], baseline_pred[label]) for label in CLASSES}
    raw_percent["global"] = percent_delta(raw_global, baseline_global)
    corrected_percent["global"] = percent_delta(corrected_global, baseline_global)

    rows = {}
    failures = []
    for label in ("global", *CLASSES):
        measured_value = measured[label]
        predicted = corrected_percent[label]
        error = predicted - measured_value
        measured_direction = direction(measured_value)
        predicted_direction = direction(predicted)
        direction_ok = predicted_direction == measured_direction
        limit = args.max_global_error_pp if label == "global" else args.max_class_error_pp
        error_ok = abs(error) <= limit
        code_ok = label != "code" or predicted >= args.min_code_retrodicted_pp
        if error > 0:
            error_direction = "retrodiction_above_measured"
        elif error < 0:
            error_direction = "retrodiction_below_measured"
        else:
            error_direction = "exact"
        rows[label] = {
            "measured_percent": measured_value,
            "raw_fantasy_predicted_percent": raw_percent[label],
            "anchor_corrected_retrodicted_percent": predicted,
            "measured_direction": measured_direction,
            "retrodicted_direction": predicted_direction,
            "direction_ok": direction_ok,
            "error_percentage_points": error,
            "absolute_error_percentage_points": abs(error),
            "error_direction": error_direction,
            "error_within_limit": error_ok,
            "error_limit_percentage_points": limit,
            "material_positive_gate_applicable": label == "code",
            "code_materially_positive_min_percent": args.min_code_retrodicted_pp if label == "code" else None,
            "code_materially_positive_ok": code_ok if label == "code" else None,
        }
        if not (direction_ok and error_ok and code_ok):
            failures.append(label)

    expected_labels = {"global", *CLASSES}
    missing_classes = sorted(expected_labels - set(rows))
    sign_reversals = sorted(label for label, row in rows.items() if not row["direction_ok"])
    count_discrepancy = assigned_cells - EXPECTED_ASSIGNED_CELLS
    selected_uncovered = {
        Q2_TIER: sorted(
            key for key, tier in wire_b.items()
            if tier == Q2_TIER and key[0] not in set(q2_layers) and key not in l000_exact_keys
        ),
        Q3_TIER: sorted(key for key, tier in wire_b.items() if tier == Q3_TIER and key[0] not in set(q3_layers)),
    }
    if missing_classes or count_discrepancy or any(selected_uncovered.values()):
        failures.extend(["contract_closure"])
    failures = sorted(set(failures))
    passed = not failures
    decision = "AUTHORIZE_WIRE_C_SOLVE_GATE_ONLY" if passed else "STOP_NO_WIRE_C_SOLVE_OR_BUILD"

    result = {
        "schema": "p860-wire-b-binding-corrected-grid-retrodiction-v3",
        "status": "PASS_RETRODICTION_GATE" if passed else "FAIL_RETRODICTION_GATE_STOP_NO_WIRE_C_SOLVE_OR_BUILD",
        "task_id": args.task_id,
        "host": platform.node(),
        "created_unix": time.time(),
        "signed_gate_decision": {
            "decision": decision,
            "signed_by": f"Hermes Kanban task {args.task_id}",
            "binding": True,
            "rule": "Wire C solve/build remains forbidden unless this receipt is PASS and every class-direction/count/pack gate passes.",
        },
        "wire_b": {
            "assignment": str(args.wire_b_assignment),
            "assignment_file_sha256": sha256(args.wire_b_assignment),
            "assignment_map_sha256": EXPECTED_WIRE_B_MAP_SHA,
            "assigned_cells_expected": EXPECTED_ASSIGNED_CELLS,
            "assigned_cells_evaluated": assigned_cells,
            "assignment_count_discrepancy": count_discrepancy,
            "tier_counts": assignment_tier_counts,
            "qtip2_cells": assignment_tier_counts.get(Q2_TIER, 0),
            "qtip3_cells": assignment_tier_counts.get(Q3_TIER, 0),
            "measured_basis": "operator-sealed 128w paired Wire-B calibration point pinned by staged measurement receipt",
            "measurement_source": {"path": str(args.wire_b_measurement), "sha256": sha256(args.wire_b_measurement)},
            "solver_result": {"path": str(args.wire_b_result), "sha256": sha256(args.wire_b_result)},
            "solver_done": {"path": str(args.wire_b_done), "sha256": sha256(args.wire_b_done)},
            "no_reassignment": True,
            "no_resampling": True,
            "no_menu_changes": True,
        },
        "corrected_grid": {
            "path": str(args.corrected_grid),
            "sha256": sha256(args.corrected_grid),
            "manifest": str(args.corrected_grid_manifest),
            "manifest_sha256": sha256(args.corrected_grid_manifest),
            "qtip2_coverage": q2["coverage_status"],
            "qtip3_coverage": q3["coverage_status"],
            "qtip15": "UNAVAILABLE_NOT_INCLUDED",
        },
        "anchors": {
            "balanced64_v1": {"path": str(args.balanced64), "sha256": EXPECTED_BALANCED64_SHA, "windows": windows},
            "qtip2": {"path": str(args.q2_receipt), "sha256": sha256(args.q2_receipt), "coverage": q2["coverage_status"]},
            "qtip3": {"path": str(args.q3_receipt), "sha256": sha256(args.q3_receipt), "coverage": q3["coverage_status"]},
            "qtip2_l000_exact_pack1": {
                "path": str(args.q2_l000_authority),
                "sha256": sha256(args.q2_l000_authority),
                "pack_fraction": 1.0,
                "exact_cells": len(l000_exact_keys),
                "identities": [f"L{k[0]:03d}/E{k[1]:03d}/{k[2]}" for k in sorted(l000_exact_keys)],
            },
        },
        "method": "freeze the exact c0eef631 22,016-cell Wire-B map; transport each corrected-grid BALANCED64 candidate/pre-repair ratio into the frozen P760 currency; apportion by frozen d4_k4096 assignment-aware cell mass over each sealed pricing domain, adding only the 17 manifest-exact pack=1.0 L000 QTIP2 identities; evaluate the unchanged map without solve, reassignment, resampling, or menu mutation",
        "solver": {
            "root": str(args.solver_root),
            "solve_p760_sha256": sha256(solver),
            "canonical_commit": pins.get("canonical_commit"),
        },
        "preexec_pins": {
            "path": str(args.pin_manifest),
            "sha256": pin_sha,
            "file_count": pins["file_count"],
            "contract_content_sha256": pins["contract_content_sha256"],
        },
        "baseline_prediction_by_class": baseline_pred,
        "raw_uniform_predictions": raw_uniform,
        "anchor_delta_apportionment": apportionment,
        "raw_wire_b_prediction_by_class": raw_wire_b_pred,
        "corrected_wire_b_prediction_by_class": corrected_wire_b_pred,
        "retrodiction_table": rows,
        "failed_labels": failures,
        "diagnostics": {
            "missing_classes": missing_classes,
            "sign_reversals": sign_reversals,
            "assignment_count_discrepancy": count_discrepancy,
            "selected_uncovered_after_exact_authority": {
                tier: [f"L{k[0]:03d}/E{k[1]:03d}/{k[2]}" for k in keys]
                for tier, keys in selected_uncovered.items()
            },
            "qtip2_anchor_missing_layers": sorted(set(range(3, 43)) - set(q2_layers)),
            "qtip2_l000_exact_pack1_cells": len(l000_exact_keys),
            "qtip3_anchor_missing_layers": sorted(set(range(3, 43)) - set(q3_layers)),
        },
        "gates": {
            "all_directions_match": all(row["direction_ok"] for row in rows.values()),
            "all_errors_within_limits": all(row["error_within_limit"] for row in rows.values()),
            "code_materially_positive": bool(rows["code"]["code_materially_positive_ok"]),
            "code_materially_positive_min_percent": args.min_code_retrodicted_pp,
            "all_22016_cells_evaluated": assigned_cells == EXPECTED_ASSIGNED_CELLS,
            "pack_fraction_1_0": True,
            "no_missing_classes": not missing_classes,
            "wire_c_solve_or_build_permitted": passed,
        },
        "measurement_source_payload_sha256": canonical(measurement_doc),
        "assignment_input_payload_sha256": canonical(assignment_doc),
    }
    result_sha = atomic_json(args.out, result)
    write_table(args.table_out, result)
    table_sha = sha256(args.table_out)
    print(json.dumps({
        "status": result["status"],
        "decision": decision,
        "out": str(args.out),
        "sha256": result_sha,
        "table": str(args.table_out),
        "table_sha256": table_sha,
        "retrodiction": rows,
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
