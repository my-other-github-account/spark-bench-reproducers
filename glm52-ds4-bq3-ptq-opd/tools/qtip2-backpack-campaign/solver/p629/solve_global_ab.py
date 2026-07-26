#!/usr/bin/env python3
"""P629 compute-node-1 canonical GENESIS class-balanced GLOBAL A/B solve.

Uses the sealed rung-2 GENESIS inputs and immutable c9fb72... assignment as the
feasible incumbent. The objective is the uniform arithmetic mean of the six
per-class predicted KLDs. The WITH arm adds exactly one all-512
qtip2_2.0117 layer option on the existing rep-16 layers. Measured layers retain
their signed TRAIN8 swap deltas; the other seven use the arithmetic mean of the
nine measured rows. The WITHOUT arm contains only the current menu.
Prediction only; no measured-transfer claim.
"""
from __future__ import annotations

import hashlib
import importlib.util
import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ortools.linear_solver import pywraplp

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
LINEAGE = ROOT / "lineage"
ANCHORS = ROOT / "anchors"
OUT = ROOT / "out"
LOGS = ROOT / "logs"

ENVELOPE = 101_346_700_411
EXPECTED_INPUT_MANIFEST_SHA = "88ebfc21a7134088cad0a9a4f09821410db29ac13cd754d4d46f8902bebecb42"
EXPECTED_ASSIGNMENT_SHA = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
EXPECTED_ORIGINAL_SOLVER_SHA = "81e2c5eb54a14c978c8324373be6d4c0a2f96e518427b8f26c5e44a33c39a68d"
MEASURED = (0, 2, 4, 6, 11, 14, 16, 19, 22)
ELIGIBLE = (0, 2, 4, 6, 11, 14, 16, 19, 22, 25, 27, 30, 34, 35, 38, 42)
QTIP_TIER = "qtip2_2.0117"
QTIP_LAYER_BYTES = 1_617_954_816
QTIP_BYTES = {"fused13": 4_210_692, "down": 2_109_444}
TIME_LIMIT_SECONDS = 678.0
CANON_GLOBAL_BASELINE = 0.08395
CANON_CODE_BASELINE = 0.04170458531457378


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return sha256(path)


def load_original():
    p = LINEAGE / "original_genesis_code_solve.py"
    if sha256(p) != EXPECTED_ORIGINAL_SOLVER_SHA:
        raise RuntimeError("original solver SHA drift")
    spec = importlib.util.spec_from_file_location("genesis_original", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def validate_inputs(gs):
    manifest = INPUTS / "INPUT_MANIFEST.json"
    if sha256(manifest) != EXPECTED_INPUT_MANIFEST_SHA:
        raise RuntimeError("input manifest SHA drift")
    doc = json.loads(manifest.read_text())
    verified = []
    for row in doc["files"]:
        p = ROOT / row["path"]
        actual = {"path": row["path"], "bytes": p.stat().st_size, "sha256": sha256(p)}
        if actual["bytes"] != int(row["bytes"]) or actual["sha256"] != row["sha256"]:
            raise RuntimeError({"sealed_input_drift": actual, "expected": row})
        verified.append(actual)
    assignment_path = LINEAGE / "NOMINATED_ASSIGNMENT.json"
    if sha256(assignment_path) != EXPECTED_ASSIGNMENT_SHA:
        raise RuntimeError("assignment SHA drift")
    return doc, verified


def read_assignment(gs):
    doc = json.loads((LINEAGE / "NOMINATED_ASSIGNMENT.json").read_text())
    assignment = {}
    for layer in range(gs.LAYERS):
        for expert in range(gs.EXPERTS):
            pair = doc["assignment"][str(layer)][str(expert)]
            for projection in gs.PROJECTIONS:
                assignment[(layer, expert, projection)] = str(pair[projection])
    return doc, assignment


def load_anchor_deltas(gs):
    rows = []
    for layer in MEASURED:
        p = ANCHORS / f"QTIP2_ANCHOR_L{layer:03d}.json"
        d = json.loads(p.read_text())
        if int(d["layer"]) != layer or not str(d.get("status", "")).startswith("PASS_MEASURED"):
            raise RuntimeError({"bad_anchor": str(p)})
        q = d["qtip2"]
        logical_bytes = q.get("logical_bytes_exact", q.get("logical_bytes"))
        units = q.get("units", q.get("projection_units"))
        if int(logical_bytes) != QTIP_LAYER_BYTES or int(units) != 512:
            raise RuntimeError({"anchor_byte_drift": str(p)})
        delta = {c: 0.0 for c in gs.CLASSES}
        if "damage_by_class" in d:
            class_rows = d["damage_by_class"]
            global_delta = d["candidate"]["global_delta_kld_qtip2_minus_baseline"]
            for c, r in class_rows.items():
                if c in delta:
                    delta[c] = float(r["delta_kld_qtip2_minus_baseline"])
        else:
            class_rows = d["measurement"]["observed_train8_classes"]
            global_delta = d["measurement"]["global"]["delta_kld_vs_baseline"]
            for c, r in class_rows.items():
                if c in delta:
                    delta[c] = float(r["delta_kld_vs_baseline"])
        rows.append({
            "layer": layer,
            "basis": "MEASURED_ARTIFACT_RELATIVE_SWAP",
            "delta_by_class": delta,
            "global_delta": float(global_delta),
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        })
    family_mean = {
        c: math.fsum(r["delta_by_class"][c] for r in rows) / len(rows)
        for c in gs.CLASSES
    }
    family_global = math.fsum(r["global_delta"] for r in rows) / len(rows)
    by_layer = {r["layer"]: r for r in rows}
    for layer in ELIGIBLE:
        if layer not in by_layer:
            by_layer[layer] = {
                "layer": layer,
                "basis": "ARITHMETIC_FAMILY_MEAN_OF_9_MEASURED_SWAP_ROWS",
                "delta_by_class": dict(family_mean),
                "global_delta": family_global,
                "source_layers": list(MEASURED),
            }
    return by_layer, rows, family_mean, family_global


def weighted_global_weights(step0_doc, classes):
    by = step0_doc["by_class"]
    counts = {c: int(by[c]["n_positions"]) for c in classes}
    total = sum(counts.values())
    return {c: counts[c] / total for c in classes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("WITHOUT", "WITH"), required=True)
    parser.add_argument("--hint-qtip-layers", default="")
    args = parser.parse_args()
    arm = args.arm
    hint_qtip_layers = {
        int(x) for x in args.hint_qtip_layers.split(",") if x.strip()
    }
    if hint_qtip_layers - set(ELIGIBLE) or (arm == "WITHOUT" and hint_qtip_layers):
        raise ValueError({"invalid_hint_qtip_layers": sorted(hint_qtip_layers), "arm": arm})
    global OUT
    OUT = ROOT / "out" / arm.lower()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    gs = load_original()
    manifest_doc, verified_inputs = validate_inputs(gs)

    anchors_grid = gs.load_anchor_grid(INPUTS / "rung1" / "ANCHOR_VERTICAL_GRID.csv")
    rows = gs.load_profile(INPUTS / "profile" / "PROFILE_ROWS.jsonl")
    importance, normalization = gs.normalize_profile_rows(rows)
    step0 = gs.step0_means(INPUTS / "baseline" / "BQ3_STEP0_PER_CLASS.json")
    old_incumbent, _ = gs.map_incumbent(INPUTS / "baseline" / "DUALVQ_K4096MENU_BQ3_BIN_MANIFEST.json")
    corrections, correction_fit = gs.fit_projection_corrections(old_incumbent, importance, anchors_grid, step0)
    cells = gs.make_cells(importance, anchors_grid, corrections)
    assignment_doc, incumbent = read_assignment(gs)
    delta_surface, measured_rows, family_mean, family_global = load_anchor_deltas(gs)

    # Exact canonical sanity: replay c9fb assignment through the sealed objective.
    incumbent_pred = gs.predict_assignment(incumbent, importance, anchors_grid, gs.CLASSES, corrections=corrections)
    lineage = json.loads((LINEAGE / "FRONTIER.json").read_text())
    expected_pred = lineage["nomination"]["predicted_class_mean_kld"]
    errors = {c: abs(incumbent_pred[c] - float(expected_pred[c])) for c in gs.CLASSES}
    if max(errors.values()) > 1e-12:
        raise RuntimeError({"canonical_reproduction_mismatch": errors})
    expected_counts = Counter(lineage["nomination"]["tier_counts"])
    actual_counts = Counter(incumbent.values())
    if actual_counts != expected_counts:
        raise RuntimeError({"tier_count_reproduction_mismatch": {"actual": actual_counts, "expected": expected_counts}})

    status_receipt = {
        "schema": "p629-progress-v1",
        "arm": arm,
        "status": f"SOLVING_{arm}_CLASS_BALANCED_GLOBAL",
        "host": os.uname().nodename,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA,
        "incumbent_assignment_sha256": EXPECTED_ASSIGNMENT_SHA,
        "incumbent_exact_physical_bytes": ENVELOPE,
        "incumbent_model_prediction": incumbent_pred,
        "canonical_reproduction_max_abs_error": max(errors.values()),
        "eligible_layers": list(ELIGIBLE),
        "measured_layers": list(MEASURED),
        "candidate": QTIP_TIER,
        "hint_qtip_layers": sorted(hint_qtip_layers),
        "started_unix": started,
    }
    atomic_json(OUT / "PROGRESS.json", status_receipt)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP backend unavailable")
    solver.SetTimeLimit(int(TIME_LIMIT_SECONDS * 1000))
    solver.SetNumThreads(1)
    solver.SetSolverSpecificParametersAsString(
        "parallel/maxnthreads = 1\n"
        "randomization/randomseedshift = 0\n"
        "limits/gap = 0.0001\n"
        "display/verblevel = 4\n"
    )

    variables = {}
    options = {}
    old_option_key = {}
    qtip_layer_vars = (
        {layer: solver.BoolVar(f"qtip2_all512_L{layer:03d}") for layer in ELIGIBLE}
        if arm == "WITH" else {}
    )
    byte_constraint = solver.RowConstraint(-solver.infinity(), 0.0, "exact_physical_envelope_delta_le_zero")
    class_constraints = {
        c: solver.RowConstraint(
            0.0,
            float(incumbent_pred["code"] if c == "code" else step0[c]),
            f"class_nonnegative_and_shipped_cap_{c}",
        )
        for c in gs.CLASSES
    }
    objective = solver.Objective()
    objective.SetMinimization()
    global_weights = weighted_global_weights(json.loads((INPUTS / "baseline" / "BQ3_STEP0_PER_CLASS.json").read_text()), gs.CLASSES)

    cell_by_key = {tuple(cell["key"]): cell for cell in cells}
    old_global_model = math.fsum(global_weights[c] * incumbent_pred[c] for c in gs.CLASSES)
    hints_vars, hints_vals = [], []

    for key, cell in cell_by_key.items():
        layer, expert, projection = key
        old_tier = incumbent[key]
        local_vars = []
        by_tier = {opt["tier"]: opt for opt in cell["options"]}
        if old_tier not in by_tier:
            raise RuntimeError({"incumbent_tier_missing": key, "tier": old_tier})
        old_opt = by_tier[old_tier]
        for opt in cell["options"]:
            tier = opt["tier"]
            v = solver.BoolVar(f"L{layer:03d}_E{expert:03d}_{projection}_{tier}")
            variables[(key, tier)] = v
            options[(key, tier)] = opt
            local_vars.append(v)
            byte_constraint.SetCoefficient(v, int(opt["bytes"]) - int(old_opt["bytes"]))
            for c in gs.CLASSES:
                class_constraints[c].SetCoefficient(v, float(opt["costs"][c]))
            objective.SetCoefficient(
                v,
                math.fsum(float(opt["costs"][c]) for c in gs.CLASSES) / len(gs.CLASSES),
            )
            hints_vars.append(v)
            hints_vals.append(1.0 if tier == old_tier and layer not in hint_qtip_layers else 0.0)
            if tier == old_tier:
                old_option_key[key] = (key, tier)

        if arm == "WITH" and layer in ELIGIBLE:
            surf = delta_surface[layer]
            qcost = {c: float(old_opt["costs"][c]) + float(surf["delta_by_class"][c]) / 512.0 for c in gs.CLASSES}
            qopt = {"tier": QTIP_TIER, "bytes": QTIP_BYTES[projection], "costs": qcost}
            v = solver.BoolVar(f"L{layer:03d}_E{expert:03d}_{projection}_{QTIP_TIER}")
            variables[(key, QTIP_TIER)] = v
            options[(key, QTIP_TIER)] = qopt
            local_vars.append(v)
            byte_constraint.SetCoefficient(v, int(qopt["bytes"]) - int(old_opt["bytes"]))
            for c in gs.CLASSES:
                class_constraints[c].SetCoefficient(v, qcost[c])
            objective.SetCoefficient(
                v,
                math.fsum(qcost[c] for c in gs.CLASSES) / len(gs.CLASSES),
            )
            group = solver.RowConstraint(0.0, 0.0, f"qtip2_group_L{layer:03d}_E{expert:03d}_{projection}")
            group.SetCoefficient(v, 1.0)
            group.SetCoefficient(qtip_layer_vars[layer], -1.0)
            hints_vars.append(v); hints_vals.append(1.0 if layer in hint_qtip_layers else 0.0)
        one = solver.RowConstraint(1.0, 1.0, f"one_L{layer:03d}_E{expert:03d}_{projection}")
        for v in local_vars:
            one.SetCoefficient(v, 1.0)

    for layer, v in qtip_layer_vars.items():
        hints_vars.append(v); hints_vals.append(1.0 if layer in hint_qtip_layers else 0.0)
    solver.SetHint(hints_vars, hints_vals)

    status_code = solver.Solve()
    status_names = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status = status_names.get(status_code, str(status_code))
    if status_code not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        atomic_json(OUT / "DONE.json", {"status": status, "elapsed_seconds": time.time() - started})
        return 2

    selected = {}
    predicted = {c: 0.0 for c in gs.CLASSES}
    byte_delta = 0
    qtip_layers = []
    tier_counts = Counter()
    changed_cells = []
    for key, cell in cell_by_key.items():
        old_tier = incumbent[key]
        old_opt = next(o for o in cell["options"] if o["tier"] == old_tier)
        picked = None
        for (k, tier), v in variables.items():
            if k == key and v.solution_value() > 0.5:
                picked = options[(k, tier)]
                break
        if picked is None:
            raise RuntimeError({"unselected_cell": key})
        tier = picked["tier"]
        selected[key] = tier
        tier_counts[tier] += 1
        byte_delta += int(picked["bytes"]) - int(old_opt["bytes"])
        for c in gs.CLASSES:
            predicted[c] += float(picked["costs"][c])
        if tier != old_tier:
            changed_cells.append({"layer": key[0], "expert": key[1], "projection": key[2], "from": old_tier, "to": tier, "delta_bytes": int(picked["bytes"]) - int(old_opt["bytes"])})
    for layer in ELIGIBLE:
        count = sum(1 for (l, _e, _p), tier in selected.items() if l == layer and tier == QTIP_TIER)
        if count not in (0, 512):
            raise RuntimeError({"partial_qtip2_layer": layer, "units": count})
        if count == 512:
            qtip_layers.append(layer)

    final_bytes = ENVELOPE + byte_delta
    if final_bytes > ENVELOPE:
        raise RuntimeError({"byte_gate_failed": final_bytes})
    objective_value = math.fsum(predicted[c] for c in gs.CLASSES) / len(gs.CLASSES)
    incumbent_objective = math.fsum(incumbent_pred[c] for c in gs.CLASSES) / len(gs.CLASSES)
    if objective_value > incumbent_objective + 1e-10:
        raise RuntimeError({"nonregression_failed": {"with": objective_value, "without": incumbent_objective}})

    model_global = objective_value
    old_global_model = incumbent_objective
    canon_global = CANON_GLOBAL_BASELINE + (model_global - old_global_model)
    canon_code = CANON_CODE_BASELINE + (predicted["code"] - incumbent_pred["code"])

    assignment_map = {
        str(layer): {
            str(expert): {p: selected[(layer, expert, p)] for p in gs.PROJECTIONS}
            for expert in range(gs.EXPERTS)
        }
        for layer in range(gs.LAYERS)
    }
    assignment_map_text = json.dumps(assignment_map, sort_keys=True, separators=(",", ":")).encode()
    assignment_map_sha = hashlib.sha256(assignment_map_text).hexdigest()
    assignment_receipt = {
        "schema": "p629-class-balanced-global-assignment-v1",
        "arm": arm,
        "measurement_label": "PREDICTION_ONLY_ARTIFACT_RELATIVE_SWAP_BASIS",
        "assignment": assignment_map,
        "assignment_map_sha256": assignment_map_sha,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA,
        "incumbent_assignment_sha256": EXPECTED_ASSIGNMENT_SHA,
    }
    assignment_receipt_sha = atomic_json(OUT / f"ASSIGNMENT_{arm}.json", assignment_receipt)

    qtip_gross_freed = 0
    for layer in qtip_layers:
        old_layer = sum(
            next(o for o in cell_by_key[(layer, e, p)]["options"] if o["tier"] == incumbent[(layer, e, p)])["bytes"]
            for e in range(gs.EXPERTS) for p in gs.PROJECTIONS
        )
        qtip_gross_freed += int(old_layer) - QTIP_LAYER_BYTES
    net_freed = ENVELOPE - final_bytes
    reallocated = qtip_gross_freed - net_freed

    elapsed = time.time() - started
    result = {
        "schema": "p629-canonical-class-balanced-global-solve-v1",
        "arm": arm,
        "status": status,
        "solver": "OR-Tools MPSolver + SCIP; one thread; incumbent-hinted; 678s limit",
        "elapsed_seconds": elapsed,
        "measurement_label": "PREDICTION_ONLY_ARTIFACT_RELATIVE_SWAP_BASIS",
        "scientific_transfer_claim": False,
        "objective": {
            "name": "CLASS-BALANCED GLOBAL: uniform mean of six per-class predicted KLDs",
            "without": incumbent_objective,
            "with": objective_value,
            "delta_with_minus_without": objective_value - incumbent_objective,
            "nonregression_pass": objective_value <= incumbent_objective + 1e-10,
            "solver_value": solver.Objective().Value(),
            "best_bound": solver.Objective().BestBound(),
        },
        "bytes": {
            "envelope": ENVELOPE,
            "without_exact": ENVELOPE,
            "with_exact": final_bytes,
            "net_freed": net_freed,
            "qtip2_gross_freed": qtip_gross_freed,
            "reallocated_to_existing_tiers": reallocated,
            "delta_with_minus_without": byte_delta,
        },
        "prediction": {
            "canonical_solver_model_by_class": predicted,
            "canonical_solver_incumbent_by_class": incumbent_pred,
            "model_delta_by_class": {c: predicted[c] - incumbent_pred[c] for c in gs.CLASSES},
            "model_class_balanced_global": model_global,
            "model_incumbent_class_balanced_global": old_global_model,
            "canon_adjusted_global_from_0.08395": canon_global,
            "canon_adjusted_code_from_sealed_0.04170458531457378": canon_code,
        },
        "qtip2": {
            "candidate": QTIP_TIER,
            "rate_bpw": 2.0117,
            "eligible_layers": list(ELIGIBLE),
            "measured_layers": list(MEASURED),
            "family_mean_layers": [x for x in ELIGIBLE if x not in MEASURED],
            "selected_layers": qtip_layers,
            "selected_units": len(qtip_layers) * 512,
            "exact_logical_bytes_per_layer": QTIP_LAYER_BYTES,
            "family_mean_delta_by_class": family_mean,
            "family_mean_global_delta": family_global,
            "price_table": [delta_surface[x] for x in ELIGIBLE],
        },
        "tier_counts": dict(sorted(tier_counts.items())),
        "changed_cell_count": len(changed_cells),
        "changed_cells": changed_cells,
        "assignment_receipt": str(OUT / f"ASSIGNMENT_{arm}.json"),
        "assignment_receipt_sha256": assignment_receipt_sha,
        "assignment_map_sha256": assignment_map_sha,
        "reproduction_gate": {
            "max_abs_error": max(errors.values()),
            "per_class_abs_error": errors,
            "tier_counts_exact": True,
            "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA,
            "incumbent_assignment_sha256": EXPECTED_ASSIGNMENT_SHA,
            "old_assignment_feasible": True,
        },
        "input_receipts": {
            "manifest": {"path": str(INPUTS / "INPUT_MANIFEST.json"), "sha256": EXPECTED_INPUT_MANIFEST_SHA},
            "verified_manifest_files": verified_inputs,
            "original_solver": {"path": str(LINEAGE / "original_genesis_code_solve.py"), "sha256": EXPECTED_ORIGINAL_SOLVER_SHA},
            "incumbent_assignment": {"path": str(LINEAGE / "NOMINATED_ASSIGNMENT.json"), "sha256": EXPECTED_ASSIGNMENT_SHA},
            "anchors": measured_rows,
        },
        "normalization_receipt": normalization,
        "correction_fit": correction_fit,
        "started_unix": started,
        "finished_unix": time.time(),
    }
    result_sha = atomic_json(OUT / f"{arm}_RESULT.json", result)
    done = {
        "status": f"PASS_{arm}_CLASS_BALANCED_GLOBAL_PREDICTION_ONLY",
        "arm": arm,
        "result_sha256": result_sha,
        "assignment_receipt_sha256": assignment_receipt_sha,
        "assignment_map_sha256": assignment_map_sha,
        "exact_bytes": final_bytes,
        "objective": objective_value,
        "qtip2_selected_layers": qtip_layers,
        "elapsed_seconds": elapsed,
    }
    atomic_json(OUT / "DONE.json", done)
    print(json.dumps(done, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
