#!/usr/bin/env python3
"""P637 actual class-balanced per-expert QTIP2 backpack solve on compute-node-3.

This is the minimal merge of P629's uniform-six-class objective/ceilings and
P634's exact ordinary per-expert QTIP2 menu/pricing.  The immutable P629
WITHOUT reproduction is the warm start and comparison arm.  QTIP2 remains a
normal per-cell option; no whole-layer binary, refit, or zero-flooring is used.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ortools.linear_solver import pywraplp

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
LINEAGE = ROOT / "lineage"
ANCHORS = ROOT / "anchors"
OUT = ROOT / "bound"
LOGS = ROOT / "logs"

ENVELOPE = 101_346_700_411
EXPECTED_INPUT_MANIFEST_SHA = "88ebfc21a7134088cad0a9a4f09821410db29ac13cd754d4d46f8902bebecb42"
EXPECTED_ASSIGNMENT_SHA = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
EXPECTED_ORIGINAL_SOLVER_SHA = "81e2c5eb54a14c978c8324373be6d4c0a2f96e518427b8f26c5e44a33c39a68d"
EXPECTED_P620_RESULT_SHA = "758da87e3a73e3115e8a9eddb50ddfb146924aaebd2811e3acfd91c15013e848"
EXPECTED_STEP1_SHA = "0928bb3b923c12c758adb40ffc7f57cda3a17d79f23444de05dab370041d45ce"
EXPECTED_EXISTING_MENU_SHA = "9b4b46a2fa694aa7fbdbdbf03273eeece6b3887a4b20b58d83003d8786a8d5b7"
EXPECTED_D4_K256_CELLS = 2_744
MEASURED = (0, 2, 4, 6, 11, 14, 16, 19, 22)
ELIGIBLE = (0, 2, 4, 6, 11, 14, 16, 19, 22, 25, 27, 30, 34, 35, 38, 42)
QTIP_TIER = "qtip2_2.0117"
TIME_LIMIT_SECONDS = 900.0
SOLVER_THREADS = 16
CANON_GLOBAL_BASELINE = 0.08395
CANON_CODE_BASELINE = 0.04170458531457378
LP_BYTE_SCALE = 1_000_000.0
LP_COST_SCALE = 1_000_000.0

# Exact physical .pt bytes per expert-projection from the sealed QTIP2 build
# manifest.  L022's sealed physical total closes the two values used for the
# remaining same-geometry/two-digit rep-16 layers.
QTIP_PHYSICAL_BYTES_BY_LAYER = {
    0: {"fused13": 4_213_837, "down": 2_112_559},
    2: {"fused13": 4_213_837, "down": 2_112_559},
    4: {"fused13": 4_213_837, "down": 2_112_559},
    6: {"fused13": 4_213_837, "down": 2_112_331},
    11: {"fused13": 4_213_837, "down": 2_112_559},
    14: {"fused13": 4_213_837, "down": 2_112_559},
    16: {"fused13": 4_213_837, "down": 2_112_559},
    19: {"fused13": 4_213_609, "down": 2_112_331},
    22: {"fused13": 4_213_609, "down": 2_112_331},
    25: {"fused13": 4_213_609, "down": 2_112_331},
    27: {"fused13": 4_213_609, "down": 2_112_331},
    30: {"fused13": 4_213_609, "down": 2_112_331},
    34: {"fused13": 4_213_609, "down": 2_112_331},
    35: {"fused13": 4_213_609, "down": 2_112_331},
    38: {"fused13": 4_213_609, "down": 2_112_331},
    42: {"fused13": 4_213_609, "down": 2_112_331},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    payload = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha256(path)


def load_original():
    p = LINEAGE / "original_banana_smasher_code_solve.py"
    if sha256(p) != EXPECTED_ORIGINAL_SOLVER_SHA:
        raise RuntimeError("original solver SHA drift")
    spec = importlib.util.spec_from_file_location("banana_smasher_original", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def validate_inputs():
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
        logical = int(q.get("logical_bytes_exact", q.get("logical_bytes")))
        physical = int(q.get("physical_bytes_exact", q.get("physical_bytes")))
        units = int(q.get("units", q.get("projection_units")))
        expected_physical = sum(QTIP_PHYSICAL_BYTES_BY_LAYER[layer].values()) * gs.EXPERTS
        if logical != 1_617_954_816 or units != 512 or physical != expected_physical:
            raise RuntimeError({
                "anchor_byte_drift": str(p),
                "logical": logical,
                "physical": physical,
                "expected_physical": expected_physical,
                "units": units,
            })
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
            "anchor_path": str(p),
            "anchor_bytes": p.stat().st_size,
            "anchor_sha256": sha256(p),
            "qtip_physical_bytes_layer": physical,
            "qtip_physical_bytes_per_projection": dict(QTIP_PHYSICAL_BYTES_BY_LAYER[layer]),
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
                "qtip_physical_bytes_layer": sum(QTIP_PHYSICAL_BYTES_BY_LAYER[layer].values()) * gs.EXPERTS,
                "qtip_physical_bytes_per_projection": dict(QTIP_PHYSICAL_BYTES_BY_LAYER[layer]),
            }
    return by_layer, rows, family_mean, family_global


def weighted_global_weights(step0_doc, classes):
    by = step0_doc["by_class"]
    counts = {c: int(by[c]["n_positions"]) for c in classes}
    total = sum(counts.values())
    return {c: counts[c] / total for c in classes}


def build_qtip_surface(gs, importance, corrections, delta_surface, global_weights):
    """Apportion each measured layer row by the existing BANANA_SMASHER mass law."""
    class_increment = {}
    global_increment = {}
    closure = []
    for layer in ELIGIBLE:
        denominators = {}
        mass = {}
        for c in gs.CLASSES:
            vals = {}
            for expert in range(gs.EXPERTS):
                for projection in gs.PROJECTIONS:
                    key = (layer, expert, projection)
                    vals[key] = (
                        importance[(layer, expert, c)]
                        * gs.PROJECTION_WEIGHTS[projection]
                        * corrections[c][projection]
                    )
            denom = math.fsum(vals.values())
            if not math.isfinite(denom) or denom <= 0:
                raise RuntimeError({"bad_mass_denominator": [layer, c], "value": denom})
            denominators[c] = denom
            mass[c] = vals
        global_mass = {}
        for expert in range(gs.EXPERTS):
            for projection in gs.PROJECTIONS:
                key = (layer, expert, projection)
                global_mass[key] = math.fsum(global_weights[c] * mass[c][key] for c in gs.CLASSES)
        global_denom = math.fsum(global_mass.values())
        for expert in range(gs.EXPERTS):
            for projection in gs.PROJECTIONS:
                key = (layer, expert, projection)
                class_increment[key] = {
                    c: float(delta_surface[layer]["delta_by_class"][c]) * mass[c][key] / denominators[c]
                    for c in gs.CLASSES
                }
                global_increment[key] = float(delta_surface[layer]["global_delta"]) * global_mass[key] / global_denom
        class_sums = {
            c: math.fsum(class_increment[(layer, e, p)][c] for e in range(gs.EXPERTS) for p in gs.PROJECTIONS)
            for c in gs.CLASSES
        }
        global_sum = math.fsum(global_increment[(layer, e, p)] for e in range(gs.EXPERTS) for p in gs.PROJECTIONS)
        class_errors = {c: class_sums[c] - float(delta_surface[layer]["delta_by_class"][c]) for c in gs.CLASSES}
        global_error = global_sum - float(delta_surface[layer]["global_delta"])
        if max([abs(global_error), *(abs(v) for v in class_errors.values())]) > 1e-12:
            raise RuntimeError({"mass_law_closure_failed": layer, "class_errors": class_errors, "global_error": global_error})
        closure.append({
            "layer": layer,
            "basis": delta_surface[layer]["basis"],
            "class_sum_errors": class_errors,
            "global_sum_error": global_error,
            "class_mass_denominators": denominators,
            "global_mass_denominator": global_denom,
        })
    return class_increment, global_increment, closure


def option_by_tier(cell):
    return {str(opt["tier"]): opt for opt in cell["options"]}


def existing_menu_sha256(cells):
    digest = hashlib.sha256()
    for cell in cells:
        row = {"key": list(cell["key"]), "options": cell["options"]}
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    gs = load_original()
    _manifest_doc, verified_inputs = validate_inputs()

    anchors_grid = gs.load_anchor_grid(INPUTS / "rung1" / "ANCHOR_VERTICAL_GRID.csv")
    rows = gs.load_profile(INPUTS / "profile" / "PROFILE_ROWS.jsonl")
    importance, normalization = gs.normalize_profile_rows(rows)
    step0_path = INPUTS / "baseline" / "BQ3_STEP0_PER_CLASS.json"
    step0_doc = json.loads(step0_path.read_text())
    step0 = gs.step0_means(step0_path)
    old_incumbent, _ = gs.map_incumbent(INPUTS / "baseline" / "DUALVQ_K4096MENU_BQ3_BIN_MANIFEST.json")
    corrections, correction_fit = gs.fit_projection_corrections(old_incumbent, importance, anchors_grid, step0)
    cells = gs.make_cells(importance, anchors_grid, corrections)
    assignment_doc, incumbent = read_assignment(gs)
    delta_surface, measured_rows, family_mean, family_global = load_anchor_deltas(gs)
    global_weights = weighted_global_weights(step0_doc, gs.CLASSES)
    qtip_class_increment, qtip_global_increment, pricing_closure = build_qtip_surface(
        gs, importance, corrections, delta_surface, global_weights
    )

    # STEP 1 fail-closed canonical replay.
    incumbent_pred = gs.predict_assignment(incumbent, importance, anchors_grid, gs.CLASSES, corrections=corrections)
    lineage = json.loads((LINEAGE / "FRONTIER.json").read_text())
    expected_pred = lineage["nomination"]["predicted_class_mean_kld"]
    errors = {c: abs(incumbent_pred[c] - float(expected_pred[c])) for c in gs.CLASSES}
    expected_counts = Counter(lineage["nomination"]["tier_counts"])
    actual_counts = Counter(incumbent.values())
    incumbent_objective = math.fsum(incumbent_pred[c] for c in gs.CLASSES) / len(gs.CLASSES)
    p620_result_path = LINEAGE / "P620_WITH_QTIP2_RESULT.json"
    p620_result = json.loads(p620_result_path.read_text())
    p620_result_sha = sha256(p620_result_path)
    step1_path = ROOT / "receipts" / "STEP1_WITHOUT_REPRO.json"
    step1 = json.loads(step1_path.read_text())
    step1_sha = sha256(step1_path)
    menu_sha = existing_menu_sha256(cells)
    lineage_nomination_wire_bytes = int(lineage["nomination"]["total_wire_bytes"])
    canonical_incumbent_bytes = int(p620_result["bytes"]["without_exact"])
    sanity_checks = {
        "input_manifest_sha_exact": sha256(INPUTS / "INPUT_MANIFEST.json") == EXPECTED_INPUT_MANIFEST_SHA,
        "incumbent_assignment_sha_exact": sha256(LINEAGE / "NOMINATED_ASSIGNMENT.json") == EXPECTED_ASSIGNMENT_SHA,
        "incumbent_assignment_objective_exact": max(errors.values()) <= 1e-12,
        "incumbent_tier_counts_exact": actual_counts == expected_counts,
        "p620_result_sha_exact": p620_result_sha == EXPECTED_P620_RESULT_SHA,
        "p629_step1_receipt_sha_exact": step1_sha == EXPECTED_STEP1_SHA,
        "p629_step1_status_pass": step1["status"] == "PASS" and all(step1["checks"].values()),
        "existing_menu_sha_exact_before_qtip": menu_sha == step1["existing_menu_sha256"] == EXPECTED_EXISTING_MENU_SHA,
        "incumbent_exact_bytes": canonical_incumbent_bytes == ENVELOPE,
        "incumbent_objective_matches_p629_without": abs(incumbent_objective - float(step1["objective"])) <= 1e-15,
        "incumbent_d4_k256_cells_2744": actual_counts["d4_k256"] == EXPECTED_D4_K256_CELLS,
    }
    sanity = {
        "schema": "p637-merged-step1-sanity-v1",
        "status": "PASS" if all(sanity_checks.values()) else "FAIL",
        "checks": sanity_checks,
        "host": os.uname().nodename,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA,
        "incumbent_assignment_sha256": EXPECTED_ASSIGNMENT_SHA,
        "incumbent_exact_physical_bytes": canonical_incumbent_bytes,
        "lineage_nomination_wire_bytes_before_canonical_exact_overhead": lineage_nomination_wire_bytes,
        "canonical_exact_overhead_vs_lineage_nomination": canonical_incumbent_bytes - lineage_nomination_wire_bytes,
        "p620_result_sha256": p620_result_sha,
        "p629_step1_receipt_sha256": step1_sha,
        "existing_menu_sha256": menu_sha,
        "incumbent_objective": incumbent_objective,
        "objective_name": "CLASS-BALANCED GLOBAL: uniform mean of six per-class predicted KLDs",
        "incumbent_prediction_by_class": incumbent_pred,
        "canonical_reproduction_abs_errors": errors,
        "incumbent_tier_counts": dict(sorted(actual_counts.items())),
        "d4_k256_per_expert_projection_cells": actual_counts["d4_k256"],
        "d4_k256_nominal_index_bpw": 2.0,
        "d4_k256_canonical_menu_physical_rate_bpw": 2.25,
        "qtip2_ordinary_candidate_cells": len(ELIGIBLE) * gs.EXPERTS * len(gs.PROJECTIONS),
        "finished_unix": time.time(),
    }
    sanity_sha = atomic_json(OUT / "SANITY.json", sanity)
    if sanity["status"] != "PASS":
        atomic_json(OUT / "DONE.json", {"status": "STOP_STEP1_SANITY_FAIL", "sanity_sha256": sanity_sha})
        return 2
    if os.environ.get("P637_SANITY_ONLY") == "1":
        print(json.dumps({"status": "PASS_STEP1_SANITY_ONLY", "sanity_sha256": sanity_sha}, sort_keys=True))
        return 0

    cell_by_key = {tuple(cell["key"]): cell for cell in cells}
    qtip_options = {}
    for key, cell in cell_by_key.items():
        layer, _expert, projection = key
        if layer not in ELIGIBLE:
            continue
        old_tier = incumbent[key]
        old_opt = option_by_tier(cell)[old_tier]
        qtip_options[key] = {
            "tier": QTIP_TIER,
            "bytes": int(QTIP_PHYSICAL_BYTES_BY_LAYER[layer][projection]),
            "costs": {
                c: float(old_opt["costs"][c]) + float(qtip_class_increment[key][c])
                for c in gs.CLASSES
            },
            "incremental_global": float(qtip_global_increment[key]),
            "pricing_basis": delta_surface[layer]["basis"],
        }

    # Deterministic QTIP-containing feasible seed, derived only by one-cell moves
    # from the incumbent.  This is the machine-readable first incumbent.
    seed = dict(incumbent)
    seed_pred = dict(incumbent_pred)
    seed_byte_delta = 0
    seed_qtip = []
    candidates = sorted(
        qtip_options,
        key=lambda key: (
            math.fsum(qtip_class_increment[key][c] for c in gs.CLASSES) / len(gs.CLASSES),
            key,
        ),
    )
    for key in candidates:
        objective_delta = math.fsum(qtip_class_increment[key][c] for c in gs.CLASSES) / len(gs.CLASSES)
        if objective_delta >= 0.0:
            continue
        old_opt = option_by_tier(cell_by_key[key])[incumbent[key]]
        delta_bytes = int(qtip_options[key]["bytes"]) - int(old_opt["bytes"])
        candidate_pred = {c: seed_pred[c] + qtip_class_increment[key][c] for c in gs.CLASSES}
        if ENVELOPE + seed_byte_delta + delta_bytes > ENVELOPE:
            continue
        if any(
            candidate_pred[c] < -1e-15
            or candidate_pred[c] > (incumbent_pred["code"] if c == "code" else step0[c]) + 1e-12
            for c in gs.CLASSES
        ):
            continue
        seed[key] = QTIP_TIER
        seed_pred = candidate_pred
        seed_byte_delta += delta_bytes
        seed_qtip.append(key)

    # Re-spend the QTIP2-freed bytes on ordinary existing-menu upgrades before
    # handing the incumbent to SCIP.  This is only a deterministic warm-start
    # heuristic: the SCIP model below still contains the complete ordinary
    # per-cell menu and exactly the same primary six-class-mean objective and
    # global constraints.  In particular, there is no per-cell/per-move code
    # veto; every trial is adjudicated against the aggregate shipped ceilings.
    respend_candidates = []
    for key, cell in cell_by_key.items():
        if seed[key] == QTIP_TIER:
            continue
        by_tier = option_by_tier(cell)
        current = by_tier[seed[key]]
        for opt in cell["options"]:
            if str(opt["tier"]) == seed[key]:
                continue
            delta_bytes = int(opt["bytes"]) - int(current["bytes"])
            delta_by_class = {
                c: float(opt["costs"][c]) - float(current["costs"][c])
                for c in gs.CLASSES
            }
            objective_delta = math.fsum(delta_by_class.values()) / len(gs.CLASSES)
            if delta_bytes <= 0 or objective_delta >= 0.0:
                continue
            respend_candidates.append((objective_delta / delta_bytes, objective_delta, key, opt, delta_bytes, delta_by_class))
    respend_candidates.sort(key=lambda row: (row[0], row[1], row[2], str(row[3]["tier"])))
    seed_respend = []
    moved = set()
    for _ratio, objective_delta, key, opt, delta_bytes, delta_by_class in respend_candidates:
        if key in moved or ENVELOPE + seed_byte_delta + delta_bytes > ENVELOPE:
            continue
        candidate_pred = {c: seed_pred[c] + delta_by_class[c] for c in gs.CLASSES}
        if any(
            candidate_pred[c] < -1e-15
            or candidate_pred[c] > (incumbent_pred["code"] if c == "code" else step0[c]) + 1e-12
            for c in gs.CLASSES
        ):
            continue
        old_tier = seed[key]
        seed[key] = str(opt["tier"])
        seed_pred = candidate_pred
        seed_byte_delta += delta_bytes
        moved.add(key)
        seed_respend.append({
            "layer": key[0], "expert": key[1], "projection": key[2],
            "from": old_tier, "to": str(opt["tier"]),
            "delta_bytes": delta_bytes, "objective_delta": objective_delta,
        })

    seed_objective = math.fsum(seed_pred[c] for c in gs.CLASSES) / len(gs.CLASSES)
    if not seed_qtip or seed_objective > incumbent_objective + 1e-12:
        raise RuntimeError({"qtip_seed_failed": {"qtip_cells": len(seed_qtip), "objective": seed_objective}})
    seed_layer_counts = Counter(k[0] for k in seed_qtip)
    seed_into_counts = Counter(incumbent[k] for k in seed_qtip)
    seed_doc = {
        "schema": "p637-first-incumbent-v1",
        "status": "FEASIBLE_QTIP2_SEED",
        "derived_from_incumbent": EXPECTED_ASSIGNMENT_SHA,
        "objective": seed_objective,
        "incumbent_objective": incumbent_objective,
        "objective_delta": seed_objective - incumbent_objective,
        "exact_bytes": ENVELOPE + seed_byte_delta,
        "byte_delta": seed_byte_delta,
        "qtip2_selected_cells": len(seed_qtip),
        "qtip2_selected_by_layer": {str(k): seed_layer_counts[k] for k in sorted(seed_layer_counts)},
        "qtip2_into_by_from_tier": dict(sorted(seed_into_counts.items())),
        "reallocated_to_existing_tiers": sum(int(row["delta_bytes"]) for row in seed_respend),
        "respend_move_count": len(seed_respend),
        "respend_moves": seed_respend,
        "qtip2_selected_cell_keys": [
            {"layer": k[0], "expert": k[1], "projection": k[2], "from": incumbent[k]}
            for k in seed_qtip
        ],
        "prediction_by_class": seed_pred,
        "incumbent_prediction_by_class": incumbent_pred,
        "best_bound": 0.0,
        "relative_gap": (seed_objective - 0.0) / max(abs(seed_objective), 1e-30),
        "selection_evidence": "deterministic class-balanced feasible warm start derived cellwise from sealed WITHOUT",
        "sanity_sha256": sanity_sha,
        "created_unix": time.time(),
    }
    seed_sha = atomic_json(OUT / "FIRST_INCUMBENT.json", seed_doc)
    atomic_json(OUT / "PROGRESS.json", {
        **seed_doc,
        "schema": "p637-progress-v1",
        "status": "BUILDING_SCIP_MODEL",
        "seed_sha256": seed_sha,
        "solver_pid": os.getpid(),
        "solver_pgid": os.getpgid(0),
        "updated_unix": time.time(),
    })

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("GLOP backend unavailable")
    solver.SetTimeLimit(int(TIME_LIMIT_SECONDS * 1000))
    solver.SetSolverSpecificParametersAsString(
        "use_dual_simplex: true\n"
        "primal_feasibility_tolerance: 1e-9\n"
        "dual_feasibility_tolerance: 1e-9\n"
        "solution_feasibility_tolerance: 1e-8\n"
    )

    vars_by_key = {}
    options = {}
    byte_constraint = solver.RowConstraint(-solver.infinity(), 0.0, "exact_physical_envelope_delta_le_zero")
    class_constraints = {
        c: solver.RowConstraint(
            0.0,
            float(incumbent_pred["code"] if c == "code" else step0[c]) * LP_COST_SCALE,
            f"class_nonnegative_and_shipped_cap_{c}",
        )
        for c in gs.CLASSES
    }
    objective = solver.Objective()
    objective.SetMinimization()
    hints_vars, hints_vals = [], []

    for key, cell in cell_by_key.items():
        old_tier = incumbent[key]
        by_tier = option_by_tier(cell)
        if old_tier not in by_tier:
            raise RuntimeError({"incumbent_tier_missing": key, "tier": old_tier})
        old_opt = by_tier[old_tier]
        local = []
        candidate_options = list(cell["options"])
        if key in qtip_options:
            candidate_options.append(qtip_options[key])
        for opt in candidate_options:
            tier = str(opt["tier"])
            v = solver.NumVar(0.0, 1.0, f"L{key[0]:03d}_E{key[1]:03d}_{key[2]}_{tier}")
            local.append((tier, v, opt))
            options[(key, tier)] = opt
            byte_constraint.SetCoefficient(v, (int(opt["bytes"]) - int(old_opt["bytes"])) / LP_BYTE_SCALE)
            for c in gs.CLASSES:
                class_constraints[c].SetCoefficient(v, float(opt["costs"][c]) * LP_COST_SCALE)
            objective.SetCoefficient(
                v,
                math.fsum(float(opt["costs"][c]) for c in gs.CLASSES) / len(gs.CLASSES) * LP_COST_SCALE,
            )
            hints_vars.append(v)
            hints_vals.append(1.0 if tier == seed[key] else 0.0)
        one = solver.RowConstraint(1.0, 1.0, f"one_L{key[0]:03d}_E{key[1]:03d}_{key[2]}")
        for _tier, v, _opt in local:
            one.SetCoefficient(v, 1.0)
        vars_by_key[key] = local
    solver.SetHint(hints_vars, hints_vals)
    solver.EnableOutput()

    atomic_json(OUT / "PROGRESS.json", {
        **seed_doc,
        "schema": "p637-progress-v1",
        "status": "SOLVING",
        "seed_sha256": seed_sha,
        "solver_pid": os.getpid(),
        "solver_pgid": os.getpgid(0),
        "model_variables": solver.NumVariables(),
        "model_constraints": solver.NumConstraints(),
        "threads": SOLVER_THREADS,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "updated_unix": time.time(),
    })

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

    lp_bound = {
        "schema": "p637-glop-lp-relaxation-bound-v1",
        "status": status,
        "valid_lower_bound_for_integer_primary_objective": status == "OPTIMAL",
        "objective_lower_bound": solver.Objective().Value() / LP_COST_SCALE,
        "byte_scale": LP_BYTE_SCALE,
        "cost_scale": LP_COST_SCALE,
        "variables": solver.NumVariables(),
        "constraints": solver.NumConstraints(),
        "elapsed_seconds": time.time() - started,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA,
        "incumbent_assignment_sha256": EXPECTED_ASSIGNMENT_SHA,
        "existing_menu_sha256": EXPECTED_EXISTING_MENU_SHA,
        "p629_step1_receipt_sha256": EXPECTED_STEP1_SHA,
        "created_unix": time.time(),
    }
    bound_sha = atomic_json(OUT / "LP_BOUND.json", lp_bound)
    print(json.dumps({"path": str(OUT / "LP_BOUND.json"), "sha256": bound_sha, **lp_bound}, sort_keys=True))
    return 0

    selected = {}
    predicted = {c: 0.0 for c in gs.CLASSES}
    byte_delta = 0
    tier_counts = Counter()
    changed_cells = []
    qtip_global_direct = 0.0
    qtip_global_reconstructed = 0.0
    for key, local in vars_by_key.items():
        old_tier = incumbent[key]
        old_opt = option_by_tier(cell_by_key[key])[old_tier]
        picked = [(tier, opt) for tier, v, opt in local if v.solution_value() > 0.5]
        if len(picked) != 1:
            raise RuntimeError({"selection_cardinality": key, "picked": [x[0] for x in picked]})
        tier, opt = picked[0]
        selected[key] = tier
        tier_counts[tier] += 1
        delta_bytes = int(opt["bytes"]) - int(old_opt["bytes"])
        byte_delta += delta_bytes
        for c in gs.CLASSES:
            predicted[c] += float(opt["costs"][c])
        if tier == QTIP_TIER:
            qtip_global_direct += qtip_global_increment[key]
            qtip_global_reconstructed += math.fsum(global_weights[c] * qtip_class_increment[key][c] for c in gs.CLASSES)
        if tier != old_tier:
            changed_cells.append({
                "layer": key[0], "expert": key[1], "projection": key[2],
                "from": old_tier, "to": tier, "delta_bytes": delta_bytes,
                "delta_by_class": {c: float(opt["costs"][c]) - float(old_opt["costs"][c]) for c in gs.CLASSES},
            })

    final_bytes = ENVELOPE + byte_delta
    objective_value = math.fsum(predicted[c] for c in gs.CLASSES) / len(gs.CLASSES)
    best_bound = solver.Objective().BestBound()
    relative_gap = max(0.0, objective_value - best_bound) / max(abs(objective_value), 1e-30)
    if final_bytes > ENVELOPE:
        raise RuntimeError({"byte_gate_failed": final_bytes})
    if objective_value > incumbent_objective + 1e-10:
        raise RuntimeError({"nonregression_failed": {"with": objective_value, "without": incumbent_objective}})

    qtip_selected = [key for key, tier in selected.items() if tier == QTIP_TIER]
    qtip_by_layer = Counter(key[0] for key in qtip_selected)
    qtip_cell_rows = [
        {"layer": key[0], "expert": key[1], "projection": key[2], "from": incumbent[key]}
        for key in sorted(qtip_selected)
    ]
    qtip_experts_by_layer = {}
    for layer in ELIGIBLE:
        experts = defaultdict(list)
        for key in qtip_selected:
            if key[0] == layer:
                experts[key[1]].append(key[2])
        qtip_experts_by_layer[str(layer)] = {
            str(expert): sorted(projections) for expert, projections in sorted(experts.items())
        }
    transition_counts = Counter((r["from"], r["to"]) for r in changed_cells)
    qtip_into_counts = Counter(r["from"] for r in changed_cells if r["to"] == QTIP_TIER)
    qtip_out_counts = Counter(r["to"] for r in changed_cells if r["from"] == QTIP_TIER)
    qtip_transition_rows = [
        r for r in changed_cells if r["to"] == QTIP_TIER or r["from"] == QTIP_TIER
    ]
    qtip_byte_delta = sum(r["delta_bytes"] for r in qtip_transition_rows)
    qtip_gross_freed = sum(max(0, -int(r["delta_bytes"])) for r in qtip_transition_rows)
    qtip_gross_added = sum(max(0, int(r["delta_bytes"])) for r in qtip_transition_rows)
    existing_rows = [
        r for r in changed_cells if r["to"] != QTIP_TIER and r["from"] != QTIP_TIER
    ]
    existing_byte_delta = sum(int(r["delta_bytes"]) for r in existing_rows)
    existing_gross_freed = sum(max(0, -int(r["delta_bytes"])) for r in existing_rows)
    existing_gross_added = sum(max(0, int(r["delta_bytes"])) for r in existing_rows)

    old_global_model = math.fsum(global_weights[c] * incumbent_pred[c] for c in gs.CLASSES)
    weighted_class_global = math.fsum(global_weights[c] * predicted[c] for c in gs.CLASSES)
    model_global = weighted_class_global + qtip_global_direct - qtip_global_reconstructed
    canon_global = CANON_GLOBAL_BASELINE + (model_global - old_global_model)
    canon_code = CANON_CODE_BASELINE + (predicted["code"] - incumbent_pred["code"])

    assignment_map = {
        str(layer): {
            str(expert): {p: selected[(layer, expert, p)] for p in gs.PROJECTIONS}
            for expert in range(gs.EXPERTS)
        }
        for layer in range(gs.LAYERS)
    }
    assignment_map_sha = hashlib.sha256(
        json.dumps(assignment_map, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assignment_receipt = {
        "schema": "p637-class-balanced-per-expert-qtip2-assignment-v1",
        "measurement_label": "PREDICTION_ONLY_ARTIFACT_RELATIVE_SWAP_MASS_LAW",
        "assignment": assignment_map,
        "assignment_map_sha256": assignment_map_sha,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA,
        "incumbent_assignment_sha256": EXPECTED_ASSIGNMENT_SHA,
        "without_assignment_map_sha256": step1["assignment_map_sha256"],
        "existing_menu_sha256": menu_sha,
    }
    assignment_receipt_sha = atomic_json(OUT / "ASSIGNMENT_WITH.json", assignment_receipt)

    elapsed = time.time() - started
    result = {
        "schema": "p637-actual-class-balanced-per-expert-qtip2-solve-v1",
        "status": status,
        "solver": {
            "kind": "OR-Tools MPSolver + SCIP",
            "threads": SOLVER_THREADS,
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "warm_start": "deterministic feasible per-expert QTIP2 seed derived from sealed P629 WITHOUT",
            "variables": solver.NumVariables(),
            "constraints": solver.NumConstraints(),
        },
        "elapsed_seconds": elapsed,
        "measurement_label": "PREDICTION_ONLY_ARTIFACT_RELATIVE_SWAP_MASS_LAW",
        "scientific_transfer_claim": False,
        "objective": {
            "name": "CLASS-BALANCED GLOBAL: uniform mean of six per-class predicted KLDs",
            "without": incumbent_objective,
            "with": objective_value,
            "delta_with_minus_without": objective_value - incumbent_objective,
            "nonregression_pass": objective_value <= incumbent_objective + 1e-10,
            "solver_value": solver.Objective().Value(),
            "best_bound": best_bound,
            "relative_gap": relative_gap,
        },
        "bytes": {
            "envelope": ENVELOPE,
            "without_exact": ENVELOPE,
            "with_exact": final_bytes,
            "delta_with_minus_without": byte_delta,
            "slack": ENVELOPE - final_bytes,
            "qtip2_transition_delta": qtip_byte_delta,
            "qtip2_gross_freed": qtip_gross_freed,
            "qtip2_gross_added": qtip_gross_added,
            "qtip2_net_freed": -qtip_byte_delta,
            "existing_tier_transition_delta": existing_byte_delta,
            "existing_tier_gross_freed": existing_gross_freed,
            "existing_tier_gross_added": existing_gross_added,
            "reallocated_to_existing_tiers": max(0, existing_byte_delta),
            "closure_qtip_plus_existing_equals_total_delta": qtip_byte_delta + existing_byte_delta == byte_delta,
        },
        "prediction": {
            "canonical_solver_model_by_class": predicted,
            "canonical_solver_incumbent_by_class": incumbent_pred,
            "model_delta_by_class": {c: predicted[c] - incumbent_pred[c] for c in gs.CLASSES},
            "model_class_balanced_global": objective_value,
            "model_incumbent_class_balanced_global": incumbent_objective,
            "model_class_balanced_global_delta": objective_value - incumbent_objective,
            "model_weighted_global": model_global,
            "model_incumbent_weighted_global": old_global_model,
            "model_global_delta": model_global - old_global_model,
            "canon_adjusted_global_from_0.08395": canon_global,
            "canon_adjusted_code_from_sealed_0.04170458531457378": canon_code,
            "qtip_direct_global_increment": qtip_global_direct,
            "qtip_class_reconstructed_global_increment": qtip_global_reconstructed,
        },
        "qtip2": {
            "candidate": QTIP_TIER,
            "ordinary_per_expert_projection_menu_column": True,
            "eligible_layers": list(ELIGIBLE),
            "eligible_candidate_cells": len(ELIGIBLE) * gs.EXPERTS * len(gs.PROJECTIONS),
            "measured_layers": list(MEASURED),
            "family_mean_layers": [x for x in ELIGIBLE if x not in MEASURED],
            "selected_cells": len(qtip_selected),
            "selected_experts_any_projection": len({(l, e) for l, e, _p in qtip_selected}),
            "selected_by_layer": {str(layer): qtip_by_layer[layer] for layer in ELIGIBLE},
            "selected_cell_detail": qtip_cell_rows,
            "selected_experts_by_layer": qtip_experts_by_layer,
            "into_qtip2_by_from_tier": dict(sorted(qtip_into_counts.items())),
            "out_of_qtip2_by_to_tier": dict(sorted(qtip_out_counts.items())),
            "d4_k256_replaced_by_qtip2": qtip_into_counts["d4_k256"],
            "d4_k1024_replaced_by_qtip2": qtip_into_counts["d4_k1024"],
            "physical_bytes_by_layer_projection": {str(k): v for k, v in QTIP_PHYSICAL_BYTES_BY_LAYER.items()},
            "family_mean_delta_by_class": family_mean,
            "family_mean_global_delta": family_global,
            "price_table": [delta_surface[x] for x in ELIGIBLE],
            "mass_law_closure": pricing_closure,
        },
        "tier_counts": dict(sorted(tier_counts.items())),
        "transition_counts": {f"{a}->{b}": n for (a, b), n in sorted(transition_counts.items())},
        "changed_cell_count": len(changed_cells),
        "changed_cells": changed_cells,
        "assignment_receipt": str(OUT / "ASSIGNMENT_WITH.json"),
        "assignment_receipt_sha256": assignment_receipt_sha,
        "assignment_map_sha256": assignment_map_sha,
        "reproduction_gate": {
            "status": "PASS",
            "sanity_sha256": sanity_sha,
            "max_abs_error": max(errors.values()),
            "per_class_abs_error": errors,
            "tier_counts_exact": True,
            "exact_bytes": canonical_incumbent_bytes,
            "d4_k256_cells": actual_counts["d4_k256"],
            "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA,
            "incumbent_assignment_sha256": EXPECTED_ASSIGNMENT_SHA,
            "p629_step1_receipt_sha256": step1_sha,
            "existing_menu_sha256": menu_sha,
            "old_assignment_feasible": True,
        },
        "input_receipts": {
            "manifest": {"path": str(INPUTS / "INPUT_MANIFEST.json"), "sha256": EXPECTED_INPUT_MANIFEST_SHA},
            "verified_manifest_files": verified_inputs,
            "original_solver": {"path": str(LINEAGE / "original_banana_smasher_code_solve.py"), "sha256": EXPECTED_ORIGINAL_SOLVER_SHA},
            "incumbent_assignment": {"path": str(LINEAGE / "NOMINATED_ASSIGNMENT.json"), "sha256": EXPECTED_ASSIGNMENT_SHA},
            "p620_result": {"path": str(p620_result_path), "sha256": EXPECTED_P620_RESULT_SHA},
            "p629_without_reproduction": {"path": str(step1_path), "sha256": step1_sha},
            "p634_pricing_source_code": {
                "path": str(ROOT / "sources/P634/code/solve_per_expert_qtip2.py"),
                "sha256": sha256(ROOT / "sources/P634/code/solve_per_expert_qtip2.py"),
            },
            "anchors": measured_rows,
        },
        "normalization_receipt": normalization,
        "correction_fit": correction_fit,
        "first_incumbent_sha256": seed_sha,
        "started_unix": started,
        "finished_unix": time.time(),
    }
    result_path = OUT / "WITH_RESULT.json"
    result_sha = atomic_json(result_path, result)
    code_sha = sha256(Path(__file__))
    done = {
        "status": "PASS_P637_ACTUAL_CLASS_BALANCED_PER_EXPERT_QTIP2_PREDICTION_ONLY",
        "solver_status": status,
        "result_sha256": result_sha,
        "assignment_receipt_sha256": assignment_receipt_sha,
        "assignment_map_sha256": assignment_map_sha,
        "code_sha256": code_sha,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA,
        "incumbent_assignment_sha256": EXPECTED_ASSIGNMENT_SHA,
        "without_assignment_map_sha256": step1["assignment_map_sha256"],
        "existing_menu_sha256": menu_sha,
        "p629_step1_receipt_sha256": step1_sha,
        "p629_code_sha256": sha256(ROOT / "sources/P629/code/solve_global_ab.py"),
        "p634_code_sha256": sha256(ROOT / "sources/P634/code/solve_per_expert_qtip2.py"),
        "exact_bytes": final_bytes,
        "objective": objective_value,
        "best_bound": best_bound,
        "relative_gap": relative_gap,
        "qtip2_selected_cells": len(qtip_selected),
        "qtip2_selected_by_layer": {str(layer): qtip_by_layer[layer] for layer in ELIGIBLE},
        "elapsed_seconds": elapsed,
    }
    done_sha = atomic_json(OUT / "DONE.json", done)
    atomic_json(OUT / "PROGRESS.json", {
        **done,
        "schema": "p637-progress-v1",
        "status": "SOLVER_EXITED_RESULT_READY",
        "done_sha256": done_sha,
        "updated_unix": time.time(),
    })
    print(json.dumps(done, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
