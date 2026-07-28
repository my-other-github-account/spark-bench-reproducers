#!/usr/bin/env python3
"""P693: P637-identical backpack solve with one additional QTIP3 3.0117 column.

The sealed P637 2-bit-only result is the warm feasible incumbent and comparison arm.
QTIP3 is available only where a PASS 512-unit archive receipt exists. Capacity uses
exact logical bytes from those receipts; quality is prediction-only on the labeled
artifact-relative QTIP SSE-transfer basis, never a measured full-wire KLD claim.
"""
from __future__ import annotations

import csv
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
OUT = ROOT / "out_p693"
Q3 = ROOT / "inputs" / "qtip3"
P637_OUT = ROOT / "source_p637_out"
ENVELOPE = 101_346_700_411
QTIP3_TIER = "qtip3_3.0117"
TIME_LIMIT_SECONDS = 678.0
SOLVER_THREADS = int(os.environ.get("P693_SOLVER_THREADS", "1"))
CONSTRAINT_PRESERVING_DOMINANCE_PRUNING = False
PRIMAL_HEAVY_HEURISTICS = False
NONREGRESSION_REFERENCE = "p637"
CLASS_WEIGHTS = {
    "agentic": 1.0,
    "chat": 1.0,
    "code": 1.0,
    "multilingual": 1.0,
    "prose": 1.0,
    "reasoning": 1.0,
}
EXPECTED_Q3_LAYERS = tuple(x for x in range(3, 43) if x not in (21, 26, 28))
EXPECTED_Q3_LAYER_LOGICAL_BYTES = 2_423_261_184
EXPECTED_Q3_BYTES = {"fused13": 6_307_844, "down": 3_158_020}
WEIGHTED512_COUNTS = {
    "agentic": 154,
    "chat": 52,
    "code": 76,
    "multilingual": 76,
    "prose": 78,
    "reasoning": 76,
}
PRICE_LABEL = "PREDICTION_ONLY_ARTIFACT_RELATIVE_QTIP3_SSE_TRANSFER_SAME_CURRENCY_AS_P637"


def objective_value(values: dict[str, float], classes) -> float:
    """Return the normalized configured class-weighted objective."""
    missing = set(classes) - set(CLASS_WEIGHTS)
    if missing:
        raise RuntimeError({"missing_objective_weights": sorted(missing)})
    denominator = math.fsum(float(CLASS_WEIGHTS[c]) for c in classes)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError({"invalid_objective_weight_sum": denominator})
    return math.fsum(float(CLASS_WEIGHTS[c]) * float(values[c]) for c in classes) / denominator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    payload = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
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


def import_base():
    path = ROOT / "code" / "solve_actual.py"
    spec = importlib.util.spec_from_file_location("p637_base", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def option_map(cell: dict) -> dict[str, dict]:
    return {str(o["tier"]): dict(o) for o in cell["options"]}


def load_archive_receipts() -> dict:
    rows = []
    for path in sorted(Q3.glob("ARCHIVE_L*.json")):
        d = json.loads(path.read_text())
        row = {
            "layer": int(d["layer"]),
            "status": str(d["status"]),
            "unit_count": int(d["unit_count"]),
            "logical_layer_bytes": int(d["logical_wire_bytes_excluding_shared_tlut"]),
            "archive_bytes": int(d["remote_archive"]["bytes"]),
            "archive_sha256": str(d["remote_archive"]["sha256"]),
            "receipt_path": str(path),
            "receipt_sha256": sha256(path),
            "prearchive_sha256": str(d["prearchive"]["sha256"]),
            "tlut_sha256": str(d["tlut_sha256"]),
        }
        rows.append(row)
    layers = tuple(sorted(r["layer"] for r in rows))
    if layers != EXPECTED_Q3_LAYERS:
        raise RuntimeError({"qtip3_archive_layer_gate": layers, "expected": EXPECTED_Q3_LAYERS})
    for row in rows:
        if row["status"] != "PASS" or row["unit_count"] != 512:
            raise RuntimeError({"bad_qtip3_archive": row})
        if row["logical_layer_bytes"] != EXPECTED_Q3_LAYER_LOGICAL_BYTES:
            raise RuntimeError({"qtip3_logical_byte_drift": row})
    tls = {r["tlut_sha256"] for r in rows}
    if len(tls) != 1:
        raise RuntimeError({"qtip3_tlut_drift": sorted(tls)})

    lineage = []
    for layer in (32, 34, 36):
        path = Q3 / f"PREARCHIVE_L{layer:03d}.json"
        d = json.loads(path.read_text())
        if d["status"] != "PASS" or int(d["unit_count"]) != 512:
            raise RuntimeError({"bad_lineage_prearchive": str(path)})
        by_projection = Counter()
        unit_bytes = defaultdict(set)
        for unit in d["units"]:
            p = str(unit["identity"]["projection"])
            by_projection[p] += 1
            unit_bytes[p].add(int(unit["logical_bytes"]))
        expected_sets = {p: {n} for p, n in EXPECTED_Q3_BYTES.items()}
        actual_sets = {p: set(v) for p, v in unit_bytes.items()}
        if by_projection != Counter({"fused13": 256, "down": 256}) or actual_sets != expected_sets:
            raise RuntimeError({"qtip3_lineage_unit_byte_drift": layer, "counts": by_projection, "bytes": actual_sets})
        lineage.append({
            "layer": layer,
            "path": str(path),
            "sha256": sha256(path),
            "units_canonical_sha256": d["units_canonical_sha256"],
            "logical_layer_bytes": int(d["logical_wire_bytes_excluding_shared_tlut"]),
            "projection_unit_bytes": dict(EXPECTED_Q3_BYTES),
        })
    return {
        "eligible_layers": list(layers),
        "receipts": rows,
        "lineage_32_34_36": lineage,
        "logical_bytes_per_layer": EXPECTED_Q3_LAYER_LOGICAL_BYTES,
        "logical_bytes_per_projection_unit": dict(EXPECTED_Q3_BYTES),
        "archive_container_bytes_total": sum(r["archive_bytes"] for r in rows),
        "archive_receipt_set_sha256": hashlib.sha256(
            "".join(f"{r['layer']}\0{r['receipt_sha256']}\0{r['archive_sha256']}\n" for r in rows).encode()
        ).hexdigest(),
        "tlut_sha256": next(iter(tls)),
    }


def load_price_basis(gs, anchors: dict) -> dict:
    ext_path = Q3 / "QTIP_MENU_EXTENSION.json"
    ext = json.loads(ext_path.read_text())
    if ext["status"] != "PREDICTION_ONLY_READY_FOR_EXPLORATORY_SOLVE" or len(ext["rows"]) != 1:
        raise RuntimeError("bad QTIP3 menu extension")
    row = ext["rows"][0]
    transfer = row["anchor_price_transfer"]
    pooled = float(transfer["pooled_sse_ratio_vs_d4_k4096"])
    projection = {k: float(v) for k, v in transfer["projection_sse_ratio_vs_d4_k4096"].items()}
    central = {c: float(transfer["central"][c]) for c in gs.CLASSES}
    for c in gs.CLASSES:
        expected = float(anchors["d4_k4096"][c]) * pooled
        if abs(expected - central[c]) > 5e-12:
            raise RuntimeError({"qtip3_family_scale_drift": c, "expected": expected, "central": central[c]})
    csv_path = Q3 / "QTIP_TIER_PRICE_ROWS.csv"
    rows = list(csv.DictReader(csv_path.open(newline="")))
    if len(rows) != 1 or rows[0]["tier"] != "qtip_c2_hyb_l16_k3_v2":
        raise RuntimeError("bad QTIP3 price CSV")
    return {
        "measurement_label": PRICE_LABEL,
        "scientific_transfer_claim": False,
        "source_extension": {"path": str(ext_path), "sha256": sha256(ext_path)},
        "source_price_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        "family_reference_tier": "d4_k4096",
        "pooled_sse_ratio": pooled,
        "projection_sse_ratio": projection,
        "central_family_anchor_by_class": central,
        "limitation": transfer["limitation"],
        "cell_rank_law": "P637 frozen per-cell importance/projection/correction rank; multiply the same-currency d4_k4096 cell cost by the sealed QTIP3 projection SSE ratio",
        "lineage_note": "L032/L034/L036 PASS unit archives bind the exact geometry/rate/bytes; this remains artifact-relative SSE transfer, not a direct full-wire KLD family anchor",
    }


def load_assignment(path: Path, gs) -> dict:
    d = json.loads(path.read_text())
    amap = d["assignment"]
    return {
        (l, e, p): str(amap[str(l)][str(e)][p])
        for l in range(gs.LAYERS)
        for e in range(gs.EXPERTS)
        for p in gs.PROJECTIONS
    }


def weighted512(pred: dict[str, float]) -> float:
    total = sum(WEIGHTED512_COUNTS.values())
    return math.fsum(WEIGHTED512_COUNTS[c] * pred[c] for c in WEIGHTED512_COUNTS) / total


def summarize(gs, opts, original, selected) -> tuple[dict, int]:
    pred = {c: 0.0 for c in gs.CLASSES}
    delta_bytes = 0
    for key, tier in selected.items():
        opt = opts[key][tier]
        old = opts[key][original[key]]
        delta_bytes += int(opt["bytes"]) - int(old["bytes"])
        for c in gs.CLASSES:
            pred[c] += float(opt["costs"][c])
    return pred, ENVELOPE + delta_bytes


def build_surface():
    base = import_base()
    gs = base.load_original()
    _manifest, verified_inputs = base.validate_inputs()
    anchors = gs.load_anchor_grid(base.INPUTS / "rung1" / "ANCHOR_VERTICAL_GRID.csv")
    rows = gs.load_profile(base.INPUTS / "profile" / "PROFILE_ROWS.jsonl")
    importance, normalization = gs.normalize_profile_rows(rows)
    step0_path = base.INPUTS / "baseline" / "BQ3_STEP0_PER_CLASS.json"
    step0_doc = json.loads(step0_path.read_text())
    step0 = gs.step0_means(step0_path)
    old, _ = gs.map_incumbent(base.INPUTS / "baseline" / "DUALVQ_K4096MENU_BQ3_BIN_MANIFEST.json")
    corrections, correction_fit = gs.fit_projection_corrections(old, importance, anchors, step0)
    cells = gs.make_cells(importance, anchors, corrections)
    _inc_doc, original = base.read_assignment(gs)
    original_pred = gs.predict_assignment(original, importance, anchors, gs.CLASSES, corrections=corrections)
    ceilings = {c: (original_pred["code"] if c == "code" else step0[c]) for c in gs.CLASSES}

    delta_surface, q2_measured, q2_family_mean, q2_family_global = base.load_anchor_deltas(gs)
    gw = base.weighted_global_weights(step0_doc, gs.CLASSES)
    q2_ci, _q2_gi, q2_closure = base.build_qtip_surface(gs, importance, corrections, delta_surface, gw)
    archives = load_archive_receipts()
    price = load_price_basis(gs, anchors)

    opts: dict[tuple, dict[str, dict]] = {}
    for cell in cells:
        key = tuple(cell["key"])
        local = option_map(cell)
        if key in q2_ci:
            old_opt = local[original[key]]
            local[base.QTIP_TIER] = {
                "tier": base.QTIP_TIER,
                "bytes": int(base.QTIP_PHYSICAL_BYTES_BY_LAYER[key[0]][key[2]]),
                "costs": {c: float(old_opt["costs"][c]) + float(q2_ci[key][c]) for c in gs.CLASSES},
                "pricing_basis": delta_surface[key[0]]["basis"],
            }
        if key[0] in EXPECTED_Q3_LAYERS:
            reference = local["d4_k4096"]
            ratio = price["projection_sse_ratio"][key[2]]
            local[QTIP3_TIER] = {
                "tier": QTIP3_TIER,
                "bytes": EXPECTED_Q3_BYTES[key[2]],
                "costs": {c: max(0.0, float(reference["costs"][c]) * ratio) for c in gs.CLASSES},
                "pricing_basis": PRICE_LABEL,
                "reference_tier": "d4_k4096",
                "sse_ratio": ratio,
            }
        opts[key] = local

    p637_final_path = P637_OUT / "FINAL_TABLE.json"
    p637_assignment_path = P637_OUT / "ASSIGNMENT_RESPENT.json"
    p637_table = json.loads(p637_final_path.read_text())
    p637 = load_assignment(p637_assignment_path, gs)
    p637_pred, p637_bytes = summarize(gs, opts, original, p637)
    p637_reproduction_objective = math.fsum(p637_pred.values()) / len(gs.CLASSES)
    checks = {
        "p637_final_table_sha_exact": sha256(p637_final_path) == "978dc051fe6272d3d1a63ad3bc6933abf265aa3d9385f6ce92131d20009f0965",
        "p637_assignment_sha_exact": sha256(p637_assignment_path) == "c030883fddb1217529d67444d08257c4a1df18e2adbc93be092aba3d3611bc65",
        "p637_bytes_reproduced": p637_bytes == int(p637_table["with"]["exact_bytes"]) == 101_346_521_679,
        "p637_objective_reproduced": abs(p637_reproduction_objective - float(p637_table["with"]["objective"])) <= 2e-14,
        "p637_classes_reproduced": max(abs(p637_pred[c] - float(p637_table["with"]["predicted_classes"][c])) for c in gs.CLASSES) <= 2e-14,
        "qtip3_archive_coverage_37": archives["eligible_layers"] == list(EXPECTED_Q3_LAYERS),
        "qtip3_exact_logical_layer_bytes": archives["logical_bytes_per_layer"] == EXPECTED_Q3_LAYER_LOGICAL_BYTES,
    }
    if not all(checks.values()):
        raise RuntimeError({"surface_reproduction_gate": checks})
    return {
        "base": base,
        "gs": gs,
        "opts": opts,
        "original": original,
        "original_pred": original_pred,
        "p637": p637,
        "p637_pred": p637_pred,
        "p637_bytes": p637_bytes,
        "p637_table": p637_table,
        "ceilings": ceilings,
        "archives": archives,
        "price": price,
        "normalization": normalization,
        "correction_fit": correction_fit,
        "verified_inputs": verified_inputs,
        "q2_measured": q2_measured,
        "q2_family_mean": q2_family_mean,
        "q2_family_global": q2_family_global,
        "q2_closure": q2_closure,
        "checks": checks,
    }


def paired_greedy_postpass(gs, opts, selected, pred, payload, ceilings):
    """Apply only atomic freeing-sell + spending-buy pairs.

    The bounded candidate pools keep the seed fast on the 304k-option surface.
    Every accepted pair is checked against the aggregate byte and six-class
    constraints; neither leg is ever committed alone.
    """
    ledger = []
    eps = 1e-14
    seller_pool_limit = 2048
    buyer_pool_limit = 4096
    for round_no in range(1, 31):
        sellers = []
        buyers = []
        for key, current_tier in selected.items():
            current = opts[key][current_tier]
            for tier, opt in opts[key].items():
                if tier == current_tier:
                    continue
                delta_class = {
                    c: float(opt["costs"][c]) - float(current["costs"][c])
                    for c in gs.CLASSES
                }
                delta_obj = objective_value(delta_class, gs.CLASSES)
                delta_bytes = int(opt["bytes"]) - int(current["bytes"])
                row = (delta_obj, delta_bytes, key, tier, delta_class)
                if delta_bytes < 0:
                    sellers.append(row)
                elif delta_bytes > 0:
                    buyers.append(row)

        seller_by_efficiency = sorted(
            sellers,
            key=lambda r: (r[0] / -r[1], r[0], r[1], r[2], r[3]),
        )[:seller_pool_limit]
        seller_by_freed = sorted(
            sellers,
            key=lambda r: (r[1], r[0], r[2], r[3]),
        )[:seller_pool_limit]
        seller_pool = []
        seen_sellers = set()
        for row in (*seller_by_efficiency, *seller_by_freed):
            identity = (row[2], row[3])
            if identity not in seen_sellers:
                seen_sellers.add(identity)
                seller_pool.append(row)
        buyers.sort(key=lambda r: (r[0] / r[1], r[0], r[1], r[2], r[3]))
        buyer_pool = buyers[:buyer_pool_limit]

        used = set()
        changed_pairs = 0
        for buy in buyer_pool:
            buy_obj, buy_bytes, buy_key, buy_tier, buy_class = buy
            if buy_key in used or selected[buy_key] == buy_tier:
                continue
            best = None
            for sell in seller_pool:
                sell_obj, sell_bytes, sell_key, sell_tier, sell_class = sell
                if sell_key == buy_key or sell_key in used or selected[sell_key] == sell_tier:
                    continue
                combined_bytes = sell_bytes + buy_bytes
                if payload + combined_bytes > ENVELOPE:
                    continue
                combined_obj = sell_obj + buy_obj
                if combined_obj >= -eps:
                    continue
                combined_class = {
                    c: sell_class[c] + buy_class[c]
                    for c in gs.CLASSES
                }
                if any(
                    pred[c] + combined_class[c] < -1e-12
                    or pred[c] + combined_class[c] > ceilings[c] + 1e-12
                    for c in gs.CLASSES
                ):
                    continue
                rank = (combined_obj, combined_bytes, sell_obj, sell_key, sell_tier)
                if best is None or rank < best[0]:
                    best = (rank, sell, combined_class)
            if best is None:
                continue

            _rank, sell, combined_class = best
            sell_obj, sell_bytes, sell_key, sell_tier, sell_class = sell
            old_sell = selected[sell_key]
            old_buy = selected[buy_key]
            selected[sell_key] = sell_tier
            selected[buy_key] = buy_tier
            combined_bytes = sell_bytes + buy_bytes
            combined_obj = sell_obj + buy_obj
            payload += combined_bytes
            for c in gs.CLASSES:
                pred[c] += combined_class[c]
            ledger.append({
                "round": round_no,
                "pair_index": len(ledger) + 1,
                "sell": {
                    "key": list(sell_key),
                    "from": old_sell,
                    "to": sell_tier,
                    "delta_bytes": sell_bytes,
                    "delta_objective": sell_obj,
                    "delta_by_class": sell_class,
                },
                "buy": {
                    "key": list(buy_key),
                    "from": old_buy,
                    "to": buy_tier,
                    "delta_bytes": buy_bytes,
                    "delta_objective": buy_obj,
                    "delta_by_class": buy_class,
                },
                "combined_delta_bytes": combined_bytes,
                "combined_delta_objective": combined_obj,
                "combined_delta_by_class": combined_class,
            })
            used.update((sell_key, buy_key))
            changed_pairs += 1
        if not changed_pairs:
            break
    return selected, pred, payload, ledger


def greedy_postpass(gs, opts, selected, pred, payload, ceilings):
    """Compatibility entry point; P924 deliberately uses paired moves only."""
    return paired_greedy_postpass(gs, opts, selected, pred, payload, ceilings)


def constraint_preserving_dominance_prune(gs, opts, original, hint):
    """Prune only options dominated in bytes and every constrained class cost.

    The original option remains available because byte coefficients and receipts use
    it as the exact reference.  The feasible hint also remains available.  Unlike a
    scalar objective-only prune, this vector rule cannot remove an option needed to
    satisfy one of the six hard class ceilings.
    """
    model_opts = {}
    removed = []
    before = 0
    for key, local in opts.items():
        before += len(local)
        protected = {original[key], hint[key]}
        kept = {}
        rows = list(local.items())
        for tier_b, opt_b in rows:
            if tier_b in protected:
                kept[tier_b] = opt_b
                continue
            dominator = None
            for tier_a, opt_a in rows:
                if tier_a == tier_b:
                    continue
                bytes_a, bytes_b = int(opt_a["bytes"]), int(opt_b["bytes"])
                costs_a = [float(opt_a["costs"][c]) for c in gs.CLASSES]
                costs_b = [float(opt_b["costs"][c]) for c in gs.CLASSES]
                weak = bytes_a <= bytes_b and all(a <= b + 1e-18 for a, b in zip(costs_a, costs_b))
                strict = bytes_a < bytes_b or any(a < b - 1e-18 for a, b in zip(costs_a, costs_b))
                if weak and strict:
                    dominator = tier_a
                    break
            if dominator is None:
                kept[tier_b] = opt_b
            else:
                removed.append({
                    "layer": key[0],
                    "expert": key[1],
                    "projection": key[2],
                    "tier": tier_b,
                    "dominated_by": dominator,
                })
        model_opts[key] = kept
    after = sum(len(local) for local in model_opts.values())
    report = {
        "schema": "p860-constraint-preserving-dominance-pruning-v1",
        "status": "PASS",
        "law": "drop B only when A has <= bytes and <= price in every one of the six constrained classes, with at least one strict inequality; preserve exact-original and feasible-hint options",
        "scalar_objective_only_pruning": False,
        "feasible_region_preserved": True,
        "option_variables_before": before,
        "option_variables_after": after,
        "removed_option_variables": before - after,
        "cells": len(model_opts),
        "removed": removed,
    }
    return model_opts, report


def lp_bound(gs, opts, original, ceilings):
    lp = pywraplp.Solver.CreateSolver("GLOP")
    if lp is None:
        raise RuntimeError("GLOP unavailable")
    byte = lp.RowConstraint(-lp.infinity(), 0.0, "envelope_delta_le_zero")
    classes = {c: lp.RowConstraint(0.0, ceilings[c] * 1e6, f"global_{c}") for c in gs.CLASSES}
    objective = lp.Objective()
    objective.SetMinimization()
    nvars = 0
    for key, local in opts.items():
        one = lp.RowConstraint(1.0, 1.0, f"one_{key[0]}_{key[1]}_{key[2]}")
        old = local[original[key]]
        for tier, opt in local.items():
            v = lp.NumVar(0.0, 1.0, f"x_{key[0]}_{key[1]}_{key[2]}_{tier}")
            nvars += 1
            one.SetCoefficient(v, 1.0)
            byte.SetCoefficient(v, (int(opt["bytes"]) - int(old["bytes"])) / 1e6)
            for c in gs.CLASSES:
                classes[c].SetCoefficient(v, float(opt["costs"][c]) * 1e6)
            objective.SetCoefficient(v, objective_value(opt["costs"], gs.CLASSES) * 1e6)
    status_code = lp.Solve()
    names = {lp.OPTIMAL: "OPTIMAL", lp.FEASIBLE: "FEASIBLE", lp.INFEASIBLE: "INFEASIBLE", lp.ABNORMAL: "ABNORMAL", lp.NOT_SOLVED: "NOT_SOLVED"}
    status = names.get(status_code, str(status_code))
    if status_code != lp.OPTIMAL:
        raise RuntimeError({"lp_status": status})
    return {
        "kind": "GLOP full continuous relaxation of exact expanded menu/constraints",
        "status": status,
        "lower_bound": objective.Value() / 1e6,
        "wall_time_ms": lp.wall_time(),
        "iterations": lp.iterations(),
        "variables": nvars,
        "constraints": lp.NumConstraints(),
        "rigorous_relation": "LP feasible region is a superset of integer assignments",
    }


def rung_summary(gs, opts, original, selected, rung: str, layers: tuple[int, ...]) -> dict:
    keys = [k for k, t in selected.items() if t == rung]
    by_layer = Counter(k[0] for k in keys)
    transitions = Counter(original[k] for k in keys)
    deltas = [int(opts[k][selected[k]]["bytes"]) - int(opts[k][original[k]]["bytes"]) for k in keys]
    return {
        "tier": rung,
        "selected_cells": len(keys),
        "selected_experts_any_projection": len({(k[0], k[1]) for k in keys}),
        "selected_by_layer": {str(l): by_layer[l] for l in layers},
        "into_by_from_tier": dict(sorted(transitions.items())),
        "gross_bytes_freed": sum(max(0, -d) for d in deltas),
        "gross_bytes_spent": sum(max(0, d) for d in deltas),
        "net_byte_delta": sum(deltas),
        "net_bytes_freed": -sum(deltas),
        "selected_cell_detail": [{"layer": k[0], "expert": k[1], "projection": k[2], "from": original[k]} for k in sorted(keys)],
    }


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    surface = build_surface()
    base, gs, opts = surface["base"], surface["gs"], surface["opts"]
    original, p637 = surface["original"], surface["p637"]
    hint_path = Path(os.environ.get("P693_HINT_ASSIGNMENT", str(P637_OUT / "ASSIGNMENT_RESPENT.json")))
    hint = load_assignment(hint_path, gs)
    hint_pred, hint_bytes = summarize(gs, opts, original, hint)
    if hint_bytes > ENVELOPE or any(hint_pred[c] < -1e-12 or hint_pred[c] > surface["ceilings"][c] + 1e-12 for c in gs.CLASSES):
        raise RuntimeError({"invalid_warm_start": str(hint_path), "bytes": hint_bytes, "prediction": hint_pred})
    original_pred, p637_pred = surface["original_pred"], surface["p637_pred"]
    ceilings = surface["ceilings"]
    if CONSTRAINT_PRESERVING_DOMINANCE_PRUNING:
        model_opts, dominance_report = constraint_preserving_dominance_prune(gs, opts, original, hint)
    else:
        model_opts = opts
        before = sum(len(local) for local in opts.values())
        dominance_report = {
            "schema": "p860-constraint-preserving-dominance-pruning-v1",
            "status": "DISABLED",
            "option_variables_before": before,
            "option_variables_after": before,
            "removed_option_variables": 0,
            "cells": len(opts),
        }
    dominance_sha = atomic_json(OUT / "DOMINANCE_PRUNING.json", dominance_report)
    sanity = {
        "schema": "p693-expanded-menu-sanity-v1",
        "status": "PASS",
        "checks": surface["checks"],
        "host": os.uname().nodename,
        "qtip3_eligible_layers": list(EXPECTED_Q3_LAYERS),
        "qtip3_logical_bytes_per_layer": EXPECTED_Q3_LAYER_LOGICAL_BYTES,
        "qtip3_logical_bytes_per_projection_unit": EXPECTED_Q3_BYTES,
        "p637_final_table_sha256": sha256(P637_OUT / "FINAL_TABLE.json"),
        "p637_assignment_sha256": sha256(P637_OUT / "ASSIGNMENT_RESPENT.json"),
        "price_basis": surface["price"],
        "archive_receipt_set_sha256": surface["archives"]["archive_receipt_set_sha256"],
    }
    sanity_sha = atomic_json(OUT / "SANITY.json", sanity)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP unavailable")
    solver.SetTimeLimit(int(TIME_LIMIT_SECONDS * 1000))
    solver.SetNumThreads(SOLVER_THREADS)
    parameter_lines = [
        f"parallel/maxnthreads = {SOLVER_THREADS}",
        "randomization/randomseedshift = 0",
        "limits/gap = 0.0000001",
        "display/verblevel = 4",
    ]
    if PRIMAL_HEAVY_HEURISTICS:
        parameter_lines += [
            "heuristics/rens/freq = 5",
            "heuristics/rins/freq = 5",
            "heuristics/feaspump/freq = 5",
            "heuristics/rounding/freq = 1",
        ]
    parameter_text = "\n".join(parameter_lines) + "\n"
    if not solver.SetSolverSpecificParametersAsString(parameter_text):
        raise RuntimeError({"invalid_scip_parameters": parameter_text})
    byte = solver.RowConstraint(-solver.infinity(), 0.0, "exact_envelope_delta_le_zero")
    class_rows = {c: solver.RowConstraint(0.0, ceilings[c] * 1e6, f"class_{c}_hard") for c in gs.CLASSES}
    objective = solver.Objective()
    objective.SetMinimization()
    vars_by_key = {}
    hint_vars, hint_vals = [], []
    for key, local in model_opts.items():
        one = solver.RowConstraint(1.0, 1.0, f"one_{key[0]}_{key[1]}_{key[2]}")
        old = local[original[key]]
        vars_by_key[key] = []
        for tier, opt in local.items():
            v = solver.BoolVar(f"L{key[0]:03d}_E{key[1]:03d}_{key[2]}_{tier}")
            vars_by_key[key].append((tier, v))
            one.SetCoefficient(v, 1.0)
            byte.SetCoefficient(v, (int(opt["bytes"]) - int(old["bytes"])) / 1e6)
            for c in gs.CLASSES:
                class_rows[c].SetCoefficient(v, float(opt["costs"][c]) * 1e6)
            objective.SetCoefficient(v, objective_value(opt["costs"], gs.CLASSES) * 1e6)
            hint_vars.append(v)
            hint_vals.append(1.0 if tier == hint[key] else 0.0)
    solver.SetHint(hint_vars, hint_vals)
    solver.EnableOutput()
    atomic_json(OUT / "PROGRESS.json", {
        "schema": "p693-progress-v1",
        "status": "SOLVING_SCIP_678S",
        "solver_pid": os.getpid(),
        "solver_pgid": os.getpgid(0),
        "variables": solver.NumVariables(),
        "constraints": solver.NumConstraints(),
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "threads": SOLVER_THREADS,
        "scip_parameters": parameter_lines,
        "primal_heavy_heuristics": PRIMAL_HEAVY_HEURISTICS,
        "dominance_pruning_sha256": dominance_sha,
        "dominance_pruning": {k: v for k, v in dominance_report.items() if k != "removed"},
        "warm_start": {"path": str(hint_path), "sha256": sha256(hint_path), "objective": objective_value(hint_pred, gs.CLASSES), "exact_bytes": hint_bytes},
        "sanity_sha256": sanity_sha,
        "started_unix": started,
    })
    status_code = solver.Solve()
    names = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status = names.get(status_code, str(status_code))
    if status_code not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        atomic_json(OUT / "DONE.json", {"status": status, "elapsed_seconds": time.time() - started})
        return 2

    selected = {}
    for key, local in vars_by_key.items():
        picked = [tier for tier, v in local if v.solution_value() > 0.5]
        if len(picked) != 1:
            raise RuntimeError({"selection_cardinality": key, "picked": picked})
        selected[key] = picked[0]
    mip_pred, mip_bytes = summarize(gs, opts, original, selected)
    mip_objective = objective_value(mip_pred, gs.CLASSES)
    scip_bound = solver.Objective().BestBound() / 1e6

    selected, final_pred, final_bytes, greedy_ledger = greedy_postpass(
        gs, opts, selected, dict(mip_pred), mip_bytes, ceilings
    )
    final_objective = objective_value(final_pred, gs.CLASSES)
    lp = lp_bound(gs, opts, original, ceilings)
    strongest_bound = max(float(scip_bound), float(lp["lower_bound"]))
    relative_gap = max(0.0, final_objective - strongest_bound) / max(abs(final_objective), 1e-30)

    p637_objective = objective_value(p637_pred, gs.CLASSES)
    original_objective = objective_value(original_pred, gs.CLASSES)
    if NONREGRESSION_REFERENCE == "legal_hint":
        nonregression_objective = objective_value(hint_pred, gs.CLASSES)
        nonregression_label = "legal current-menu greedy hint"
    elif NONREGRESSION_REFERENCE == "p637":
        nonregression_objective = p637_objective
        nonregression_label = "sealed P637 2-bit comparator"
    else:
        raise RuntimeError({"unknown_nonregression_reference": NONREGRESSION_REFERENCE})
    if final_bytes > ENVELOPE or final_objective > nonregression_objective + 1e-10:
        raise RuntimeError({"nonregression_or_byte_gate": {
            "bytes": final_bytes,
            "objective": final_objective,
            "reference_objective": nonregression_objective,
            "reference_label": nonregression_label,
        }})
    if any(final_pred[c] < -1e-12 or final_pred[c] > ceilings[c] + 1e-12 for c in gs.CLASSES):
        raise RuntimeError({"class_gate": final_pred, "ceilings": ceilings})

    amap = {
        str(l): {
            str(e): {p: selected[(l, e, p)] for p in gs.PROJECTIONS}
            for e in range(gs.EXPERTS)
        }
        for l in range(gs.LAYERS)
    }
    map_sha = hashlib.sha256(json.dumps(amap, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assignment_receipt = {
        "schema": "p693-qtip2-qtip3-assignment-v1",
        "measurement_label": PRICE_LABEL,
        "assignment": amap,
        "assignment_map_sha256": map_sha,
        "source_p637_assignment_sha256": sha256(P637_OUT / "ASSIGNMENT_RESPENT.json"),
        "input_manifest_sha256": base.EXPECTED_INPUT_MANIFEST_SHA,
        "qtip3_archive_receipt_set_sha256": surface["archives"]["archive_receipt_set_sha256"],
    }
    assignment_sha = atomic_json(OUT / "ASSIGNMENT_QTIP2_QTIP3.json", assignment_receipt)

    q2 = rung_summary(gs, opts, original, selected, base.QTIP_TIER, tuple(base.ELIGIBLE))
    q3 = rung_summary(gs, opts, original, selected, QTIP3_TIER, EXPECTED_Q3_LAYERS)
    transitions = Counter((original[k], selected[k]) for k in selected if original[k] != selected[k])
    vs_p637_transitions = Counter((p637[k], selected[k]) for k in selected if p637[k] != selected[k])
    tier_counts = Counter(selected.values())
    all_changed_deltas = [int(opts[k][selected[k]]["bytes"]) - int(opts[k][original[k]]["bytes"]) for k in selected if selected[k] != original[k]]
    qkeys = {k for k, t in selected.items() if t in (base.QTIP_TIER, QTIP3_TIER)}
    qdelta = sum(int(opts[k][selected[k]]["bytes"]) - int(opts[k][original[k]]["bytes"]) for k in qkeys)
    ordinary_delta = final_bytes - ENVELOPE - qdelta

    result = {
        "schema": "p693-p637-expanded-qtip2-qtip3-solve-v1",
        "status": "PASS_FEASIBLE_EXPANDED_MENU_WITH_RIGOROUS_LP_BOUND",
        "host": os.uname().nodename,
        "measurement_label": PRICE_LABEL,
        "scientific_transfer_claim": False,
        "objective_name": "V2_REWEIGHTED_PRE_V3 normalized weighted mean of six per-class predicted KLDs",
        "objective_class_weights": dict(CLASS_WEIGHTS),
        "constraints": {
            "envelope": ENVELOPE,
            "per_class_hard_ceilings": ceilings,
            "class_lower_bound": 0.0,
            "code_ceiling": "sealed P637 incumbent code prediction",
            "other_class_ceilings": "sealed step0 means",
            "full_existing_menu_both_directions": True,
            "cross_class_trades_permitted": True,
            "nonregression_reference": {
                "label": nonregression_label,
                "objective": nonregression_objective,
                "mode": NONREGRESSION_REFERENCE,
            },
        },
        "arms": {
            "legal_current_menu_greedy_hint": {
                "objective": objective_value(hint_pred, gs.CLASSES),
                "prediction_by_class": hint_pred,
                "weighted512_global": weighted512(hint_pred),
                "exact_bytes": hint_bytes,
                "assignment_path": str(hint_path),
                "assignment_sha256": sha256(hint_path),
            },
            "incumbent_without_qtip": {
                "objective": original_objective,
                "prediction_by_class": original_pred,
                "weighted512_global": weighted512(original_pred),
                "exact_bytes": ENVELOPE,
            },
            "with_2bit_only_sealed_p637": {
                "objective": p637_objective,
                "prediction_by_class": p637_pred,
                "weighted512_global": weighted512(p637_pred),
                "exact_bytes": surface["p637_bytes"],
                "final_table_sha256": sha256(P637_OUT / "FINAL_TABLE.json"),
                "assignment_sha256": sha256(P637_OUT / "ASSIGNMENT_RESPENT.json"),
                "qtip2_cells": int(surface["p637_table"]["with"]["qtip2_selected_cells"]),
            },
            "with_2bit_plus_3bit": {
                "objective": final_objective,
                "prediction_by_class": final_pred,
                "weighted512_global": weighted512(final_pred),
                "exact_bytes": final_bytes,
                "slack": ENVELOPE - final_bytes,
            },
        },
        "delta_expanded_minus_incumbent": {
            "objective": final_objective - original_objective,
            "prediction_by_class": {c: final_pred[c] - original_pred[c] for c in gs.CLASSES},
            "weighted512_global": weighted512(final_pred) - weighted512(original_pred),
            "exact_bytes": final_bytes - ENVELOPE,
        },
        "delta_expanded_minus_2bit_only": {
            "objective": final_objective - p637_objective,
            "prediction_by_class": {c: final_pred[c] - p637_pred[c] for c in gs.CLASSES},
            "weighted512_global": weighted512(final_pred) - weighted512(p637_pred),
            "exact_bytes": final_bytes - surface["p637_bytes"],
        },
        "rungs": {"qtip2_2.0117": q2, "qtip3_3.0117": q3},
        "qtip3_pricing": surface["price"],
        "qtip3_archive_inventory": surface["archives"],
        "bytes": {
            "envelope": ENVELOPE,
            "with_exact": final_bytes,
            "slack": ENVELOPE - final_bytes,
            "gross_changed_bytes_freed": sum(max(0, -d) for d in all_changed_deltas),
            "gross_changed_bytes_spent": sum(max(0, d) for d in all_changed_deltas),
            "net_with_minus_incumbent": final_bytes - ENVELOPE,
            "all_qtip_net_delta": qdelta,
            "all_qtip_net_bytes_freed": -qdelta,
            "ordinary_tier_net_delta": ordinary_delta,
            "ordinary_tier_net_bytes_spent": max(0, ordinary_delta),
            "closure_qtip_plus_ordinary_equals_total": qdelta + ordinary_delta == final_bytes - ENVELOPE,
        },
        "solver": {
            "kind": "OR-Tools MPSolver SCIP integer solve + deterministic feasible postpass + GLOP full LP relaxation",
            "integer_status": status,
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "threads": SOLVER_THREADS,
            "scip_parameters": parameter_lines,
            "primal_heavy_heuristics": PRIMAL_HEAVY_HEURISTICS,
            "dominance_pruning_sha256": dominance_sha,
            "dominance_pruning": {k: v for k, v in dominance_report.items() if k != "removed"},
            "variables": solver.NumVariables(),
            "constraints": solver.NumConstraints(),
            "mip_incumbent_objective_before_postpass": mip_objective,
            "mip_incumbent_exact_bytes_before_postpass": mip_bytes,
            "scip_best_bound": scip_bound,
            "lp": lp,
            "strongest_rigorous_lower_bound": strongest_bound,
            "relative_gap": relative_gap,
            "optimality_claimed": status == "OPTIMAL" and relative_gap <= 1e-7,
            "greedy_postpass_moves": len(greedy_ledger),
            "greedy_postpass_rounds": max((r["round"] for r in greedy_ledger), default=0),
            "wall_seconds": time.time() - started,
        },
        "tier_counts": dict(sorted(tier_counts.items())),
        "transition_counts_vs_incumbent": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
        "transition_counts_vs_p637_2bit_only": {f"{a}->{b}": n for (a, b), n in sorted(vs_p637_transitions.items())},
        "greedy_postpass_ledger": greedy_ledger,
        "assignment_receipt": str(OUT / "ASSIGNMENT_QTIP2_QTIP3.json"),
        "assignment_receipt_sha256": assignment_sha,
        "assignment_map_sha256": map_sha,
        "reproduction_gate": {"status": "PASS", "sanity_sha256": sanity_sha, "checks": surface["checks"]},
        "input_receipts": {
            "verified_p637_inputs": surface["verified_inputs"],
            "p637_final_table": {"path": str(P637_OUT / "FINAL_TABLE.json"), "sha256": sha256(P637_OUT / "FINAL_TABLE.json")},
            "p637_assignment": {"path": str(P637_OUT / "ASSIGNMENT_RESPENT.json"), "sha256": sha256(P637_OUT / "ASSIGNMENT_RESPENT.json")},
            "qtip3_archive_receipt_set_sha256": surface["archives"]["archive_receipt_set_sha256"],
        },
        "normalization_receipt": surface["normalization"],
        "correction_fit": surface["correction_fit"],
        "started_unix": started,
        "finished_unix": time.time(),
    }
    result_sha = atomic_json(OUT / "RESULT.json", result)
    done = {
        "schema": "p693-done-v1",
        "status": result["status"],
        "result_sha256": result_sha,
        "assignment_receipt_sha256": assignment_sha,
        "assignment_map_sha256": map_sha,
        "objective": final_objective,
        "exact_bytes": final_bytes,
        "qtip2_cells": q2["selected_cells"],
        "qtip3_cells": q3["selected_cells"],
        "strongest_rigorous_lower_bound": strongest_bound,
        "relative_gap": relative_gap,
        "elapsed_seconds": time.time() - started,
        "code_sha256": sha256(Path(__file__)),
    }
    done_sha = atomic_json(OUT / "DONE.json", done)
    atomic_json(OUT / "PROGRESS.json", {**done, "status": "SOLVER_EXITED_RESULT_READY", "done_sha256": done_sha})
    print(json.dumps({**done, "done_sha256": done_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
