#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
BOUND = ROOT / "bound"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> str:
    payload = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha(path)


result_path = OUT / "WITH_RESULT.json"
done_path = OUT / "DONE.json"
verifier_path = OUT / "VERIFIER.json"
assignment_path = OUT / "ASSIGNMENT_WITH.json"
step1_path = ROOT / "receipts" / "STEP1_WITHOUT_REPRO.json"
lp_path = BOUND / "LP_BOUND.json"

r = json.loads(result_path.read_text())
d = json.loads(done_path.read_text())
v = json.loads(verifier_path.read_text())
s = json.loads(step1_path.read_text())
lp = json.loads(lp_path.read_text()) if lp_path.exists() else None

without_obj = float(r["objective"]["without"])
with_obj = float(r["objective"]["with"])
scip_bound = float(r["objective"]["best_bound"])
scip_gap = float(r["objective"]["relative_gap"])
lp_bound = float(lp["objective_lower_bound"]) if lp and lp.get("valid_lower_bound_for_integer_primary_objective") else None
lp_gap = max(0.0, with_obj - lp_bound) / max(abs(with_obj), 1e-30) if lp_bound is not None else None
without_classes = r["prediction"]["canonical_solver_incumbent_by_class"]
with_classes = r["prediction"]["canonical_solver_model_by_class"]
class_delta = {k: float(with_classes[k]) - float(without_classes[k]) for k in without_classes}

final = {
    "schema": "p637-final-with-without-table-v1",
    "status": "PASS" if v["status"] == "PASS" and d["solver_status"] in {"FEASIBLE", "OPTIMAL"} else "FAIL",
    "host": "compute-node-3",
    "acceptance_case": v["acceptance_case"],
    "objective": {
        "name": r["objective"]["name"],
        "without": without_obj,
        "with": with_obj,
        "delta": with_obj - without_obj,
    },
    "prediction": {
        "without_by_class": without_classes,
        "with_by_class": with_classes,
        "delta_by_class": class_delta,
    },
    "bytes": r["bytes"],
    "qtip2": {
        "selected_cells": r["qtip2"]["selected_cells"],
        "selected_by_layer": r["qtip2"]["selected_by_layer"],
        "selected_experts_by_layer": r["qtip2"]["selected_experts_by_layer"],
        "selected_cell_detail": r["qtip2"]["selected_cell_detail"],
        "displaced_by_tier": r["qtip2"]["into_qtip2_by_from_tier"],
        "eligible_layers": r["qtip2"]["eligible_layers"],
        "ordinary_per_expert_projection_menu_column": r["qtip2"]["ordinary_per_expert_projection_menu_column"],
    },
    "tier_counts": {
        "without": s["tier_counts"],
        "with": r["tier_counts"],
    },
    "transition_counts": r["transition_counts"],
    "solver": {
        "status": d["solver_status"],
        "wall_seconds": r["elapsed_seconds"],
        "scip_best_bound": scip_bound,
        "scip_reported_relative_gap": scip_gap,
        "lp_relaxation_status": lp.get("status") if lp else None,
        "lp_relaxation_valid_lower_bound": lp_bound,
        "lp_certified_relative_gap": lp_gap,
        "lp_bound_receipt_sha256": sha(lp_path) if lp_path.exists() else None,
    },
    "constraints": {
        "class_balanced_uniform_six_class_mean_is_primary_objective": True,
        "per_class_ceilings_are_global_aggregate_constraints": True,
        "per_cell_or_per_move_code_nonworsening_veto": False,
        "cross_class_trades_allowed_subject_to_final_global_ceilings": True,
    },
    "answer": {
        "does_backpack_buy_qtip2_at_bit_size_parity": int(r["qtip2"]["selected_cells"]) > 0 and int(r["bytes"]["with_exact"]) <= int(r["bytes"]["envelope"]),
        "qtip2_cells_bought": r["qtip2"]["selected_cells"],
        "qtip2_bytes_freed": r["bytes"]["qtip2_net_freed"],
        "bytes_reallocated_to_existing_tiers": r["bytes"]["reallocated_to_existing_tiers"],
        "net_slack": r["bytes"]["slack"],
        "what_freed_bytes_buy": r["transition_counts"],
    },
    "sha256": {
        "input_manifest": d["input_manifest_sha256"],
        "incumbent_assignment": d["incumbent_assignment_sha256"],
        "without_assignment_map": d["without_assignment_map_sha256"],
        "with_assignment_map": d["assignment_map_sha256"],
        "with_assignment_receipt": sha(assignment_path),
        "existing_menu_before_qtip2": d["existing_menu_sha256"],
        "p629_step1_without_receipt": d["p629_step1_receipt_sha256"],
        "p629_source_code": d["p629_code_sha256"],
        "p634_source_code": d["p634_code_sha256"],
        "merged_solver_code": d["code_sha256"],
        "with_result": sha(result_path),
        "done": sha(done_path),
        "verifier": sha(verifier_path),
    },
    "verifier": {
        "status": v["status"],
        "failed_checks": v["failed_checks"],
        "checks": v["checks"],
    },
}

out_path = OUT / "FINAL_TABLE.json"
out_sha = atomic_json(out_path, final)
print(json.dumps({"path": str(out_path), "sha256": out_sha, "status": final["status"]}, sort_keys=True))
raise SystemExit(0 if final["status"] == "PASS" else 2)
