#!/usr/bin/env python3
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
LINEAGE = ROOT / "lineage"
EXPECTED_MANIFEST = "88ebfc21a7134088cad0a9a4f09821410db29ac13cd754d4d46f8902bebecb42"
EXPECTED_INCUMBENT = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
EXPECTED_STEP1 = "0928bb3b923c12c758adb40ffc7f57cda3a17d79f23444de05dab370041d45ce"
EXPECTED_MENU = "9b4b46a2fa694aa7fbdbdbf03273eeece6b3887a4b20b58d83003d8786a8d5b7"
EXPECTED_D4 = 2744
ENVELOPE = 101346700411
QTIP = "qtip2_2.0117"
GAP_TARGET = 1e-7
CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


rpath = OUT / "WITH_RESULT.json"
apath = OUT / "ASSIGNMENT_WITH.json"
dpath = OUT / "DONE.json"
spath = OUT / "SANITY.json"
step1_path = ROOT / "receipts" / "STEP1_WITHOUT_REPRO.json"
r = json.loads(rpath.read_text())
a = json.loads(apath.read_text())
d = json.loads(dpath.read_text())
s = json.loads(spath.read_text())
step1 = json.loads(step1_path.read_text())
step0_doc = json.loads((ROOT / "inputs/baseline/BQ3_STEP0_PER_CLASS.json").read_text())
step0 = {c: float(step0_doc["by_class"][c]["mean"]) for c in CLASSES}
base = json.loads((LINEAGE / "NOMINATED_ASSIGNMENT.json").read_text())
assignment = a["assignment"]
base_assignment = base["assignment"]
map_sha = hashlib.sha256(json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
qtip_by_layer = Counter()
transition_counts = Counter()
for layer_s, experts in assignment.items():
    for expert_s, projmap in experts.items():
        for projection, tier in projmap.items():
            old = base_assignment[layer_s][expert_s][projection]
            if tier == QTIP:
                qtip_by_layer[int(layer_s)] += 1
            if tier != old:
                transition_counts[(old, tier)] += 1
qtip_count = sum(qtip_by_layer.values())
objective = float(r["objective"]["with"])
bound = float(r["objective"]["best_bound"])
gap = max(0.0, objective - bound) / max(abs(objective), 1e-30)
case_a = qtip_count > 0 and objective <= float(r["objective"]["without"]) + 1e-10
case_b = qtip_count == 0 and (r["status"] == "OPTIMAL" or gap <= GAP_TARGET)
closure_errors = []
for row in r["qtip2"]["mass_law_closure"]:
    closure_errors.append(abs(float(row["global_sum_error"])))
    closure_errors.extend(abs(float(v)) for v in row["class_sum_errors"].values())
changed_delta = sum(int(row["delta_bytes"]) for row in r["changed_cells"])
result_transition_counts = Counter()
for key, value in r["transition_counts"].items():
    before, after = key.split("->", 1)
    result_transition_counts[(before, after)] = int(value)
predicted = {c: float(r["prediction"]["canonical_solver_model_by_class"][c]) for c in CLASSES}
without_predicted = {c: float(r["prediction"]["canonical_solver_incumbent_by_class"][c]) for c in CLASSES}
uniform_objective = math.fsum(predicted.values()) / len(CLASSES)
qtip_detail = sorted(
    (int(row["layer"]), int(row["expert"]), str(row["projection"]), str(row["from"]))
    for row in r["qtip2"]["selected_cell_detail"]
)
qtip_from_assignment = sorted(
    (int(layer_s), int(expert_s), projection, base_assignment[layer_s][expert_s][projection])
    for layer_s, experts in assignment.items()
    for expert_s, projections in experts.items()
    for projection, tier in projections.items()
    if tier == QTIP
)
price_by_layer = {int(row["layer"]): row for row in r["qtip2"]["price_table"]}
expected_signed = {
    0: -0.00029121718633007276,
    2: -0.006373947346279607,
    16: -0.008513247534104096,
}
checks = {
    "step1_sanity_pass": s["status"] == "PASS" and all(s["checks"].values()),
    "p629_step1_receipt_exact": sha(step1_path) == EXPECTED_STEP1 == r["reproduction_gate"]["p629_step1_receipt_sha256"] == d["p629_step1_receipt_sha256"],
    "existing_menu_exact": s["existing_menu_sha256"] == r["reproduction_gate"]["existing_menu_sha256"] == d["existing_menu_sha256"] == EXPECTED_MENU,
    "without_assignment_map_exact": step1["assignment_map_sha256"] == a["without_assignment_map_sha256"] == d["without_assignment_map_sha256"],
    "incumbent_d4_k256_2744": int(s["d4_k256_per_expert_projection_cells"]) == EXPECTED_D4,
    "result_sha_matches_done": sha(rpath) == d["result_sha256"],
    "assignment_receipt_sha_matches_done": sha(apath) == d["assignment_receipt_sha256"],
    "assignment_map_sha_closure": map_sha == a["assignment_map_sha256"] == r["assignment_map_sha256"] == d["assignment_map_sha256"],
    "manifest_sha_exact": d["input_manifest_sha256"] == r["reproduction_gate"]["input_manifest_sha256"] == EXPECTED_MANIFEST,
    "incumbent_sha_exact": d["incumbent_assignment_sha256"] == r["reproduction_gate"]["incumbent_assignment_sha256"] == EXPECTED_INCUMBENT,
    "canonical_replay_zero": float(r["reproduction_gate"]["max_abs_error"]) <= 1e-12,
    "ordinary_per_expert_column": bool(r["qtip2"]["ordinary_per_expert_projection_menu_column"]),
    "eligible_candidate_count": int(r["qtip2"]["eligible_candidate_cells"]) == 8192,
    "mass_law_closure": max(closure_errors, default=0.0) <= 1e-12,
    "solver_status_feasible_or_optimal": r["status"] in {"FEASIBLE", "OPTIMAL"},
    "objective_finite": math.isfinite(objective) and math.isfinite(bound),
    "objective_is_uniform_six_class_mean": abs(objective - uniform_objective) <= 1e-15 and "uniform mean of six" in r["objective"]["name"],
    "signed_measured_improvements_preserved": all(abs(float(price_by_layer[layer]["global_delta"]) - expected) <= 1e-15 for layer, expected in expected_signed.items()),
    "code_ceiling_preserved": predicted["code"] <= without_predicted["code"] + 1e-12,
    "non_code_step0_ceilings_preserved": all(predicted[c] <= step0[c] + 1e-12 for c in CLASSES if c != "code"),
    "objective_nonregression": bool(r["objective"]["nonregression_pass"]) and objective <= float(r["objective"]["without"]) + 1e-10,
    "byte_cap": int(r["bytes"]["with_exact"]) <= int(r["bytes"]["envelope"]) == ENVELOPE,
    "byte_delta_closure": int(r["bytes"]["with_exact"]) == ENVELOPE + int(r["bytes"]["delta_with_minus_without"]) == ENVELOPE + changed_delta,
    "byte_component_closure": bool(r["bytes"]["closure_qtip_plus_existing_equals_total_delta"]) and int(r["bytes"]["qtip2_transition_delta"]) + int(r["bytes"]["existing_tier_transition_delta"]) == changed_delta,
    "gross_qtip_byte_closure": int(r["bytes"]["qtip2_gross_added"]) - int(r["bytes"]["qtip2_gross_freed"]) == int(r["bytes"]["qtip2_transition_delta"]),
    "tier_count_total": sum(int(v) for v in r["tier_counts"].values()) == 43 * 256 * 2,
    "qtip_count_closure": qtip_count == int(r["qtip2"]["selected_cells"]) == int(d["qtip2_selected_cells"]),
    "qtip_detail_closure": qtip_detail == qtip_from_assignment,
    "qtip_by_layer_closure": {str(k): qtip_by_layer[k] for k in r["qtip2"]["eligible_layers"]} == r["qtip2"]["selected_by_layer"] == d["qtip2_selected_by_layer"],
    "transition_counts_closure": transition_counts == result_transition_counts,
    "d4_k256_into_qtip_closure": transition_counts[("d4_k256", QTIP)] == int(r["qtip2"]["d4_k256_replaced_by_qtip2"]),
    "d4_k1024_into_qtip_closure": transition_counts[("d4_k1024", QTIP)] == int(r["qtip2"]["d4_k1024_replaced_by_qtip2"]),
    "acceptance_a_or_b": case_a or case_b,
}
receipt = {
    "schema": "p637-independent-verifier-v1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "acceptance_case": "A" if case_a else ("B" if case_b else None),
    "checks": checks,
    "failed_checks": [k for k, v in checks.items() if not v],
    "qtip2_selected_cells": qtip_count,
    "qtip2_selected_by_layer": {str(k): qtip_by_layer[k] for k in r["qtip2"]["eligible_layers"]},
    "objective": objective,
    "best_bound": bound,
    "relative_gap": gap,
    "exact_bytes": int(r["bytes"]["with_exact"]),
    "result_sha256": sha(rpath),
    "assignment_receipt_sha256": sha(apath),
    "assignment_map_sha256": map_sha,
    "done_sha256": sha(dpath),
    "sanity_sha256": sha(spath),
    "solver_code_sha256": d["code_sha256"],
    "input_manifest_sha256": EXPECTED_MANIFEST,
    "incumbent_assignment_sha256": EXPECTED_INCUMBENT,
}
vpath = OUT / "VERIFIER.json"
atomic_json(vpath, receipt)
print(json.dumps({"path": str(vpath), "sha256": sha(vpath), **receipt}, sort_keys=True))
raise SystemExit(0 if receipt["status"] == "PASS" else 2)
