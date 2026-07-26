#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sources/P629"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("p629", SRC / "code/solve_global_ab.py")
p629 = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(p629)
gs = p629.load_original()
_manifest, _verified = p629.validate_inputs(gs)
anchors = gs.load_anchor_grid(SRC / "inputs/rung1/ANCHOR_VERTICAL_GRID.csv")
rows = gs.load_profile(SRC / "inputs/profile/PROFILE_ROWS.jsonl")
importance, _norm = gs.normalize_profile_rows(rows)
step0 = gs.step0_means(SRC / "inputs/baseline/BQ3_STEP0_PER_CLASS.json")
old, _ = gs.map_incumbent(SRC / "inputs/baseline/DUALVQ_K4096MENU_BQ3_BIN_MANIFEST.json")
corrections, _ = gs.fit_projection_corrections(old, importance, anchors, step0)
cells = gs.make_cells(importance, anchors, corrections)
_assignment_doc, incumbent = p629.read_assignment(gs)
predicted = gs.predict_assignment(
    incumbent, importance, anchors, gs.CLASSES, corrections=corrections
)
objective = math.fsum(predicted[c] for c in gs.CLASSES) / len(gs.CLASSES)
lineage = json.loads((SRC / "lineage/FRONTIER.json").read_text())
errors = {
    c: abs(predicted[c] - float(lineage["nomination"]["predicted_class_mean_kld"][c]))
    for c in gs.CLASSES
}
counts = Counter(incumbent.values())
expected_counts = Counter(lineage["nomination"]["tier_counts"])

# Seal the complete pre-QTIP menu before constructing the WITH arm.
h = hashlib.sha256()
option_counts = Counter()
for cell in cells:
    row = {"key": list(cell["key"]), "options": cell["options"]}
    h.update(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    h.update(b"\n")
    for option in cell["options"]:
        option_counts[option["tier"]] += 1
menu_sha = h.hexdigest()

assignment_map = {
    str(layer): {
        str(expert): {
            projection: incumbent[(layer, expert, projection)]
            for projection in gs.PROJECTIONS
        }
        for expert in range(gs.EXPERTS)
    }
    for layer in range(gs.LAYERS)
}
map_sha = hashlib.sha256(
    json.dumps(assignment_map, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
assignment_receipt = {
    "schema": "p629-class-balanced-global-assignment-v1",
    "arm": "WITHOUT",
    "measurement_label": "PREDICTION_ONLY_ARTIFACT_RELATIVE_SWAP_BASIS",
    "assignment": assignment_map,
    "assignment_map_sha256": map_sha,
    "input_manifest_sha256": p629.EXPECTED_INPUT_MANIFEST_SHA,
    "incumbent_assignment_sha256": p629.EXPECTED_ASSIGNMENT_SHA,
}
assignment_receipt_payload = (
    json.dumps(assignment_receipt, indent=2, sort_keys=True) + "\n"
).encode()
assignment_receipt_sha = hashlib.sha256(assignment_receipt_payload).hexdigest()

sealed_result = SRC / "out/without/WITHOUT_RESULT.json"
sealed_assignment = SRC / "out/without/ASSIGNMENT_WITHOUT.json"
sealed_done = json.loads((SRC / "out/without/DONE.json").read_text())
byte_contract = {
    "envelope": p629.ENVELOPE,
    "without_exact": p629.ENVELOPE,
    "modeled_payload": int(lineage["nomination"]["total_wire_bytes"]),
    "canonical_overhead": p629.ENVELOPE - int(lineage["nomination"]["total_wire_bytes"]),
}
byte_sha = hashlib.sha256(
    json.dumps(byte_contract, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
checks = {
    "input_manifest_sha": sha(SRC / "inputs/INPUT_MANIFEST.json")
    == p629.EXPECTED_INPUT_MANIFEST_SHA,
    "incumbent_assignment_sha": sha(SRC / "lineage/NOMINATED_ASSIGNMENT.json")
    == p629.EXPECTED_ASSIGNMENT_SHA,
    "canonical_replay": max(errors.values()) <= 1e-12,
    "tier_counts": counts == expected_counts,
    "class_balanced_objective": abs(objective - float(sealed_done["objective"])) <= 1e-15,
    "exact_bytes": int(sealed_done["exact_bytes"]) == p629.ENVELOPE,
    "assignment_map_sha": map_sha == sealed_done["assignment_map_sha256"],
    "assignment_receipt_sha": assignment_receipt_sha
    == sha(sealed_assignment)
    == sealed_done["assignment_receipt_sha256"],
    "sealed_result_sha": sha(sealed_result) == sealed_done["result_sha256"],
    "no_qtip_in_existing_menu": all(k != p629.QTIP_TIER for k in option_counts),
}
doc = {
    "schema": "p637-step1-p629-without-reproduction-v1",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "host": os.uname().nodename,
    "objective_name": "uniform mean of six predicted class KLDs",
    "objective": objective,
    "prediction_by_class": predicted,
    "exact_physical_bytes": p629.ENVELOPE,
    "byte_contract": byte_contract,
    "byte_contract_sha256": byte_sha,
    "tier_counts": dict(sorted(counts.items())),
    "input_manifest_sha256": p629.EXPECTED_INPUT_MANIFEST_SHA,
    "incumbent_assignment_sha256": p629.EXPECTED_ASSIGNMENT_SHA,
    "assignment_map_sha256": map_sha,
    "assignment_receipt_sha256": assignment_receipt_sha,
    "existing_menu_sha256": menu_sha,
    "existing_menu_cells": len(cells),
    "existing_menu_option_counts": dict(sorted(option_counts.items())),
    "sealed_p629_without_result_sha256": sha(sealed_result),
    "p629_code_sha256": sha(SRC / "code/solve_global_ab.py"),
    "canonical_reproduction_max_abs_error": max(errors.values()),
}
out = ROOT / "receipts/STEP1_WITHOUT_REPRO.json"
tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out)
summary = {
    "receipt": str(out),
    "receipt_sha256": sha(out),
    **{
        key: doc[key]
        for key in (
            "status",
            "objective",
            "exact_physical_bytes",
            "input_manifest_sha256",
            "incumbent_assignment_sha256",
            "assignment_map_sha256",
            "assignment_receipt_sha256",
            "existing_menu_sha256",
            "p629_code_sha256",
        )
    },
}
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if doc["status"] == "PASS" else 2)
