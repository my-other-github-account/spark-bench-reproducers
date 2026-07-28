#!/usr/bin/env python3
"""P924 reweighted PRE-V3 solve on the frozen P860/P760 full-menu solver.

This wrapper keeps every option, byte coefficient, hard constraint, and corrected-grid
price unchanged while applying the preregistered normalized class-weighted objective
and atomic sell+buy paired greedy seeding before primal-heavy SCIP.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT.parent
RETRO_SCRIPT = PREVIEW / "retrodiction" / "retrodict_wire_b.py"
P760_SCRIPT = ROOT / "code" / "solve_p760_turbo.py"
CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
Q2_TIER = "qtip2_2.0117"
Q3_TIER = "qtip3_3.0117"
Q15_TIER = "qtip15_1.509117"
ENVELOPE_BYTES = 101_346_700_411
PREVIEW_LABEL = "V2_REWEIGHTED_PRE_V3"
TASK_ID = "PUBLIC_TASK"
DEFINITIVE_RERUN = "Re-solve on CORRECTED_PRICING_V3 when P923 lands."
CLASS_WEIGHTS = {
    "agentic": 1.0,
    "chat": 1.0,
    "code": 1.5,
    "multilingual": 2.0,
    "prose": 1.5,
    "reasoning": 1.0,
}
BOUND_SOURCE_ROOT: Path | None = None


def objective_value(prediction: dict[str, float]) -> float:
    denominator = math.fsum(CLASS_WEIGHTS.values())
    return math.fsum(CLASS_WEIGHTS[label] * float(prediction[label]) for label in CLASSES) / denominator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def retro_module():
    return load_module("p860_retrodiction_pricing", RETRO_SCRIPT)


def apply_measured_tier(
    opts: dict[tuple[int, int, str], dict[str, dict[str, Any]]],
    original: dict[tuple[int, int, str], str],
    tier: str,
    coverage_layers: list[int],
    receipt: dict[str, Any],
    logical_bytes_per_projection: dict[str, int],
) -> dict[str, Any]:
    layers = sorted(int(x) for x in coverage_layers)
    if not layers or len(layers) != len(set(layers)):
        raise RuntimeError(f"{tier} current inventory layer closure drift")
    byte_map = {str(k): int(v) for k, v in logical_bytes_per_projection.items()}
    if set(byte_map) != {"down", "fused13"} or any(value <= 0 for value in byte_map.values()):
        raise RuntimeError(f"{tier} logical-byte map drift")
    retro = retro_module()
    increments, closure = retro.anchor_apportionment(opts, original, layers, receipt)
    layer_set = set(layers)
    changed = 0
    for key, local in opts.items():
        if key[0] not in layer_set:
            local.pop(tier, None)
            continue
        base = local[original[key]]
        costs = {}
        for label in CLASSES:
            value = float(base["costs"][label]) + float(increments[key][label])
            if not math.isfinite(value) or value < -1e-12:
                raise RuntimeError(f"{tier} negative/nonfinite repriced cost at {key}/{label}: {value}")
            costs[label] = max(0.0, value)
        local[tier] = {
            "tier": tier,
            "bytes": byte_map[key[2]],
            "costs": costs,
            "pricing_basis": "P860_HONEST_BALANCED64_CANDIDATE_TO_PRE_REPAIR_RATIO_TRANSPORT",
            "coverage_status": receipt["coverage_status"],
        }
        changed += 1
    expected_cells = sum(1 for key in opts if key[0] in layer_set)
    if changed != expected_cells:
        raise RuntimeError(f"{tier} current inventory cell closure drift")
    baseline = {
        label: math.fsum(float(opts[key][original[key]]["costs"][label]) for key in opts)
        for label in CLASSES
    }
    uniform = dict(original)
    for key in opts:
        if key[0] in layer_set:
            uniform[key] = tier
    measured_ratio = {label: retro.anchor_ratio(receipt, label) for label in CLASSES}
    observed = {
        label: math.fsum(float(opts[key][uniform[key]]["costs"][label]) for key in opts)
        for label in CLASSES
    }
    ratio_error = {
        label: observed[label] - baseline[label] * measured_ratio[label]
        for label in CLASSES
    }
    if max(abs(value) for value in ratio_error.values()) > 1e-11:
        raise RuntimeError(f"{tier} transported ratio closure drift: {ratio_error}")
    return {
        "tier": tier,
        "coverage_status": receipt["coverage_status"],
        "coverage_layers": layers,
        "coverage_layer_count": len(layers),
        "current_inventory_cells": changed,
        "logical_bytes_per_projection": byte_map,
        "candidate_to_pre_repair_ratio_by_class": measured_ratio,
        "transported_model_delta_by_class": closure,
        "ratio_closure_error_by_class": ratio_error,
    }


def validate_bound_source(row: dict[str, Any]) -> dict[str, Any]:
    original_path = Path(row["path"])
    path = (
        (BOUND_SOURCE_ROOT / original_path.name).resolve()
        if BOUND_SOURCE_ROOT is not None
        else original_path.resolve()
    )
    expected = str(row["sha256"])
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"current-menu source binding drift: {path}")
    return {
        "path": str(path),
        "sealed_origin_path": str(original_path),
        "sha256": expected,
        "bytes": path.stat().st_size,
    }


def canonical_payload_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_v2_guard_config(
    config_path: Path,
    first_feasible_path: Path,
    p887_path: Path,
) -> tuple[dict[str, Any], float]:
    config = json.loads(config_path.read_text())
    first = json.loads(first_feasible_path.read_text())
    p887 = json.loads(p887_path.read_text())
    expected = {
        "schema": "wire-c-v2-reweighted-pre-v3-config-v1",
        "status": "PREREGISTERED_V2_REWEIGHTED_PRE_V3",
        "task_id": TASK_ID,
        "objective": "normalized class-weighted mean with multilingual protection and code moat",
        "class_weights": CLASS_WEIGHTS,
        "envelope_bytes": ENVELOPE_BYTES,
        "pack_fraction": 1.0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"V2 config drift at {key}")
    guard = config.get("hard_code_guard", {})
    if guard.get("class") != "code" or guard.get("operator") != "<=":
        raise RuntimeError("V2 hard CODE guard schema drift")
    cap = float(guard.get("cap_predicted_kld"))
    sealed_cap = float(first.get("prediction_by_class", {}).get("code"))
    if not math.isfinite(cap) or cap < 0 or abs(cap - sealed_cap) > 1e-15:
        raise RuntimeError({"V2_CODE_cap_binding_drift": {"config": cap, "sealed_first_feasible": sealed_cap}})
    pins = config.get("sealed_input_shas", {})
    actual = {
        "first_feasible": sha256(first_feasible_path),
        "p887_late_band": sha256(p887_path),
    }
    if any(pins.get(key) != value for key, value in actual.items()):
        raise RuntimeError({"V2_guard_input_pin_drift": {"expected": pins, "actual": actual}})
    observed = float(p887.get("six_classes", {}).get("code", {}).get("mean"))
    if abs(observed - 0.055876973966477825) > 1e-15:
        raise RuntimeError("P887 sealed code row drift")
    if abs(float(guard.get("p887_uniform_q3_code_kld")) - 0.053585325180321663) > 1e-15:
        raise RuntimeError("V2 config P887 uniform-q3 comparison drift")
    return config, cap


def validate_solve_authority(
    retro_gate: dict[str, Any],
    operator_override: dict[str, Any] | None,
) -> dict[str, Any]:
    """Accept the binding PASS or a narrowly scoped, fully disclosed baseline override."""
    if (
        retro_gate.get("status") == "PASS_RETRODICTION_GATE"
        and retro_gate.get("gates", {}).get("wire_c_build_permitted")
    ):
        return {
            "mode": "BINDING_RETRODICTION_PASS",
            "output_label": PREVIEW_LABEL,
            "retrodiction_gate_passed": True,
            "known_blind_spot_note": None,
            "definitive_rerun_required_after": DEFINITIVE_RERUN,
        }
    if operator_override is None:
        raise RuntimeError("retrodiction gate is not PASS and no operator baseline override was supplied")
    required = {
        "status": "AUTHORIZED_OPERATOR_BASELINE_PREVIEW",
        "task_id": "PUBLIC_TASK",
        "output_label": "WIRE_C_BASELINE_PREVIEW",
        "retrodiction_fail_disclosed": True,
        "known_blind_spot": "code late-band risk",
        "definitive_rerun_required_after": "120/120 seals",
    }
    for key, expected in required.items():
        if operator_override.get(key) != expected:
            raise RuntimeError(f"operator baseline override drift at {key}")
    decision = retro_gate.get("signed_gate_decision", {}).get("decision")
    if (
        retro_gate.get("status") != "FAIL_RETRODICTION_GATE_STOP_NO_WIRE_C_SOLVE_OR_BUILD"
        or decision != "STOP_NO_WIRE_C_SOLVE_OR_BUILD"
    ):
        raise RuntimeError("operator baseline override is only valid for the disclosed binding STOP receipt")
    table = retro_gate.get("retrodiction_table", {})
    code = table.get("code", {})
    chat = table.get("chat", {})
    code_measured = float(code.get("measured_percent"))
    code_retrodicted = float(code.get("anchor_corrected_retrodicted_percent"))
    code_error = float(code.get("absolute_error_percentage_points"))
    chat_error = float(chat.get("absolute_error_percentage_points"))
    observed = (code_measured, code_retrodicted, code_error, chat_error)
    if not all(math.isfinite(value) for value in observed):
        raise RuntimeError("nonfinite disclosed retrodiction failure")
    note = (
        "KNOWN BLIND SPOT — code late-band risk: binding Wire-B retrodiction FAIL was "
        f"measured {code_measured:+.1f}% versus retrodicted {code_retrodicted:+.12f}% "
        f"({code_error:.12f} pp absolute miss); chat miss {chat_error:.12f} pp. "
        "This operator-authorized baseline is diagnostic only, not a repaired pricing model."
    )
    return {
        "mode": "OPERATOR_AUTHORIZED_BASELINE_PREVIEW",
        "output_label": PREVIEW_LABEL,
        "retrodiction_gate_passed": False,
        "retrodiction_fail_disclosed": True,
        "known_blind_spot_note": note,
        "definitive_rerun_required_after": DEFINITIVE_RERUN,
        "override_receipt": operator_override,
    }


def stamp_auxiliary_preview_outputs(out: Path, authority: dict[str, Any]) -> None:
    excluded = {"ASSIGNMENT_QTIP2_QTIP3.json", "RESULT.json", "DONE.json", "PROGRESS.json"}
    for path in sorted(out.glob("*.json")):
        if path.name in excluded:
            continue
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise RuntimeError(f"unexpected non-object solver output: {path}")
        value.update({
            "output_label": PREVIEW_LABEL,
            "preview": True,
            "solve_authority": authority,
            "definitive_rerun_required_after": DEFINITIVE_RERUN,
        })
        atomic_json(path, value)


def first_feasible_preview_receipt(
    prediction: dict[str, float],
    exact_bytes: int,
    hint_path: str,
    hint_sha256: str,
    authority: dict[str, Any],
) -> dict[str, Any]:
    objective = objective_value(prediction)
    return {
        "schema": "v2-reweighted-pre-v3-first-feasible-v1",
        "status": "V2_REWEIGHTED_PRE_V3_FIRST_FEASIBLE",
        "task_id": TASK_ID,
        "output_label": PREVIEW_LABEL,
        "preview": True,
        "pre_v3": True,
        "objective_reweighted": objective,
        "objective_class_weights": CLASS_WEIGHTS,
        "solve_authority": authority,
        "known_blind_spot_note": authority.get("known_blind_spot_note"),
        "definitive_rerun_required_after": DEFINITIVE_RERUN,
        "prediction_by_class": prediction,
        "exact_bytes": int(exact_bytes),
        "envelope_bytes": ENVELOPE_BYTES,
        "envelope_slack_bytes": ENVELOPE_BYTES - int(exact_bytes),
        "feasibility": {
            "objective_finite_nonnegative": math.isfinite(objective) and objective >= 0.0,
            "all_classes_present": set(prediction) == set(CLASSES),
            "exact_bytes_at_or_below_envelope": int(exact_bytes) <= ENVELOPE_BYTES,
        },
        "warm_start_assignment": {"path": hint_path, "sha256": hint_sha256},
    }


def main() -> int:
    started_unix = time.time()
    started_monotonic = time.monotonic()
    rusage_before = resource.getrusage(resource.RUSAGE_SELF)
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-menu", type=Path, required=True)
    parser.add_argument("--q2-anchor", type=Path, required=True)
    parser.add_argument("--q3-anchor", type=Path, required=True)
    parser.add_argument("--q15-anchor", type=Path)
    parser.add_argument("--retrodiction", type=Path, required=True)
    parser.add_argument("--operator-baseline-override", type=Path)
    parser.add_argument("--corrected-grid-receipt", type=Path, required=True)
    parser.add_argument("--v2-config", type=Path, required=True)
    parser.add_argument("--first-feasible", type=Path, required=True)
    parser.add_argument("--p887-receipt", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out_p860_v2")
    parser.add_argument("--time-limit-seconds", type=float, default=3600.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="load, bind, reprice, prune and seed; never call SCIP")
    args = parser.parse_args()

    global BOUND_SOURCE_ROOT
    BOUND_SOURCE_ROOT = args.source_root.resolve()
    if not BOUND_SOURCE_ROOT.is_dir():
        raise RuntimeError(f"source root is not a directory: {BOUND_SOURCE_ROOT}")
    v2_config, code_cap = load_v2_guard_config(args.v2_config, args.first_feasible, args.p887_receipt)

    menu = json.loads(args.current_menu.read_text())
    if menu.get("status") != "PASS_CURRENT_SNAPSHOT" or menu.get("task_id") != "PUBLIC_TASK":
        raise RuntimeError("current-menu seal drift")
    objective_config = menu.get("objective", {})
    if int(objective_config.get("envelope_bytes", -1)) != ENVELOPE_BYTES:
        raise RuntimeError("Wire C exact envelope drift")
    if objective_config.get("class_weights") != CLASS_WEIGHTS:
        raise RuntimeError("Wire C V2 reweighted objective drift")
    sources = [validate_bound_source(row) for row in menu.get("sources", [])]
    if not sources:
        raise RuntimeError("current-menu source set is empty")
    retro_gate = json.loads(args.retrodiction.read_text())
    operator_override = (
        json.loads(args.operator_baseline_override.read_text())
        if args.operator_baseline_override is not None
        else None
    )
    solve_authority = validate_solve_authority(retro_gate, operator_override)
    grid_receipt = json.loads(args.corrected_grid_receipt.read_text())
    if grid_receipt.get("status") != "PASS_CORRECTED_GRID":
        raise RuntimeError("corrected-grid receipt is not PASS")
    if float(grid_receipt.get("safety", {}).get("pack_fraction_required", -1)) != 1.0:
        raise RuntimeError("pack=1.0 grid requirement drift")

    anchors = {
        Q2_TIER: (args.q2_anchor, json.loads(args.q2_anchor.read_text())),
        Q3_TIER: (args.q3_anchor, json.loads(args.q3_anchor.read_text())),
    }
    if Q15_TIER in menu["tiers"]:
        if args.q15_anchor is None:
            raise RuntimeError("current K1 inventory requires an honest q15 anchor")
        anchors[Q15_TIER] = (args.q15_anchor, json.loads(args.q15_anchor.read_text()))
    expected_anchor_kind = {Q2_TIER: "qtip2", Q3_TIER: "qtip3", Q15_TIER: "qtip15"}
    for tier, (path, receipt) in anchors.items():
        spec = menu["tiers"].get(tier)
        if spec is None:
            raise RuntimeError(f"anchor supplied for tier absent from current menu: {tier}")
        if receipt.get("status") != "PASS_BALANCED64_V1" or receipt.get("anchor") != expected_anchor_kind[tier]:
            raise RuntimeError(f"unsealed/mismatched anchor for {tier}")
        if sorted(receipt.get("coverage", {}).get("coverage_layers", [])) != sorted(spec["layers"]):
            raise RuntimeError(f"{tier} anchor/menu layer drift")
        grid_anchor = grid_receipt.get("anchors", {}).get(expected_anchor_kind[tier], {})
        if grid_anchor.get("sha256") != sha256(path):
            raise RuntimeError(f"{tier} corrected-grid anchor binding drift")
    if sorted(menu["tiers"][Q3_TIER]["layers"]) != list(range(3, 43)):
        raise RuntimeError("current qtip3 menu must be exact 40/40 L003-L042")

    input_pins = {
        "solver": {
            "solve_p860": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "solve_p760": {"path": str(P760_SCRIPT), "sha256": sha256(P760_SCRIPT)},
            "solve_p693": {"path": str(ROOT / "code" / "solve_p693_turbo.py"), "sha256": sha256(ROOT / "code" / "solve_p693_turbo.py")},
            "retrodiction_pricing": {"path": str(RETRO_SCRIPT), "sha256": sha256(RETRO_SCRIPT)},
        },
        "data": {
            "current_menu": {"path": str(args.current_menu.resolve()), "sha256": sha256(args.current_menu)},
            "qtip2_anchor": {"path": str(args.q2_anchor.resolve()), "sha256": sha256(args.q2_anchor)},
            "qtip3_anchor": {"path": str(args.q3_anchor.resolve()), "sha256": sha256(args.q3_anchor)},
            "retrodiction": {"path": str(args.retrodiction.resolve()), "sha256": sha256(args.retrodiction)},
            "corrected_grid_receipt": {"path": str(args.corrected_grid_receipt.resolve()), "sha256": sha256(args.corrected_grid_receipt)},
            "v2_config": {"path": str(args.v2_config.resolve()), "sha256": sha256(args.v2_config)},
            "first_feasible_guard_basis": {"path": str(args.first_feasible.resolve()), "sha256": sha256(args.first_feasible)},
            "p887_late_band": {"path": str(args.p887_receipt.resolve()), "sha256": sha256(args.p887_receipt)},
            "source_root": str(BOUND_SOURCE_ROOT),
            "bound_menu_sources": sources,
        },
        "objective": {
            "payload": objective_config,
            "sha256": canonical_payload_sha256(objective_config),
        },
        "hard_code_guard": v2_config["hard_code_guard"],
        "configuration": {
            "payload": {
                "output_label": PREVIEW_LABEL,
                "time_limit_seconds": float(args.time_limit_seconds),
                "threads": int(args.threads),
                "pack_fraction": 1.0,
                "current_snapshot_is_menu": True,
                "qtip15_included": Q15_TIER in menu["tiers"],
                "constraint_preserving_dominance_pruning": True,
                "primal_heavy_heuristics": ["RENS", "RINS", "feaspump", "rounding"],
                "nonregression_reference": "legal_current_menu_greedy_hint",
                "hard_code_cap_predicted_kld": code_cap,
                "dry_run": bool(args.dry_run),
            },
        },
    }
    input_pins["configuration"]["sha256"] = canonical_payload_sha256(input_pins["configuration"]["payload"])
    if args.operator_baseline_override is not None:
        input_pins["data"]["operator_baseline_override"] = {
            "path": str(args.operator_baseline_override.resolve()),
            "sha256": sha256(args.operator_baseline_override),
        }

    p760 = load_module("p924_frozen_p760", P760_SCRIPT)
    p760.p693.CLASS_WEIGHTS = dict(CLASS_WEIGHTS)
    frozen_build_surface = p760.p693.build_surface
    repricing_cache: dict[str, Any] = {}

    def build_surface_p860():
        surface = frozen_build_surface()
        opts, original = surface["opts"], surface["original"]
        reports = {}
        for tier, (_path, receipt) in anchors.items():
            spec = menu["tiers"][tier]
            reports[tier] = apply_measured_tier(
                opts,
                original,
                tier,
                [int(x) for x in spec["layers"]],
                receipt,
                spec["logical_bytes_per_projection"],
            )
        for tier in (Q2_TIER, Q3_TIER, Q15_TIER):
            if tier not in anchors:
                for local in opts.values():
                    local.pop(tier, None)
        repricing_cache.clear()
        repricing_cache.update(reports)
        original_code = float(surface["original_pred"]["code"])
        if abs(original_code - code_cap) > 1e-12:
            raise RuntimeError({"V2_CODE_guard_surface_drift": {"surface_original_code": original_code, "sealed_cap": code_cap}})
        surface["ceilings"]["code"] = code_cap
        surface["wire_c_v2_code_guard"] = {
            **v2_config["hard_code_guard"],
            "surface_original_code_predicted_kld": original_code,
            "enforced_solver_ceiling": float(surface["ceilings"]["code"]),
        }
        surface["p860_current_menu"] = menu
        surface["p860_repricing"] = reports
        return surface

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "PREVIEW_NOTICE.json", {
        "schema": "wire-c-baseline-preview-notice-v1",
        "status": "WIRE_C_BASELINE_PREVIEW_AUTHORIZED",
        "output_label": PREVIEW_LABEL,
        "preview": True,
        "solve_authority": solve_authority,
        "known_blind_spot_note": solve_authority.get("known_blind_spot_note"),
        "definitive_rerun_required_after": DEFINITIVE_RERUN,
        "pins": input_pins,
    })
    p760.p693.build_surface = build_surface_p860
    p760.p693.OUT = out
    p760.p693.TIME_LIMIT_SECONDS = float(args.time_limit_seconds)
    p760.p693.SOLVER_THREADS = int(args.threads)
    p760.p693.CONSTRAINT_PRESERVING_DOMINANCE_PRUNING = True
    p760.p693.PRIMAL_HEAVY_HEURISTICS = True
    p760.p693.NONREGRESSION_REFERENCE = "legal_hint"
    first_surface = build_surface_p860()
    requested_hint_path = Path(os.environ.get(
        "P693_HINT_ASSIGNMENT",
        str(ROOT / "source_p637_out" / "ASSIGNMENT_RESPENT.json"),
    ))
    requested_hint = p760.p693.load_assignment(requested_hint_path, first_surface["gs"])
    invalid_hint_cells = [
        {"layer": key[0], "expert": key[1], "projection": key[2], "tier": tier}
        for key, tier in sorted(requested_hint.items())
        if tier not in first_surface["opts"][key]
    ]
    first_selected = dict(first_surface["original"]) if invalid_hint_cells else requested_hint
    gs = first_surface["gs"]
    hint_payload = {
        "schema": "wire-c-baseline-preview-warm-start-v1",
        "status": "PASS_FEASIBLE_WARM_START",
        "output_label": PREVIEW_LABEL,
        "preview": True,
        "source_requested_hint": {"path": str(requested_hint_path.resolve()), "sha256": sha256(requested_hint_path)},
        "fallback_to_exact_original_incumbent": bool(invalid_hint_cells),
        "fallback_reason": "requested P637 hint selects qtip2 on a layer absent from the exact sealed current menu" if invalid_hint_cells else None,
        "invalid_requested_hint_cells": invalid_hint_cells,
        "assignment": {
            str(layer): {
                str(expert): {projection: first_selected[(layer, expert, projection)] for projection in gs.PROJECTIONS}
                for expert in range(gs.EXPERTS)
            }
            for layer in range(gs.LAYERS)
        },
        "solve_authority": solve_authority,
        "known_blind_spot_note": solve_authority.get("known_blind_spot_note"),
        "definitive_rerun_required_after": DEFINITIVE_RERUN,
    }
    hint_path = out / "WIRE_C_BASELINE_PREVIEW_WARM_START.json"
    atomic_json(hint_path, hint_payload)
    os.environ["P693_HINT_ASSIGNMENT"] = str(hint_path)
    first_prediction, first_bytes = p760.p693.summarize(
        first_surface["gs"], first_surface["opts"], first_surface["original"], first_selected
    )
    first_receipt = first_feasible_preview_receipt(
        first_prediction, first_bytes, str(hint_path.resolve()), sha256(hint_path), solve_authority
    )
    if not all(first_receipt["feasibility"].values()):
        raise RuntimeError({"invalid_first_feasible_preview": first_receipt})
    atomic_json(out / "FIRST_FEASIBLE_PREVIEW.json", first_receipt)
    print(json.dumps(first_receipt, sort_keys=True), flush=True)
    greedy_selected, greedy_prediction, greedy_bytes, greedy_ledger = p760.p693.paired_greedy_postpass(
        first_surface["gs"],
        first_surface["opts"],
        dict(first_selected),
        dict(first_prediction),
        first_bytes,
        first_surface["ceilings"],
    )
    greedy_objective = objective_value(greedy_prediction)
    first_objective = float(first_receipt["objective_reweighted"])
    changed_keys = [key for key in greedy_selected if greedy_selected[key] != first_selected[key]]
    greedy_feasibility = {
        "exact_bytes_at_or_below_envelope": greedy_bytes <= ENVELOPE_BYTES,
        "objective_finite_nonnegative": math.isfinite(greedy_objective) and greedy_objective >= 0.0,
        "uniform_six_classes_complete": set(greedy_prediction) == set(CLASSES),
        "per_class_hard_ceilings_satisfied": all(
            -1e-12 <= greedy_prediction[label] <= first_surface["ceilings"][label] + 1e-12
            for label in CLASSES
        ),
        "strictly_improves_first_feasible": greedy_objective < first_objective - 1e-14,
        "nonzero_changed_cells": bool(changed_keys),
    }
    greedy_seed_gate_keys = (
        "exact_bytes_at_or_below_envelope",
        "objective_finite_nonnegative",
        "uniform_six_classes_complete",
        "per_class_hard_ceilings_satisfied",
    )
    if not all(greedy_feasibility[key] for key in greedy_seed_gate_keys):
        raise RuntimeError({
            "greedy_feasible_seed_gate": greedy_feasibility,
            "first_objective": first_objective,
            "greedy_objective": greedy_objective,
            "changed_cells": len(changed_keys),
        })
    greedy_map = {
        str(layer): {
            str(expert): {
                projection: greedy_selected[(layer, expert, projection)]
                for projection in gs.PROJECTIONS
            }
            for expert in range(gs.EXPERTS)
        }
        for layer in range(gs.LAYERS)
    }
    greedy_map_sha = canonical_payload_sha256(greedy_map)
    greedy_tier_counts = Counter(greedy_selected.values())
    greedy_assignment = {
        "schema": "v2-reweighted-pre-v3-paired-seed-assignment-v1",
        "status": "V2_REWEIGHTED_PRE_V3_PAIRED_GREEDY_SEED",
        "task_id": TASK_ID,
        "output_label": PREVIEW_LABEL,
        "preview": True,
        "pre_v3": True,
        "objective_class_weights": CLASS_WEIGHTS,
        "assignment": greedy_map,
        "assignment_map_sha256": greedy_map_sha,
        "source_first_feasible_sha256": sha256(out / "FIRST_FEASIBLE_PREVIEW.json"),
        "current_menu_sha256": sha256(args.current_menu),
        "solve_authority": solve_authority,
        "known_blind_spot_note": solve_authority.get("known_blind_spot_note"),
        "definitive_rerun_required_after": DEFINITIVE_RERUN,
    }
    greedy_path = out / "V2_REWEIGHTED_PRE_V3_PAIRED_GREEDY_SEED.json"
    greedy_assignment_sha = atomic_json(greedy_path, greedy_assignment)
    paired_ledger_path = out / "PAIRED_GREEDY_LEDGER.json"
    paired_ledger_sha = atomic_json(paired_ledger_path, {
        "schema": "v2-reweighted-pre-v3-paired-greedy-ledger-v1",
        "task_id": TASK_ID,
        "output_label": PREVIEW_LABEL,
        "pair_count": len(greedy_ledger),
        "pairs": greedy_ledger,
    })
    first_improved = {
        "schema": "v2-reweighted-pre-v3-paired-greedy-feasible-seed-v1",
        "status": (
            "V2_REWEIGHTED_PRE_V3_FIRST_IMPROVED"
            if greedy_feasibility["strictly_improves_first_feasible"]
            else "V2_REWEIGHTED_PRE_V3_FEASIBLE_SEED_UNCHANGED"
        ),
        "task_id": TASK_ID,
        "output_label": PREVIEW_LABEL,
        "preview": True,
        "pre_v3": True,
        "objective_reweighted": greedy_objective,
        "objective_class_weights": CLASS_WEIGHTS,
        "objective_delta_vs_first_feasible": greedy_objective - first_objective,
        "prediction_by_class": greedy_prediction,
        "exact_bytes": greedy_bytes,
        "envelope_bytes": ENVELOPE_BYTES,
        "envelope_slack_bytes": ENVELOPE_BYTES - greedy_bytes,
        "changed_cell_count_vs_seed": len(changed_keys),
        "paired_swap_count": len(greedy_ledger),
        "paired_swap_rounds": max((int(row["round"]) for row in greedy_ledger), default=0),
        "tier_counts": dict(sorted(greedy_tier_counts.items())),
        "selected_cell_counts": {
            "QTIP2": greedy_tier_counts[Q2_TIER],
            "QTIP3": greedy_tier_counts[Q3_TIER],
            "QTIP15": greedy_tier_counts[Q15_TIER],
        },
        "paired_swap_ledger": {"path": str(paired_ledger_path.resolve()), "sha256": paired_ledger_sha},
        "feasibility": greedy_feasibility,
        "assignment": {"path": str(greedy_path.resolve()), "sha256": greedy_assignment_sha, "map_sha256": greedy_map_sha},
        "current_menu_sha256": sha256(args.current_menu),
        "corrected_grid_csv_sha256": "74869b5f8e3ef4eb43dc98c6ee060c2d9ad048bb215cadd308fb2c9983933dda",
        "solve_authority": solve_authority,
        "known_blind_spot_note": solve_authority.get("known_blind_spot_note"),
        "definitive_rerun_required_after": DEFINITIVE_RERUN,
    }
    atomic_json(out / "FIRST_IMPROVED_PAIRED_GREEDY.json", first_improved)
    print(json.dumps(first_improved, sort_keys=True), flush=True)
    if args.dry_run:
        model_opts, dominance_report = p760.p693.constraint_preserving_dominance_prune(
            gs,
            first_surface["opts"],
            first_surface["original"],
            greedy_selected,
        )
        dominance_sha = atomic_json(out / "DOMINANCE_PRUNING_DRY_RUN.json", dominance_report)
        checks = {
            "metadata_sources_loaded_and_hashed": bool(sources),
            "current_menu_exact_q3_40_q2_39_k1_absent": (
                len(menu["tiers"][Q3_TIER]["layers"]) == 40
                and len(menu["tiers"][Q2_TIER]["layers"]) == 39
                and Q15_TIER not in menu["tiers"]
            ),
            "reweighted_objective_exact": objective_config.get("class_weights") == CLASS_WEIGHTS,
            "exact_envelope_preserved": int(objective_config.get("envelope_bytes")) == ENVELOPE_BYTES,
            "hard_code_guard_bound_to_sealed_incumbent": abs(float(first_surface["ceilings"]["code"]) - code_cap) <= 1e-15,
            "greedy_seed_satisfies_hard_code_guard": float(greedy_prediction["code"]) <= code_cap + 1e-12,
            "greedy_seed_is_feasible": all(greedy_feasibility[key] for key in greedy_seed_gate_keys),
            "dominance_pruning_pass": dominance_report.get("status") == "PASS",
            "dominance_pruning_removed_options": int(dominance_report.get("removed_option_variables", 0)) > 0,
            "original_and_hint_options_retained": all(
                first_surface["original"][key] in model_opts[key] and greedy_selected[key] in model_opts[key]
                for key in model_opts
            ),
            "scip_not_invoked": True,
        }
        dry_receipt = {
            "schema": "v2-reweighted-pre-v3-metadata-pruning-dry-run-v1",
            "status": "PASS_V2_REWEIGHTED_PRE_V3_DRY_RUN" if all(checks.values()) else "FAIL_V2_REWEIGHTED_PRE_V3_DRY_RUN",
            "task_id": TASK_ID,
            "output_label": PREVIEW_LABEL,
            "checks": checks,
            "pins": input_pins,
            "hard_code_guard": first_surface["wire_c_v2_code_guard"],
            "repricing": repricing_cache,
            "greedy_seed": {
                "objective_reweighted": greedy_objective,
                "objective_class_weights": CLASS_WEIGHTS,
                "prediction_by_class": greedy_prediction,
                "exact_bytes": greedy_bytes,
                "slack_bytes": ENVELOPE_BYTES - greedy_bytes,
                "changed_cells": len(changed_keys),
                "assignment_sha256": greedy_assignment_sha,
                "assignment_map_sha256": greedy_map_sha,
            },
            "dominance_pruning": {k: v for k, v in dominance_report.items() if k != "removed"},
            "dominance_pruning_receipt_sha256": dominance_sha,
            "solver_call_count": 0,
        }
        dry_sha = atomic_json(out / "V2_REWEIGHTED_PRE_V3_DRY_RUN.json", dry_receipt)
        if dry_receipt["status"] != "PASS_V2_REWEIGHTED_PRE_V3_DRY_RUN":
            raise RuntimeError(dry_receipt)
        print(json.dumps({"status": dry_receipt["status"], "receipt_sha256": dry_sha}, sort_keys=True))
        return 0
    os.environ["P693_HINT_ASSIGNMENT"] = str(greedy_path)
    rc = int(p760.p693.main() or 0)
    if rc != 0:
        return rc

    stamp_auxiliary_preview_outputs(out, solve_authority)

    surface = build_surface_p860()
    gs, opts, original = surface["gs"], surface["opts"], surface["original"]
    assignment_path = out / "ASSIGNMENT_QTIP2_QTIP3.json"
    assignment = json.loads(assignment_path.read_text())
    assignment.update({
        "schema": "v2-reweighted-pre-v3-current-snapshot-assignment-v1",
        "task_id": TASK_ID,
        "output_label": PREVIEW_LABEL,
        "preview": True,
        "pre_v3": True,
        "solve_authority": solve_authority,
        "known_blind_spot_note": solve_authority.get("known_blind_spot_note"),
        "definitive_rerun_required_after": DEFINITIVE_RERUN,
        "current_menu_sha256": sha256(args.current_menu),
        "retrodiction_sha256": sha256(args.retrodiction),
        "corrected_grid_receipt_sha256": sha256(args.corrected_grid_receipt),
    })
    assignment_sha = atomic_json(assignment_path, assignment)
    selected = p760.p693.load_assignment(assignment_path, gs)
    pred, exact_bytes = p760.p693.summarize(gs, opts, original, selected)
    objective = objective_value(pred)

    menu_law_violations = []
    for key, tier in selected.items():
        if tier in (Q2_TIER, Q3_TIER, Q15_TIER):
            if tier not in menu["tiers"] or int(key[0]) not in set(menu["tiers"][tier]["layers"]):
                menu_law_violations.append({"key": list(key), "tier": tier})

    result_path = out / "RESULT.json"
    result = json.loads(result_path.read_text())
    arm = result["arms"]["with_2bit_plus_3bit"]
    if abs(float(arm["objective"]) - objective) > 1e-12 or int(arm["exact_bytes"]) != exact_bytes:
        raise RuntimeError("P860 postprocess solve reproduction drift")
    ceilings = result.get("constraints", {}).get("per_class_hard_ceilings", {})
    predictions = arm.get("prediction_by_class", {})
    feasibility = {
        "exact_bytes_at_or_below_envelope": exact_bytes <= ENVELOPE_BYTES,
        "exact_bytes": exact_bytes,
        "envelope_bytes": ENVELOPE_BYTES,
        "envelope_slack_bytes": ENVELOPE_BYTES - exact_bytes,
        "objective_finite_nonnegative": math.isfinite(objective) and objective >= 0.0,
        "uniform_six_classes_complete": set(predictions) == set(CLASSES),
        "per_class_hard_ceilings_complete": set(ceilings) == set(CLASSES),
        "per_class_hard_ceilings_satisfied": (
            set(predictions) == set(CLASSES)
            and set(ceilings) == set(CLASSES)
            and all(float(predictions[label]) <= float(ceilings[label]) + 1e-12 for label in CLASSES)
        ),
        "current_snapshot_is_menu": not menu_law_violations,
        "menu_law_violations": menu_law_violations,
        "qtip3_exact_40_of_40": sorted(menu["tiers"][Q3_TIER]["layers"]) == list(range(3, 43)),
        "qtip2_exact_as_sealed": sorted(menu["tiers"][Q2_TIER]["layers"]) == sorted(anchors[Q2_TIER][1]["coverage"]["coverage_layers"]),
        "k1_as_sealed": (Q15_TIER in menu["tiers"]) == (Q15_TIER in anchors),
        "pack_fraction": 1.0,
        "hard_code_guard_satisfied": float(predictions.get("code", math.inf)) <= code_cap + 1e-12,
        "hard_code_guard_cap": code_cap,
        "hard_code_guard_slack": code_cap - float(predictions.get("code", math.inf)),
    }
    if not all(value for key, value in feasibility.items() if key not in {
        "exact_bytes", "envelope_bytes", "envelope_slack_bytes", "menu_law_violations",
        "pack_fraction", "hard_code_guard_cap", "hard_code_guard_slack",
    }):
        raise RuntimeError({"Wire_C_preview_feasibility_failure": feasibility})
    rusage_after = resource.getrusage(resource.RUSAGE_SELF)
    runtime = {
        "started_unix": started_unix,
        "completed_unix": time.time(),
        "wall_seconds": time.monotonic() - started_monotonic,
        "user_cpu_seconds": rusage_after.ru_utime - rusage_before.ru_utime,
        "system_cpu_seconds": rusage_after.ru_stime - rusage_before.ru_stime,
        "max_rss_kib": rusage_after.ru_maxrss,
    }
    result.update({
        "schema": "v2-reweighted-pre-v3-current-snapshot-solve-v1",
        "status": "PASS_V2_REWEIGHTED_PRE_V3_FEASIBLE_CURRENT_SNAPSHOT",
        "task_id": TASK_ID,
        "output_label": PREVIEW_LABEL,
        "preview": True,
        "pre_v3": True,
        "objective_reweighted": objective,
        "objective_class_weights": CLASS_WEIGHTS,
        "solve_authority": solve_authority,
        "known_blind_spot_note": solve_authority.get("known_blind_spot_note"),
        "definitive_rerun_required_after": DEFINITIVE_RERUN,
        "pins": input_pins,
        "objective_configuration": objective_config,
        "preview_feasibility": feasibility,
        "runtime": runtime,
        "current_menu": {"path": str(args.current_menu.resolve()), "sha256": sha256(args.current_menu), "payload": menu},
        "corrected_grid_receipt": {"path": str(args.corrected_grid_receipt.resolve()), "sha256": sha256(args.corrected_grid_receipt)},
        "retrodiction_gate": {"path": str(args.retrodiction.resolve()), "sha256": sha256(args.retrodiction)},
        "anchor_repricing": repricing_cache,
        "source_bindings": sources,
        "assignment_receipt_sha256": assignment_sha,
    })
    arm["label"] = "V2_REWEIGHTED_PRE_V3_current_exact_qtip3_qtip2_inventory"
    result["rungs"] = {
        tier: p760.p693.rung_summary(gs, opts, original, selected, tier, tuple(menu["tiers"][tier]["layers"]))
        for tier in anchors
    }
    qtiers = set(anchors)
    qkeys = {key for key, tier in selected.items() if tier in qtiers}
    qdelta = sum(int(opts[key][selected[key]]["bytes"]) - int(opts[key][original[key]]["bytes"]) for key in qkeys)
    result["bytes"]["all_current_qtip_net_delta"] = qdelta
    result["bytes"]["all_current_qtip_net_bytes_freed"] = -qdelta
    changed_cells = []
    for key in sorted(selected):
        if selected[key] == original[key]:
            continue
        old_opt = opts[key][original[key]]
        new_opt = opts[key][selected[key]]
        changed_cells.append({
            "layer": key[0],
            "expert": key[1],
            "projection": key[2],
            "from": original[key],
            "to": selected[key],
            "delta_bytes": int(new_opt["bytes"]) - int(old_opt["bytes"]),
            "delta_by_class": {
                label: float(new_opt["costs"][label]) - float(old_opt["costs"][label])
                for label in CLASSES
            },
        })
    purchase_tiers = {}
    for tier in (Q2_TIER, Q3_TIER, Q15_TIER):
        keys = [key for key, selected_tier in selected.items() if selected_tier == tier]
        deltas = [int(opts[key][tier]["bytes"]) - int(opts[key][original[key]]["bytes"]) for key in keys]
        purchase_tiers[tier] = {
            "selected_cells": len(keys),
            "selected_logical_bytes": sum(int(opts[key][tier]["bytes"]) for key in keys),
            "gross_bytes_freed": sum(max(0, -delta) for delta in deltas),
            "gross_bytes_spent": sum(max(0, delta) for delta in deltas),
            "net_byte_delta": sum(deltas),
        }
    purchase_table = {
        "schema": "v2-reweighted-pre-v3-purchase-table-v1",
        "status": "PASS_V2_REWEIGHTED_PRE_V3_PURCHASE_TABLE",
        "task_id": TASK_ID,
        "changed_cell_count": len(changed_cells),
        "changed_cells": changed_cells,
        "selected_cell_counts": {
            "QTIP2": purchase_tiers[Q2_TIER]["selected_cells"],
            "QTIP3": purchase_tiers[Q3_TIER]["selected_cells"],
            "QTIP15": purchase_tiers[Q15_TIER]["selected_cells"],
        },
        "bytes_by_tier": purchase_tiers,
        "total_qtip_selected_logical_bytes": sum(row["selected_logical_bytes"] for row in purchase_tiers.values()),
        "total_package_bytes": exact_bytes,
        "package_slack_bytes": ENVELOPE_BYTES - exact_bytes,
        "objective_reweighted": objective,
        "objective_class_weights": CLASS_WEIGHTS,
        "prediction_by_class": pred,
        "assignment_sha256": assignment_sha,
        "assignment_map_sha256": assignment["assignment_map_sha256"],
        "hard_code_guard": {
            "cap_predicted_kld": code_cap,
            "observed_predicted_kld": float(pred["code"]),
            "slack": code_cap - float(pred["code"]),
            "status": "PASS" if float(pred["code"]) <= code_cap + 1e-12 else "FAIL",
        },
    }
    purchase_table_sha = atomic_json(out / "PURCHASE_TABLE.json", purchase_table)
    result["purchase_table"] = {"path": str(out / "PURCHASE_TABLE.json"), "sha256": purchase_table_sha}
    result_sha = atomic_json(result_path, result)
    done = {
        "schema": "v2-reweighted-pre-v3-current-snapshot-done-v1",
        "status": result["status"],
        "task_id": TASK_ID,
        "output_label": PREVIEW_LABEL,
        "preview": True,
        "pre_v3": True,
        "solve_authority": solve_authority,
        "known_blind_spot_note": solve_authority.get("known_blind_spot_note"),
        "definitive_rerun_required_after": DEFINITIVE_RERUN,
        "preview_feasibility": feasibility,
        "runtime": runtime,
        "pins": input_pins,
        "result": str(result_path),
        "result_sha256": result_sha,
        "assignment": str(assignment_path),
        "assignment_sha256": assignment_sha,
        "assignment_map_sha256": assignment["assignment_map_sha256"],
        "purchase_table": str(out / "PURCHASE_TABLE.json"),
        "purchase_table_sha256": purchase_table_sha,
        "exact_logical_bytes": exact_bytes,
        "objective_reweighted": objective,
        "objective_class_weights": CLASS_WEIGHTS,
        "current_menu_sha256": sha256(args.current_menu),
        "retrodiction_sha256": sha256(args.retrodiction),
    }
    done_sha = atomic_json(out / "DONE.json", done)
    atomic_json(out / "PROGRESS.json", {
        **done,
        "status": "V2_REWEIGHTED_PRE_V3_SOLVER_EXITED_RESULT_READY",
        "done_sha256": done_sha,
    })
    print(json.dumps({**done, "done_sha256": done_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
