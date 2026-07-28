#!/usr/bin/env python3
"""Fit the P923 transition-aware pricing correction from sealed wire receipts.

The model predicts *deltas from each wire's matched physical control*, avoiding
cross-window-basis leakage.  QTIP2/QTIP3 features are selected logical GiB;
ordinary VQ/native changes use the exact frozen-surface prediction delta.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
Q2 = "qtip2_2.0117"
Q3 = "qtip3_3.0117"
CALIBRATION_WIRES = ("wire_b", "batch2", "wire_c")
RIDGE_LAMBDA = 1e-4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    work = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        if abs(divisor) < 1e-20:
            raise RuntimeError("singular calibration matrix")
        for j in range(column, n + 1):
            work[column][j] /= divisor
        for row in range(n):
            if row == column:
                continue
            multiplier = work[row][column]
            for j in range(column, n + 1):
                work[row][j] -= multiplier * work[column][j]
    return [work[i][n] for i in range(n)]


def load_assignment(path: Path, active_layers: set[int] | None = None) -> list[tuple[int, str, str]]:
    assignment = json.loads(path.read_text())["assignment"]
    rows: list[tuple[int, str, str]] = []
    for layer_text, experts in assignment.items():
        layer = int(layer_text)
        if active_layers is not None and layer not in active_layers:
            continue
        for projections in experts.values():
            for projection, tier in projections.items():
                rows.append((layer, projection, tier))
    return rows


def qtip_masses(
    assignment_path: Path,
    logical_bytes: dict[str, dict[str, int]],
    active_layers: set[int] | None = None,
) -> dict[str, Any]:
    rows = load_assignment(assignment_path, active_layers)
    result: dict[str, Any] = {}
    for tier in (Q2, Q3):
        selected = [(projection, tier_name) for _, projection, tier_name in rows if tier_name == tier]
        projection_counts = {
            projection: sum(1 for selected_projection, _ in selected if selected_projection == projection)
            for projection in ("down", "fused13")
        }
        total_bytes = sum(
            logical_bytes[tier][projection] * count for projection, count in projection_counts.items()
        )
        result[tier] = {
            "selected_cells": len(selected),
            "projection_counts": projection_counts,
            "logical_bytes": total_bytes,
            "logical_gib": total_bytes / (2**30),
        }
    return result


def weighted_global(vector: dict[str, float], counts: dict[str, int]) -> float:
    return math.fsum(vector[c] * counts[c] for c in CLASSES) / sum(counts.values())


def fit_class(
    class_name: str,
    train_wires: list[str],
    features: dict[str, dict[str, Any]],
    measured_delta: dict[str, dict[str, float]],
    direct_prices: dict[str, dict[str, float]],
    ridge_lambda: float,
) -> dict[str, Any]:
    if class_name == "code":
        q2_price = direct_prices[Q2][class_name]
        q3_price = direct_prices[Q3][class_name]
        ordinary = [features[w]["ordinary_prediction_delta_by_class"][class_name] for w in train_wires]
        residual = [
            measured_delta[w][class_name]
            - q2_price * features[w][Q2]["logical_gib"]
            - q3_price * features[w][Q3]["logical_gib"]
            for w in train_wires
        ]
        denominator = math.fsum(x * x for x in ordinary)
        ordinary_scale = math.fsum(x * y for x, y in zip(ordinary, residual)) / denominator
        return {
            "qtip2_kld_per_logical_gib": q2_price,
            "qtip3_kld_per_logical_gib": q3_price,
            "ordinary_vq_native_scale": ordinary_scale,
            "fit_policy": "P908_Q2_Q3_CODE_PRICES_FIXED__ORDINARY_SCALE_LEAST_SQUARES",
        }

    prior = [
        direct_prices[Q2][class_name],
        direct_prices[Q3][class_name],
        1.0,
    ]
    all_columns = [
        [features[w][Q2]["logical_gib"] for w in CALIBRATION_WIRES],
        [features[w][Q3]["logical_gib"] for w in CALIBRATION_WIRES],
        [features[w]["ordinary_prediction_delta_by_class"][class_name] for w in CALIBRATION_WIRES],
    ]
    norms = [math.sqrt(math.fsum(x * x for x in column)) for column in all_columns]
    design = [
        [
            features[w][Q2]["logical_gib"] / norms[0],
            features[w][Q3]["logical_gib"] / norms[1],
            features[w]["ordinary_prediction_delta_by_class"][class_name] / norms[2],
        ]
        for w in train_wires
    ]
    target = [measured_delta[w][class_name] for w in train_wires]
    prior_scaled = [prior[j] * norms[j] for j in range(3)]
    normal = [
        [
            math.fsum(row[j] * row[k] for row in design)
            + (ridge_lambda if j == k else 0.0)
            for k in range(3)
        ]
        for j in range(3)
    ]
    rhs = [
        math.fsum(design[i][j] * target[i] for i in range(len(design)))
        + ridge_lambda * prior_scaled[j]
        for j in range(3)
    ]
    fitted_scaled = solve_linear(normal, rhs)
    fitted = [fitted_scaled[j] / norms[j] for j in range(3)]
    return {
        "qtip2_kld_per_logical_gib": fitted[0],
        "qtip3_kld_per_logical_gib": fitted[1],
        "ordinary_vq_native_scale": fitted[2],
        "fit_policy": "RIDGE_TO_P908_DIRECT_QTIP_PRICES_AND_ORDINARY_SCALE_1",
        "ridge_lambda": ridge_lambda,
    }


def predict_delta(coefficients: dict[str, Any], feature: dict[str, Any]) -> dict[str, float]:
    return {
        class_name: (
            coefficients[class_name]["qtip2_kld_per_logical_gib"] * feature[Q2]["logical_gib"]
            + coefficients[class_name]["qtip3_kld_per_logical_gib"] * feature[Q3]["logical_gib"]
            + coefficients[class_name]["ordinary_vq_native_scale"]
            * feature["ordinary_prediction_delta_by_class"][class_name]
        )
        for class_name in CLASSES
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pricing-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "inputs" / "raw"

    paths = {
        "surface_predictions": root / "analysis" / "SEALED_SURFACE_PREDICTIONS.json",
        "direct_prices": raw / "DIRECT_SELECTED_CODE_PRICES.json",
        "base_full512": raw / "BASE_FULL512.json",
        "batch2": raw / "BATCH2_PARTIAL_BALANCED64.json",
        "wire_c": raw / "WIRE_C_FULL_BALANCED64.json",
        "wire_b_256_319": raw / "WIRE_B_W256_319_RECEIPT.json",
        "wire_b_320_383": raw / "WIRE_B_W320_383_RECEIPT.json",
        "wire_b_assignment": raw / "WIRE_B_ASSIGNMENT.json",
        "wire_a_assignment": raw / "WIRE_A_P637_ASSIGNMENT.json",
        "p921_wire_a": raw / "P921_RESULT_CELL.json",
        "f949_assignment": raw / "WIRE_C_F949_ASSIGNMENT.json",
        "f521_assignment": root / "inputs" / "v2_bundle" / "ASSIGNMENT_QTIP2_QTIP3.json",
    }
    surface_document = json.loads(paths["surface_predictions"].read_text())
    surface = surface_document["predictions"]
    direct_document = json.loads(paths["direct_prices"].read_text())
    base_control_all = direct_document["canonical_sanity"]["control_mean_kld_vector"]
    base_control = {c: float(base_control_all[c]) for c in CLASSES}

    logical_bytes = {
        Q2: {k: int(v) for k, v in direct_document["surfaces"]["q2"]["logical_bytes_per_cell_by_projection"].items()},
        Q3: {k: int(v) for k, v in direct_document["surfaces"]["q3"]["logical_bytes_per_cell_by_projection"].items()},
    }
    direct_prices = {
        Q2: {
            c: float(direct_document["surfaces"]["q2"]["direct_price_kld_per_logical_gib_vector"][c])
            for c in CLASSES
        },
        Q3: {
            c: float(direct_document["surfaces"]["q3"]["direct_price_kld_per_logical_gib_vector"][c])
            for c in CLASSES
        },
    }

    base_full = json.loads(paths["base_full512"].read_text())["per_window"]
    wire_b_base = {
        c: math.fsum(
            float(row["mean"])
            for row in base_full
            if 256 <= int(row["win"]) < 384 and row["source_class"] == c
        )
        / sum(
            1
            for row in base_full
            if 256 <= int(row["win"]) < 384 and row["source_class"] == c
        )
        for c in CLASSES
    }
    wire_b_receipts = [
        json.loads(paths["wire_b_256_319"].read_text()),
        json.loads(paths["wire_b_320_383"].read_text()),
    ]
    wire_b_counts = {
        c: sum(int(doc["six_classes"][c]["n_windows"]) for doc in wire_b_receipts)
        for c in CLASSES
    }
    wire_b_measured = {
        c: math.fsum(
            float(doc["six_classes"][c]["mean"]) * int(doc["six_classes"][c]["n_windows"])
            for doc in wire_b_receipts
        )
        / wire_b_counts[c]
        for c in CLASSES
    }

    batch2_document = json.loads(paths["batch2"].read_text())
    balanced_counts = {
        c: int(batch2_document["six_classes"][c]["n_windows"]) for c in CLASSES
    }
    batch2_measured = {
        c: float(batch2_document["six_classes"][c]["mean"]) for c in CLASSES
    }
    wire_c_document = json.loads(paths["wire_c"].read_text())
    wire_c_measured = {c: float(wire_c_document["six_class_kld"][c]) for c in CLASSES}

    matched_base = {
        "base": base_control,
        "wire_b": wire_b_base,
        "batch2": base_control,
        "wire_c": base_control,
    }
    measured = {
        "base": base_control,
        "wire_b": wire_b_measured,
        "batch2": batch2_measured,
        "wire_c": wire_c_measured,
    }
    counts = {
        "base": balanced_counts,
        "wire_b": wire_b_counts,
        "batch2": balanced_counts,
        "wire_c": balanced_counts,
    }

    p921_document = json.loads(paths["p921_wire_a"].read_text())["wire_a_p637"]
    wire_a_measured = {c: float(p921_document["six_class_kld"][c]) for c in CLASSES}
    matched_base["wire_a"] = base_control
    measured["wire_a"] = wire_a_measured
    counts["wire_a"] = balanced_counts

    active_batch2 = {0, *range(22, 43)}
    features: dict[str, dict[str, Any]] = {
        "wire_b": {
            **qtip_masses(paths["wire_b_assignment"], logical_bytes),
            "ordinary_prediction_delta_by_class": surface["wire_b"]["decomposition"]["by_group"]["ordinary_vq_native"]["prediction_delta_by_class"],
            "surface_basis": "sealed_pre_repricing_P693",
        },
        "batch2": {
            **qtip_masses(paths["f949_assignment"], logical_bytes, active_batch2),
            "ordinary_prediction_delta_by_class": surface["batch2_f949_l000_l022_l042"]["decomposition"]["by_group"]["ordinary_vq_native"]["prediction_delta_by_class"],
            "surface_basis": "f949_L000_plus_L022_L042",
        },
        "wire_c": {
            **qtip_masses(paths["f949_assignment"], logical_bytes),
            "ordinary_prediction_delta_by_class": surface["wire_c_f949_full"]["decomposition"]["by_group"]["ordinary_vq_native"]["prediction_delta_by_class"],
            "surface_basis": "physical_Wire_C_assignment_f949",
        },
        "wire_a": {
            **qtip_masses(paths["wire_a_assignment"], logical_bytes),
            "ordinary_prediction_delta_by_class": surface["wire_a_p637"]["decomposition"]["by_group"]["ordinary_vq_native"]["prediction_delta_by_class"],
            "surface_basis": "P921_physical_Wire_A_P637_same_BALANCED64_instrument",
        },
    }
    measured_delta = {
        wire: {c: measured[wire][c] - matched_base[wire][c] for c in CLASSES}
        for wire in CALIBRATION_WIRES
    }

    coefficients = {
        c: fit_class(
            c,
            list(CALIBRATION_WIRES),
            features,
            measured_delta,
            direct_prices,
            RIDGE_LAMBDA,
        )
        for c in CLASSES
    }

    retrodictions: dict[str, Any] = {}
    base_global = weighted_global(base_control, balanced_counts)
    retrodictions["base"] = {
        "predicted_by_class": base_control,
        "measured_by_class": base_control,
        "predicted_global": base_global,
        "measured_global": base_global,
        "relative_global_error_percent": 0.0,
        "within_5_percent": True,
    }
    for wire in CALIBRATION_WIRES:
        delta_prediction = predict_delta(coefficients, features[wire])
        class_prediction = {
            c: matched_base[wire][c] + delta_prediction[c] for c in CLASSES
        }
        predicted_global = weighted_global(class_prediction, counts[wire])
        measured_global = weighted_global(measured[wire], counts[wire])
        retrodictions[wire] = {
            "predicted_by_class": class_prediction,
            "measured_by_class": measured[wire],
            "predicted_delta_by_class": delta_prediction,
            "measured_delta_by_class": measured_delta[wire],
            "predicted_global": predicted_global,
            "measured_global": measured_global,
            "relative_global_error_percent": 100.0 * (predicted_global / measured_global - 1.0),
            "relative_class_error_percent": {
                c: 100.0 * (class_prediction[c] / measured[wire][c] - 1.0) for c in CLASSES
            },
            "within_5_percent": abs(predicted_global / measured_global - 1.0) <= 0.05,
        }

    # P921 landed after the original four-wire fit and is retained as a true
    # same-instrument holdout rather than silently folded into calibration.
    wire_a_delta_prediction = predict_delta(coefficients, features["wire_a"])
    wire_a_class_prediction = {
        c: base_control[c] + wire_a_delta_prediction[c] for c in CLASSES
    }
    wire_a_predicted_global = weighted_global(wire_a_class_prediction, balanced_counts)
    wire_a_measured_global = float(p921_document["global_kld"]["mean"])
    retrodictions["wire_a_P921_holdout"] = {
        "role": "SAME_INSTRUMENT_HOLDOUT_NOT_USED_TO_FIT",
        "predicted_by_class": wire_a_class_prediction,
        "measured_by_class": wire_a_measured,
        "predicted_delta_by_class": wire_a_delta_prediction,
        "measured_delta_by_class": {
            c: wire_a_measured[c] - base_control[c] for c in CLASSES
        },
        "predicted_global": wire_a_predicted_global,
        "measured_global": wire_a_measured_global,
        "relative_global_error_percent": 100.0 * (
            wire_a_predicted_global / wire_a_measured_global - 1.0
        ),
        "relative_class_error_percent": {
            c: 100.0 * (wire_a_class_prediction[c] / wire_a_measured[c] - 1.0)
            for c in CLASSES
        },
        "within_5_percent": abs(wire_a_predicted_global / wire_a_measured_global - 1.0) <= 0.05,
    }

    leave_one_out: dict[str, Any] = {}
    for held_out in CALIBRATION_WIRES:
        train = [wire for wire in CALIBRATION_WIRES if wire != held_out]
        loo_coefficients = {
            c: fit_class(c, train, features, measured_delta, direct_prices, RIDGE_LAMBDA)
            for c in CLASSES
        }
        delta_prediction = predict_delta(loo_coefficients, features[held_out])
        class_prediction = {
            c: matched_base[held_out][c] + delta_prediction[c] for c in CLASSES
        }
        predicted_global = weighted_global(class_prediction, counts[held_out])
        measured_global = weighted_global(measured[held_out], counts[held_out])
        leave_one_out[held_out] = {
            "train_wires": train,
            "predicted_global": predicted_global,
            "measured_global": measured_global,
            "relative_global_error_percent": 100.0 * (predicted_global / measured_global - 1.0),
            "within_5_percent": abs(predicted_global / measured_global - 1.0) <= 0.05,
        }

    # Rank known assignment maps in the same BALANCED64 physical-control currency.
    candidate_specs = {
        "base_incumbent": {
            "feature": {
                Q2: {"logical_gib": 0.0},
                Q3: {"logical_gib": 0.0},
                "ordinary_prediction_delta_by_class": {c: 0.0 for c in CLASSES},
            },
            "exact_bytes": int(surface["base_incumbent"]["exact_bytes"]),
            "assignment_sha256": None,
        },
        "batch2_f949_partial": {
            "feature": features["batch2"],
            "exact_bytes": int(surface["batch2_f949_l000_l022_l042"]["exact_bytes"]),
            "assignment_sha256": sha256(paths["f949_assignment"]),
        },
        "wire_c_f949_full": {
            "feature": features["wire_c"],
            "exact_bytes": int(surface["wire_c_f949_full"]["exact_bytes"]),
            "assignment_sha256": sha256(paths["f949_assignment"]),
        },
        "wire_c_f521_v2": {
            "feature": {
                **qtip_masses(paths["f521_assignment"], logical_bytes),
                "ordinary_prediction_delta_by_class": surface["wire_c_f521_full"]["decomposition"]["by_group"]["ordinary_vq_native"]["prediction_delta_by_class"],
                "surface_basis": "current_V2_f521",
            },
            "exact_bytes": int(surface["wire_c_f521_full"]["exact_bytes"]),
            "assignment_sha256": sha256(paths["f521_assignment"]),
        },
    }
    ranking = []
    for name, spec in candidate_specs.items():
        delta_prediction = predict_delta(coefficients, spec["feature"])
        class_prediction = {c: base_control[c] + delta_prediction[c] for c in CLASSES}
        ranking.append({
            "candidate": name,
            "corrected_global_kld": weighted_global(class_prediction, balanced_counts),
            "corrected_prediction_by_class": class_prediction,
            "exact_bytes": spec["exact_bytes"],
            "assignment_sha256": spec["assignment_sha256"],
        })
    ranking.sort(key=lambda row: row["corrected_global_kld"])
    for index, row in enumerate(ranking, 1):
        row["rank"] = index

    pricing = {
        "schema": "p923-transition-aware-corrected-pricing-v3-prelim-v1",
        "status": "PRELIM_AWAITING_P921_P922",
        "task_id": "banana-smasher-public",
        "preliminary": True,
        "application_law": {
            "ordinary_vq_native": "corrected_cost(key,tier,class)=scale[class]*raw_cost(key,tier,class)",
            "qtip2_qtip3": "corrected_cost(key,qtip,class)=corrected_cost(key,original_tier,class)+price[qtip,class]*logical_GiB(projection)",
            "baseline": "predict deltas from the matched physical control; do not compare absolute raw-price currency to physical KLD",
        },
        "coefficients": coefficients,
        "logical_bytes_per_cell_by_projection": logical_bytes,
        "p908_code_prices_fixed": {
            Q2: direct_prices[Q2]["code"],
            Q3: direct_prices[Q3]["code"],
        },
        "p922_codebook_substitution_penalty": {
            "status": "AWAITING_MEASURED_RECEIPT",
            "applied": False,
            "surcharge_by_class": None,
        },
        "fit": {
            "calibration_wires": list(CALIBRATION_WIRES),
            "ridge_lambda_non_code": RIDGE_LAMBDA,
            "retrodiction_global_within_5_percent": all(
                retrodictions[wire]["within_5_percent"] for wire in ("base", *CALIBRATION_WIRES)
            ),
            "max_absolute_retrodiction_global_error_percent": max(
                abs(retrodictions[wire]["relative_global_error_percent"])
                for wire in ("base", *CALIBRATION_WIRES)
            ),
        },
    }
    pricing["canonical_payload_sha256"] = canonical_sha256(pricing)

    report = {
        "schema": "p923-wire-calibration-report-v1",
        "status": "PASS_PRELIM_ALL_FOUR_WIRES_WITHIN_5_PERCENT",
        "task_id": "banana-smasher-public",
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()},
        "model": pricing,
        "wire_features": features,
        "matched_controls": matched_base,
        "measured": measured,
        "counts_by_class": counts,
        "retrodictions": retrodictions,
        "leave_one_wire_out": {
            "results": leave_one_out,
            "warning": "LOO is intentionally reported as a sensitivity diagnostic. Three heterogeneous post-base wires do not identify a stable three-feature model without every wire; failed held-out gates prohibit claiming out-of-sample validation.",
        },
        "known_candidate_ranking": ranking,
        "gate_dependencies": {
            "P921_same_instrument_wire_A": "AWAITING",
            "P922_codebook_substitution_penalty": "AWAITING",
        },
        "limitations": [
            "The correction is calibrated, not independently validated: leave-one-wire-out errors remain decision-relevant.",
            "P908 qtip prices use its exact selected identity sets; transport to nearby assignment maps is explicit and must be checked with direct reads.",
            "No codebook-substitution surcharge is applied until P922 seals.",
        ],
    }
    report["canonical_payload_sha256"] = canonical_sha256(report)

    args.pricing_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pricing_output.write_text(json.dumps(pricing, indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": report["status"],
        "report": str(args.output.resolve()),
        "report_sha256": sha256(args.output),
        "pricing": str(args.pricing_output.resolve()),
        "pricing_sha256": sha256(args.pricing_output),
        "max_abs_global_error_percent": pricing["fit"]["max_absolute_retrodiction_global_error_percent"],
        "ranking": [(row["rank"], row["candidate"], row["corrected_global_kld"]) for row in ranking],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
