#!/usr/bin/env python3
"""Recompute all public comparison rows from bundled receipts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"


def load(name: str) -> dict:
    return json.loads((A / name).read_text())


def weighted(values: dict[str, float], counts: dict[str, int]) -> float:
    return math.fsum(float(values[c]) * int(counts[c]) for c in CLASSES) / sum(
        int(counts[c]) for c in CLASSES
    )


def close(name: str, got: float, expected: float, tolerance: float = 1e-14) -> None:
    if not math.isclose(float(got), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise SystemExit(f"mismatch {name}: got={got!r} expected={expected!r}")


def build() -> dict:
    quota = load("BALANCED64_V1.json")["quota"]
    direct = load("DIRECT_SELECTED_CODE_PRICES.public.json")
    bq3 = {
        c: float(direct["canonical_sanity"]["control_mean_kld_vector"][c])
        for c in CLASSES
    }

    wire_c_doc = load("WIRE_C_FULL_BALANCED64.public.json")
    wire_c = {c: float(wire_c_doc["six_class_kld"][c]) for c in CLASSES}

    p921 = load("P921_RESULT_CELL.public.json")
    wire_a = {c: float(p921["wire_a_p637"]["six_class_kld"][c]) for c in CLASSES}

    q3_doc = load("P819_QTIP3_UNIFORM_BALANCED64.public.json")
    q3 = {c: float(q3_doc["six_classes"][c]["mean"]) for c in CLASSES}

    q2_doc = load("P880_QTIP2_ASSEALED_BALANCED64.public.json")
    q2 = {c: float(q2_doc["six_classes"][c]["mean"]) for c in CLASSES}

    wire_b_receipts = [
        load("WIRE_B_W256_319_RECEIPT.public.json"),
        load("WIRE_B_W320_383_RECEIPT.public.json"),
    ]
    wire_b_counts = {
        c: sum(int(d["six_classes"][c]["n_windows"]) for d in wire_b_receipts)
        for c in CLASSES
    }
    wire_b = {
        c: math.fsum(
            float(d["six_classes"][c]["mean"])
            * int(d["six_classes"][c]["n_windows"])
            for d in wire_b_receipts
        )
        / wire_b_counts[c]
        for c in CLASSES
    }
    base_windows = load("BQ3_FULL512.public.json")["per_window"]
    wire_b_control = {
        c: math.fsum(
            float(row["mean"])
            for row in base_windows
            if 256 <= int(row["win"]) < 384 and row["source_class"] == c
        )
        / sum(
            1
            for row in base_windows
            if 256 <= int(row["win"]) < 384 and row["source_class"] == c
        )
        for c in CLASSES
    }

    p922 = load("P922_RESULT_SUMMARY.public.json")
    penalty = float(
        p922["parsed_metrics"]["measured_minus_priced_substitution_penalty_kld"]
    )

    return {
        "rows": {
            "bq3_balanced64_control": {
                "kld_by_class": bq3,
                "global_kld": weighted(bq3, quota),
            },
            "wire_a_balanced64": {
                "kld_by_class": wire_a,
                "global_kld": weighted(wire_a, quota),
            },
            "wire_c_r_balanced64": {
                "kld_by_class": wire_c,
                "global_kld": weighted(wire_c, quota),
            },
            "uniform_qtip3_balanced64": {
                "kld_by_class": q3,
                "global_kld": weighted(q3, quota),
            },
            "uniform_qtip2_balanced64": {
                "kld_by_class": q2,
                "global_kld": weighted(q2, quota),
            },
            "wire_b_w256_383": {
                "kld_by_class": wire_b,
                "global_kld": weighted(wire_b, wire_b_counts),
                "matched_bq3_global_kld": weighted(wire_b_control, wire_b_counts),
            },
        },
        "p922_penalty_kld": penalty,
        "mechanical_true_c_point_kld": weighted(wire_c, quota) - penalty,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    got = build()
    if not args.check:
        print(json.dumps(got, indent=2, sort_keys=True))
        return

    expected = load("SAME_INSTRUMENT_RESULTS.json")
    expected_rows = {row["key"]: row for row in expected["rows"]}
    key_map = {
        "bq3_balanced64_control": "genesis_base",
        "wire_a_balanced64": "wire_a",
        "wire_c_r_balanced64": "wire_c_r",
        "uniform_qtip3_balanced64": "qtip3_vertical",
        "uniform_qtip2_balanced64": "qtip2_vertical",
        "wire_b_w256_383": "wire_b",
    }
    for name, row in got["rows"].items():
        expected_row = expected_rows[key_map[name]]
        close(f"{name}.global_kld", row["global_kld"], expected_row["global_kld"])
        for class_name in CLASSES:
            close(
                f"{name}.{class_name}",
                row["kld_by_class"][class_name],
                expected_row["six_class_kld"][class_name],
            )

    diagnostic = expected["diagnostics"]["p922_substitution_penalty"]
    estimate = expected_rows["wire_c_true_estimate"]
    close("p922_penalty", got["p922_penalty_kld"], diagnostic["global_kld"])
    close(
        "mechanical_true_c_point",
        got["mechanical_true_c_point_kld"],
        estimate["global_kld_point_estimate"],
    )
    if estimate["status"] != "ESTIMATE_NOT_MEASURED":
        raise SystemExit("true-C row lost ESTIMATE_NOT_MEASURED label")

    # Bind receipt-level global rows independently of class-weighted recomputation.
    close(
        "P921 Wire A receipt",
        got["rows"]["wire_a_balanced64"]["global_kld"],
        load("P921_RESULT_CELL.public.json")["wire_a_p637"]["global_kld"]["mean"],
    )
    close(
        "P819 QTIP3 receipt",
        got["rows"]["uniform_qtip3_balanced64"]["global_kld"],
        load("P819_QTIP3_UNIFORM_BALANCED64.public.json")["global"]["mean"],
    )
    close(
        "P880 QTIP2 receipt",
        got["rows"]["uniform_qtip2_balanced64"]["global_kld"],
        load("P880_QTIP2_ASSEALED_BALANCED64.public.json")["global"]["mean"],
    )
    print("SAME_INSTRUMENT_RECOMPUTE_PASS rows=6 true_c=ESTIMATE_NOT_MEASURED")


if __name__ == "__main__":
    main()
