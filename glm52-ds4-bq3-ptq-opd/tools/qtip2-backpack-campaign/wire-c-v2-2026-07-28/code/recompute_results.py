#!/usr/bin/env python3
"""Recompute the package's public numeric claims from bundled receipts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def load(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def class_means(document: dict[str, Any], key: str = "six_classes") -> dict[str, float]:
    values = document[key]
    result: dict[str, float] = {}
    for class_name in CLASSES:
        value = values[class_name]
        result[class_name] = float(value["mean"] if isinstance(value, dict) else value)
    return result


def weighted(values: dict[str, float], counts: dict[str, int]) -> float:
    denominator = sum(int(counts[name]) for name in CLASSES)
    return math.fsum(float(values[name]) * int(counts[name]) for name in CLASSES) / denominator


def require_close(name: str, actual: float, expected: float, tolerance: float = 1e-14) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"{name} mismatch: actual={actual!r} expected={expected!r}")


def build() -> dict[str, Any]:
    quota = {key: int(value) for key, value in load("BALANCED64_V1.json")["quota"].items()}
    qtip2_doc = load("P880_QTIP2_ASSEALED_BALANCED64.public.json")
    p951_doc = load("P951_TRUE_C_BALANCED64.public.json")
    p931_doc = load("P931_V3_DEFINITIVE.public.json")
    p963_doc = load("P963_EXACT_ACCELERATION_SEAL.public.json")
    p922_doc = load("P922_RESULT_SUMMARY.public.json")
    wire_c_doc = load("WIRE_C_FULL_BALANCED64.public.json")

    qtip2 = class_means(qtip2_doc)
    terminal = class_means(p951_doc)
    qtip2_global = weighted(qtip2, quota)
    terminal_global = weighted(terminal, quota)

    baseline_seconds = float(p963_doc["baseline"]["elapsed_seconds"])
    accelerated_seconds = float(p963_doc["accelerated"]["elapsed_seconds"])
    speedup = baseline_seconds / accelerated_seconds
    reduction_percent = (baseline_seconds - accelerated_seconds) / baseline_seconds * 100.0

    wire_c_r = {name: float(wire_c_doc["six_class_kld"][name]) for name in CLASSES}
    wire_c_r_global = weighted(wire_c_r, quota)
    p922_penalty = float(
        p922_doc["parsed_metrics"]["measured_minus_priced_substitution_penalty_kld"]
    )

    solver = p931_doc["solver"]
    return {
        "balanced64": {
            "uniform_qtip2": {"global_kld": qtip2_global, "six_class_kld": qtip2},
            "terminal_true_c": {
                "global_kld": terminal_global,
                "receipt_global_kld": float(p951_doc["global"]["mean"]),
                "six_class_kld": terminal,
            },
            "terminal_minus_uniform_delta": terminal_global - qtip2_global,
            "relative_kld_reduction": (qtip2_global - terminal_global) / qtip2_global,
            "terminal_to_uniform_ratio": terminal_global / qtip2_global,
        },
        "historical_estimate": {
            "wire_c_r_global_kld": wire_c_r_global,
            "p922_substitution_penalty_kld": p922_penalty,
            "mechanical_true_c_point_kld": wire_c_r_global - p922_penalty,
        },
        "p931_projection": {
            "objective_reweighted": float(solver["objective_reweighted"]),
            "best_bound": float(solver["best_bound"]),
            "relative_gap": float(solver["relative_gap"]),
            "exact_bytes": int(solver["exact_bytes"]),
            "slack_bytes": int(solver["slack_bytes"]),
            "prediction_by_class": {name: float(p931_doc["prediction_by_class"][name]) for name in CLASSES},
        },
        "p963_acceleration": {
            "baseline_elapsed_seconds": baseline_seconds,
            "accelerated_elapsed_seconds": accelerated_seconds,
            "speedup": speedup,
            "wall_clock_reduction_percent": reduction_percent,
            "seconds_saved": baseline_seconds - accelerated_seconds,
        },
    }


def check(computed: dict[str, Any]) -> None:
    same = load("SAME_INSTRUMENT_RESULTS.json")
    rows = {row["key"]: row for row in same["rows"]}
    table = load("CAMPAIGN_COMPARISON_TABLE.json")
    campaigns = {row["campaign"]: row for row in table["campaigns"]}
    verdict = table["one_instrument_verdicts"]["balanced64_uniform_qtip2_vs_terminal_true_c"]
    p931 = load("P931_V3_DEFINITIVE.public.json")
    p963 = load("P963_EXACT_ACCELERATION_SEAL.public.json")

    balanced = computed["balanced64"]
    for computed_key, registry_key in (
        ("uniform_qtip2", "qtip2_vertical"),
        ("terminal_true_c", "wire_c_true_terminal_balanced64"),
    ):
        actual = balanced[computed_key]
        expected = rows[registry_key]
        require_close(f"{registry_key}.global_kld", actual["global_kld"], expected["global_kld"])
        for class_name in CLASSES:
            require_close(
                f"{registry_key}.{class_name}",
                actual["six_class_kld"][class_name],
                expected["six_class_kld"][class_name],
            )

    require_close(
        "P951 receipt global",
        balanced["terminal_true_c"]["global_kld"],
        balanced["terminal_true_c"]["receipt_global_kld"],
    )
    require_close("comparison delta", balanced["terminal_minus_uniform_delta"], verdict["delta_terminal_minus_baseline"])
    require_close("comparison reduction", balanced["relative_kld_reduction"], verdict["relative_kld_reduction"])
    require_close("comparison ratio", balanced["terminal_to_uniform_ratio"], verdict["terminal_to_baseline_ratio"])

    direct = table["one_instrument_verdicts"]["terminal_true_c_vs_direct_iq4"]
    iq4_kld = float(rows["iq4_reference"]["global_kld"])
    true_c_kld = float(rows["wire_c_true_terminal_balanced64"]["global_kld"])
    require_close("direct IQ4 delta", true_c_kld - iq4_kld, direct["delta_true_c_minus_iq4"], tolerance=0.0)
    require_close(
        "direct IQ4 relative reduction",
        (iq4_kld - true_c_kld) / iq4_kld,
        direct["true_c_lower_than_iq4_fraction"],
        tolerance=0.0,
    )
    if rows["iq4_reference"]["receipt_sha256"] != "abb2031865874c0025719889064f5b0e4f7c5a55cfb3ee2916a924ed348bdf07":
        raise RuntimeError("direct IQ4 receipt SHA drift")
    if rows["iq4_reference"]["sealed_finite_positions"] != 524288:
        raise RuntimeError("direct IQ4 population disclosure drift")

    estimate = computed["historical_estimate"]
    require_close(
        "historical mechanical point",
        estimate["mechanical_true_c_point_kld"],
        rows["wire_c_true_estimate"]["global_kld_point_estimate"],
    )
    if rows["wire_c_true_estimate"]["status"] != "ESTIMATE_NOT_MEASURED":
        raise RuntimeError("historical estimate lost ESTIMATE_NOT_MEASURED label")

    projection = computed["p931_projection"]
    campaign_projection = campaigns["C_corrected_pricing_solver"]
    for key in ("objective_reweighted", "best_bound", "relative_gap"):
        require_close(f"P931 {key}", projection[key], campaign_projection[key])
    for key in ("exact_bytes", "slack_bytes"):
        if projection[key] != campaign_projection[key]:
            raise RuntimeError(f"P931 {key} mismatch")
    if p931["public_validity"] != {
        "measured": False,
        "optimality_proven": False,
        "physical_checkpoint_scored": False,
        "result_type": "PROJECTED_SOLVER_RESULT",
        "source_payloads_redistributed": False,
        "status": "PROJECTED_DEFINITIVE_FEASIBLE__TIME_LIMIT_INCUMBENT__NOT_PHYSICAL_MEASUREMENT",
        "true_c_measurement_status": "PENDING_DIRECT_P937_P939",
    }:
        raise RuntimeError("P931 public-validity contract drift")

    acceleration = computed["p963_acceleration"]
    campaign_acceleration = campaigns["E_exact_equal_scorer_acceleration"]
    for key in ("baseline_elapsed_seconds", "accelerated_elapsed_seconds", "speedup", "wall_clock_reduction_percent"):
        require_close(f"P963 {key}", acceleration[key], campaign_acceleration[key], tolerance=1e-12)
    require_close("P963 sealed speedup", acceleration["speedup"], p963["comparison"]["speedup"], tolerance=1e-12)
    if p963["baseline"]["output_set_sha256"] != p963["accelerated"]["output_set_sha256"]:
        raise RuntimeError("P963 output-set SHA mismatch")
    if p963["exactness"]["maximum_absolute_per_position_delta"] != 0.0:
        raise RuntimeError("P963 per-position exactness drift")

    if table["one_instrument_verdicts"]["p931_projection_vs_p951_measurement"]["verdict"] != "NOT_COMPARABLE":
        raise RuntimeError("P931/P951 comparability label drift")
    if campaigns["F_inference_and_eval_protocol"]["sampled_n_per_task"] != 5:
        raise RuntimeError("binding sampled n drift")
    if campaigns["F_inference_and_eval_protocol"]["greedy_repeats"] != 3:
        raise RuntimeError("binding greedy-repeat count drift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="retained for compatibility; checking is the default")
    parser.add_argument("--json", action="store_true", help="print recomputed values after checking")
    args = parser.parse_args()
    computed = build()
    try:
        check(computed)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"WIRE_C_V2_RECOMPUTE_FAIL: {exc}") from exc
    if args.json:
        print(json.dumps(computed, indent=2, sort_keys=True))
    else:
        print(
            "WIRE_C_V2_RECOMPUTE_PASS "
            f"terminal_true_c={computed['balanced64']['terminal_true_c']['global_kld']:.17g} "
            f"p931_projected={computed['p931_projection']['objective_reweighted']:.17g} "
            f"p963_speedup={computed['p963_acceleration']['speedup']:.15g}"
        )


if __name__ == "__main__":
    main()
