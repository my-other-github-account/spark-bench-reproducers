#!/usr/bin/env python3
"""Verify the public P930 corrected-pricing fan-in and exact source pins."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def load(name: str) -> dict:
    return json.loads((A / name).read_text())


def sha256(name: str) -> str:
    return hashlib.sha256((A / name).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CORRECTED_PRICING_VERIFY_FAIL: {message}")


def close(a: float, b: float, tolerance: float = 1e-14) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    manifest = load_manifest()
    pricing = load("P930_CORRECTED_PRICING_V3.public.json")
    report = load("P930_WIRE_CALIBRATION_FINAL_REPORT.public.json")
    validation = load("P930_FINAL_VALIDATION.public.json")
    p922 = load("P922_RESULT_SUMMARY.public.json")

    require(pricing["status"] == "PASS_FINAL_SOLVER_CONSUMABLE", "pricing status")
    require(
        report["status"]
        == "PASS_FINAL_ALL_FOUR_GLOBAL_WITHIN_5_PERCENT_WITH_CLASS_RESIDUAL_DISCLOSURE",
        "report status",
    )
    require(validation["status"] == "PASS", "validation status")
    require(pricing["classes"] == list(CLASSES), "six-class order")

    grid_name = "P930_CORRECTED_VERTICAL_GRID_V3.csv"
    require(
        sha256(grid_name) == pricing["corrected_vertical_grid"]["sha256"],
        "corrected grid hash",
    )
    with (A / grid_name).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 14, "corrected grid row count")
    require(len({row["tier"] for row in rows}) == 14, "corrected grid unique tiers")
    for row in rows:
        for key in ("global_mean_kld", *CLASSES):
            value = float(row[key])
            require(math.isfinite(value) and value >= 0.0, f"nonphysical grid value {row['tier']}.{key}")

    pricing_penalty = pricing["p922_codebook_substitution_surcharge"]
    measured_penalty = p922["parsed_metrics"]["measured_minus_priced_substitution_penalty_kld"]
    require(close(pricing_penalty["surcharge_global_kld"], measured_penalty), "P922 global penalty")
    require(set(pricing_penalty["surcharge_kld_by_class"]) == set(CLASSES), "P922 six classes")
    require(
        pricing_penalty["selection_sha256"]
        == source_pin(manifest, "artifacts/P930_P922_RESTORED_VQ_SELECTION_PINNED.public.json"),
        "P922 selection source pin",
    )

    interaction = pricing["p928_mixed_tier_interaction"]
    require(set(interaction["interaction_kld_by_class"]) == set(CLASSES), "P928 six classes")
    require(
        interaction["anchor_assignment_sha256"]
        == source_pin(manifest, "artifacts/P930_P928_MIX_C_PATTERN_ASSIGNMENT_PINNED.public.json"),
        "P928 assignment source pin",
    )

    require(validation["gates"]["all_global_within_5_percent"], "global retrodiction gate")
    require(validation["per_class_miss_count"] == 6, "disclosed class miss count")
    require(report["limitations"][0].startswith("A is the only strict"), "strict holdout disclosure")
    require(
        pricing["surface_modes"]["solver_option_surface"]["do_not_double_count"],
        "double-counting guard",
    )

    expected_sources = {
        "artifacts/P930_CORRECTED_PRICING_V3.public.json": "c8673867b0fb7626232721d4939a9fdf95ef6d1a3de69698fd2a3d42398606c0",
        "artifacts/P930_WIRE_CALIBRATION_FINAL_REPORT.public.json": "6213107d728ac0df48be7121a082a6efa6f894d30c800e8db94315589c86a0d9",
        "artifacts/P930_CORRECTED_VERTICAL_GRID_V3.csv": "49407ff0114c5bcf9f7a68fbfc2a4822fee1839852aff5d89b8ce12d1251c203",
        "artifacts/P930_FINAL_VALIDATION.public.json": "9666d979b79ec576f55a4ea685bb1311b29910875fc227a24f470370e516b379",
    }
    for path, expected in expected_sources.items():
        require(source_pin(manifest, path) == expected, f"source hash {path}")

    print("CORRECTED_PRICING_VERIFY_PASS rows=14 classes=6 p922=EXPLICIT p928=EXPLICIT")


def load_manifest() -> dict:
    return json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text())


def source_pin(manifest: dict, path: str) -> str:
    for row in manifest["files"]:
        if row["path"] == path:
            return row["source_sha256"]
    raise SystemExit(f"CORRECTED_PRICING_VERIFY_FAIL: missing manifest row {path}")


if __name__ == "__main__":
    main()
