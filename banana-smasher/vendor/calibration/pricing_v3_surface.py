#!/usr/bin/env python3
"""P931 pricing-only adapter for the frozen P924 solver surface.

The solver objective, admissible menu, bytes, ceilings, paired-greedy routine, and
SCIP configuration remain owned by solve_p924_reweighted.py / solve_p693_turbo.py.
This module only replaces the three P930-modified absolute vertical-grid rows and
adds the separately specified P922 frozen-codebook selection surcharge.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
GRID_VALUE_COLUMNS = ("global_mean_kld", *CLASSES)
EXPECTED_COLUMNS = (
    "tier", "family", "k", "menu_rate_bpw", "global_mean_kld",
    "agentic", "chat", "code", "multilingual", "prose", "reasoning",
)
EXPECTED_MODIFIED_TIERS = {"d4_k4096", "qtip2_2.0117", "qtip3_3.0117"}
EXPECTED_SHA256 = {
    "pricing_v3": "c8673867b0fb7626232721d4939a9fdf95ef6d1a3de69698fd2a3d42398606c0",
    "pre_v3_grid": "74869b5f8e3ef4eb43dc98c6ee060c2d9ad048bb215cadd308fb2c9983933dda",
    "v3_grid": "49407ff0114c5bcf9f7a68fbfc2a4822fee1839852aff5d89b8ce12d1251c203",
    "p922_selection": "e776c293be491f080a630f7ba1d066ea0cc420c773be6758de2b4c92a3fb9818",
    "p928_assignment": "62c26b9ea8f53aa2a2be84ff55b0e444100625f900832e096624ea178d9f9122",
    "v3_validation": "9666d979b79ec576f55a4ea685bb1311b29910875fc227a24f470370e516b379",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_grid(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or ())
        rows = list(reader)
    if tuple(columns) != EXPECTED_COLUMNS:
        raise RuntimeError({"grid_column_drift": {"path": str(path), "actual": columns, "expected": EXPECTED_COLUMNS}})
    if len(rows) != 14:
        raise RuntimeError({"grid_row_count_drift": {"path": str(path), "actual": len(rows), "expected": 14}})
    by_tier = {str(row["tier"]): row for row in rows}
    if len(by_tier) != 14:
        raise RuntimeError({"grid_duplicate_tiers": str(path)})
    for tier, row in by_tier.items():
        for label in GRID_VALUE_COLUMNS:
            value = float(row[label])
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError({"invalid_grid_value": {"path": str(path), "tier": tier, "label": label, "value": value}})
    return columns, by_tier


def _summarize(opts: dict, selected: dict, classes: tuple[str, ...]) -> dict[str, float]:
    return {
        label: math.fsum(float(opts[key][tier]["costs"][label]) for key, tier in selected.items())
        for label in classes
    }


def apply_v3_pricing_surface(
    surface: dict[str, Any],
    *,
    pre_v3_grid_path: Path,
    v3_grid_path: Path,
    pricing_v3_path: Path,
    p922_selection_path: Path,
    p928_assignment_path: Path,
    v3_validation_path: Path,
) -> dict[str, Any]:
    """Mutate only option costs on an already validated frozen P924 surface."""
    paths = {
        "pricing_v3": pricing_v3_path.resolve(),
        "pre_v3_grid": pre_v3_grid_path.resolve(),
        "v3_grid": v3_grid_path.resolve(),
        "p922_selection": p922_selection_path.resolve(),
        "p928_assignment": p928_assignment_path.resolve(),
        "v3_validation": v3_validation_path.resolve(),
    }
    observed_shas = {name: sha256(path) for name, path in paths.items()}
    if observed_shas != EXPECTED_SHA256:
        raise RuntimeError({"P930_input_sha_drift": {"expected": EXPECTED_SHA256, "actual": observed_shas}})

    pricing = json.loads(paths["pricing_v3"].read_text())
    selection = json.loads(paths["p922_selection"].read_text())
    p928_assignment = json.loads(paths["p928_assignment"].read_text())
    validation = json.loads(paths["v3_validation"].read_text())
    if pricing.get("status") != "PASS_FINAL_SOLVER_CONSUMABLE" or pricing.get("schema") != "p930-corrected-pricing-v3-final-v1":
        raise RuntimeError("P930 corrected-pricing V3 is not final solver-consumable")
    if validation.get("status") != "PASS" or validation.get("failures"):
        raise RuntimeError({"P930_final_validation_not_pass": validation})
    if pricing.get("corrected_vertical_grid", {}).get("sha256") != observed_shas["v3_grid"]:
        raise RuntimeError("P930 pricing/grid binding drift")
    if pricing.get("p922_codebook_substitution_surcharge", {}).get("selection_sha256") != observed_shas["p922_selection"]:
        raise RuntimeError("P930 pricing/P922-selection binding drift")
    if pricing.get("p928_mixed_tier_interaction", {}).get("anchor_assignment_sha256") != observed_shas["p928_assignment"]:
        raise RuntimeError("P930 pricing/P928-assignment binding drift")
    if p928_assignment.get("tier_assignment_sha256") != "bbc7b122b666881fc6765bab4a1fcbcd3a99cac3eb562e97f5348c1516d33909":
        raise RuntimeError("P928 assignment tier-assignment pin drift")

    _columns_old, old_grid = _load_grid(paths["pre_v3_grid"])
    _columns_new, new_grid = _load_grid(paths["v3_grid"])
    if set(old_grid) != set(new_grid):
        raise RuntimeError({"V3_grid_tier_set_drift": {"old": sorted(old_grid), "new": sorted(new_grid)}})
    modified = {
        tier for tier in old_grid
        if any(float(old_grid[tier][label]) != float(new_grid[tier][label]) for label in GRID_VALUE_COLUMNS)
    }
    if modified != EXPECTED_MODIFIED_TIERS:
        raise RuntimeError({"V3_modified_tier_drift": {"expected": sorted(EXPECTED_MODIFIED_TIERS), "actual": sorted(modified)}})
    for tier in set(old_grid) - modified:
        if old_grid[tier] != new_grid[tier]:
            raise RuntimeError({"V3_nonpricing_row_text_drift": tier})

    opts = surface["opts"]
    original = surface["original"]
    classes = tuple(surface["gs"].CLASSES)
    if classes != CLASSES:
        raise RuntimeError({"solver_class_drift": classes})
    before_original_prediction = _summarize(opts, original, classes)

    factors: dict[str, dict[str, float]] = {}
    scaled_option_counts: dict[str, int] = {}
    tier_sums_before: dict[str, dict[str, float]] = {}
    tier_sums_after: dict[str, dict[str, float]] = {}
    for tier in sorted(modified):
        factors[tier] = {}
        for label in CLASSES:
            old_value = float(old_grid[tier][label])
            new_value = float(new_grid[tier][label])
            if old_value <= 0.0:
                raise RuntimeError({"cannot_scale_zero_PRE_V3_price": {"tier": tier, "label": label}})
            factors[tier][label] = new_value / old_value
        matching = [(key, local[tier]) for key, local in opts.items() if tier in local]
        if not matching:
            raise RuntimeError({"V3_modified_tier_absent_from_solver_surface": tier})
        scaled_option_counts[tier] = len(matching)
        tier_sums_before[tier] = {
            label: math.fsum(float(option["costs"][label]) for _key, option in matching)
            for label in CLASSES
        }
        for _key, option in matching:
            for label in CLASSES:
                option["costs"][label] = float(option["costs"][label]) * factors[tier][label]
            option["pricing_v3_grid_factor_by_class"] = dict(factors[tier])
            option["pricing_v3_grid_sha256"] = observed_shas["v3_grid"]
        tier_sums_after[tier] = {
            label: math.fsum(float(option["costs"][label]) for _key, option in matching)
            for label in CLASSES
        }
        closure_error = {
            label: tier_sums_after[tier][label] - tier_sums_before[tier][label] * factors[tier][label]
            for label in CLASSES
        }
        if max(abs(value) for value in closure_error.values()) > 5e-15:
            raise RuntimeError({"V3_grid_factor_closure_drift": {"tier": tier, "error": closure_error}})

    p922 = pricing["p922_codebook_substitution_surcharge"]
    linear = p922["solver_linearization"]
    surcharge = {label: float(linear["surcharge_kld_per_selected_identity_by_class"][label]) for label in CLASSES}
    rows = selection.get("rows", [])
    if selection.get("status") != "PASS_EXACT_RESTORED_VQ_SELECTION" or len(rows) != 3803:
        raise RuntimeError({"P922_selection_row_closure_drift": {"status": selection.get("status"), "rows": len(rows)}})
    seen: set[tuple[int, int, str, str]] = set()
    counts_by_tier: dict[str, int] = {}
    for row in rows:
        key = (int(row["layer"]), int(row["expert"]), str(row["projection"]))
        tier = str(row["new_tier"])
        join = (*key, tier)
        if join in seen:
            raise RuntimeError({"P922_duplicate_join": join})
        seen.add(join)
        if key not in opts or tier not in opts[key]:
            raise RuntimeError({"P922_join_absent_from_solver_surface": join})
        option = opts[key][tier]
        for label in CLASSES:
            option["costs"][label] = float(option["costs"][label]) + surcharge[label]
        option["p922_frozen_codebook_surcharge_by_class"] = dict(surcharge)
        option["p922_selection_sha256"] = observed_shas["p922_selection"]
        counts_by_tier[tier] = counts_by_tier.get(tier, 0) + 1

    expected_counts = {str(k): int(v) for k, v in p922["counts_by_new_tier"].items()}
    if counts_by_tier != expected_counts:
        raise RuntimeError({"P922_counts_by_tier_drift": {"expected": expected_counts, "actual": counts_by_tier}})
    surcharge_closure = {label: math.fsum(surcharge[label] for _ in rows) for label in CLASSES}
    expected_surcharge = {label: float(p922["surcharge_kld_by_class"][label]) for label in CLASSES}
    surcharge_error = {label: surcharge_closure[label] - expected_surcharge[label] for label in CLASSES}
    if max(abs(value) for value in surcharge_error.values()) > 5e-15:
        raise RuntimeError({"P922_surcharge_closure_drift": surcharge_error})

    after_original_prediction = _summarize(opts, original, classes)
    source_code_cap = float(surface["ceilings"]["code"])
    effective_code_cap = float(after_original_prediction["code"])
    surface["original_pred"] = after_original_prediction
    surface["ceilings"]["code"] = effective_code_cap
    surface["p931_v3_pricing"] = {
        "schema": "p931-pricing-only-surface-adapter-v1",
        "status": "PASS_P930_V3_GRID_AND_P922_SURCHARGE_APPLIED",
        "task_id": "banana-smasher-public",
        "input_shas": observed_shas,
        "source_p924_solver_sha256": "fc99323ecea76d52b3640e96894ded0ee684987d66abea38cde6cc8be4dfd355",
        "modified_grid_tiers": sorted(modified),
        "unchanged_grid_tiers": sorted(set(old_grid) - modified),
        "grid_factor_by_tier_and_class": factors,
        "scaled_option_counts": scaled_option_counts,
        "tier_option_sums_before_surcharge": tier_sums_before,
        "tier_option_sums_after_grid_factor_before_surcharge": tier_sums_after,
        "p928_interaction_application": "ALREADY_BAKED_INTO_THE_THREE_V3_GRID_ROWS__NOT_ADDED_TWICE",
        "p922_application": "SEPARATE_EXACT_FROZEN_CODEBOOK_JOIN",
        "p922_join_key": ["layer", "expert", "projection", "new_tier"],
        "p922_joined_rows": len(rows),
        "p922_counts_by_tier": counts_by_tier,
        "p922_surcharge_per_identity_by_class": surcharge,
        "p922_surcharge_closure_by_class": surcharge_closure,
        "p922_surcharge_closure_error_by_class": surcharge_error,
        "original_prediction_before_v3": before_original_prediction,
        "original_prediction_after_v3": after_original_prediction,
        "hard_code_guard_semantics": "same no-regression-to-exact-original-incumbent rule, repriced on the V3 surface",
        "source_pre_v3_code_cap_predicted_kld": source_code_cap,
        "effective_v3_code_cap_predicted_kld": effective_code_cap,
        "pricing_report_canonical_payload_sha256": pricing.get("canonical_payload_sha256"),
        "surface_report_canonical_sha256": None,
    }
    report = surface["p931_v3_pricing"]
    report["surface_report_canonical_sha256"] = canonical_sha256({k: v for k, v in report.items() if k != "surface_report_canonical_sha256"})
    return report
