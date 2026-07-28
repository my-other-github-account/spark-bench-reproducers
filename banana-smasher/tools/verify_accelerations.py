#!/usr/bin/env python3
"""Fail closed on acceleration evidence, gates, and lineage dispositions."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text())


def close(actual: float, expected: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)


def verify() -> dict[str, Any]:
    ledger = load("receipts/ACCELERATION_RECEIPTS.json")
    expected = load("configs/EXPECTED_PERF.json")
    rows = ledger.get("rows", [])
    by_id = {row.get("id"): row for row in rows}
    failures = []

    if len(by_id) != len(rows):
        failures.append({"reason": "duplicate or missing acceleration id"})
    required = expected.get("required_accelerations", [])
    if set(by_id) != set(required):
        failures.append({
            "reason": "required acceleration set mismatch",
            "missing": sorted(set(required) - set(by_id)),
            "extra": sorted(set(by_id) - set(required)),
        })

    measured_receipts = 0
    contract_gates = 0
    for identifier, row in by_id.items():
        implementations = row.get("implementation", [])
        if isinstance(implementations, str):
            implementations = [implementations]
        for relative in implementations:
            if not (ROOT / relative).is_file():
                failures.append({"id": identifier, "missing_implementation": relative})
        gate = row.get("regression_gate") or row.get("gate")
        if not gate or not (ROOT / gate).is_file():
            failures.append({"id": identifier, "missing_regression_gate": gate})
        if row.get("measured") is True:
            receipt = row.get("receipt")
            if not receipt or not (ROOT / receipt).is_file():
                failures.append({"id": identifier, "missing_measured_receipt": receipt})
            else:
                measured_receipts += 1
                try:
                    load(receipt)
                except (OSError, json.JSONDecodeError) as error:
                    failures.append({"id": identifier, "invalid_receipt": str(error)})
        else:
            if not row.get("contract"):
                failures.append({"id": identifier, "reason": "unmeasured row lacks contract"})
            else:
                contract_gates += 1
            for forbidden in ("speedup", "cells_per_minute", "seconds"):
                if forbidden in row:
                    failures.append({
                        "id": identifier,
                        "reason": "unmeasured contract contains an evidence-like numeric claim",
                        "field": forbidden,
                    })

    p526 = load(by_id["packed_qtip_shared_prefill"]["receipt"])
    if p526.get("verdict") != "MISS" or by_id["packed_qtip_shared_prefill"].get("disposition") != "HOLD_REJECTED_LT_GATE":
        failures.append({"id": "packed_qtip_shared_prefill", "reason": "P526 MISS is not fail-closed"})
    p526_speeds = [row["speedup"] for row in p526.get("rows", [])]
    expected_range = by_id["packed_qtip_shared_prefill"]["measured_speedup_range"]
    if not p526_speeds or not close(min(p526_speeds), expected_range[0]) or not close(max(p526_speeds), expected_range[1]):
        failures.append({"id": "packed_qtip_shared_prefill", "reason": "P526 speed range drift"})

    p530 = load(by_id["dense_all_prefill"]["receipt"])
    p530_row = next((row for row in p530.get("final_rows", []) if row.get("prompt_tokens") == 2048), None)
    dense = by_id["dense_all_prefill"]
    ladder = p530.get("ladder", [])
    checks = [
        p530.get("status") == "PASS_GE_200_TOK_S",
        len(ladder) == 3,
        close(ladder[0].get("pp2048_prefill_tok_s", -1), dense["baseline_pp2048_tok_s"]),
        close(ladder[1].get("pp2048_prefill_tok_s", -1), dense["intermediate_pp2048_tok_s"]),
        close(ladder[2].get("pp2048_probe_prefill_tok_s", -1), dense["final_probe_pp2048_tok_s"]),
        p530_row is not None and close(p530_row.get("prefill_tok_s_median", -1), dense["final_gate_pp2048_tok_s"]),
    ]
    if not all(checks):
        failures.append({"id": "dense_all_prefill", "reason": "P530 ladder/final receipt drift"})

    p963 = load(by_id["accelerated_measurement_rail"]["receipt"])
    rail = by_id["accelerated_measurement_rail"]
    if not all([
        p963.get("status") == "PASS_EXACT_EQUAL_ACCELERATION_GE_2X",
        close(p963.get("baseline", {}).get("elapsed_seconds", -1), rail["baseline_elapsed_seconds"]),
        close(p963.get("accelerated", {}).get("elapsed_seconds", -1), rail["accelerated_elapsed_seconds"]),
        close(p963.get("comparison", {}).get("speedup", -1), rail["speedup"]),
        close(p963.get("exactness", {}).get("maximum_absolute_per_position_delta", -1), rail["replay_max_abs"]),
    ]):
        failures.append({"id": "accelerated_measurement_rail", "reason": "P963 exact receipt drift"})

    mover_receipt = load(by_id["bulk_mover_8_streams"]["receipt"])
    mover = by_id["bulk_mover_8_streams"]
    if not all([
        mover_receipt.get("status") == "PASS",
        mover_receipt.get("stage_retired") is True,
        mover_receipt.get("bytes") == mover["bytes"],
        close(mover_receipt.get("elapsed_seconds", -1), mover["seconds"]),
        close(mover_receipt.get("speedup", -1), mover["speedup"]),
    ]):
        failures.append({"id": "bulk_mover_8_streams", "reason": "P963 mover canary drift"})

    eval_receipt = load(by_id["p486_full164_eval"]["receipt"])
    eval_row = by_id["p486_full164_eval"]
    if not all([
        eval_receipt.get("status") == "PASS",
        eval_receipt.get("coverage", {}).get("completed") == eval_row["coverage"],
        eval_receipt.get("scores", {}).get("base_passes") == eval_row["base_passes"],
        eval_receipt.get("scores", {}).get("plus_passes") == eval_row["plus_passes"],
    ]):
        failures.append({"id": "p486_full164_eval", "reason": "P486 result drift"})

    baselines = expected.get("historical_baselines", {})
    for name, baseline in baselines.items():
        if baseline.get("validity") != "historical-measured-baseline":
            failures.append({"baseline": name, "reason": "historical baseline validity drift"})
        receipt = baseline.get("receipt")
        if not receipt or not (ROOT / receipt).is_file():
            failures.append({"baseline": name, "missing_receipt": receipt})
    p602 = load(baselines["uniform_decode_tok_s"]["receipt"])
    if not close(p602["rows"][0]["decode_tok_s"], baselines["uniform_decode_tok_s"]["value"]):
        failures.append({"baseline": "uniform_decode_tok_s", "reason": "P602 value drift"})
    if not close(p602["rows"][1]["decode_tok_s_median"], baselines["mixed_decode_tok_s"]["value"]):
        failures.append({"baseline": "mixed_decode_tok_s", "reason": "P602 value drift"})
    if p530_row is None or not close(p530_row["decode_tok_s_median"], baselines["integrated_decode_tok_s_at_pp2048"]["value"]):
        failures.append({"baseline": "integrated_decode_tok_s_at_pp2048", "reason": "P530 value drift"})
    if p530_row is None or not close(p530_row["prefill_tok_s_median"], baselines["prefill_tok_s_at_pp2048"]["value"]):
        failures.append({"baseline": "prefill_tok_s_at_pp2048", "reason": "P530 value drift"})

    lineage = {row["id"]: row for row in ledger.get("lineage_dispositions", [])}
    required_lineage = {"P526", "P530", "P948", "P950", "P951", "P959", "P963", "P486"}
    if set(lineage) != required_lineage:
        failures.append({"reason": "lineage disposition set mismatch"})
    else:
        p948 = load(lineage["P948"]["receipt"])
        p950 = load(lineage["P950"]["receipt"])
        p951 = load(lineage["P951"]["receipt"])
        p959 = load(lineage["P959"]["receipt"])
        if not (p948.get("must_not_seed_or_promote") is True and "SPECULATIVE_REVOKED" in p948.get("classification", "")):
            failures.append({"id": "P948", "reason": "speculative boundary drift"})
        if not (p950.get("strict_pass_codebooks") == 77 and p950.get("current_vs_terminal_gap", {}).get("terminal_result_pending") is True):
            failures.append({"id": "P950", "reason": "preliminary boundary drift"})
        if p951.get("status") != "PASS_INDEPENDENT_TERMINAL_TRUE_C_80_OF_80":
            failures.append({"id": "P951", "reason": "terminal baseline drift"})
        if p959.get("status") != "PASS_TERMINAL_UPDATE_000_REBUILT_FROM_WIRE" or p959.get("speculative_seed_used") is not False:
            failures.append({"id": "P959", "reason": "terminal seed rebuild drift"})

    return {
        "schema": "banana-smasher-acceleration-verification-v1",
        "status": "PASS" if not failures else "FAIL",
        "accelerations": len(rows),
        "measured_receipts": measured_receipts,
        "contract_gates": contract_gates,
        "lineage_receipts": len(lineage),
        "historical_baselines": len(baselines),
        "failures": failures,
    }


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
