#!/usr/bin/env python3
"""Aggregate PANEL_GATE v1 from matched candidate-own rollout JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

REQUIRED = {
    "prompt_id",
    "budget_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "teacher_nll",
    "teacher_approval",
    "finish_reason",
    "content_is_null",
}


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing {sorted(missing)}")
            prompt_id = str(row["prompt_id"])
            if prompt_id in rows:
                raise ValueError(f"{path}:{line_no}: duplicate prompt_id {prompt_id}")
            for field in ("budget_tokens", "completion_tokens", "reasoning_tokens"):
                if int(row[field]) < 0:
                    raise ValueError(f"{path}:{line_no}: negative {field}")
            if int(row["completion_tokens"]) > int(row["budget_tokens"]):
                raise ValueError(f"{path}:{line_no}: completion exceeds frozen budget")
            for field in ("teacher_nll", "teacher_approval"):
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"{path}:{line_no}: non-finite {field}")
            rows[prompt_id] = row
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return values[lo]
    frac = index - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = [rows[key] for key in sorted(rows)]
    nll = [float(row["teacher_nll"]) for row in ordered]
    approval = [float(row["teacher_approval"]) for row in ordered]
    completion = [int(row["completion_tokens"]) for row in ordered]
    reasoning = [int(row["reasoning_tokens"]) for row in ordered]
    return {
        "prompts": len(ordered),
        "prompt_macro_nll": statistics.fmean(nll),
        "prompt_macro_approval": statistics.fmean(approval),
        "prompt_nll_p95": quantile(nll, 0.95),
        "prompt_nll_p99": quantile(nll, 0.99),
        "mean_completion_tokens": statistics.fmean(completion),
        "mean_reasoning_tokens": statistics.fmean(reasoning),
        "null_count": sum(bool(row["content_is_null"]) for row in ordered),
        "cap_exhaustion_count": sum(
            int(row["completion_tokens"]) >= int(row["budget_tokens"]) for row in ordered
        ),
        "finish_reason_counts": {
            reason: sum(str(row["finish_reason"]) == reason for row in ordered)
            for reason in sorted({str(row["finish_reason"]) for row in ordered})
        },
        "worst_prompts": [
            {"prompt_id": str(row["prompt_id"]), "teacher_nll": float(row["teacher_nll"])}
            for row in sorted(ordered, key=lambda item: float(item["teacher_nll"]), reverse=True)[:5]
        ],
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate matched frozen-budget own rollouts and compute prompt-macro PANEL_GATE v1."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_rows(args.baseline)
    candidate = load_rows(args.candidate)
    if baseline.keys() != candidate.keys():
        missing_b = sorted(candidate.keys() - baseline.keys())
        missing_c = sorted(baseline.keys() - candidate.keys())
        raise ValueError(f"prompt mismatch: baseline_missing={missing_b}, candidate_missing={missing_c}")

    for prompt_id in baseline:
        b, c = baseline[prompt_id], candidate[prompt_id]
        if int(b["budget_tokens"]) != int(c["budget_tokens"]):
            raise ValueError(f"{prompt_id}: frozen budget mismatch")
        for optional_identity in ("prompt_sha256", "request_sha256"):
            if optional_identity in b or optional_identity in c:
                if b.get(optional_identity) != c.get(optional_identity):
                    raise ValueError(f"{prompt_id}: {optional_identity} mismatch")

    bsum, csum = summarize(baseline), summarize(candidate)
    baseline_reasoning = bsum["mean_reasoning_tokens"]
    result = {
        "schema": "panel-gate-v1",
        "aggregation": "prompt-macro; one vote per prompt",
        "baseline": bsum,
        "candidate": csum,
        "delta": {
            "prompt_macro_nll": csum["prompt_macro_nll"] - bsum["prompt_macro_nll"],
            "prompt_macro_approval": csum["prompt_macro_approval"] - bsum["prompt_macro_approval"],
            "reasoning_length_ratio": (
                csum["mean_reasoning_tokens"] / baseline_reasoning if baseline_reasoning else None
            ),
            "null_count": csum["null_count"] - bsum["null_count"],
            "cap_exhaustion_count": csum["cap_exhaustion_count"] - bsum["cap_exhaustion_count"],
        },
        "inputs": {
            "baseline_sha256": file_sha256(args.baseline),
            "candidate_sha256": file_sha256(args.candidate),
        },
        "note": "Static per-class KLD is a separate damage-location diagnostic and is not ranked here.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
