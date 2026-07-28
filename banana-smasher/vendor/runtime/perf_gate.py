#!/usr/bin/env python3
"""Fail-closed serving readiness gate for fresh 2K measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def evaluate(metrics: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    gates = expected["ready_gates"]
    target = expected["target"]
    required_kernels = set(expected["required_decode_kernel_classes"])
    observed_kernels = set(metrics.get("decode_kernel_classes", []))
    prompt_tokens = int(metrics.get("prompt_tokens", 0) or 0)
    prefill = float(metrics.get("prefill_tok_s", 0.0) or 0.0)
    decode = float(metrics.get("decode_tok_s", 0.0) or 0.0)
    ttft = float(metrics.get("ttft_seconds", float("inf")) or float("inf"))
    checks = {
        "fresh_measurement": metrics.get("validity") == "fresh-measurement",
        "prompt_tokens": prompt_tokens == int(target["prompt_tokens"]),
        "prefill_tok_s": prefill >= float(gates["prefill_tok_s_min"]),
        "decode_tok_s": decode >= float(gates["decode_tok_s_min"]),
        "ttft_seconds": ttft <= float(gates["ttft_seconds_max"]),
        "decode_kernel_classes": required_kernels <= observed_kernels,
        "cache_verified": metrics.get("cache_verified") is True,
        "resident_envelope_verified": metrics.get("resident_envelope_verified") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": "banana-smasher-serving-readiness-v1",
        "status": "READY" if not failed else "DEGRADED",
        "checks": checks,
        "failed": failed,
        "observed": {
            "prompt_tokens": prompt_tokens,
            "prefill_tok_s": prefill,
            "decode_tok_s": decode,
            "ttft_seconds": ttft,
            "decode_kernel_classes": sorted(observed_kernels),
        },
        "required": {
            "prompt_tokens": target["prompt_tokens"],
            "prefill_tok_s_min": gates["prefill_tok_s_min"],
            "decode_tok_s_min": gates["decode_tok_s_min"],
            "ttft_seconds_max": gates["ttft_seconds_max"],
            "decode_kernel_classes": sorted(required_kernels),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.metrics.read_text()), json.loads(args.expected.read_text()))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if result["status"] == "READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
