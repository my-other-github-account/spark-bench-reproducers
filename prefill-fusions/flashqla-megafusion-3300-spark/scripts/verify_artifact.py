#!/usr/bin/env python3
import json
import math
import pathlib
import statistics
import sys

EXPECTED = {
    "latency_mode": "api",
    "prefix_caching_enabled": False,
    "prompt_size": 2048,
    "response_size": 32,
    "concurrency": 1,
    "runs": 30,
}
CANONICAL = {
    "mean": 3315.9676412971107,
    "median": 3315.498017640833,
    "std": 10.305973182328897,
    "min": 3294.5744437385347,
    "max": 3340.3987329331135,
}

def die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)

def main() -> None:
    if len(sys.argv) != 2:
        die("usage: verify_artifact.py results/result-*.json")
    p = pathlib.Path(sys.argv[1])
    d = json.loads(p.read_text())
    b = d["benchmarks"][0]
    vals = b["pp_throughput"]["values"]

    checks = [
        (d.get("latency_mode") == EXPECTED["latency_mode"], f"latency_mode={d.get('latency_mode')!r}"),
        (d.get("prefix_caching_enabled") is EXPECTED["prefix_caching_enabled"], f"prefix_caching_enabled={d.get('prefix_caching_enabled')!r}"),
        (b.get("prompt_size") == EXPECTED["prompt_size"], f"prompt_size={b.get('prompt_size')!r}"),
        (b.get("response_size") == EXPECTED["response_size"], f"response_size={b.get('response_size')!r}"),
        (b.get("concurrency") == EXPECTED["concurrency"], f"concurrency={b.get('concurrency')!r}"),
        (len(vals) == EXPECTED["runs"], f"runs={len(vals)!r}"),
    ]
    bad = [detail for ok, detail in checks if not ok]
    if bad:
        die("contract mismatch: " + ", ".join(bad))

    mean = statistics.mean(vals)
    median = statistics.median(vals)
    std = b["pp_throughput"].get("std", statistics.pstdev(vals))
    summary = {"mean": mean, "median": median, "std": std, "min": min(vals), "max": max(vals)}

    print("contract: PASS")
    print(f"artifact: {p}")
    print(f"mean={mean:.2f} median={median:.2f} std={std:.2f} min={min(vals):.2f} max={max(vals):.2f} n={len(vals)}")
    print("canonical:", " ".join(f"{k}={v:.2f}" for k, v in CANONICAL.items()))

    # Fresh reruns are allowed normal system variance. The canonical artifact itself should be exact;
    # rerun artifacts should stay close enough to catch contract drift.
    rel = abs(mean - CANONICAL["mean"]) / CANONICAL["mean"]
    if rel > 0.10:
        die(f"mean is >10% from canonical: rel_delta={rel:.3%}")
    print(f"canonical_delta_mean={rel:.3%}")

if __name__ == "__main__":
    main()
