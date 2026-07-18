#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROMPT = (
    "Write a detailed numbered list of practical ways to validate a distributed "
    "inference server. Continue until the token limit; do not conclude early."
)


def request_once(endpoint: str, request_id: str) -> dict:
    payload = {
        "model": "deepseek-v4-flash-ud-iq4-xs",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 64,
        "temperature": 0,
        "seed": 42,
        "stream": False,
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as response:
        raw = response.read()
    wall = time.perf_counter() - t0
    body = json.loads(raw)
    timings = body.get("timings", {})
    usage = body.get("usage", {})
    predicted_n = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
    message = (body.get("choices") or [{}])[0].get("message") or {}
    text = (message.get("reasoning_content") or "") + (message.get("content") or "")
    return {
        "request_id": request_id,
        "wall_seconds": wall,
        "completion_tokens": int(usage.get("completion_tokens") or predicted_n),
        "predicted_n": predicted_n,
        "server_predicted_tok_s": timings.get("predicted_per_second"),
        "server_prompt_tok_s": timings.get("prompt_per_second"),
        "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
        "nonempty": bool(text.strip()),
        "response": body,
    }


def run_batch(endpoint: str, concurrency: int, count: int, label: str) -> dict:
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_once, endpoint, f"{label}-{i+1}") for i in range(count)]
        rows = [f.result() for f in futures]
    wall = time.perf_counter() - t0
    total_tokens = sum(r["predicted_n"] for r in rows)
    return {
        "label": label,
        "concurrency": concurrency,
        "request_count": count,
        "batch_wall_seconds": wall,
        "total_predicted_tokens": total_tokens,
        "aggregate_predicted_tok_s": total_tokens / wall,
        "nonempty_count": sum(r["nonempty"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8356")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    health_req = urllib.request.Request(args.endpoint.rstrip("/") + "/health")
    with urllib.request.urlopen(health_req, timeout=10) as r:
        health = json.loads(r.read())
    warmup = request_once(args.endpoint, "warmup-excluded")
    single = run_batch(args.endpoint, concurrency=1, count=5, label="single-5x64")
    p2 = run_batch(args.endpoint, concurrency=2, count=6, label="parallel2-6x64")
    p4 = run_batch(args.endpoint, concurrency=4, count=8, label="parallel4-8x64")

    single_server_tps = [r["server_predicted_tok_s"] for r in single["rows"] if r["server_predicted_tok_s"]]
    report = {
        "schema": "iq4xs-fast-config-throughput-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "health": health,
        "request_shape": {"max_tokens": 64, "temperature": 0, "seed": 42, "prompt": PROMPT},
        "warmup_excluded": warmup,
        "single_5x64": single,
        "single_server_predicted_tok_s": {
            "values": single_server_tps,
            "mean": statistics.fmean(single_server_tps),
            "median": statistics.median(single_server_tps),
            "min": min(single_server_tps),
            "max": max(single_server_tps),
        },
        "parallel2": p2,
        "parallel4": p4,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    concise = {
        "output": args.output,
        "single_5x64_server_tps": report["single_server_predicted_tok_s"],
        "single_5x64_full_wall_aggregate_tps": single["aggregate_predicted_tok_s"],
        "parallel2_aggregate_tps": p2["aggregate_predicted_tok_s"],
        "parallel4_aggregate_tps": p4["aggregate_predicted_tok_s"],
        "nonempty": {"single": single["nonempty_count"], "p2": p2["nonempty_count"], "p4": p4["nonempty_count"]},
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
