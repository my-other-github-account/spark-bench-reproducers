#!/usr/bin/env python3
"""Optional exact-2K C1/C2 performance check for the stock vLLM server.

The check is informational: it never changes serve command semantics. It sends
OpenAI-compatible completion requests with exactly 2,048 prompt token IDs and
requires authoritative usage accounting; missing usage is a hard failure.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

SEED_PROMPT = "Write a Python module that implements an LRU cache with TTL support, unit tests included."
PROVENANCE = "P943 overlay 9a4b7098 / pack 3650fe7e / planes b524c5a; P1321 ladder be0453e1d6081a87a0288c8611b9ee5ec33a4b2ba927cb68c358e71a10b242f7"


def pid_start_ticks(pid: int = 1) -> int:
    suffix = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1]
    return int(suffix.split()[19])


def get_json(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {response.status}")
        return json.load(response)


def exact_prompt_ids(model_root: str, count: int) -> list[int]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True)
    seed = tokenizer.encode(SEED_PROMPT, add_special_tokens=False)
    if not seed:
        raise RuntimeError("tokenizer returned an empty seed")
    return (seed * ((count + len(seed) - 1) // len(seed)))[:count]


def stream_row(base: str, model: str, prompt_ids: list[int], max_tokens: int = 256) -> dict:
    body = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = last = None
    usage = None
    chunks = 0
    done_seen = False
    finish_reason = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        http_status = response.status
        for raw in response:
            line = raw.decode("utf-8", "strict").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done_seen = True
                break
            obj = json.loads(payload)
            if obj.get("usage") is not None:
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            if choice.get("text"):
                now = time.monotonic()
                if first is None:
                    first = now
                last = now
                chunks += 1
    end = time.monotonic()
    if http_status != 200:
        raise RuntimeError(f"completion returned HTTP {http_status}")
    if not done_seen:
        raise RuntimeError("completion stream ended without [DONE]")
    if not isinstance(usage, dict):
        raise RuntimeError("completion stream omitted authoritative usage")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens != len(prompt_ids):
        raise RuntimeError(f"prompt token mismatch: {prompt_tokens} != {len(prompt_ids)}")
    if completion_tokens != max_tokens:
        raise RuntimeError(f"completion token mismatch: {completion_tokens} != {max_tokens}")
    if first is None or last is None or chunks < 2:
        raise RuntimeError("completion stream did not contain a measurable decode interval")
    decode_span = last - first
    if decode_span <= 0:
        raise RuntimeError("completion decode interval is not positive")
    return {
        "http_status": http_status,
        "start": start,
        "first": first,
        "last": last,
        "end": end,
        "ttft_s": first - start,
        "wall_s": end - start,
        "decode_span_s": decode_span,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "aggregate_tok_s": completion_tokens / (end - start),
        "decode_tok_s_after_first": (completion_tokens - 1) / decode_span,
        "chunks": chunks,
        "done_seen": done_seen,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def concurrent_row(
    base: str, model: str, prompt_ids: list[int], max_tokens: int, concurrency: int
) -> dict:
    rows: list[dict | None] = [None] * concurrency
    errors: list[str] = []

    def worker(index: int) -> None:
        try:
            rows[index] = stream_row(base, model, prompt_ids, max_tokens)
        except Exception as exc:
            errors.append(f"stream{index}: {type(exc).__name__}: {exc}")

    start = time.monotonic()
    threads = [
        threading.Thread(target=worker, args=(i,), daemon=False)
        for i in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    end = time.monotonic()
    complete = [row for row in rows if row is not None]
    total = sum(int(row["completion_tokens"]) for row in complete)
    wall = end - start
    # Decode overlap, not merely request-lifetime overlap.
    decode_overlap = (
        len(complete) == concurrency
        and max(row["first"] for row in complete) < min(row["last"] for row in complete)
    )
    return {
        "streams": complete,
        "errors": errors,
        "batch_wall_s": wall,
        "total_completion_tokens": total,
        "aggregate_tok_s": total / wall if wall > 0 else 0.0,
        "decode_overlap": decode_overlap,
    }


def c2_row(base: str, model: str, prompt_ids: list[int], max_tokens: int) -> dict:
    return concurrent_row(base, model, prompt_ids, max_tokens, 2)


def c4_row(base: str, model: str, prompt_ids: list[int], max_tokens: int) -> dict:
    return concurrent_row(base, model, prompt_ids, max_tokens, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model-root", default="/model")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--output", type=Path, default=Path("/tmp/GOLDEN_PERF_HEALTH.json"))
    ap.add_argument("--c1-warmups", type=int, default=1)
    ap.add_argument("--c2-warmups", type=int, default=1)
    ap.add_argument("--c4-warmups", type=int, default=1)
    ap.add_argument("--c1-rows", type=int, default=3)
    ap.add_argument("--c2-rows", type=int, default=3)
    ap.add_argument("--c4-rows", type=int, default=3)
    ap.add_argument("--c1-bar", type=float, default=13.0)
    ap.add_argument("--c2-bar", type=float, default=18.4223808768)
    ap.add_argument("--c4-bar", type=float, default=27.0)
    ap.add_argument("--ttft-bar", type=float, default=2.5)
    ap.add_argument("--warm-only", action="store_true")
    args = ap.parse_args()
    created = time.time()
    health_ok = False
    errors: list[str] = []
    try:
        with urllib.request.urlopen(args.base + "/health", timeout=10) as response:
            health_ok = response.status == 200
        model_data = get_json(args.base + "/v1/models")
        model = model_data["data"][0]["id"]
        prompt_ids = exact_prompt_ids(args.model_root, args.prompt_tokens)
        if len(prompt_ids) != args.prompt_tokens:
            raise RuntimeError("failed to construct exact prompt token count")
        c1_warmups = [stream_row(args.base, model, prompt_ids, args.max_tokens) for _ in range(args.c1_warmups)]
        c2_warmups = [c2_row(args.base, model, prompt_ids, args.max_tokens) for _ in range(args.c2_warmups)]
        c4_warmups = [c4_row(args.base, model, prompt_ids, args.max_tokens) for _ in range(args.c4_warmups)]
        if args.warm_only:
            c1, c2, c4 = [], [], []
        else:
            c1 = [stream_row(args.base, model, prompt_ids, args.max_tokens) for _ in range(args.c1_rows)]
            c2 = [c2_row(args.base, model, prompt_ids, args.max_tokens) for _ in range(args.c2_rows)]
            c4 = [c4_row(args.base, model, prompt_ids, args.max_tokens) for _ in range(args.c4_rows)]
    except Exception as exc:
        model = None
        prompt_ids = []
        c1_warmups, c2_warmups, c4_warmups, c1, c2, c4 = [], [], [], [], [], []
        errors.append(f"{type(exc).__name__}: {exc}")
    if args.warm_only and not errors:
        receipt = {
            "schema": "banana_smasher-golden-perf-warmup-v2", "status": "WARMED",
            "created_unix": created, "pid1_start_ticks": pid_start_ticks(), "model": model, "provenance": PROVENANCE,
            "prompt_tokens_exact": len(prompt_ids), "max_tokens": args.max_tokens,
            "excluded_warmups": {"c1": c1_warmups, "c2": c2_warmups, "c4": c4_warmups}, "errors": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "WARMED", "output": str(args.output)}, sort_keys=True))
        return 0
    # P1321/box-10 canon uses completion tokens divided by full batch wall for
    # both C1 and C2; never substitute post-first-token decode speed for C1.
    c1_values = [row["aggregate_tok_s"] for row in c1]
    ttft_values = [row["ttft_s"] for row in c1]
    c2_values = [row["aggregate_tok_s"] for row in c2]
    c4_values = [row["aggregate_tok_s"] for row in c4]
    c1_median = statistics.median(c1_values) if c1_values else 0.0
    ttft_median = statistics.median(ttft_values) if ttft_values else float("inf")
    c2_median = statistics.median(c2_values) if c2_values else 0.0
    c4_median = statistics.median(c4_values) if c4_values else 0.0
    gates = {
        "health_200": health_ok,
        "exact_prompt_tokens": len(prompt_ids) == args.prompt_tokens,
        "c1_rows_complete": len(c1) == args.c1_rows,
        "c2_rows_complete": len(c2) == args.c2_rows,
        "c4_rows_complete": len(c4) == args.c4_rows,
        "c1_median_ge_bar": c1_median >= args.c1_bar,
        "c1_median_ttft_le_bar": ttft_median <= args.ttft_bar,
        "c2_median_gt_bar": c2_median > args.c2_bar,
        "c4_median_gt_bar": c4_median > args.c4_bar,
        "c4_median_gt_c2": c4_median > c2_median,
        "c2_http_usage_and_decode_overlap": bool(c2) and all(
            not row["errors"] and row["decode_overlap"] and len(row["streams"]) == 2
            for row in c2
        ),
        "c4_http_usage_and_decode_overlap": bool(c4) and all(
            not row["errors"] and row["decode_overlap"] and len(row["streams"]) == 4
            for row in c4
        ),
    }
    ready = not errors and all(gates.values())
    receipt = {
        "schema": "banana_smasher-golden-perf-health-v3", "status": "READY" if ready else "DEGRADED",
        "created_unix": created, "pid1_start_ticks": pid_start_ticks(), "model": model, "provenance": PROVENANCE,
        "contract": {"prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens, "endpoint": "/v1/completions", "aggregate_formula": "completion_tokens/batch_wall", "warmups_excluded": True, "shape_gates": "C1x3/C2x3/C4x3 medians"},
        "bars": {"c1": args.c1_bar, "c2": args.c2_bar, "c4": args.c4_bar, "ttft_s": args.ttft_bar},
        "deltas": {"c1_minus_bar": c1_median - args.c1_bar, "c2_minus_bar": c2_median - args.c2_bar, "c4_minus_bar": c4_median - args.c4_bar, "c4_minus_c2": c4_median - c2_median, "ttft_headroom_s": args.ttft_bar - ttft_median},
        "summary": {"c1_median_tok_s": c1_median, "c2_median_aggregate_tok_s": c2_median, "c4_median_aggregate_tok_s": c4_median, "c1_median_ttft_s": ttft_median},
        "gates": gates, "errors": errors,
        "excluded_warmups": {"c1": c1_warmups, "c2": c2_warmups, "c4": c4_warmups}, "measured": {"c1": c1, "c2": c2, "c4": c4},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], **receipt["summary"], "output": str(args.output)}, sort_keys=True))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
