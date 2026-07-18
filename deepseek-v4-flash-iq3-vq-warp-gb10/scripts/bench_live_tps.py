#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.request
from pathlib import Path

_SEQUENCE = re.compile(r"^\s*\d+(?:\s*,\s*\d+)*\s*,?\s*$")


def validate_sequence(text: str) -> dict:
    if not _SEQUENCE.fullmatch(text):
        raise RuntimeError(f"output is not a comma-separated integer sequence: {text[:160]!r}")
    values = [int(value) for value in re.findall(r"\d+", text)]
    expected = list(range(1, len(values) + 1))
    if values != expected:
        raise RuntimeError(f"sequence mismatch: got {values[:16]!r}")
    return {"integer_count": len(values), "starts_at_one": True, "strictly_ascending": True}


def one(url: str, model: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write an endless comma-separated sequence of positive integers "
                    "starting at 1. Output only the sequence and continue until the "
                    "response token limit."
                ),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    start = time.perf_counter()
    first_token = None
    usage = None
    finish_reason = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    completion_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            event = json.loads(body)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if content or reasoning:
                    if first_token is None:
                        first_token = time.perf_counter()
                    if content:
                        content_parts.append(content)
                        completion_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        completion_parts.append(reasoning)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    end = time.perf_counter()
    if not usage or first_token is None:
        raise RuntimeError({"usage": usage, "first_token": first_token})
    completion_tokens = int(usage["completion_tokens"])
    if completion_tokens != max_tokens:
        raise RuntimeError(f"expected {max_tokens} completion tokens, got {completion_tokens}")
    if finish_reason != "length":
        raise RuntimeError(f"expected finish_reason='length', got {finish_reason!r}")
    content_text = "".join(content_parts)
    completion_text = "".join(completion_parts)
    sequence = validate_sequence(completion_text)
    decode_elapsed = max(end - first_token, 1e-9)
    return {
        "wall_s": end - start,
        "ttft_s": first_token - start,
        "decode_s_after_first": decode_elapsed,
        "completion_tokens": completion_tokens,
        "expected_completion_tokens": max_tokens,
        "wall_tok_s": completion_tokens / (end - start),
        "decode_tok_s_after_first": max(completion_tokens - 1, 0) / decode_elapsed,
        "finish_reason": finish_reason,
        "output_prefix": completion_text[:160],
        "content_prefix": content_text[:160],
        "reasoning_prefix": "".join(reasoning_parts)[:160],
        "sequence_validation": sequence,
        "valid": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-iq3-arm4")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = args.base_url.rstrip("/") + "/chat/completions"
    warmups = [one(url, args.model, args.warmup_tokens) for _ in range(args.warmup)]
    runs = [one(url, args.model, args.max_tokens) for _ in range(args.reps)]
    result = {
        "format": "iq3-live-stream-tps-v2",
        "url": url,
        "model": args.model,
        "protocol": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": args.max_tokens,
            "warmup_count": args.warmup,
            "warmup_tokens": args.warmup_tokens,
            "measured_repetitions": args.reps,
            "required_finish_reason": "length",
            "required_completion_tokens": args.max_tokens,
            "required_behavior": "comma-separated integers 1..N, strictly ascending",
        },
        "warmups": warmups,
        "runs": runs,
        "all_runs_valid": all(run["valid"] for run in runs),
        "median_wall_tok_s": statistics.median(run["wall_tok_s"] for run in runs),
        "median_decode_tok_s_after_first": statistics.median(
            run["decode_tok_s_after_first"] for run in runs
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
