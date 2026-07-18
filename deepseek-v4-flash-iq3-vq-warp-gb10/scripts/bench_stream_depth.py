#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Single-request streaming decode-depth benchmark with per-token timestamps."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def window_stats(arrivals: list[float], start: int, end: int) -> dict[str, Any]:
    """Return rates for half-open token window [start, end).

    interval_tok_s uses only intervals between tokens inside the window.
    boundary_tok_s includes the interval entering the window when start > 0.
    """
    end = min(end, len(arrivals))
    if start < 0 or end <= start or end > len(arrivals):
        return {"start": start, "end": end, "available": False}
    count = end - start
    first = arrivals[start]
    last = arrivals[end - 1]
    internal_intervals = max(count - 1, 0)
    internal_elapsed = max(last - first, 1e-12)
    result: dict[str, Any] = {
        "start": start,
        "end": end,
        "available": True,
        "token_count": count,
        "elapsed_s_first_to_last": internal_elapsed,
        "interval_tok_s": internal_intervals / internal_elapsed,
    }
    if start > 0:
        boundary_elapsed = max(last - arrivals[start - 1], 1e-12)
        result["elapsed_s_including_entry_interval"] = boundary_elapsed
        result["boundary_tok_s"] = count / boundary_elapsed
    else:
        result["boundary_tok_s"] = result["interval_tok_s"]
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
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
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    arrivals: list[float] = []
    token_ids: list[int] = []
    usage = None
    finish_reason = None
    text_prefix_parts: list[str] = []
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
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
            now = time.perf_counter()
            for choice in event.get("choices", []):
                delta_ids = choice.get("token_ids") or []
                token_ids.extend(int(x) for x in delta_ids)
                arrivals.extend([now] * len(delta_ids))
                delta = choice.get("delta", {})
                content = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or ""
                )
                if content and sum(map(len, text_prefix_parts)) < 256:
                    text_prefix_parts.append(content)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    ended = time.perf_counter()
    if not usage:
        raise RuntimeError("stream ended without usage")
    completion_tokens = int(usage["completion_tokens"])
    if len(arrivals) != completion_tokens:
        raise RuntimeError(
            f"token timestamp count mismatch: arrivals={len(arrivals)} "
            f"usage.completion_tokens={completion_tokens}"
        )
    if len(arrivals) < 2:
        raise RuntimeError(f"too few generated tokens: {len(arrivals)}")
    first = arrivals[0]
    last = arrivals[-1]
    decode_elapsed = max(last - first, 1e-12)
    result = {
        "format": "iq3-stream-depth-v1",
        "url": url,
        "model": args.model,
        "request": payload,
        "started_perf_counter": started,
        "wall_s": ended - started,
        "ttft_s": first - started,
        "stream_tail_s": ended - last,
        "completion_tokens": completion_tokens,
        "decode_s_after_first_to_last": decode_elapsed,
        "decode_tok_s_after_first": (completion_tokens - 1) / decode_elapsed,
        "wall_tok_s": completion_tokens / (ended - started),
        "finish_reason": finish_reason,
        "text_prefix": "".join(text_prefix_parts)[:256],
        "windows": {
            "tokens_0_256": window_stats(arrivals, 0, 256),
            "tokens_1024_1280": window_stats(arrivals, 1024, 1280),
            "tokens_3840_4096": window_stats(arrivals, 3840, 4096),
        },
        "arrival_s_after_request": [t - started for t in arrivals],
        "token_ids": token_ids,
        "usage": usage,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-iq3-arm4")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {k: v for k, v in result.items() if k not in {"arrival_s_after_request", "token_ids"}}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
