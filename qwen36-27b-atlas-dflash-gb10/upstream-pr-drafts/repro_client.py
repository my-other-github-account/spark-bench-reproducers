#!/usr/bin/env python3
"""OpenAI-compatible streaming receipt client.

The script intentionally uses only the Python standard library so it can run
from stripped-down repro hosts. It records streamed text, final usage, optional
token ids if the server exposes them, elapsed time, and stable hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_tokens(tokens: list[int]) -> str | None:
    if not tokens:
        return None
    payload = ",".join(str(t) for t in tokens)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _extract_token_ids(obj: Any) -> list[int]:
    found: list[int] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"token_id", "token_ids", "tokens"}:
                    if isinstance(value, int):
                        found.append(value)
                    elif isinstance(value, list):
                        found.extend(v for v in value if isinstance(v, int))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return found


def _collect_delta_text(chunk: dict[str, Any]) -> str:
    parts: list[str] = []
    for choice in chunk.get("choices", []) or []:
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            parts.append(content)
        text = choice.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def run_request(base_url: str, request_body: dict[str, Any], api_key: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(request_body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    started = time.perf_counter()
    chunks = 0
    content_parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reasons: list[str] = []
    token_ids: list[int] = []
    raw_events: list[dict[str, Any]] = []

    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                chunks += 1
                raw_events.append(event)
                content_parts.append(_collect_delta_text(event))
                token_ids.extend(_extract_token_ids(event))
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices", []) or []:
                    reason = choice.get("finish_reason")
                    if isinstance(reason, str):
                        finish_reasons.append(reason)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc

    elapsed = time.perf_counter() - started
    text = "".join(content_parts)
    return {
        "base_url": base_url,
        "request": request_body,
        "elapsed_seconds": elapsed,
        "stream_chunks": chunks,
        "content": text,
        "content_sha256": _sha256_text(text),
        "usage": usage,
        "finish_reasons": finish_reasons,
        "token_ids": token_ids or None,
        "token_ids_sha256": _sha256_tokens(token_ids),
        "raw_events": raw_events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--request-json", required=True, help="Path to request JSON body.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--api-key", default="[REDACTED]")
    args = parser.parse_args()

    with open(args.request_json, "r", encoding="utf-8") as fh:
        request_body = json.load(fh)
    receipt = run_request(args.base_url, request_body, args.api_key)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: receipt[k] for k in ("elapsed_seconds", "content_sha256", "usage")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
