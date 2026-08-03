#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def get_json(url: str, *, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, payload: dict[str, Any], *, timeout: int = 120) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    api_base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    service_base = api_base[:-3] if api_base.endswith("/v1") else api_base
    model = os.environ.get("MODEL", "banana-smasher-v5")

    with urllib.request.urlopen(f"{service_base}/health", timeout=30) as response:
        response.read()

    models = get_json(f"{api_base}/models")
    model_ids: set[str] = set()
    for item in models.get("data", []):
        if isinstance(item, dict):
            identifier = item.get("id")
            if isinstance(identifier, str):
                model_ids.add(identifier)
    if model not in model_ids:
        raise SystemExit(
            f"expected served model {model!r} is absent from /v1/models: {sorted(model_ids)}"
        )

    body = post_json(
        f"{api_base}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 8,
        },
    )
    message = body["choices"][0]["message"]["content"]
    if not isinstance(message, str) or not message.strip():
        raise SystemExit("empty chat completion")
    print(message)


if __name__ == "__main__":
    main()
