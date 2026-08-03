#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.request

base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
model = os.environ.get("MODEL", "banana-smasher-v5")
payload = json.dumps(
    {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 8,
    }
).encode()
request = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    body = json.load(response)
message = body["choices"][0]["message"]["content"]
if not isinstance(message, str) or not message.strip():
    raise SystemExit("empty chat completion")
print(message)
