#!/usr/bin/env python3
"""HTTP + one-time P1321 performance HEALTHCHECK sidecar.

The sidecar never rewrites PID 1 argv, environment, or serving state. On the
first health check for a specific container boot it runs one excluded warmup
per shape followed by C1x3/C2x3/C4x3 and persists a boot-bound receipt. Later checks
only verify HTTP health and that exact READY receipt.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def argv_value(argv: list[str], flag: str, default: str) -> str:
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return default


def pid_start_ticks(pid: int) -> int:
    # /proc/PID/stat field 22. The comm field may contain spaces, so split only
    # after the final ') '.
    suffix = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1]
    return int(suffix.split()[19])


try:
    argv = [part.decode() for part in Path("/proc/1/cmdline").read_bytes().split(b"\0") if part]
except Exception:
    argv = []
port = argv_value(argv, "--port", "8000")
if not port.isdecimal() or not (1 <= int(port) <= 65535):
    print(f"NOT_READY: invalid --port in process argv: {port}", file=sys.stderr)
    raise SystemExit(1)
base = os.environ.get("VLLM_HEALTHCHECK_URL", f"http://127.0.0.1:{port}")

try:
    with urllib.request.urlopen(base + "/health", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"health HTTP {response.status}")
    with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"models HTTP {response.status}")
        models = json.load(response)
    if not (models.get("data") or []):
        raise RuntimeError("models list is empty")
except Exception as exc:
    print(f"NOT_READY: {exc}", file=sys.stderr)
    raise SystemExit(1)

receipt_path = Path(os.environ.get("BANANA_SMASHER_PERF_HEALTH_RECEIPT", "/tmp/GOLDEN_PERF_HEALTH.json"))
lock_path = Path("/tmp/GOLDEN_PERF_HEALTH.lock")
current_start = pid_start_ticks(1)


def receipt_ready() -> bool:
    try:
        data = json.loads(receipt_path.read_text())
    except Exception:
        return False
    return (
        data.get("status") == "READY"
        and data.get("pid1_start_ticks") == current_start
        and len((data.get("measured") or {}).get("c1") or []) == 3
        and len((data.get("measured") or {}).get("c2") or []) == 3
        and len((data.get("measured") or {}).get("c4") or []) == 3
        and ((data.get("summary") or {}).get("c1_median_tok_s") or 0) >= 13.0
        and ((data.get("summary") or {}).get("c2_median_aggregate_tok_s") or 0) > 18.4223808768
        and ((data.get("summary") or {}).get("c4_median_aggregate_tok_s") or 0) >= 27.0
        and ((data.get("summary") or {}).get("c4_median_aggregate_tok_s") or 0)
        > ((data.get("summary") or {}).get("c2_median_aggregate_tok_s") or 0)
    )

lock_path.touch(exist_ok=True)
with lock_path.open("r+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not receipt_ready():
        command = [
            sys.executable,
            "/opt/banana_smasher/bin/golden_perf_check.py",
            "--base", base,
            "--model-root", "/model",
            "--c1-warmups", "1",
            "--c2-warmups", "1",
            "--c4-warmups", "1",
            "--c1-rows", "3",
            "--c2-rows", "3",
            "--c4-rows", "3",
            "--c1-bar", "13.0",
            "--c2-bar", "18.4223808768",
            "--c4-bar", "27.0",
            "--output", str(receipt_path),
        ]
        completed = subprocess.run(command, timeout=1700, check=False)
        if completed.returncode != 0 or not receipt_ready():
            print(f"NOT_READY: performance self-check rc={completed.returncode}", file=sys.stderr)
            raise SystemExit(1)
print("READY")
