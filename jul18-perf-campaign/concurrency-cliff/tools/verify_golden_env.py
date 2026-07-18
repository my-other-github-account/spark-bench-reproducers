#!/usr/bin/env python3
"""Task-local Golden Serve verifier for the c=2 small-M/batch-graph rail."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

EXPECTED = {
    "moe_w2_cubit.py": "2cc85159",
    "moe_vq_triton.py": "7b25bf83",
    "_C.cpython-312-aarch64-linux-gnu.so": "2b0baea3",
}
REQUIRED_ENV = {
    "VLLM_MOE_W2": "1",
    "VLLM_MOE_W2_NUM_LAYERS": "43",
    "VLLM_MOE_VQ_CUDA_WARP": "1",
    "VLLM_MOE_VQ_CUDA_WARP_MAX_M": "4",
    "VLLM_MOE_VQ_FAST": "1",
    "VLLM_MOE_VQ_GROUP_FAST": "1",
    "VLLM_MOE_VQ_D4_FAST": "1",
    "VLLM_MOE_VQ_M1_FAST": "0",
    "VLLM_MOE_W2_DECODE_GRAPH": "1",
    "VLLM_MOE_W2_DECODE_GRAPH_MAX_T": "4",
}
FORBIDDEN_ENV = ("VLLM_MOE_VQ_CUDA_WARP_MAX_LAYER", "VLLM_MOE_VQ_BN")
MODEL_PATH = "$HOME/models/hf/DeepSeek-V4-Flash"
M2_DISPATCH_PROBE = "VQ DISPATCH PROBE path=cuda_warp_m2"
GRAPH_REPLAY_PROBE = "DECODE-GRAPH REPLAY PROBE"


def sha256_prefix(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--require-batch2", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []

    proc = Path(f"/proc/{args.pid}")
    if not proc.exists():
        failures.append(f"PID {args.pid} is absent")
        env: dict[str, str] = {}
        cmdline = ""
    else:
        env = dict(
            item.split("=", 1)
            for item in (proc / "environ").read_text().split("\0")
            if "=" in item
        )
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    if MODEL_PATH not in cmdline or "--port 8001" not in cmdline:
        failures.append("MODEL/CMDLINE: exact full checkpoint and loopback :8001 serve not proven")

    for key, value in REQUIRED_ENV.items():
        if env.get(key) != value:
            failures.append(f"ENV {key}={env.get(key)!r} != {value!r}")
    for key in FORBIDDEN_ENV:
        if key in env:
            failures.append(f"ENV {key} is forbidden")

    pythonpath = env.get("PYTHONPATH", "").split(":")
    if len(pythonpath) < 2:
        failures.append("PYTHONPATH does not bind task runtime + kernel")
    else:
        runtime, kernel = map(Path, pythonpath[:2])
        files = {
            "moe_w2_cubit.py": runtime / "vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py",
            "moe_vq_triton.py": runtime / "vllm/model_executor/layers/quantization/utils/moe_vq_triton.py",
            "_C.cpython-312-aarch64-linux-gnu.so": kernel / "vq_warp_gemv/_C.cpython-312-aarch64-linux-gnu.so",
        }
        for name, path in files.items():
            if not path.is_file():
                failures.append(f"ARTIFACT missing: {path}")
            elif sha256_prefix(path) != EXPECTED[name]:
                failures.append(
                    f"ARTIFACT {name} sha256={sha256_prefix(path)} != {EXPECTED[name]}"
                )

    if not args.server_log.is_file():
        failures.append(f"LOG missing: {args.server_log}")
    else:
        log = args.server_log.read_text(errors="ignore")
        if "DECODE-GRAPH ON-PATH sentinel" not in log:
            failures.append("LOG: no private decode-graph capture sentinel")
        if "decode-graph capture FAILED" in log:
            failures.append("LOG: decode-graph capture failure present")
        if "EngineCore encountered an issue" in log:
            failures.append("LOG: EngineCore failure present")
        if args.require_batch2:
            if M2_DISPATCH_PROBE not in log:
                failures.append("LOG: c=2 did not prove cuda_warp_m2 dispatch")
            replay_lines = [line for line in log.splitlines() if GRAPH_REPLAY_PROBE in line]
            if not any("T=2" in line for line in replay_lines):
                failures.append("LOG: c=2 batch-shape graph replay not proven")

    if failures:
        print("GOLDEN PREFLIGHT FAIL:")
        for failure in failures:
            print(" -", failure)
        return 1
    mode = "post-c2" if args.require_batch2 else "prebench"
    print(f"GOLDEN PREFLIGHT PASS ({mode}; env+hashes+graph sentinels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
