#!/usr/bin/env python3
"""Create RELEASE.json from the only two late-bound release facts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
SHAPES = ("C1", "C2", "C4", "C8", "C16")
APPROVED_SOURCE_COMMIT = "05016b598ae45f7b162277710f6076ef76cf31c2"


def _walk(value: Any, key: str = ""):
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child, key)
    else:
        yield key, value


def _pass_status(receipt: dict[str, Any]) -> str:
    for key in ("status", "verdict", "result"):
        value = receipt.get(key)
        if isinstance(value, str) and value.upper().startswith("PASS"):
            return value
    raise ValueError("gate receipt lacks a top-level PASS status/verdict/result")


def _pack_manifest_sha(receipt: dict[str, Any]) -> str:
    candidates = []
    for key, value in _walk(receipt):
        normalized = key.lower().replace("-", "_")
        if "pack" in normalized and "manifest" in normalized and isinstance(value, str):
            candidate = value.removeprefix("sha256:")
            if SHA_RE.fullmatch(candidate):
                candidates.append(candidate)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError("gate receipt must bind exactly one pack-manifest SHA-256")
    return unique[0]


def build_release(image_digest: str, gate_path: Path) -> dict[str, Any]:
    if not DIGEST_RE.fullmatch(image_digest):
        raise ValueError("image digest must be sha256: followed by 64 lowercase hex digits")
    if not gate_path.is_file() or gate_path.is_symlink():
        raise ValueError("gate receipt must be one regular, non-symlink file")
    raw = gate_path.read_bytes()
    receipt = json.loads(raw)
    if not isinstance(receipt, dict):
        raise ValueError("gate receipt must be a JSON object")
    status = _pass_status(receipt)
    values = [value for _, value in _walk(receipt)]
    if image_digest not in values:
        raise ValueError("gate receipt does not bind the supplied image digest")
    serialized = json.dumps(receipt, sort_keys=True).upper()
    missing = [shape for shape in SHAPES if shape not in serialized]
    if missing:
        raise ValueError("gate receipt lacks ladder shapes: " + ", ".join(missing))
    repo = Path(__file__).resolve().parents[3]
    public_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "schema": "banana-smasher-release-v1",
        "status": "GOLDEN_FULL_PACK_GATE_PASS",
        "public_source_commit": public_commit,
        "approved_banana_smasher_source_commit": APPROVED_SOURCE_COMMIT,
        "image_digest": image_digest,
        "pack_manifest_sha256": _pack_manifest_sha(receipt),
        "measured_full_pack_gate": {
            "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "status": status,
            "required_shapes": list(SHAPES),
            "warmup_excluded": True,
            "aggregate_formula": "completion_tokens/batch_wall",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("RELEASE.json"))
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"refusing to overwrite immutable release: {args.output}")
    try:
        release = build_release(args.image_digest, args.gate_receipt)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"release seal refused: {exc}") from exc
    args.output.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": release["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
