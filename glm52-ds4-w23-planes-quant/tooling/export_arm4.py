#!/usr/bin/env python3
"""Export a BINREPAIR checkpoint's codebooks into a complete VQ3U plane set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch checkpoint state['L{n}']['cb13'/'cb2'] into base "
            "vq3u_layer_NNN.pt files and emit a hash-bound receipt."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-planes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=43)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    checkpoint = args.checkpoint.expanduser().resolve()
    base_planes = args.base_planes.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a dict at key 'state'")
    expected = {f"L{layer}" for layer in range(args.layers)}
    missing = sorted(expected - set(state))
    if missing:
        raise ValueError(f"checkpoint is missing layers: {missing}")

    receipt: dict[str, Any] = {
        "format": "binrepair-vq3u-export-v1",
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "manifest_md5": payload.get("manifest_md5"),
        "best_probe_mean": payload.get("best_probe_mean"),
        "exported_at_unix": time.time(),
        "layers": [],
    }

    for layer in range(args.layers):
        src = base_planes / f"vq3u_layer_{layer:03d}.pt"
        dst = output / src.name
        if not src.is_file():
            raise FileNotFoundError(src)
        plane = torch.load(src, map_location="cpu", weights_only=False)
        entry = state[f"L{layer}"]
        if not isinstance(entry, dict) or not {"cb13", "cb2"} <= set(entry):
            raise ValueError(f"L{layer} is missing cb13/cb2")

        new13 = entry["cb13"].to(torch.float16).contiguous()
        new2 = entry["cb2"].to(torch.float16).contiguous()
        if new13.shape != plane["cb13"].shape:
            raise ValueError(f"L{layer} cb13 shape mismatch")
        if new2.shape != plane["cb2"].shape:
            raise ValueError(f"L{layer} cb2 shape mismatch")

        delta13 = float((new13.float() - plane["cb13"].float()).abs().max())
        delta2 = float((new2.float() - plane["cb2"].float()).abs().max())
        plane["cb13"] = new13
        plane["cb2"] = new2
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        torch.save(plane, tmp)
        os.replace(tmp, dst)
        receipt["layers"].append(
            {
                "layer": layer,
                "file": dst.name,
                "bytes": dst.stat().st_size,
                "md5": digest(dst, "md5"),
                "sha256": digest(dst),
                "max_delta13": delta13,
                "max_delta2": delta2,
            }
        )
        if layer % 10 == 0 or layer + 1 == args.layers:
            print(
                f"L{layer:03d}: max_delta13={delta13:.6g} "
                f"max_delta2={delta2:.6g}",
                flush=True,
            )

    atomic_json(output / "EXPORT_META.json", receipt)
    print(f"EXPORT_COMPLETE layers={args.layers} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
