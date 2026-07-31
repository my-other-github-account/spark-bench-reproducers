#!/usr/bin/env python3
"""Create the one-volume external model+wire artifact used by vllm serve /model.

On one filesystem this uses hard links, so the prepared artifact consumes only
metadata plus a patched config.json. Every resulting regular file is SHA-256
sealed in GOLDEN_PACK_MANIFEST.json. No source file is modified.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha_row(path: Path, root: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            h.update(block)
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def link_tree(source: Path, target: Path, *, skip: set[str] | None = None) -> None:
    skip = skip or set()
    for current in sorted(source.rglob("*")):
        rel = current.relative_to(source)
        if str(rel) in skip:
            continue
        destination = target / rel
        if current.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if current.is_symlink():
            raise RuntimeError(f"source symlink is forbidden: {current} -> {os.readlink(current)}")
        if not current.is_file():
            raise RuntimeError(f"unsupported source member: {current}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(current, destination)


def contained(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--wire-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    model = args.model_root.resolve()
    wire = args.wire_root.resolve()
    output = args.output.resolve()
    if model == wire or contained(model, wire) or contained(wire, model):
        raise SystemExit("model-root and wire-root must be disjoint")
    for source in (model, wire):
        if output == source or contained(output, source) or contained(source, output):
            raise SystemExit(f"output and source trees must be disjoint: {output} vs {source}")
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    for required in (model / "config.json", wire / "PACK_MANIFEST.json", wire / "PACK_COMPLETE"):
        if not required.is_file():
            raise SystemExit(f"missing required input: {required}")
    temp = output.with_name(output.name + ".tmp")
    if temp.exists():
        raise SystemExit(f"refusing stale temp output: {temp}")
    temp.mkdir(parents=True)
    link_tree(model, temp, skip={"config.json"})
    link_tree(wire, temp / "wire_v4-step32")
    config = json.loads((model / "config.json").read_text())
    qconfig = dict(config.get("quantization_config") or {})
    qconfig.update({
        "moe_quant_algo": "IQ3_WIRE",
        "moe_pack_root": "wire_v4-step32",
        "golden_runtime": "P1268_PUBLIC_CANON_IQ3_WIRE",
        "golden_runtime_result_sha256": "9b1d42fe3f4dcb28e7f8660b37f800fdbfdcd7f721fb4bc57ca31a0dda313860",
    })
    config["quantization_config"] = qconfig
    (temp / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    files = []
    for path in temp.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"prepared artifact contains a symlink: {path}")
        if path.is_file():
            files.append(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda p: sha_row(p, temp), files))
    rows.sort(key=lambda row: row["path"])
    manifest = {
        "schema": "genesis-golden-external-pack-v1",
        "truth_label": "PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C",
        "model_source": str(model),
        "wire_source": str(wire),
        "quant_method_auto_detected": "deepseek_v4_fp8 + moe_quant_algo=IQ3_WIRE",
        "p1268_result_sha256": "9b1d42fe3f4dcb28e7f8660b37f800fdbfdcd7f721fb4bc57ca31a0dda313860",
        "files": rows,
    }
    (temp / "GOLDEN_PACK_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temp.rename(output)
    print(json.dumps({"status": "PASS", "output": str(output), "files": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
