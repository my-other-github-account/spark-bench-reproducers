#!/usr/bin/env python3
"""Deterministically seal the complete final vendored Python closure."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess

SITE = Path("lib/python3.12/site-packages")
M4_DEST = SITE / "vq_warp_gemv"
QUANT_DEST = SITE / "vllm/models/deepseek_v4/quant_config.py"
ATTENTION_DEST = SITE / "vllm/models/deepseek_v4/attention.py"
FLASHINFER_CORE_DEST = SITE / "flashinfer/jit/core.py"


def scan(root: Path, prefix: Path, context: str) -> list[tuple[str, Path, str, str | None]]:
    if not root.is_dir():
        raise SystemExit(f"missing context {context}: {root}")
    found = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        dest = (prefix / rel).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            found.append((dest, path, context, target))
        elif path.is_file():
            found.append((dest, path, context, None))
        elif not path.is_dir():
            raise SystemExit(f"unsupported member in {context}: {path}")
    if not found:
        raise SystemExit(f"empty context is forbidden: {context}")
    return found


def hash_row(item: tuple[str, Path, str, str | None]) -> dict:
    dest, path, context, target = item
    if target is not None:
        return {"path": dest, "kind": "symlink", "target": target, "source_context": context}
    if context == "pyoverlay" and dest == QUANT_DEST.as_posix():
        # The Docker build applies this exact source-hash-gated transform before
        # static verification. Seal the final image bytes, not the input overlay.
        from patch_quant_method_defaults import patched_bytes
        data = patched_bytes(path)
        return {
            "path": dest, "kind": "file", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "source_context": "pyoverlay+quant_patch",
        }
    if context == "pyoverlay" and dest == ATTENTION_DEST.as_posix():
        from patch_quant_method_defaults import patched_attention_bytes
        data = patched_attention_bytes(path)
        return {
            "path": dest, "kind": "file", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "source_context": "pyoverlay+iq3_config_admission_patch",
        }
    if context == "venv" and dest == FLASHINFER_CORE_DEST.as_posix():
        from patch_flashinfer_cache_load import patched_bytes
        data = patched_bytes(path)
        return {
            "path": dest, "kind": "file", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "source_context": "venv+flashinfer_sealed_cache_load_patch",
        }
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return {
        "path": dest, "kind": "file", "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(), "source_context": context,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--venv", type=Path, required=True)
    ap.add_argument("--pyoverlay", type=Path, required=True)
    ap.add_argument("--m4-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    script = r'''
import importlib.metadata as md, json, sys
names = ["torch", "vllm", "triton", "flashinfer-python", "numpy"]
out = {"python": sys.version, "distributions": {}}
for name in names:
    try:
        d = md.distribution(name)
        out["distributions"][name] = {"version": d.version, "metadata_name": d.metadata.get("Name")}
    except md.PackageNotFoundError:
        out["distributions"][name] = None
print(json.dumps(out, sort_keys=True))
'''
    meta = json.loads(subprocess.check_output([str(args.python), "-c", script], text=True))
    merged: dict[str, tuple[str, Path, str, str | None]] = {}
    for item in scan(args.venv.resolve(), Path(), "venv"):
        merged[item[0]] = item
    for item in scan(args.pyoverlay.resolve(), SITE, "pyoverlay"):
        merged[item[0]] = item
    m4 = args.m4_root.resolve()
    for name in ("_C.so", "__init__.py"):
        path = m4 / name
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing regular M4 member: {path}")
        item = ((M4_DEST / name).as_posix(), path, "m4", None)
        merged[item[0]] = item
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(hash_row, merged.values()))
    rows.sort(key=lambda row: row["path"])
    meta.update({
        "schema": "banana_smasher-golden-wheel-manifest-v2",
        "destination_root": "/work/build/venvs/vllm-banana",
        "file_count": len(rows), "files": rows,
        "truth_label": "PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C",
        "p1268_launcher_sha256": "d7aff83634ef5456385419523b3ecaf3a8213fecb8155f34053d65366953ebb2",
        "p1268_result_sha256": "9b1d42fe3f4dcb28e7f8660b37f800fdbfdcd7f721fb4bc57ca31a0dda313860",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "file_count": len(rows), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
