#!/usr/bin/env python3
"""Assemble a self-describing, fail-closed GENESIS checkpoint export pack."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from pack_contract import SCHEMA_VERSION, canonical_inventory_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def export_pack(
    *,
    source_root: str | Path,
    overlay: str | Path,
    tokenizer: str | Path,
    output: str | Path,
    model_id: str,
    expected_bytes: int,
    expected_files: int,
    expected_inventory_sha256: str | None,
    sealed_source_inventory_sha256: str | None = None,
    preassembled_planes: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    overlay_path = Path(overlay).resolve()
    tokenizer_path = Path(tokenizer).resolve()
    destination = Path(output).resolve()
    if (
        not isinstance(sealed_source_inventory_sha256, str)
        or len(sealed_source_inventory_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in sealed_source_inventory_sha256
        )
    ):
        raise RuntimeError(
            "--sealed-source-inventory-sha256 must be a lowercase SHA-256"
        )
    if preassembled_planes:
        expected_source = destination / "planes"
        if not expected_source.is_dir() or source != expected_source.resolve():
            raise RuntimeError(
                "preassembled mode requires --source-root to be OUTPUT/planes"
            )
        unexpected = [
            path.name for path in destination.iterdir() if path.name != "planes"
        ]
        if unexpected:
            raise RuntimeError(
                f"preassembled output contains unexpected entries: {unexpected}"
            )
    elif destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    plane_sources = sorted(path for path in source.rglob("*") if path.is_file())
    if len(plane_sources) != expected_files:
        raise RuntimeError(
            f"resident file count mismatch: expected={expected_files} actual={len(plane_sources)}"
        )
    resident_bytes = sum(path.stat().st_size for path in plane_sources)
    if resident_bytes != expected_bytes:
        raise RuntimeError(
            f"resident byte count mismatch: expected={expected_bytes} actual={resident_bytes}"
        )

    installs: list[tuple[Path, Path, str]] = []
    for path in plane_sources:
        relative = path.relative_to(source)
        installs.append((path, Path("planes") / relative, "plane"))
    installs.extend([
        (overlay_path, Path("overlay/mixed_tier_compact.pt"), "mixed_tier_overlay"),
        (tokenizer_path, Path("tokenizer/tokenizer.json"), "tokenizer"),
    ])
    def describe(item: tuple[Path, Path, str]) -> dict[str, Any]:
        item_source, relative, role = item
        return {
            "path": relative.as_posix(),
            "role": role,
            "bytes": item_source.stat().st_size,
            "sha256": _sha256(item_source),
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        files = list(pool.map(describe, installs))
    resident = [row for row in files if row["role"] == "plane"]
    inventory = canonical_inventory_sha256(resident)
    if expected_inventory_sha256 and inventory != expected_inventory_sha256:
        raise RuntimeError(
            "resident inventory mismatch: "
            f"expected={expected_inventory_sha256} actual={inventory}"
        )
    for path, relative, role in installs:
        if preassembled_planes and role == "plane":
            continue
        _install_file(path, destination / relative)
    manifest = {
        "schema": "genesis-pack",
        "schema_version": SCHEMA_VERSION,
        "container_schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "validation_scope": "systems-serving-only",
        "quality_validated": False,
        "resident_envelope": {
            "root": "planes",
            "bytes": resident_bytes,
            "files": len(resident),
            "inventory_sha256": inventory,
            "sealed_source_inventory_sha256": sealed_source_inventory_sha256,
        },
        "serving": {
            "artifact": "overlay/mixed_tier_compact.pt",
            "tokenizer": "tokenizer/tokenizer.json",
            "layers": 43,
            "experts": 256,
            "topk": 6,
            "tier_names": ["qtip", "truevq_d4", "truevq_d8", "native_mxfp4"],
            "prefill_mode": "dense_all",
            "dense_threshold": 64,
            "layer_stride": 43,
        },
        "files": files,
    }
    (destination / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="deepseek-v4-mixed-tier-prefill-ladder")
    parser.add_argument("--expected-bytes", type=int, default=101_346_700_411)
    parser.add_argument("--expected-files", type=int, default=1_645)
    parser.add_argument("--expected-inventory-sha256")
    parser.add_argument("--sealed-source-inventory-sha256")
    parser.add_argument(
        "--preassembled-planes",
        action="store_true",
        help="hash an existing read-only OUTPUT/planes mount instead of copying it",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = export_pack(
        source_root=args.source_root,
        overlay=args.overlay,
        tokenizer=args.tokenizer,
        output=args.output,
        model_id=args.model_id,
        expected_bytes=args.expected_bytes,
        expected_files=args.expected_files,
        expected_inventory_sha256=args.expected_inventory_sha256,
        sealed_source_inventory_sha256=args.sealed_source_inventory_sha256,
        preassembled_planes=args.preassembled_planes,
        workers=args.workers,
    )
    print(json.dumps({
        "status": "PASS",
        "model_id": manifest["model_id"],
        "resident_envelope": manifest["resident_envelope"],
        "files": len(manifest["files"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
