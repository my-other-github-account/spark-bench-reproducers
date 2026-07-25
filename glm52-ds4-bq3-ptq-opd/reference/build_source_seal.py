#!/usr/bin/env python3
"""Build or verify the committed PTQ-OPD production-source seal."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import train_contracts as C

ROOT = Path(__file__).resolve().parent
SEAL_PATH = ROOT / "SOURCE_SEAL.json"


def production_paths() -> list[str]:
    paths = [
        "build_source_seal.py",
        "ptq_opd.py",
        "train_contracts.py",
        "train_ptq_opd.py",
        "plans/static_anchor_control.json",
    ]
    paths.extend(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "adapter").rglob("*.py"))
    )
    return sorted(paths)


def write_atomic(path: Path, value: object) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def check() -> dict[str, object]:
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    expected = production_paths()
    if sorted(seal.get("files", {})) != expected:
        raise ValueError("source seal file set does not match production source set")
    C.verify_source_seal(ROOT, seal)
    return seal


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.write:
        seal = C.build_source_seal(ROOT, production_paths())
        write_atomic(SEAL_PATH, seal)
        action = "wrote"
    else:
        seal = check()
        action = "verified"
    print(json.dumps({
        "action": action,
        "files": len(seal["files"]),
        "seal_sha256": seal["seal_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
