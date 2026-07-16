#!/usr/bin/env python3
"""Idempotently make run_pilot.sh's campaign inputs environment-overridable."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

TARGETS = (
    "BR_VQ3B_DIR",
    "BR_TRAIN",
    "BR_PROBE",
    "BR_REF_KLD",
    "BR_OUTDIR",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def patch_text(text: str, path: Path) -> tuple[str, list[str]]:
    changed: list[str] = []
    for variable in TARGETS:
        pattern = re.compile(
            rf"^export {variable}=(\"[^\"]*\"|'[^']*'|\S+)(.*)$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise ValueError(
                f"{path}: expected one export for {variable}, found {len(matches)}"
            )
        value = matches[0].group(1)
        suffix = matches[0].group(2)
        if f"${{{variable}:-" in value:
            continue
        default = value
        if default.startswith('"') and default.endswith('"'):
            default = default[1:-1]
        replacement = f'export {variable}="${{{variable}:-{default}}}"{suffix}'
        text = pattern.sub(replacement, text, count=1)
        changed.append(variable)
    return text, changed


def main() -> int:
    args = parse_args()
    pending = False
    for raw_path in args.paths:
        path = raw_path.expanduser()
        original = path.read_text()
        updated, changed = patch_text(original, path)
        if args.check:
            if changed:
                pending = True
                print(f"NEEDS_PATCH {path}: {','.join(changed)}")
            else:
                print(f"PASS {path}")
            continue
        if changed:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(updated)
            os.chmod(tmp, path.stat().st_mode)
            os.replace(tmp, path)
            print(f"PATCHED {path}: {','.join(changed)}")
        else:
            print(f"NOCHANGE {path}")
    return 1 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
