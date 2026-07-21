#!/usr/bin/env python3
"""Fail-closed privacy and public-naming audit for this reproducer package."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
TEXT_SUFFIXES = {"", ".json", ".md", ".py", ".sha256", ".txt"}

# Construct sensitive literals so this audit does not flag its own policy table.
JOINED_FORBIDDEN = [
    "mac" + "mini",
    "d" + "nola",
    "Da" + "vid",
    "banana_bae" + "ee",
]
PATTERNS = {
    "absolute home path": re.compile(r"/(?:Users|home)/[^\s\"'`]+"),
    "private host": re.compile(r"\bspark-[0-9]+(?:\b|[-_])", re.IGNORECASE),
    "task identifier": re.compile(r"\bt_[0-9a-f]{8}\b"),
    "IPv4 address": re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"),
    "private mission path": re.compile(r"(?:^|[/\\])missions(?:[/\\]|$)", re.IGNORECASE),
    "BQ3 misnamed as IQ3": re.compile(r"IQ3_BIN|repaired-IQ3|IQ3 artifact", re.IGNORECASE),
}


def text_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == SELF:
            continue
        if any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return sorted(files)


def main() -> None:
    failures = []
    files = text_files()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    failures.append(f"{relative}:{line_number}: {label}")
            for forbidden in JOINED_FORBIDDEN:
                if forbidden.lower() in line.lower():
                    failures.append(f"{relative}:{line_number}: forbidden identity")
    if failures:
        print("PUBLICATION_AUDIT_FAIL")
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"PUBLICATION_AUDIT_PASS files={len(files)}")


if __name__ == "__main__":
    main()
