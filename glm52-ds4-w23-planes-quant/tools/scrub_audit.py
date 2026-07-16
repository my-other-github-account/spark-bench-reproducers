#!/usr/bin/env python3
"""Fail closed when campaign publication contains private infrastructure or secrets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".sh", ".toml", ".yaml", ".yml"}
SKIP_NAMES = {"scrub_audit.py"}
PATTERNS = {
    "private IPv4 address": re.compile(
        r"(?<![0-9])(?:10(?:\.[0-9]{1,3}){3}|192\.168(?:\.[0-9]{1,3}){2}|"
        r"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2}|"
        r"100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])(?:\.[0-9]{1,3}){2})(?![0-9])"
    ),
    "private home path": re.compile(r"/home/" + "banana_bae(?:/|\b)"),
    "noncanonical host alias": re.compile(r"\b(?:spark-work|swork|[A-Za-z0-9-]+\.local)\b"),
    "GitHub token": re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "cloud access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
DOC_HOST_ALIAS = re.compile(r"(?<![A-Za-z0-9_-])s([1-9])(?![A-Za-z0-9_-])")


def iter_text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def audit(root: Path) -> list[str]:
    failures: list[str] = []
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(root)
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{rel}:{line}: {label}: {match.group(0)!r}")
        if path.suffix.lower() in {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml"}:
            for match in DOC_HOST_ALIAS.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{rel}:{line}: noncanonical host alias: {match.group(0)!r}; use spark-{match.group(1)}"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    failures = audit(root)
    if failures:
        print("SCRUB_AUDIT_FAIL")
        print("\n".join(failures))
        return 1
    print(f"SCRUB_AUDIT_PASS files={sum(1 for _ in iter_text_files(root))} root={root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
