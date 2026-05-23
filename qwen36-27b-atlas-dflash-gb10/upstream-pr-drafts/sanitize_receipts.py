#!/usr/bin/env python3
"""Fail if receipt/draft files contain credential-looking material."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PATTERNS = {
    "api_key_cli": re.compile(r"(?i)(--api-key\s+)(?!\[REDACTED\])([^\s\"']+)"),
    "api_key_json": re.compile(
        r"(?i)([\"']api[_-]?key[\"']\s*:\s*[\"'])(?!\[REDACTED\])([^\"']+)"
    ),
    "api_key_assignment": re.compile(
        r"(?i)(api[_-]?key\s*=\s*[\"'])(?!\[REDACTED\])([^\"']+)"
    ),
    "bearer_token": re.compile(r"(?i)bearer\s+(?!\[REDACTED\])([A-Za-z0-9._=-]{16,})"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "password_value": re.compile(
        r"(?i)(password|passwd|secret)[\"']?\s*[:=]\s*[\"']?(?!\[REDACTED\])([^\s\"']{8,})"
    ),
}

SKIP_PARTS = {".git", "__pycache__"}
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".diff",
    ".patch",
}


def iter_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not (set(child.parts) & SKIP_PARTS):
                    if child.suffix in TEXT_SUFFIXES:
                        files.append(child)
        elif path.is_file():
            files.append(path)
    return sorted(files)


def scan(paths: list[str]) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for path in iter_files(paths):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for name, regex in PATTERNS.items():
                if regex.search(line):
                    findings.append(
                        {
                            "file": str(path),
                            "line": line_no,
                            "pattern": name,
                            "redacted_line": regex.sub(r"\1[REDACTED]", line),
                        }
                    )
    return {"ok": not findings, "files_scanned": len(iter_files(paths)), "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = scan(args.paths)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
