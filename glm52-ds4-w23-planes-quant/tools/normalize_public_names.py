#!/usr/bin/env python3
"""Normalize copied campaign prose/data to public role aliases and placeholders."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".toml", ".yaml", ".yml"}
SHORT_HOST = re.compile(r"(?<![A-Za-z0-9_-])s([1-9])(?![A-Za-z0-9_-])")
WORK_HOST = re.compile(r"\b(?:spark" + "-work|s" + r"work)\b")
PRIVATE_MISSIONS = "/home/" + "banana_bae/missions"
PRIVATE_HOME = "/home/" + "banana_bae"


def normalize(text: str) -> str:
    text = text.replace(PRIVATE_MISSIONS, "$MISSION_ROOT")
    text = text.replace(PRIVATE_HOME, "$HOME")
    text = WORK_HOST.sub("spark-5", text)
    return SHORT_HOST.sub(lambda m: f"spark-{m.group(1)}", text)


def main() -> None:
    changed = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in DOC_SUFFIXES:
            continue
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        before = path.read_text(encoding="utf-8")
        after = normalize(before)
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed += 1
            print(path.relative_to(ROOT))
    print(f"NORMALIZE_COMPLETE changed_files={changed}")


if __name__ == "__main__":
    main()
