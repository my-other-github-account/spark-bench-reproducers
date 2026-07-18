#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Redact infrastructure identifiers from a server log before publication."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_PRIVATE_IP = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
_HOME_PATH = re.compile(r"/home/[^/\s]+/(?:[^\s'\"]+)")
_TASK_ID = re.compile(r"\bt_[0-9a-f]{8}\b")


def scrub(text: str) -> str:
    text = _PRIVATE_IP.sub("<private-ip>", text)
    text = _HOME_PATH.sub("<private-path>", text)
    return _TASK_ID.sub("<task-id>", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(scrub(args.input.read_text(errors="replace")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
