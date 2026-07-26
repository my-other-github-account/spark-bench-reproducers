#!/usr/bin/env python3
"""Fail-closed privacy, provenance, and payload audit for this reproducer package."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
TEXT_SUFFIXES = {
    "",
    ".csv",
    ".json",
    ".md",
    ".patch",
    ".py",
    ".sha256",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_PUBLIC_PAYLOADS = {
    "docker/artifacts/tokenizer.json",
    "docker/provenance/pip-freeze.sanitized.txt",
}

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
    "task identifier": re.compile(r"t_[0-9a-f]{8}", re.IGNORECASE),
    "private IPv4 address": re.compile(
        r"(?<![0-9])(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}|"
        r"192\.168\.(?:[0-9]{1,3}\.)[0-9]{1,3}|"
        r"172\.(?:1[6-9]|2[0-9]|3[01])\.(?:[0-9]{1,3}\.)[0-9]{1,3}|"
        r"100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.(?:[0-9]{1,3}\.)[0-9]{1,3})(?![0-9])"
    ),
    "private mission path": re.compile(r"(?:^|[/\\])missions(?:[/\\]|$)", re.IGNORECASE),
    "BQ3 misnamed as IQ3": re.compile(r"IQ3_BIN|repaired-IQ3|IQ3 artifact", re.IGNORECASE),
}

TOOLS = ROOT / "tools/qtip2-backpack-campaign"
MANIFEST_PATH = TOOLS / "TOOLS_MANIFEST.json"
MANIFEST_MD_PATH = TOOLS / "TOOLS_MANIFEST.md"
NON_PAYLOAD_FILES = {"TOOLS_MANIFEST.json", "TOOLS_MANIFEST.md"}
FORBIDDEN_DATA_SUFFIXES = {
    ".bin",
    ".gguf",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
}
REQUIRED_PREFIXES = {
    "solver/p629/",
    "solver/p637/",
    "builders/wire/",
    "builders/qtip-rep16/",
    "builders/packers/",
    "rail/p671/",
    "dose/p600/",
    "dose/p613-acceleration/",
    "dose/p662-candidate/",
    "dose/p672-package/",
    "misc/kld/",
    "misc/qsfp/",
    "misc/teacher-sharding/",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_manifest() -> list[str]:
    failures: list[str] = []
    if not MANIFEST_PATH.is_file() or not MANIFEST_MD_PATH.is_file():
        return ["missing TOOLS_MANIFEST.json or TOOLS_MANIFEST.md"]
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"invalid TOOLS_MANIFEST.json: {error}"]

    rows = manifest.get("files")
    if not isinstance(rows, list):
        return ["TOOLS_MANIFEST.json files must be an array"]
    by_path: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"manifest row {index} is not an object")
            continue
        path_text = row.get("path")
        if not isinstance(path_text, str) or not path_text:
            failures.append(f"manifest row {index} has invalid path")
            continue
        if path_text in by_path:
            failures.append(f"duplicate manifest path: {path_text}")
            continue
        by_path[path_text] = row
        path = TOOLS / path_text
        if not path.is_file():
            failures.append(f"manifest path missing: {path_text}")
            continue
        actual = sha256(path)
        if row.get("shipped_sha256") != actual:
            failures.append(f"shipped SHA mismatch: {path_text}")
        for required in ("repro_stage", "source_sha256", "origin", "source_verification"):
            if required not in row:
                failures.append(f"manifest field missing for {path_text}: {required}")
        source_hash = row.get("source_sha256")
        if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
            failures.append(f"invalid source SHA-256: {path_text}")

    payload_paths = {
        path.relative_to(TOOLS).as_posix()
        for path in TOOLS.rglob("*")
        if path.is_file()
        and path.relative_to(TOOLS).as_posix() not in NON_PAYLOAD_FILES
        and "__pycache__" not in path.parts
    }
    listed_paths = set(by_path)
    for path_text in sorted(payload_paths - listed_paths):
        failures.append(f"unlisted tools payload: {path_text}")
    for path_text in sorted(listed_paths - payload_paths):
        failures.append(f"manifest lists non-payload path: {path_text}")

    for prefix in sorted(REQUIRED_PREFIXES):
        if not any(path.startswith(prefix) for path in listed_paths):
            failures.append(f"required reproduction family absent: {prefix}")
    if "solver/INPUT_MANIFEST_SCHEMA.json" not in listed_paths:
        failures.append("solver input manifest schema absent")
    builder = by_path.get("builders/wire/canonical_shared_builder.py", {})
    if builder.get("source_sha256") != "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e":
        failures.append("canonical_shared_builder.py source pin is not 60b594ac…")

    manifest_md = MANIFEST_MD_PATH.read_text(encoding="utf-8")
    for path_text in sorted(listed_paths):
        if f"`{path_text}`" not in manifest_md:
            failures.append(f"TOOLS_MANIFEST.md missing path: {path_text}")
    for path in TOOLS.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_DATA_SUFFIXES:
            failures.append(f"forbidden data payload bundled: {path.relative_to(TOOLS)}")
    return failures


def text_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == SELF:
            continue
        if any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED_PUBLIC_PAYLOADS:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return sorted(files)


def main() -> None:
    failures = audit_manifest()
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
    print(f"PUBLICATION_AUDIT_PASS files={len(files)} manifest_rows={len(json.loads(MANIFEST_PATH.read_text())['files'])}")


if __name__ == "__main__":
    main()
