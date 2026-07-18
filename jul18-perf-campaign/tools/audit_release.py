#!/usr/bin/env python3
"""Fail-closed public release audit for the July 18 package."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".sh", ".cu", ".cpp", ".h", ".hpp", ".toml", ".yaml", ".yml"}
FORBIDDEN = {
    "absolute_home": re.compile(r"/home/|/Users/"),
    "internal_task": re.compile(r"t_[0-9a-f]{8}"),
    "private_ipv4": re.compile(r"(?<![0-9.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![0-9.])"),
    "private_worker": re.compile(r"\bspark-(?:work|[0-9]+)\b", re.I),
    "private_username": re.compile(r"\bdnola\b|banana_baeee|\bmacmini(?:\.local)?\b", re.I),
    "personal_name": re.compile(r"\b(?:David|Nola)\b", re.I),
    "consumer_email": re.compile(r"\b[A-Z0-9._%+-]+@(?:gmail|icloud)\.com\b", re.I),
    "tailnet_hostname": re.compile(r"\b[A-Z0-9-]+\.ts\.net\b", re.I),
    "phone_shaped": re.compile(r"(?:\+\d{10,15}\b|\b\d{3}[-. ]\d{3}[-. ]\d{4}\b|\(\d{3}\)\s*\d{3}[-. ]\d{4}\b)"),
    "kasa_device_id": re.compile(r"(?i)\bkasa\b.{0,48}\b\d{4,}\b"),
    "credential_assignment": re.compile(r"(?i)(api[_-]?key|password|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_manifest(base: Path) -> list[str]:
    errors = []
    manifest = base / "MANIFEST.json"
    if not manifest.is_file():
        return [f"missing {manifest.relative_to(ROOT)}"]
    rows = json.loads(manifest.read_text())
    for row in rows:
        path = base / row["path"]
        if not path.is_file():
            errors.append(f"manifest missing file: {path.relative_to(ROOT)}")
            continue
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            errors.append(f"manifest mismatch: {path.relative_to(ROOT)}")
    return errors


def audit_links(path: Path) -> list[str]:
    errors = []
    text = path.read_text()
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#")):
            continue
        clean = target.split("#", 1)[0]
        if clean and not (path.parent / clean).resolve().exists():
            errors.append(f"broken link {path.relative_to(ROOT)} -> {target}")
    return errors


def audit_git_tracking() -> list[str]:
    """Ensure an ignored local receipt cannot satisfy a manifest by accident."""
    try:
        repo = Path(subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip())
        tracked = set(subprocess.check_output(
            ["git", "-C", str(repo), "ls-files", "-z"],
        ).decode().split("\0"))
    except (OSError, subprocess.CalledProcessError):
        return ["cannot verify git tracking state"]

    errors = []
    for path in ROOT.rglob("*"):
        if path.is_file() and str(path.relative_to(repo)) not in tracked:
            errors.append(f"untracked package file: {path.relative_to(ROOT)}")
    return errors


def main() -> None:
    errors = []
    required = ["DAY.md", "README.md"]
    for name in required:
        if not (ROOT / name).is_file():
            errors.append(f"missing {name}")
    errors.extend(audit_git_tracking())

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix == ".json":
            try:
                json.loads(path.read_text())
            except Exception as exc:
                errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        if path.suffix == ".md":
            errors.extend(audit_links(path))
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            for label, pattern in FORBIDDEN.items():
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    errors.append(f"{label}: {path.relative_to(ROOT)}:{line}: {match.group(0)!r}")

    for topic in [p for p in ROOT.iterdir() if p.is_dir() and p.name != "tools"]:
        errors.extend(audit_manifest(topic))
    errors.extend(audit_manifest(ROOT))

    if errors:
        print("RELEASE AUDIT FAIL", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print("RELEASE AUDIT PASS")


if __name__ == "__main__":
    main()
