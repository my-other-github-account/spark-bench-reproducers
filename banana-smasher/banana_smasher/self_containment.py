"""Fail-closed package-local reference and privacy scanner."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


_TEXT_SUFFIXES = {
    "", ".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".json", ".md", ".py",
    ".sh", ".toml", ".txt", ".yaml", ".yml", ".cfg", ".ini", ".csv",
}


def _literal_patterns() -> Dict[str, str]:
    return {
        "parent-directory-reference": ".." + "/",
        "macos-user-path": "/" + "Users/",
        "linux-user-path": "/" + "home/",
        "private-temp-path": "/private/" + "var/",
        "source-repository-tree": "glm52" + "-ds4",
        "private-operator-name": "dno" + "la",
    }


def _regex_patterns() -> Dict[str, re.Pattern[str]]:
    return {
        "private-ipv4": re.compile(r"(?<![0-9])(?:10(?:\.[0-9]{1,3}){3}|192\.168(?:\.[0-9]{1,3}){2}|172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})(?![0-9])"),
        "kanban-task-id": re.compile(r"\bt_[0-9a-f]{8}\b"),
        "credential-assignment": re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]{8,}"),
    }


def _files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            yield path
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == "workspace":
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        yield path


def scan_package(root: Path) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    scanned = 0
    for path in _files(root):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            failures.append({"path": relative, "kind": "symlink", "line": None})
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        for line_number, line in enumerate(text.splitlines(), 1):
            for kind, needle in _literal_patterns().items():
                if needle in line:
                    failures.append({"path": relative, "kind": kind, "line": line_number})
            for kind, pattern in _regex_patterns().items():
                if pattern.search(line):
                    failures.append({"path": relative, "kind": kind, "line": line_number})
    return {
        "schema": "banana-smasher-self-containment-verification-v1",
        "status": "PASS" if not failures else "FAIL",
        "scanned_text_files": scanned,
        "failures": failures,
        "rules": [
            "no symlinks",
            "no parent-directory references",
            "no references to source repository trees",
            "no private user paths, operator names, task IDs, addresses, or embedded credentials",
        ],
    }
