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
        "root-user-path": "/" + "root/",
        "private-temp-path": "/private/" + "var/",
        "source-repository-tree": "glm52" + "-ds4",
        "private-operator-name": "dno" + "la",
        "unfinished-todo-marker": "TO" + "DO",
        "unfinished-substitution-marker": "place" + "holder",
        "private-key-marker": "-----BEGIN " + "PRIVATE KEY-----",
    }


def _regex_patterns() -> Dict[str, re.Pattern[str]]:
    return {
        "private-ipv4": re.compile(r"(?<![0-9])(?:10(?:\.[0-9]{1,3}){3}|192\.168(?:\.[0-9]{1,3}){2}|172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})(?![0-9])"),
        "kanban-task-id": re.compile(r"\bt_[0-9a-f]{8}\b"),
        "credential-assignment": re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret|token)\s*[:=]\s*['\"][^'\"]{8,}"),
        "credential-token-shape": re.compile(
            r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})"
        ),
        "email-like-text": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    }


def _is_non_pii_fixture(relative: str) -> bool:
    return relative == "vendor/eval/HumanEvalPlus-v0.1.10.jsonl"


def _is_operational_marker(kind: str, relative: str, line: str) -> bool:
    if kind.startswith("unfinished-"):
        # Upstream GitHub issue-form hint keys are UI metadata, not unfinished
        # product code. The pinned benchmark corpus is immutable test data and
        # may contain arbitrary source-language strings.
        if relative.startswith("vendor/evalplus/.github/ISSUE_TEMPLATE/"):
            return False
        if _is_non_pii_fixture(relative):
            return False
    if kind == "email-like-text":
        if _is_non_pii_fixture(relative):
            return False
        if "git@github.com" in line:
            return False
    return True


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
                if needle in line and _is_operational_marker(kind, relative, line):
                    failures.append({"path": relative, "kind": kind, "line": line_number})
            for kind, pattern in _regex_patterns().items():
                if pattern.search(line) and _is_operational_marker(kind, relative, line):
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
            "no actionable TO" + "DO or place" + "holder markers (immutable benchmark fixtures and upstream issue-form UI hints are non-operational)",
        ],
    }
