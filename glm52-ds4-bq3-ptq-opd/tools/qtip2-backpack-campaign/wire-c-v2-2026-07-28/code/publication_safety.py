#!/usr/bin/env python3
"""Fail-closed publication-safety scan for the public Wire-C package.

The scanner is standard-library only. It rejects symlinks, hidden/cache entries,
non-regular files, non-UTF-8 bytes, non-text suffixes, manifest drift, private
infrastructure identifiers, and common credential families.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".md", ".py", ".sh"})
MANIFEST_ROW_KEYS = frozenset({
    "path", "privacy_substitution_applied", "provenance_type",
    "public_copy_bytes", "public_copy_sha256", "role", "source_sha256",
    "source_verification",
})
HEX64 = re.compile(r"[0-9a-f]{64}")
TASK_ID = re.compile(r"(?<![A-Za-z0-9])t_[0-9a-f]{8}(?![A-Za-z0-9])", re.I)
UNIX_HOME = re.compile(r"/(?:Users|home)/[^/\s\"'`]+(?:/[^\s\"'`]+)*")
WINDOWS_HOME = re.compile(r"(?i)\b[A-Z]:\\Users\\[^\s\"'`]+(?:\\[^\s\"'`]+)*")
PRIVATE_HOST = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])spark[-_]?\d+(?![A-Za-z0-9_-])|"
    r"\b(?:[A-Za-z0-9-]+\.)+(?:local|lan|internal)\b"
)
WORD = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_-]{2,31}(?![A-Za-z0-9])")
KNOWN_IDENTITY_HASHES = frozenset({
    "f37a49a9cd088403783cc38d37b72fb0abaaf4ca04bc4cff75fb3e5f46de24c6",
    "9b26c65fe33bf161eea7edd9f1e361783a2ee985b8d5036ddc5b6029f957cd0d",
    "07d046d5fac12b3f82daf5035b9aae86db5adc8275ebfbf05ec83005a4a8ba3e",
})
URL_CREDENTIALS = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@")
PEM_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")
IPV4_CANDIDATE = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
SECRET_ASSIGNMENT = re.compile(
    r"(?im)(?P<name>(?:[A-Za-z0-9_-]+[_-])?"
    r"(?:secret|password|token|api[_-]?key|access[_-]?key|private[_-]?key))"
    r"\s*(?:=|:)\s*(?P<value>[^,\n#]+)"
)
TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("openai_or_anthropic_token", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
)
HIDDEN_OR_CACHE = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"})
EXPLICIT_BINARY_SUFFIXES = frozenset({
    ".bin", ".safetensors", ".pt", ".pth", ".onnx", ".gguf", ".npz", ".npy",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".png", ".jpg", ".jpeg",
    ".gif", ".webp", ".pdf", ".dylib", ".so", ".dll", ".exe", ".pyc",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise RuntimeError({"invalid_manifest_path": value})
    pure = PurePosixPath(value)
    if pure.is_absolute() or value != pure.as_posix():
        raise RuntimeError({"noncanonical_manifest_path": value})
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise RuntimeError({"unsafe_manifest_path": value})
    if any(part.startswith(".") or part in HIDDEN_OR_CACHE for part in pure.parts):
        raise RuntimeError({"hidden_or_cache_path": value})
    suffix = pure.suffix.lower()
    if suffix in EXPLICIT_BINARY_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
        raise RuntimeError({"non_text_extension": value})
    return value


def _is_private_ipv4(token: str) -> bool:
    try:
        address = ipaddress.ip_address(token)
    except ValueError:
        return False
    cgnat = ipaddress.ip_network("100." + "64.0.0/10")
    return address.version == 4 and (address.is_private or address in cgnat)


def _contains_known_identity(text: str) -> bool:
    for match in WORD.finditer(text):
        digest = hashlib.sha256(match.group(0).lower().encode()).hexdigest()
        if digest in KNOWN_IDENTITY_HASHES:
            return True
    return False


def _placeholder_secret(value: str) -> bool:
    value = value.strip().rstrip(",").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    upper = value.upper()
    if not value or value in {"None", "null", "NULL", "''", '""'}:
        return True
    if HEX64.fullmatch(value.lower()):
        return True
    if any(marker in upper for marker in (
        "REDACTED", "PLACEHOLDER", "PUBLIC_", "EXAMPLE", "DUMMY", "CHANGEME",
        "OS.ENVIRON", "GETENV", "ENV[", "${", "$", "<", "REQUIRED",
    )):
        return True
    if value[0] in "([{r" or value.startswith(("re.compile", "frozenset", "set", "tuple")):
        return True
    if any(marker in value for marker in ("\\b", "(?", "[A-", "[0-", "Pattern", "str |", "None |", "->")):
        return True
    return False


def privacy_findings(text: str) -> list[str]:
    findings: list[str] = []
    for label, pattern in (
        ("private_unix_home", UNIX_HOME),
        ("private_windows_home", WINDOWS_HOME),
        ("private_hostname", PRIVATE_HOST),
        ("task_id", TASK_ID),
        ("url_credentials", URL_CREDENTIALS),
        ("pem_private_key", PEM_PRIVATE_KEY),
    ):
        if pattern.search(text):
            findings.append(label)
    if _contains_known_identity(text):
        findings.append("private_identity")
    if any(_is_private_ipv4(match.group(0)) for match in IPV4_CANDIDATE.finditer(text)):
        findings.append("private_or_cgnat_ipv4")
    for label, pattern in TOKEN_PATTERNS:
        if pattern.search(text):
            findings.append(label)
    for match in SECRET_ASSIGNMENT.finditer(text):
        if not _placeholder_secret(match.group("value")):
            findings.append("nonplaceholder_secret_assignment")
            break
    return sorted(set(findings))


def _walk_regular_text(root: Path) -> list[Path]:
    root_mode = root.lstat().st_mode
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise RuntimeError("package root must be a real directory, not a symlink")
    files: list[Path] = []
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept: list[str] = []
        for name in sorted(directories):
            path = current_path / name
            mode = path.lstat().st_mode
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(mode):
                raise RuntimeError({"symlink_directory": relative})
            if not stat.S_ISDIR(mode):
                raise RuntimeError({"nonregular_directory_entry": relative})
            if name.startswith(".") or name in HIDDEN_OR_CACHE:
                raise RuntimeError({"hidden_or_cache_directory": relative})
            kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError({"symlink_file": relative})
            if not stat.S_ISREG(mode):
                raise RuntimeError({"nonregular_file": relative})
            if name.startswith("."):
                raise RuntimeError({"hidden_file": relative})
            files.append(path)
    return files


def scan_package(
    root: Path,
    manifest_path: Path | None = None,
    *,
    self_files: Iterable[str] = ("PACKAGE_MANIFEST.json",),
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest_path = (manifest_path or root / "PACKAGE_MANIFEST.json").resolve(strict=True)
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("manifest resolves outside package root") from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise RuntimeError("manifest files must be a list")

    manifest_by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != MANIFEST_ROW_KEYS:
            raise RuntimeError({"manifest_row_schema": sorted(row) if isinstance(row, dict) else type(row).__name__})
        relative = validate_relative_path(row.get("path"))
        if relative in manifest_by_path:
            raise RuntimeError({"duplicate_manifest_path": relative})
        public_sha = str(row.get("public_copy_sha256", ""))
        source_sha = str(row.get("source_sha256", ""))
        if not HEX64.fullmatch(public_sha) or not HEX64.fullmatch(source_sha):
            raise RuntimeError({"invalid_manifest_sha": relative})
        transformed = row.get("privacy_substitution_applied")
        if transformed is not (source_sha != public_sha):
            raise RuntimeError({"privacy_substitution_semantics": relative})
        if not isinstance(row.get("public_copy_bytes"), int) or row["public_copy_bytes"] < 0:
            raise RuntimeError({"invalid_manifest_bytes": relative})
        manifest_by_path[relative] = row

    self_set = set(self_files)
    actual: dict[str, Path] = {}
    scanned_bytes = 0
    for path in _walk_regular_text(root):
        relative = path.relative_to(root).as_posix()
        if relative in self_set:
            continue
        validate_relative_path(relative)
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError({"resolved_path_escape": relative}) from exc
        data = path.read_bytes()
        if b"\x00" in data:
            raise RuntimeError({"nul_byte": relative})
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError({"non_utf8": relative, "offset": exc.start}) from exc
        findings = privacy_findings(text)
        if findings:
            raise RuntimeError({"privacy_findings": relative, "kinds": findings})
        actual[relative] = path
        scanned_bytes += len(data)

    if set(actual) != set(manifest_by_path):
        raise RuntimeError({
            "manifest_tree_closure": {
                "unmanifested": sorted(set(actual) - set(manifest_by_path)),
                "missing": sorted(set(manifest_by_path) - set(actual)),
            }
        })
    for relative, row in manifest_by_path.items():
        path = actual[relative]
        if path.stat().st_size != row["public_copy_bytes"]:
            raise RuntimeError({"manifest_byte_mismatch": relative})
        if sha256(path) != row["public_copy_sha256"]:
            raise RuntimeError({"manifest_sha_mismatch": relative})

    return {
        "schema": "wire-c-publication-safety-receipt-v1",
        "status": "PASS",
        "manifest_sha256": sha256(manifest_path),
        "scanned_file_count": len(actual),
        "scanned_byte_count": scanned_bytes,
        "tree_safety_status": "PASS_STRICT_TEXT_REGULAR_CONTAINED_MANIFEST_CLOSED",
        "privacy_scan_status": "PASS_NO_PRIVATE_OR_SECRET_MATERIAL",
        "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(scan_package(args.root, args.manifest), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
