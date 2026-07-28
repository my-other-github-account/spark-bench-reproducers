#!/usr/bin/env python3
"""Fail-closed privacy/credential/artifact scan for this text-only bundle."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".gguf", ".onnx", ".npy", ".npz",
    ".ckpt", ".engine", ".cubin", ".fatbin", ".so", ".dylib", ".dll",
}
SELF = "tools/privacy_scan.py"

# Construct high-risk literals in pieces so this scanner does not trigger on itself.
RULES = [
    ("aws_access_key", re.compile("AK" + "IA[0-9A-Z]{16}")),
    ("hf_token", re.compile("hf" + "_[A-Za-z0-9]{20,}")),
    ("api_secret_prefix", re.compile("sk" + "-[A-Za-z0-9_-]{20,}")),
    ("private_key_block", re.compile("BEGIN " + "(?:RSA |OPENSSH |EC )?PRIVATE KEY")),
    ("secret_assignment", re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*[\"'][^\"'\n]{12,}[\"']")),
    ("private_spark_home", re.compile("/ho" + "me/" + "dno" + "la")),
    ("private_collection_home", re.compile("/Us" + "ers/" + "mac" + "mini")),
    ("direct_qsfp_address", re.compile(r"192\.168\.20[01]\.\d{1,3}")),
    ("direct_lan_address", re.compile(r"10\.0\.0\.\d{1,3}")),
    ("direct_tailscale_address", re.compile(r"100\.(?:\d{1,3}\.){2}\d{1,3}")),
    ("email_address", re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    ("mac_address", re.compile(r"(?i)\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b")),
    ("operator_username", re.compile("\\b" + "dno" + "la" + "\\b", re.IGNORECASE)),
    ("collection_hostname", re.compile("\\b" + "mac" + "mini" + "\\b", re.IGNORECASE)),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--receipt", help="optional JSON receipt path; write only after a passing scan")
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    findings: list[dict[str, object]] = []
    scanned_files = 0
    scanned_bytes = 0

    if not root.is_dir():
        print(json.dumps({"status": "FAIL", "findings": [{"rule": "ROOT_NOT_DIRECTORY"}]}, sort_keys=True))
        return 1

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append({"rule": "SYMLINK_FORBIDDEN", "path": rel})
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            findings.append({"rule": "NON_REGULAR_ENTRY", "path": rel})
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append({"rule": "FORBIDDEN_ARTIFACT_EXTENSION", "path": rel})
        if rel == SELF:
            continue
        raw = path.read_bytes()
        scanned_files += 1
        scanned_bytes += len(raw)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            findings.append({"rule": "NON_UTF8_FILE", "path": rel})
            continue
        for rule, pattern in RULES:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append({"rule": rule, "path": rel, "line": line})

    result = {
        "schema": "glm52-privacy-scan-v1",
        "status": "PASS" if not findings else "FAIL",
        "scanned_files": scanned_files,
        "scanned_bytes": scanned_bytes,
        "self_source_excluded": SELF,
        "findings": findings,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(output, end="")
    if args.receipt and not findings:
        receipt = pathlib.Path(args.receipt)
        if not receipt.is_absolute():
            receipt = root / receipt
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(output, encoding="utf-8")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
