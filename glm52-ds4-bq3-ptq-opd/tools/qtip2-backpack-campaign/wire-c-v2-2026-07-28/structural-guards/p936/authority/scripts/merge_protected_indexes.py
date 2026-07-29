#!/usr/bin/env python3
"""Merge host protected-SHA reverse indexes into one compact fleet index."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_compact(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":")).iterencode(document):
                handle.write(chunk.encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    merged = {}
    sources = []
    sealed_documents = 0
    for raw_path in arguments.input:
        path = Path(raw_path).expanduser().resolve()
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != "p936-protected-sha-index-v1" or document.get("status") != "PASS":
            raise RuntimeError("input protected index schema/status drift: %s" % path)
        sources.append({
            "path": str(path), "sha256": sha256(path),
            "protected_sha_count": document.get("protected_sha_count"),
        })
        sealed_documents += int(document.get("sealed_documents", 0))
        for digest, references in document.get("entries", {}).items():
            bucket = merged.setdefault(digest, [])
            for reference in references:
                if reference not in bucket:
                    bucket.append(reference)
    entries = {
        digest: sorted(references, key=lambda row: (
            row.get("manifest_path", ""), row.get("json_pointer", ""), row.get("manifest_sha256", "")))
        for digest, references in sorted(merged.items())
    }
    output = {
        "schema": "p936-protected-sha-index-v1",
        "status": "PASS",
        "scope": "fleet-union",
        "sources": sources,
        "sealed_documents": sealed_documents,
        "protected_sha_count": len(entries),
        "entries": entries,
        "generated_unix": time.time(),
    }
    output_path = Path(arguments.output).expanduser().resolve()
    atomic_compact(output_path, output)
    print(json.dumps({
        "status": "PASS", "protected_sha_count": len(entries),
        "bytes": output_path.stat().st_size, "sha256": sha256(output_path),
        "output": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FAIL_CLOSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
