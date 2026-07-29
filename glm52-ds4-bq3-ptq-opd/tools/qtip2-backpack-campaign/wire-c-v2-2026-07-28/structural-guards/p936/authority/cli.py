#!/usr/bin/env python3
"""CLI for the structural authority, reclaim, and seal guards."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .authority_guard import (
    AuthorityStore,
    GuardViolation,
    _atomic_json,
    assert_reclaim_allowed,
    assert_seal_dependencies,
    build_protected_index,
    resolve_plan_codebook,
    sha256_file,
)


def _load_json(path: str) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _emit(document: Mapping[str, Any], output: str | None) -> None:
    if output:
        _atomic_json(Path(output).expanduser().resolve(), document)
    print(json.dumps(document, sort_keys=True))


def _dependency_rows(value: Any, pointer: str = ""):
    if isinstance(value, dict):
        if isinstance(value.get("sha256"), str) and isinstance(value.get("bytes"), int):
            yield {
                "sha256": value["sha256"],
                "bytes": value["bytes"],
                "role": pointer or "dependency",
            }
        for key, child in value.items():
            if isinstance(child, str) and key.endswith("_sha256"):
                base = key[: -len("_sha256")]
                byte_key = base + "_bytes"
                if isinstance(value.get(byte_key), int):
                    yield {
                        "sha256": child,
                        "bytes": value[byte_key],
                        "role": "%s/%s" % (pointer, base),
                    }
            yield from _dependency_rows(child, "%s/%s" % (pointer, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _dependency_rows(child, "%s/%d" % (pointer, index))


def _manifest_codebooks(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "codebook" and isinstance(child, dict):
                yield child
            yield from _manifest_codebooks(child)
    elif isinstance(value, list):
        for child in value:
            yield from _manifest_codebooks(child)


def _remote_probe_factory(location_document: Mapping[str, Any]):
    code = (
        "import hashlib,json,pathlib,sys;"
        "p=pathlib.Path(sys.argv[1])/'store'/(sys.argv[2]+'.bin');"
        "n=int(sys.argv[3]);"
        "ok=p.is_file() and not p.is_symlink() and p.stat().st_size==n;"
        "h=hashlib.sha256(p.read_bytes()).hexdigest() if ok else None;"
        "print(json.dumps({'ok':bool(ok and h==sys.argv[2]),'path':str(p),'bytes':n,'sha256':h}))"
    )

    def probe(host: str, root: Path, digest: str, expected_bytes: int):
        specification = location_document[host]
        mode = specification.get("mode", "local")
        if mode == "local":
            path = root / "store" / (digest + ".bin")
            if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
                return None
            if sha256_file(path) != digest:
                return None
            return {"host": host, "path": str(path), "bytes": expected_bytes, "sha256": digest}
        if mode != "ssh" or not specification.get("target"):
            raise GuardViolation("invalid seal location mode for %s" % host)
        command = "python3 -c %s %s %s %s" % (
            shlex.quote(code),
            shlex.quote(str(root)),
            shlex.quote(digest),
            shlex.quote(str(expected_bytes)),
        )
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", specification["target"], command],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            return None
        try:
            row = json.loads(result.stdout)
        except ValueError:
            return None
        if not row.get("ok"):
            return None
        return {"host": host, "path": row["path"], "bytes": expected_bytes, "sha256": digest}

    return probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="ingest files into an append-only store")
    ingest.add_argument("--root", required=True)
    ingest.add_argument("--role", default="codebook")
    ingest.add_argument("paths", nargs="+")

    ingest_manifest = subparsers.add_parser("ingest-manifests", help="ingest codebook specs from manifests")
    ingest_manifest.add_argument("--root", required=True)
    ingest_manifest.add_argument("manifests", nargs="+")

    resolve = subparsers.add_parser("resolve", help="resolve and full-read a SHA")
    resolve.add_argument("--root", required=True)
    resolve.add_argument("sha256")

    plan = subparsers.add_parser("plan-resolve", help="resolve a plan row by SHA")
    plan.add_argument("--root", required=True)
    plan.add_argument("--plan", required=True)
    plan.add_argument("--row", required=True)
    plan.add_argument("--requested-sha256")

    index = subparsers.add_parser("build-index", help="build protected reverse-reference index")
    index.add_argument("--output", required=True)
    index.add_argument("manifests", nargs="+")

    reclaim = subparsers.add_parser("reclaim-check", help="fail closed before reclaim")
    reclaim.add_argument("--index", required=True)
    reclaim.add_argument("--archive-first")
    reclaim.add_argument("paths", nargs="+")

    deps = subparsers.add_parser("manifest-dependencies", help="extract seal dependencies")
    deps.add_argument("--output", required=True)
    deps.add_argument("manifests", nargs="+")

    seal = subparsers.add_parser("seal-check", help="require byte-exact copies on >=2 hosts")
    seal.add_argument("--dependencies", required=True)
    seal.add_argument("--locations", required=True)
    seal.add_argument("--min-copies", type=int, default=2)
    seal.add_argument("--output")

    arguments = parser.parse_args(argv)
    if arguments.command == "ingest":
        store = AuthorityStore(arguments.root)
        rows = []
        for raw_path in arguments.paths:
            path = Path(raw_path).expanduser().resolve()
            digest = store.ingest(path, metadata={"role": arguments.role})
            rows.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
        _emit({"schema": "p936-authority-ingest-receipt-v1", "status": "PASS", "rows": rows}, None)
        return 0
    if arguments.command == "ingest-manifests":
        store = AuthorityStore(arguments.root)
        rows = []
        seen = set()
        for manifest in arguments.manifests:
            document = _load_json(manifest)
            for specification in _manifest_codebooks(document):
                raw_path = specification.get("path") or specification.get("local_path")
                wanted = specification.get("sha256")
                expected_bytes = specification.get("bytes")
                if not raw_path or not wanted:
                    raise GuardViolation("manifest codebook lacks path/SHA: %s" % manifest)
                key = (str(raw_path), str(wanted))
                if key in seen:
                    continue
                seen.add(key)
                path = Path(str(raw_path)).expanduser().resolve()
                actual = store.ingest(path, metadata={"role": "codebook", "manifest": str(Path(manifest).resolve())})
                if actual != wanted or (isinstance(expected_bytes, int) and path.stat().st_size != expected_bytes):
                    raise GuardViolation("manifest codebook identity drift: %s" % path)
                rows.append({"path": str(path), "sha256": actual, "bytes": path.stat().st_size})
        _emit({"schema": "p936-authority-manifest-ingest-v1", "status": "PASS", "rows": rows}, None)
        return 0
    if arguments.command == "resolve":
        path = AuthorityStore(arguments.root).resolve(arguments.sha256)
        _emit({"status": "PASS", "path": str(path), "sha256": arguments.sha256, "bytes": path.stat().st_size}, None)
        return 0
    if arguments.command == "plan-resolve":
        path = resolve_plan_codebook(
            AuthorityStore(arguments.root), arguments.plan, _load_json(arguments.row), arguments.requested_sha256
        )
        _emit({"status": "PASS", "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}, None)
        return 0
    if arguments.command == "build-index":
        _emit(build_protected_index(arguments.manifests, arguments.output), None)
        return 0
    if arguments.command == "reclaim-check":
        assert_reclaim_allowed(arguments.paths, arguments.index, arguments.archive_first)
        _emit({"schema": "p936-reclaim-check-v1", "status": "PASS", "paths": arguments.paths}, None)
        return 0
    if arguments.command == "manifest-dependencies":
        by_sha = {}
        for manifest in arguments.manifests:
            for row in _dependency_rows(_load_json(manifest)):
                previous = by_sha.setdefault(row["sha256"], row)
                if previous["bytes"] != row["bytes"]:
                    raise GuardViolation("dependency SHA has conflicting byte counts")
        document = {
            "schema": "p936-seal-dependencies-v1",
            "status": "PASS",
            "dependencies": [by_sha[digest] for digest in sorted(by_sha)],
        }
        _emit(document, arguments.output)
        return 0
    if arguments.command == "seal-check":
        dependency_document = _load_json(arguments.dependencies)
        if isinstance(dependency_document, list):
            dependencies = dependency_document
        elif isinstance(dependency_document, dict):
            dependencies = dependency_document.get("dependencies", dependency_document)
        else:
            raise GuardViolation("dependency document must be a list or object")
        location_document = _load_json(arguments.locations)
        if not isinstance(location_document, dict):
            raise GuardViolation("location document must be an object")
        roots = {
            host: specification["root"]
            for host, specification in location_document.items()
        }
        census = assert_seal_dependencies(
            dependencies,
            roots,
            min_copies=arguments.min_copies,
            probe=_remote_probe_factory(location_document),
        )
        _emit(census, arguments.output)
        return 0
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GuardViolation as exc:
        print("FAIL_CLOSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
