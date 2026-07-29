#!/usr/bin/env python3
"""Content-addressed authority primitives for the QTIP2 campaign.

The store is append-only at the API boundary: payloads are named by SHA-256,
existing payloads are byte-verified and never overwritten, and index rows are
only appended after the payload has been fsynced.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GuardViolation(RuntimeError):
    """A structural authority invariant failed."""


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: str) -> str:
    digest = str(value).lower()
    if not _SHA256_RE.fullmatch(digest):
        raise GuardViolation("invalid SHA-256 digest: %r" % (value,))
    return digest


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AuthorityStore:
    """Append-only ``store/<sha256>.bin`` authority store."""

    def __init__(self, root: os.PathLike[str] | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.payload_root = self.root / "store"
        self.index_path = self.root / "index.jsonl"
        self.lock_path = self.root / ".index.lock"
        self.payload_root.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)

    def path_for(self, digest: str) -> Path:
        return self.payload_root / ("%s.bin" % _validate_digest(digest))

    def ingest(
        self,
        source: os.PathLike[str] | str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        source_path = Path(source).expanduser().resolve()
        info = source_path.lstat()
        if not stat.S_ISREG(info.st_mode) or source_path.is_symlink():
            raise GuardViolation("authority source must be a regular non-symlink file: %s" % source_path)
        digest = sha256_file(source_path)
        destination = self.path_for(digest)

        with self.lock_path.open("r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise GuardViolation("authority destination is not a regular file: %s" % destination)
                if destination.stat().st_size != info.st_size or sha256_file(destination) != digest:
                    raise GuardViolation("content-address collision or store corruption: %s" % destination)
            else:
                fd, temporary_name = tempfile.mkstemp(prefix=".%s." % digest, dir=str(self.payload_root))
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(fd, "wb") as target, source_path.open("rb") as origin:
                        shutil.copyfileobj(origin, target, length=8 << 20)
                        target.flush()
                        os.fsync(target.fileno())
                    if temporary.stat().st_size != info.st_size or sha256_file(temporary) != digest:
                        raise GuardViolation("ingest copy changed bytes for %s" % source_path)
                    os.chmod(temporary, 0o444)
                    os.replace(temporary, destination)
                    _fsync_dir(self.payload_root)
                finally:
                    temporary.unlink(missing_ok=True)

            known = False
            if self.index_path.exists():
                with self.index_path.open("r", encoding="utf-8") as existing:
                    for line in existing:
                        if not line.strip():
                            continue
                        if json.loads(line).get("sha256") == digest:
                            known = True
                            break
            if not known:
                row = {
                    "schema": "p936-authority-store-index-v1",
                    "sha256": digest,
                    "bytes": int(info.st_size),
                    "source_path": str(source_path),
                    "ingested_unix": time.time(),
                    "metadata": dict(metadata or {}),
                }
                encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                descriptor = os.open(str(self.index_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                try:
                    os.write(descriptor, encoded)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                _fsync_dir(self.root)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return digest

    def resolve(self, digest: str) -> Path:
        wanted = _validate_digest(digest)
        path = self.path_for(wanted)
        if path.is_symlink() or not path.is_file():
            raise GuardViolation("authority SHA is absent: %s" % wanted)
        if sha256_file(path) != wanted:
            raise GuardViolation("authority payload hash drift: %s" % path)
        return path


def resolve_codebook_binding(
    store: AuthorityStore,
    plan_path: os.PathLike[str] | str,
    expected_sha256: str,
    requested_sha256: str,
) -> Path:
    """Resolve a plan binding by SHA; path substitutions are never accepted."""
    expected = _validate_digest(expected_sha256)
    requested = _validate_digest(requested_sha256)
    if requested == expected:
        return store.resolve(expected)

    plan = Path(plan_path).expanduser().resolve()
    waiver_path = plan.parent / "SUBSTITUTION_WAIVER.json"
    if waiver_path.is_symlink() or not waiver_path.is_file():
        raise GuardViolation("codebook SHA mismatch is unwaivable without adjacent SUBSTITUTION_WAIVER.json")
    try:
        waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GuardViolation("invalid substitution waiver JSON: %s" % exc) from exc
    required = {
        "expected_codebook_sha256",
        "substitute_codebook_sha256",
        "measured_delta",
        "ci95",
        "measurement_receipt_sha256",
        "windows",
    }
    if not isinstance(waiver, dict) or set(waiver) != required:
        raise GuardViolation("substitution waiver schema must contain exactly the required measured fields")
    if waiver["expected_codebook_sha256"] != expected or waiver["substitute_codebook_sha256"] != requested:
        raise GuardViolation("substitution waiver does not bind the exact expected/substitute SHA pair")
    if not isinstance(waiver["windows"], int) or isinstance(waiver["windows"], bool) or waiver["windows"] < 64:
        raise GuardViolation("substitution waiver requires windows>=64")
    receipt_sha = _validate_digest(waiver["measurement_receipt_sha256"])
    try:
        receipt_path = store.resolve(receipt_sha)
    except GuardViolation as exc:
        raise GuardViolation("measurement receipt SHA does not exist in the authority store") from exc

    def valid_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    ci95 = waiver["ci95"]
    if not valid_number(waiver["measured_delta"]):
        raise GuardViolation("measured_delta must be a finite numeric measurement, not an estimate label")
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(valid_number(value) for value in ci95)
        or ci95[0] > waiver["measured_delta"]
        or waiver["measured_delta"] > ci95[1]
    ):
        raise GuardViolation("ci95 must be a numeric interval containing measured_delta")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise GuardViolation("measurement receipt is not valid JSON") from exc
    expected_receipt_fields = {
        "schema": "p936-codebook-substitution-measurement-v1",
        "status": "PASS",
        "expected_codebook_sha256": expected,
        "substitute_codebook_sha256": requested,
        "measured_delta": waiver["measured_delta"],
        "ci95": waiver["ci95"],
        "windows": waiver["windows"],
    }
    if not isinstance(receipt, dict) or any(
        receipt.get(key) != value for key, value in expected_receipt_fields.items()
    ):
        raise GuardViolation("measurement receipt does not match the waiver and exact SHA pair")
    return store.resolve(requested)


def _is_sealed_document(document: Any) -> bool:
    if not isinstance(document, dict):
        return False
    status_value = document.get("status")
    status = str(status_value).upper() if status_value is not None else ""
    return (
        status.startswith("PASS")
        or status in {"SEALED", "DONE", "TERMINAL", "COMPLETE"}
        or document.get("sealed") is True
    )


def _sha_references(value: Any, pointer: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            child_pointer = "%s/%s" % (pointer, escaped)
            if (
                (key == "sha256" or str(key).endswith("_sha256"))
                and isinstance(child, str)
                and _SHA256_RE.fullmatch(child.lower())
            ):
                yield child.lower(), child_pointer
            yield from _sha_references(child, child_pointer)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _sha_references(child, "%s/%d" % (pointer, index))


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(
        (json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def build_protected_index(
    manifest_paths: list[os.PathLike[str] | str],
    output_path: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Build a deterministic reverse index from sealed manifest SHA fields."""
    references: dict[str, list[dict[str, Any]]] = {}
    sealed_count = 0
    input_count = 0
    for raw_path in manifest_paths:
        path = Path(raw_path).expanduser().resolve()
        input_count += 1
        if path.is_symlink() or not path.is_file():
            raise GuardViolation("manifest input must be a regular file: %s" % path)
        try:
            if path.suffix == ".jsonl":
                documents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                documents = [json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, UnicodeError, ValueError) as exc:
            raise GuardViolation("cannot parse manifest %s" % path) from exc
        manifest_sha = sha256_file(path)
        for document_index, document in enumerate(documents):
            if not _is_sealed_document(document):
                continue
            sealed_count += 1
            prefix = "/documents/%d" % document_index if path.suffix == ".jsonl" else ""
            for digest, pointer in _sha_references(document, prefix):
                row = {
                    "manifest_path": str(path),
                    "manifest_sha256": manifest_sha,
                    "json_pointer": pointer,
                }
                bucket = references.setdefault(digest, [])
                if row not in bucket:
                    bucket.append(row)
    entries = {digest: sorted(rows, key=lambda row: (row["manifest_path"], row["json_pointer"])) for digest, rows in sorted(references.items())}
    document = {
        "schema": "p936-protected-sha-index-v1",
        "status": "PASS",
        "input_files": input_count,
        "sealed_documents": sealed_count,
        "protected_sha_count": len(entries),
        "entries": entries,
        "generated_unix": time.time(),
    }
    resolved_output = Path(output_path).expanduser().resolve()
    if resolved_output.suffix == ".jsonl":
        _atomic_jsonl(resolved_output, [
            {
                "schema": "p936-protected-sha-index-v1",
                "status": "PASS",
                "sha256": digest,
                "reference_count": len(rows),
                "references": rows,
            }
            for digest, rows in entries.items()
        ])
    else:
        _atomic_json(resolved_output, document)
    return document


def _load_protected_entries(index_path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        if index_path.suffix == ".jsonl":
            rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                raise GuardViolation("protected SHA index is empty")
            entries = {}
            for row in rows:
                if (
                    not isinstance(row, dict)
                    or row.get("schema") != "p936-protected-sha-index-v1"
                    or row.get("status") != "PASS"
                ):
                    raise GuardViolation("protected SHA index schema/status drift")
                digest = _validate_digest(str(row.get("sha256", "")))
                references = row.get("references")
                if not isinstance(references, list) or row.get("reference_count") != len(references):
                    raise GuardViolation("protected SHA index references are invalid")
                if digest in entries:
                    raise GuardViolation("protected SHA index contains duplicate digest")
                entries[digest] = references
            return entries
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except GuardViolation:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise GuardViolation("protected SHA index is unavailable or invalid") from exc
    if index.get("schema") != "p936-protected-sha-index-v1" or index.get("status") != "PASS":
        raise GuardViolation("protected SHA index schema/status drift")
    entries = index.get("entries")
    if not isinstance(entries, dict):
        raise GuardViolation("protected SHA index entries are invalid")
    return entries


def _reclaim_files(paths: list[os.PathLike[str] | str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_symlink():
            raise GuardViolation("reclaim target may not be a symlink: %s" % path)
        if path.is_dir():
            files.extend(sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and not candidate.is_symlink()))
        elif path.is_file():
            files.append(path)
    return files


def assert_reclaim_allowed(
    paths: list[os.PathLike[str] | str],
    protected_index_path: os.PathLike[str] | str,
    archive_receipt_path: Optional[os.PathLike[str] | str] = None,
) -> None:
    """Fail closed before deleting any SHA referenced by a sealed manifest."""
    index_path = Path(protected_index_path).expanduser().resolve()
    protected = _load_protected_entries(index_path)
    hits = []
    for path in _reclaim_files(paths):
        digest = sha256_file(path)
        if digest in protected:
            hits.append({"path": str(path), "sha256": digest})
    if not hits:
        return
    if archive_receipt_path is None:
        raise GuardViolation("reclaim target has protected SHA without ARCHIVE_FIRST receipt: %s" % hits)
    receipt_path = Path(archive_receipt_path).expanduser().resolve()
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise GuardViolation("ARCHIVE_FIRST receipt must be a regular file")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise GuardViolation("ARCHIVE_FIRST receipt is invalid JSON") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "p936-archive-first-v1"
        or receipt.get("status") != "PASS"
        or not isinstance(receipt.get("entries"), list)
    ):
        raise GuardViolation("ARCHIVE_FIRST receipt schema/status drift")
    rows = {}
    required_archive_fields = {
        "source_path",
        "source_sha256",
        "nas_path",
        "archive_bytes",
        "readback_sha256",
    }
    for row in receipt["entries"]:
        if not isinstance(row, dict) or not required_archive_fields.issubset(row):
            raise GuardViolation("ARCHIVE_FIRST entry schema is invalid")
        source_path = str(Path(str(row["source_path"])).expanduser().resolve())
        source_sha = _validate_digest(str(row["source_sha256"]))
        rows[(source_path, source_sha)] = row
    reclaim_path_set = {hit["path"] for hit in hits}
    for hit in hits:
        row = rows.get((hit["path"], hit["sha256"]))
        if row is None:
            raise GuardViolation("ARCHIVE_FIRST receipt does not cover protected SHA: %s" % hit)
        nas_path = Path(str(row["nas_path"])).expanduser().resolve()
        if str(nas_path) in reclaim_path_set:
            raise GuardViolation("ARCHIVE_FIRST NAS copy is inside the reclaim target set")
        if nas_path.is_symlink() or not nas_path.is_file():
            raise GuardViolation("ARCHIVE_FIRST NAS copy is unavailable: %s" % nas_path)
        if (
            _validate_digest(str(row["readback_sha256"])) != hit["sha256"]
            or row["archive_bytes"] != nas_path.stat().st_size
            or sha256_file(nas_path) != hit["sha256"]
        ):
            raise GuardViolation("ARCHIVE_FIRST NAS readback mismatch: %s" % nas_path)


def assert_seal_dependencies(
    dependencies: list[Mapping[str, Any]],
    copy_locations: Mapping[str, os.PathLike[str] | str],
    min_copies: int = 2,
    probe: Optional[Any] = None,
) -> dict[str, Any]:
    """Require byte-exact copies on distinct hosts before sealing."""
    if not isinstance(min_copies, int) or isinstance(min_copies, bool) or min_copies < 2:
        raise GuardViolation("seal min_copies must be at least two")
    if not dependencies:
        raise GuardViolation("seal requires at least one codebook dependency")
    if len(copy_locations) < min_copies:
        raise GuardViolation("seal copy census has fewer distinct hosts than required")
    rows = []
    seen = set()
    for dependency in dependencies:
        if not isinstance(dependency, Mapping):
            raise GuardViolation("seal dependency entry is invalid")
        digest = _validate_digest(str(dependency.get("sha256", "")))
        expected_bytes = dependency.get("bytes")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            raise GuardViolation("seal dependency bytes are invalid for %s" % digest)
        if digest in seen:
            raise GuardViolation("duplicate seal dependency SHA: %s" % digest)
        seen.add(digest)
        copies = []
        for host, raw_root in sorted(copy_locations.items()):
            root = Path(raw_root).expanduser().resolve()
            if probe is not None:
                result = probe(host, root, digest, expected_bytes)
                if result:
                    copies.append(dict(result))
                continue
            candidate = root / "store" / ("%s.bin" % digest)
            if candidate.is_symlink() or not candidate.is_file():
                continue
            if candidate.stat().st_size != expected_bytes or sha256_file(candidate) != digest:
                continue
            copies.append(
                {
                    "host": host,
                    "path": str(candidate),
                    "bytes": expected_bytes,
                    "sha256": digest,
                }
            )
        row = {
            "sha256": digest,
            "bytes": expected_bytes,
            "role": dependency.get("role"),
            "copy_count": len(copies),
            "copies": copies,
        }
        rows.append(row)
        if len(copies) < min_copies:
            raise GuardViolation(
                "seal copy census failed for %s: have %d need %d"
                % (digest, len(copies), min_copies)
            )
    return {
        "schema": "p936-seal-dependency-census-v1",
        "status": "PASS",
        "min_copies": min_copies,
        "distinct_hosts": sorted(copy_locations),
        "dependency_count": len(rows),
        "dependencies": rows,
        "verified_unix": time.time(),
    }


def resolve_plan_codebook(
    store: AuthorityStore,
    plan_path: os.PathLike[str] | str,
    row: Mapping[str, Any],
    requested_sha256: Optional[str] = None,
) -> Path:
    """Resolve a plan row by SHA and ignore its historical mission path."""
    if not isinstance(row, Mapping) or not isinstance(row.get("codebook"), Mapping):
        raise GuardViolation("plan row lacks a structural codebook specification")
    spec = row["codebook"]
    expected = _validate_digest(str(spec.get("sha256", "")))
    requested = expected if requested_sha256 is None else _validate_digest(requested_sha256)
    path = resolve_codebook_binding(store, plan_path, expected, requested)
    expected_bytes = spec.get("bytes")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
        or path.stat().st_size != expected_bytes
    ):
        raise GuardViolation("plan codebook byte census drift for %s" % expected)
    return path
