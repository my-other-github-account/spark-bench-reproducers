#!/usr/bin/env python3
"""Immutable-SHA authority and exact layer-resume contracts for TRUE-C/F521."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

HEX64 = re.compile(r"^[0-9a-f]{64}$")
BY_SHA = re.compile(r"(?:^|/)by_sha/([0-9a-f]{64})(?:/|$)")

class AuthorityError(RuntimeError):
    pass

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _digest(value: object, field: str) -> str:
    text = str(value or "")
    if not HEX64.fullmatch(text):
        raise AuthorityError(f"invalid {field}: {text!r}")
    return text

@dataclass(frozen=True)
class AuthorityObject:
    sha256: str
    bytes: int
    host: str
    path: str
    role: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "AuthorityObject":
        digest = _digest(row.get("sha256"), "authority sha256")
        try:
            size = int(row["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthorityError(f"invalid authority bytes for {digest}") from exc
        host, path = str(row.get("host") or ""), str(row.get("path") or "")
        if size < 0 or not host or not path.startswith("/"):
            raise AuthorityError(f"invalid authority host/path/bytes for {digest}")
        embedded = BY_SHA.search(path)
        if embedded and embedded.group(1) != digest:
            raise AuthorityError(f"by_sha directory disagrees with digest: {embedded.group(1)} != {digest}")
        return cls(digest, size, host, path, str(row["role"]) if row.get("role") else None)

    def as_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "bytes": self.bytes, "host": self.host,
                "path": self.path, "role": self.role}

Verifier = Callable[[AuthorityObject], tuple[str, int] | Mapping[str, Any]]

class ImmutableSHAIndex:
    """Pinned duplicate-free digest index. Paths locate bytes; they never prove identity."""
    def __init__(self, objects: Iterable[AuthorityObject], *, index_sha256: str | None = None):
        self._objects: dict[str, AuthorityObject] = {}
        for obj in objects:
            if obj.sha256 in self._objects:
                raise AuthorityError(f"duplicate authority SHA: {obj.sha256}")
            self._objects[obj.sha256] = obj
        if not self._objects:
            raise AuthorityError("authority index contains no objects")
        self.index_sha256 = index_sha256

    @classmethod
    def load(cls, path: Path, *, expected_index_sha256: str) -> "ImmutableSHAIndex":
        expected = _digest(expected_index_sha256, "authority index sha256")
        if not path.is_file():
            raise AuthorityError(f"authority index missing: {path}")
        raw = path.read_bytes(); observed = hashlib.sha256(raw).hexdigest()
        if observed != expected:
            raise AuthorityError(f"authority index SHA drift: {observed} != {expected}")
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuthorityError(f"authority index is not JSON: {path}") from exc
        index = cls.from_document(doc); index.index_sha256 = observed; return index

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> "ImmutableSHAIndex":
        if doc.get("schema") != "true-c-immutable-sha-index-v1" or doc.get("status") != "SEALED":
            raise AuthorityError("authority index schema/status drift")
        rows = doc.get("objects")
        if not isinstance(rows, list):
            raise AuthorityError("authority index objects must be a list")
        return cls(AuthorityObject.from_mapping(row) for row in rows)

    def entry(self, expected_sha256: str, *, expected_bytes: int | None = None) -> AuthorityObject:
        digest = _digest(expected_sha256, "expected sha256")
        obj = self._objects.get(digest)
        if obj is None:
            raise AuthorityError(f"missing authority SHA: {digest}")
        if expected_bytes is not None and obj.bytes != int(expected_bytes):
            raise AuthorityError(f"authority byte-count drift for {digest}: {obj.bytes} != {int(expected_bytes)}")
        return obj

    def resolve(self, expected_sha256: str, *, expected_bytes: int | None = None,
                verifier: Verifier | None = None) -> AuthorityObject:
        obj = self.entry(expected_sha256, expected_bytes=expected_bytes)
        if verifier is None:
            path = Path(obj.path)
            if not path.is_file():
                raise AuthorityError(f"authority object missing: {obj.path}")
            observed_sha, observed_bytes = sha256_file(path), path.stat().st_size
        else:
            result = verifier(obj)
            if isinstance(result, Mapping):
                observed_sha, observed_bytes = str(result.get("sha256") or ""), int(result.get("bytes", -1))
            else:
                observed_sha, observed_bytes = str(result[0]), int(result[1])
        if observed_bytes != obj.bytes:
            raise AuthorityError(f"authority object byte drift for {obj.sha256}: {observed_bytes} != {obj.bytes}")
        if observed_sha != obj.sha256:
            raise AuthorityError(f"authority object SHA drift for {obj.sha256}: observed {observed_sha} at {obj.path}")
        return obj

    def manifest(self) -> list[dict[str, Any]]:
        return [self._objects[k].as_dict() for k in sorted(self._objects)]

def bind_stage_specs(rows: Sequence[MutableMapping[str, Any]], index: ImmutableSHAIndex) -> list[dict[str, Any]]:
    """Bind payload/codebook specs by expected SHA. Receipt paths are provenance only."""
    specs: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") == "native_mxfp4":
            continue
        for role in ("artifact", "codebook"):
            provenance, expected = row.get(role), row.get(f"{role}_sha256")
            if provenance and not expected:
                raise AuthorityError(f"{role} provenance present without expected SHA")
            if not expected:
                continue
            size = row.get(f"{role}_bytes")
            obj = index.entry(str(expected), expected_bytes=int(size) if size is not None else None)
            spec = {"host": obj.host, "source": obj.path, "rel": f"objects/{obj.sha256}",
                    "bytes": obj.bytes, "sha256": obj.sha256, "role": role}
            prior = specs.setdefault(obj.sha256, spec)
            if prior != spec:
                raise AuthorityError(f"conflicting authority binding for {obj.sha256}")
            row[f"{role}_provenance_path"] = str(provenance) if provenance else None
            row[f"{role}_authority_sha256"] = obj.sha256
            row[f"{role}_stage_rel"] = spec["rel"]
    return [specs[k] for k in sorted(specs)]

def validate_inherited_prefix(receipt: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Bind inherited PARTIAL rows only by semantic identity and exact hashes."""
    group = list(contract.get("codebook_group") or [])
    expected_rows = contract.get("rows")
    if list(receipt.get("codebook_group") or []) != group or receipt.get("status") != "PARTIAL":
        raise AuthorityError("inherited prefix group/status drift")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise AuthorityError("inherited prefix contract rows missing")
    if int(receipt.get("completed_rows", -1)) != len(expected_rows):
        raise AuthorityError("inherited prefix completed row count drift")
    if int(receipt.get("expected_rows", -1)) != int(contract.get("expected_rows", -2)):
        raise AuthorityError("inherited prefix total row count drift")
    expected_cb = _digest(contract.get("codebook_sha256"), "inherited codebook sha256")
    if receipt.get("codebook_sha256") != expected_cb:
        raise AuthorityError("inherited prefix codebook SHA drift")
    got_rows = receipt.get("rows")
    if not isinstance(got_rows, list) or len(got_rows) != len(expected_rows):
        raise AuthorityError("inherited prefix rows missing/duplicate")
    def key(row: Mapping[str, Any]) -> tuple[int, int, str]:
        ident = row.get("identity")
        return (int(ident[0]), int(ident[1]), str(ident[2])) if isinstance(ident, list) and len(ident)==3 else (int(row["layer"]), int(row["expert"]), str(row["projection"]))
    got: dict[tuple[int,int,str], Mapping[str,Any]] = {}; want: dict[tuple[int,int,str], Mapping[str,Any]] = {}
    for row in got_rows:
        k=key(row)
        if k in got: raise AuthorityError(f"duplicate inherited identity: {k}")
        got[k]=row
    for row in expected_rows:
        k=key(row)
        if k in want: raise AuthorityError(f"duplicate inherited contract identity: {k}")
        want[k]=row
    if set(got) != set(want): raise AuthorityError("inherited prefix identity set drift")
    for k, expected in want.items():
        payload = _digest(expected.get("artifact_sha256"), "inherited payload sha256")
        if got[k].get("artifact_sha256") != payload: raise AuthorityError(f"inherited payload SHA drift: {k}")
        if got[k].get("codebook_sha256") != expected_cb: raise AuthorityError(f"inherited row codebook SHA drift: {k}")
        if got[k].get("status") != "PASS": raise AuthorityError(f"inherited row status drift: {k}")
    return {"status":"PASS_EXACT_HASH_BOUND_PREFIX","codebook_group":group,
            "completed_rows":len(want),"expected_rows":int(contract["expected_rows"]),
            "codebook_sha256":expected_cb,"identity_set_sha256":canonical_sha256(sorted(want)),
            "producer_task_id_coupled":False,"path_coupled":False}

def validate_resume_progress(progress: Mapping[str, Any], *, total_layers: int = 43) -> int:
    completed, mmap_completed = progress.get("completed_layers"), progress.get("mmap_completed_layers")
    if not isinstance(completed, list) or completed != mmap_completed:
        raise AuthorityError("resume completed/mmap layer lists differ")
    if any(not isinstance(x, int) for x in completed) or len(completed)>total_layers or completed != list(range(len(completed))):
        raise AuthorityError("resume layers are not one exact contiguous prefix")
    if progress.get("active_layer") is not None: raise AuthorityError("resume progress has an active unsealed layer")
    if progress.get("local_stage_retired") is not True: raise AuthorityError("resume local stage was not retired")
    if progress.get("mmap_loader_mode") != "torch-mmap": raise AuthorityError("resume mmap loader mode drift")
    return len(completed)

def resume_layer_plan(progress: Mapping[str, Any], checkpoint_sidecar: Mapping[str, Any] | None,
                      *, total_layers: int=43, expected_binding_sha256: str | None=None) -> list[int]:
    start = validate_resume_progress(progress, total_layers=total_layers)
    if start == 0:
        if checkpoint_sidecar is not None: raise AuthorityError("fresh run unexpectedly has a resume checkpoint")
        return list(range(total_layers))
    if checkpoint_sidecar is None: raise AuthorityError("sealed progress prefix has no checkpoint sidecar")
    if checkpoint_sidecar.get("schema") != "p874-anchor-walk-ckpt-sidecar-v2" or checkpoint_sidecar.get("status") != "SEALED":
        raise AuthorityError("resume checkpoint sidecar schema/status drift")
    if int(checkpoint_sidecar.get("layer",-2)) != start-1: raise AuthorityError("resume checkpoint layer does not terminate progress prefix")
    _digest(checkpoint_sidecar.get("checkpoint_sha256"), "checkpoint sha256")
    if int(checkpoint_sidecar.get("checkpoint_bytes",-1)) <= 0: raise AuthorityError("resume checkpoint byte count invalid")
    binding = _digest(checkpoint_sidecar.get("binding_sha256"), "checkpoint binding sha256")
    if expected_binding_sha256 is not None and binding != _digest(expected_binding_sha256,"expected binding sha256"):
        raise AuthorityError("resume checkpoint binding SHA drift")
    wins=checkpoint_sidecar.get("wins")
    if not isinstance(wins,list) or not wins or len(wins)!=len(set(wins)): raise AuthorityError("resume checkpoint window identity drift")
    return list(range(start,total_layers))
