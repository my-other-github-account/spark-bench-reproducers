#!/usr/bin/env python3
"""Immutable SHA authority contracts for canonical TRUE-C/F521 integration.

Human-readable names and receipt paths are provenance only.  Selection is by a
pinned expected SHA, and bytes are hashed before a resolved object is returned.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, Tuple, Union

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BY_SHA_PATH = re.compile(r"(?:^|/)by_sha/([0-9a-f]{64})(?:/|$)")


class AuthorityError(RuntimeError):
    """The immutable authority contract is absent, ambiguous, or invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object, field: str) -> str:
    text = str(value or "")
    if not _HEX_SHA256.fullmatch(text):
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
        digest = _sha256(row.get("sha256"), "authority sha256")
        try:
            byte_count = int(row["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthorityError(f"invalid authority bytes for {digest}") from exc
        host = str(row.get("host") or "")
        path = str(row.get("path") or "")
        if byte_count < 0 or not host or not path.startswith("/"):
            raise AuthorityError(
                f"invalid authority host/path/bytes for {digest}"
            )
        embedded_digest = _BY_SHA_PATH.search(path)
        if embedded_digest and embedded_digest.group(1) != digest:
            raise AuthorityError(
                "by_sha directory disagrees with authority digest: "
                f"{embedded_digest.group(1)} != {digest}"
            )
        role = str(row["role"]) if row.get("role") else None
        return cls(digest, byte_count, host, path, role)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "bytes": self.bytes,
            "host": self.host,
            "path": self.path,
            "role": self.role,
        }


# Type-alias assignments are evaluated at import time even with postponed
# annotations. Keep this Python 3.9 compatible; PEP 604 requires Python 3.10.
Verifier = Callable[
    [AuthorityObject], Union[Tuple[str, int], Mapping[str, Any]]
]


class ImmutableSHAIndex:
    """Pinned duplicate-free SHA index; paths locate bytes but never prove them."""

    def __init__(
        self,
        objects: Iterable[AuthorityObject],
        *,
        index_sha256: str | None = None,
    ) -> None:
        self._objects: dict[str, AuthorityObject] = {}
        for authority_object in objects:
            if authority_object.sha256 in self._objects:
                raise AuthorityError(
                    f"duplicate authority SHA: {authority_object.sha256}"
                )
            self._objects[authority_object.sha256] = authority_object
        if not self._objects:
            raise AuthorityError("authority index contains no objects")
        self.index_sha256 = index_sha256

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_index_sha256: str,
    ) -> "ImmutableSHAIndex":
        expected = _sha256(
            expected_index_sha256, "authority index sha256"
        )
        if not path.is_file():
            raise AuthorityError(f"authority index missing: {path}")
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected:
            raise AuthorityError(
                f"authority index SHA drift: {observed} != {expected}"
            )
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuthorityError(
                f"authority index is not JSON: {path}"
            ) from exc
        index = cls.from_document(document)
        index.index_sha256 = observed
        return index

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any]
    ) -> "ImmutableSHAIndex":
        if (
            document.get("schema") != "true-c-immutable-sha-index-v1"
            or document.get("status") != "SEALED"
        ):
            raise AuthorityError("authority index schema/status drift")
        rows = document.get("objects")
        if not isinstance(rows, list):
            raise AuthorityError("authority index objects must be a list")
        objects: list[AuthorityObject] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise AuthorityError("authority index object must be a mapping")
            objects.append(AuthorityObject.from_mapping(row))
        return cls(objects)

    def entry(
        self,
        expected_sha256: str,
        *,
        expected_bytes: int | None = None,
    ) -> AuthorityObject:
        digest = _sha256(expected_sha256, "expected sha256")
        authority_object = self._objects.get(digest)
        if authority_object is None:
            raise AuthorityError(f"missing authority SHA: {digest}")
        if (
            expected_bytes is not None
            and authority_object.bytes != int(expected_bytes)
        ):
            raise AuthorityError(
                f"authority byte-count drift for {digest}: "
                f"{authority_object.bytes} != {int(expected_bytes)}"
            )
        return authority_object

    def resolve(
        self,
        expected_sha256: str,
        *,
        expected_bytes: int | None = None,
        verifier: Verifier | None = None,
    ) -> AuthorityObject:
        """Resolve one expected digest and hash the candidate bytes before use."""
        authority_object = self.entry(
            expected_sha256, expected_bytes=expected_bytes
        )
        if verifier is None:
            candidate = Path(authority_object.path)
            if not candidate.is_file():
                raise AuthorityError(
                    f"authority object missing: {authority_object.path}"
                )
            observed_sha256 = sha256_file(candidate)
            observed_bytes = candidate.stat().st_size
        else:
            result = verifier(authority_object)
            if isinstance(result, Mapping):
                observed_sha256 = str(result.get("sha256") or "")
                observed_bytes = int(result.get("bytes", -1))
            else:
                observed_sha256 = str(result[0])
                observed_bytes = int(result[1])
        if observed_bytes != authority_object.bytes:
            raise AuthorityError(
                f"authority object byte drift for {authority_object.sha256}: "
                f"{observed_bytes} != {authority_object.bytes}"
            )
        if observed_sha256 != authority_object.sha256:
            raise AuthorityError(
                f"authority object SHA drift for {authority_object.sha256}: "
                f"observed {observed_sha256} at {authority_object.path}"
            )
        return authority_object

    def manifest(self) -> list[dict[str, Any]]:
        return [
            self._objects[digest].as_dict()
            for digest in sorted(self._objects)
        ]


def bind_stage_specs(
    rows: Sequence[MutableMapping[str, Any]],
    index: ImmutableSHAIndex,
) -> list[dict[str, Any]]:
    """Select payload/codebook stage sources only by each expected digest."""
    specifications: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") == "native_mxfp4":
            continue
        for role in ("artifact", "codebook"):
            provenance_path = row.get(role)
            expected_sha256 = row.get(f"{role}_sha256")
            if provenance_path and not expected_sha256:
                raise AuthorityError(
                    f"{role} provenance present without expected SHA"
                )
            if not expected_sha256:
                continue
            expected_bytes = row.get(f"{role}_bytes")
            authority_object = index.entry(
                str(expected_sha256),
                expected_bytes=(
                    int(expected_bytes)
                    if expected_bytes is not None
                    else None
                ),
            )
            specification = {
                "host": authority_object.host,
                "source": authority_object.path,
                "rel": f"objects/{authority_object.sha256}",
                "bytes": authority_object.bytes,
                "sha256": authority_object.sha256,
                "role": role,
            }
            prior = specifications.setdefault(
                authority_object.sha256, specification
            )
            if prior != specification:
                raise AuthorityError(
                    "conflicting authority binding for "
                    f"{authority_object.sha256}"
                )
            row[f"{role}_provenance_path"] = (
                str(provenance_path) if provenance_path else None
            )
            row[f"{role}_authority_sha256"] = authority_object.sha256
            row[f"{role}_stage_rel"] = specification["rel"]
    return [specifications[digest] for digest in sorted(specifications)]


def _identity(row: Mapping[str, Any]) -> tuple[int, int, str]:
    explicit = row.get("identity")
    if isinstance(explicit, list) and len(explicit) == 3:
        return int(explicit[0]), int(explicit[1]), str(explicit[2])
    try:
        return (
            int(row["layer"]),
            int(row["expert"]),
            str(row["projection"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityError("inherited prefix identity is invalid") from exc


def validate_inherited_prefix(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an inherited PARTIAL prefix by semantics and exact hashes only."""
    if contract.get("schema") != "true-c-inherited-prefix-contract-v1":
        raise AuthorityError("inherited prefix contract schema drift")
    if receipt.get("schema") != "p925-true-c-refit-codebook-v2":
        raise AuthorityError("inherited prefix receipt schema drift")
    group = list(contract.get("codebook_group") or [])
    if (
        list(receipt.get("codebook_group") or []) != group
        or receipt.get("status") != "PARTIAL"
    ):
        raise AuthorityError("inherited prefix group/status drift")
    expected_rows = contract.get("rows")
    received_rows = receipt.get("rows")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise AuthorityError("inherited prefix contract rows missing")
    if not isinstance(received_rows, list):
        raise AuthorityError("inherited prefix receipt rows missing")
    if int(receipt.get("completed_rows", -1)) != len(expected_rows):
        raise AuthorityError("inherited prefix completed row count drift")
    if len(received_rows) != len(expected_rows):
        raise AuthorityError("inherited prefix rows missing/duplicate")
    if int(receipt.get("expected_rows", -1)) != int(
        contract.get("expected_rows", -2)
    ):
        raise AuthorityError("inherited prefix total row count drift")

    expected_codebook_sha256 = _sha256(
        contract.get("codebook_sha256"), "inherited codebook sha256"
    )
    if receipt.get("codebook_sha256") != expected_codebook_sha256:
        raise AuthorityError("inherited prefix codebook SHA drift")

    expected_identities = [_identity(row) for row in expected_rows]
    received_identities = [_identity(row) for row in received_rows]
    if len(set(expected_identities)) != len(expected_identities):
        raise AuthorityError("duplicate inherited contract identity")
    if len(set(received_identities)) != len(received_identities):
        raise AuthorityError("duplicate inherited identity")
    if received_identities != expected_identities:
        raise AuthorityError("inherited prefix identity/order drift")

    for identity, received, expected in zip(
        expected_identities, received_rows, expected_rows
    ):
        expected_payload_sha256 = _sha256(
            expected.get("artifact_sha256"),
            "inherited payload sha256",
        )
        if received.get("artifact_sha256") != expected_payload_sha256:
            raise AuthorityError(
                f"inherited payload SHA drift: {identity}"
            )
        if received.get("codebook_sha256") != expected_codebook_sha256:
            raise AuthorityError(
                f"inherited row codebook SHA drift: {identity}"
            )
        if received.get("status") != "PASS":
            raise AuthorityError(
                f"inherited row status drift: {identity}"
            )

    return {
        "status": "PASS_EXACT_HASH_BOUND_PREFIX",
        "codebook_group": group,
        "completed_rows": len(expected_rows),
        "expected_rows": int(contract["expected_rows"]),
        "codebook_sha256": expected_codebook_sha256,
        "identity_set_sha256": canonical_sha256(
            sorted(expected_identities)
        ),
        "producer_task_id_coupled": False,
        "path_coupled": False,
    }


def validate_resume_progress(
    progress: Mapping[str, Any], *, total_layers: int = 43
) -> int:
    """Return the first unfinished layer after validating an exact sealed prefix."""
    completed = progress.get("completed_layers")
    mmap_completed = progress.get("mmap_completed_layers")
    if not isinstance(completed, list) or completed != mmap_completed:
        raise AuthorityError("resume completed/mmap layer lists differ")
    if (
        any(not isinstance(layer, int) for layer in completed)
        or len(completed) > total_layers
        or completed != list(range(len(completed)))
    ):
        raise AuthorityError("resume layers are not one exact contiguous prefix")
    if progress.get("active_layer") is not None:
        raise AuthorityError("resume progress has an active unsealed layer")
    if progress.get("local_stage_retired") is not True:
        raise AuthorityError("resume local stage was not retired")
    if progress.get("mmap_loader_mode") != "torch-mmap":
        raise AuthorityError("resume mmap loader mode drift")
    return len(completed)


def validate_completed_layer_receipts(
    progress: Mapping[str, Any],
    *,
    expected_binding_sha256: str,
    expected_wins: Sequence[int],
    total_layers: int = 43,
) -> Mapping[str, Any] | None:
    """Validate the immutable receipt chain and sufficient statistics once each."""
    start = validate_resume_progress(progress, total_layers=total_layers)
    rows = progress.get("completed_layer_receipts")
    if start == 0:
        if rows not in (None, []):
            raise AuthorityError("fresh progress unexpectedly has layer receipts")
        return None
    if not isinstance(rows, list) or len(rows) != start:
        raise AuthorityError("completed-layer receipt manifest length drift")
    binding_sha256 = _sha256(
        expected_binding_sha256, "completed-layer binding sha256"
    )
    wins = list(expected_wins)
    if not wins or len(wins) != len(set(wins)):
        raise AuthorityError("completed-layer window identity drift")

    previous_receipt_sha256: str | None = None
    latest: Mapping[str, Any] | None = None
    for layer, reference in enumerate(rows):
        if not isinstance(reference, Mapping) or int(reference.get("layer", -1)) != layer:
            raise AuthorityError("completed-layer receipt manifest order drift")
        path = Path(str(reference.get("path") or ""))
        expected_receipt_sha256 = _sha256(
            reference.get("sha256"), "completed-layer receipt sha256"
        )
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise AuthorityError(f"completed-layer receipt missing: {path}") from exc
        observed_receipt_sha256 = hashlib.sha256(raw).hexdigest()
        if observed_receipt_sha256 != expected_receipt_sha256:
            raise AuthorityError(
                f"completed-layer receipt SHA drift L{layer:03d}: "
                f"{observed_receipt_sha256} != {expected_receipt_sha256}"
            )
        try:
            receipt = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuthorityError(
                f"completed-layer receipt is not JSON L{layer:03d}"
            ) from exc
        if not isinstance(receipt, Mapping):
            raise AuthorityError("completed-layer receipt must be an object")
        if (
            receipt.get("schema") != "p874-anchor-walk-layer-receipt-v3"
            or receipt.get("status") != "SEALED"
            or int(receipt.get("layer", -1)) != layer
            or receipt.get("wins") != wins
            or receipt.get("binding_sha256") != binding_sha256
            or receipt.get("previous_receipt_sha256")
            != previous_receipt_sha256
        ):
            raise AuthorityError(f"completed-layer receipt semantic drift L{layer:03d}")
        statistics = receipt.get("sufficient_statistics")
        if not isinstance(statistics, Mapping) or not statistics:
            raise AuthorityError(f"completed-layer statistics missing L{layer:03d}")
        if receipt.get("sufficient_statistics_sha256") != canonical_sha256(statistics):
            raise AuthorityError(f"completed-layer statistics SHA drift L{layer:03d}")
        previous_receipt_sha256 = observed_receipt_sha256
        latest = receipt

    assert latest is not None
    checkpoint = Path(str(latest.get("checkpoint") or ""))
    if not checkpoint.is_file():
        raise AuthorityError("latest completed-layer checkpoint missing")
    expected_checkpoint_sha256 = _sha256(
        latest.get("checkpoint_sha256"), "checkpoint sha256"
    )
    checkpoint_bytes = checkpoint.read_bytes()
    if (
        len(checkpoint_bytes) != int(latest.get("checkpoint_bytes", -1))
        or hashlib.sha256(checkpoint_bytes).hexdigest()
        != expected_checkpoint_sha256
    ):
        raise AuthorityError("latest completed-layer checkpoint identity drift")
    return latest


def resume_layer_plan(
    progress: Mapping[str, Any],
    latest_receipt: Mapping[str, Any] | None,
    *,
    total_layers: int = 43,
    expected_binding_sha256: str | None = None,
) -> list[int]:
    """Return only unfinished layers; never fall back to prefix recomputation."""
    start = validate_resume_progress(progress, total_layers=total_layers)
    if start == 0:
        if latest_receipt is not None:
            raise AuthorityError("fresh run unexpectedly has a resume receipt")
        return list(range(total_layers))
    if latest_receipt is None:
        raise AuthorityError("sealed progress prefix has no checkpoint receipt")
    if int(latest_receipt.get("layer", -2)) != start - 1:
        raise AuthorityError("resume checkpoint layer does not terminate progress prefix")
    if expected_binding_sha256 is not None and latest_receipt.get(
        "binding_sha256"
    ) != _sha256(expected_binding_sha256, "expected binding sha256"):
        raise AuthorityError("resume checkpoint binding SHA drift")
    return list(range(start, total_layers))
