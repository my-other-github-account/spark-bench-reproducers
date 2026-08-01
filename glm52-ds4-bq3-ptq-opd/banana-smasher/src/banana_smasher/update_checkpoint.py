from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA = "banana-smasher-update-checkpoint-v1"
MANIFEST_SCHEMA = "banana-smasher-update-checkpoint-manifest-v1"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_torch_save(path: Path, value: Any) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def commit_segment_checkpoint(
    checkpoint_dir: str | Path,
    payload: dict[str, Any],
    *,
    identity: dict[str, Any],
    backend: str,
    segment_plan: list[int],
) -> dict[str, Any]:
    """Durably commit one accumulated-gradient transaction.

    Payload publication precedes the checksummed manifest, and both renames are
    directory-fsynced. A reader therefore observes either the previous complete
    transaction or this complete transaction, never an unverified partial one.
    """
    root = Path(checkpoint_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    previous_payload = None
    if manifest_path.is_file():
        previous_payload = Path(json.loads(manifest_path.read_text())["payload_path"])
    payload_path = root / (
        f"payload-{payload['run_id']}-{int(payload['next_segment_index']):04d}-"
        f"{payload['state']}.pt"
    )
    payload_value = {"schema": CHECKPOINT_SCHEMA, **payload}
    payload_record = atomic_torch_save(payload_path, payload_value)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "COMPLETE" if payload.get("state") == "complete" else "IN_PROGRESS",
        "backend": backend,
        "identity": identity,
        "segment_plan": [int(count) for count in segment_plan],
        "logical_items": sum(int(count) for count in segment_plan),
        "next_segment_index": int(payload["next_segment_index"]),
        "completed_segments": list(payload["completed_segments"]),
        "optimizer_steps": int(payload.get("optimizer_steps", 0)),
        "transaction_state": str(payload["state"]),
        "payload_path": str(payload_path),
        "payload_sha256": payload_record["sha256"],
        "payload_bytes": payload_record["bytes"],
    }
    atomic_json(manifest_path, manifest)
    if (
        previous_payload is not None
        and previous_payload != payload_path
        and previous_payload.parent.resolve() == root
        and previous_payload.name.startswith("payload-")
    ):
        previous_payload.unlink(missing_ok=True)
        _fsync_directory(root)
    return manifest


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_identity: dict[str, Any],
    expected_backend: str,
    expected_segment_plan: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    root = Path(checkpoint_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"update checkpoint manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported update checkpoint manifest schema: {manifest.get('schema')!r}")
    if manifest.get("identity") != expected_identity:
        raise RuntimeError("checkpoint identity mismatch")
    if manifest.get("backend") != expected_backend:
        raise RuntimeError(
            f"checkpoint backend mismatch: {manifest.get('backend')!r} != {expected_backend!r}"
        )
    if expected_segment_plan is not None and manifest.get("segment_plan") != expected_segment_plan:
        raise RuntimeError("checkpoint segment geometry mismatch")
    completed = manifest.get("completed_segments")
    next_index = int(manifest.get("next_segment_index", -1))
    if completed != list(range(next_index)):
        raise RuntimeError("checkpoint completed segments are not a contiguous prefix")
    payload_path = Path(manifest["payload_path"])
    if not payload_path.is_file():
        raise RuntimeError(f"checkpoint payload is missing: {payload_path}")
    if payload_path.stat().st_size != int(manifest["payload_bytes"]):
        raise RuntimeError("checkpoint payload byte count mismatch")
    if _sha256(payload_path) != manifest["payload_sha256"]:
        raise RuntimeError("checkpoint payload SHA-256 mismatch")
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"checkpoint payload cannot be loaded: {exc}") from exc
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"unsupported update checkpoint payload schema: {payload.get('schema')!r}")
    if int(payload.get("next_segment_index", -1)) != next_index:
        raise RuntimeError("checkpoint payload/manifest next-segment mismatch")
    if payload.get("completed_segments") != completed:
        raise RuntimeError("checkpoint payload/manifest completed-prefix mismatch")
    return payload, manifest


def finalize_checkpoint(
    checkpoint_dir: str | Path,
    *,
    receipt: str | Path,
    output_record: dict[str, Any],
) -> dict[str, Any]:
    root = Path(checkpoint_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "status": "COMPLETE",
            "transaction_state": "complete",
            "receipt": str(Path(receipt).resolve()),
            "output": output_record,
        }
    )
    atomic_json(manifest_path, manifest)
    return manifest
