from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

BANK_SCHEMA = "bs-validation-bank-v1"


class ValidationError(ValueError):
    """A fail-closed validation ceremony error."""


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _safe_member(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValidationError(f"BANK_MEMBER_PATH_UNSAFE: {relative!r}")
    path = root / candidate
    if not path.is_file() or path.is_symlink():
        raise ValidationError(f"BANK_MEMBER_MISSING: {relative}")
    return path


def _resolve_bank(bank: str | Path) -> Path:
    path = Path(bank).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValidationError(f"BANK_NOT_DIRECTORY: {path}")
        return path.resolve()
    normalized = str(bank).lower()
    standard = Path.home() / ".cache" / "banana-smasher" / "banks" / normalized
    if not standard.is_dir():
        raise ValidationError(
            f"BANK_NOT_FOUND: {bank!s}; expected {standard} or an explicit bank path"
        )
    return standard.resolve()


def _load_bank(
    bank: str | Path,
    *,
    bank_teacher_logits: str | Path | None = None,
) -> tuple[Path, dict[str, Any], Path]:
    root = _resolve_bank(bank)
    manifest_path = root / "bank.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise ValidationError(f"BANK_MANIFEST_INVALID: {exc}") from exc
    if manifest.get("schema") != BANK_SCHEMA:
        raise ValidationError(
            f"BANK_SCHEMA_MISMATCH: expected {BANK_SCHEMA!r}, "
            f"got {manifest.get('schema')!r}"
        )
    members = manifest.get("members")
    if not isinstance(members, dict) or not members:
        raise ValidationError("BANK_MEMBERS_INVALID: non-empty object required")

    teacher_path: Path | None = None
    for relative, row in sorted(members.items()):
        if not isinstance(relative, str) or not isinstance(row, dict):
            raise ValidationError("BANK_MEMBER_ROW_INVALID")
        path = root / relative
        if not path.exists() and row.get("role") == "teacher_logits":
            if bank_teacher_logits is None:
                raise ValidationError(
                    "TEACHER_LOGITS_MISSING: bank them once with "
                    f"smash validate <artifact> --bank {root} "
                    "--bank-teacher-logits <teacher_logits.npz>"
                )
            source = Path(bank_teacher_logits).expanduser().resolve()
            if not source.is_file() or source.is_symlink():
                raise ValidationError(f"TEACHER_LOGITS_SOURCE_INVALID: {source}")
            if source.stat().st_size != row.get("bytes"):
                raise ValidationError("TEACHER_LOGITS_SOURCE_BYTES_MISMATCH")
            if _sha256_file(source) != row.get("sha256"):
                raise ValidationError("TEACHER_LOGITS_SOURCE_SHA256_MISMATCH")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, path)
        path = _safe_member(root, relative)
        if path.stat().st_size != row.get("bytes"):
            raise ValidationError(f"BANK_MEMBER_BYTES_MISMATCH: {relative}")
        if _sha256_file(path) != row.get("sha256"):
            raise ValidationError(f"BANK_MEMBER_SHA256_MISMATCH: {relative}")
        if row.get("role") == "teacher_logits":
            if teacher_path is not None:
                raise ValidationError("TEACHER_LOGITS_DUPLICATE")
            teacher_path = path
    if teacher_path is None:
        raise ValidationError("TEACHER_LOGITS_ROLE_MISSING")
    return root, manifest, teacher_path


def _load_logits(path: Path, *, require_classes: bool) -> dict[str, np.ndarray[Any, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"sample_ids", "logits"}
            if require_classes:
                required.add("classes")
            missing = required - set(archive.files)
            if missing:
                raise ValidationError(f"LOGITS_FIELDS_MISSING: {sorted(missing)}")
            result = {name: np.asarray(archive[name]) for name in required}
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"LOGITS_ARCHIVE_INVALID: {path}: {exc}") from exc
    logits = result["logits"]
    sample_ids = result["sample_ids"]
    if logits.ndim < 2 or logits.shape[0] != sample_ids.shape[0]:
        raise ValidationError("LOGITS_SHAPE_MISMATCH")
    if not np.issubdtype(logits.dtype, np.floating) or not np.isfinite(logits).all():
        raise ValidationError("LOGITS_NONFINITE_OR_NONFLOAT")
    if require_classes and result["classes"].shape != sample_ids.shape:
        raise ValidationError("LOGITS_CLASSES_SHAPE_MISMATCH")
    return result


def _student_logits_path(artifact: Path, bank_id: str) -> Path:
    if artifact.is_file():
        if artifact.suffix != ".npz":
            raise ValidationError(
                "STUDENT_BACKEND_UNSUPPORTED: file artifacts must be .npz logits"
            )
        return artifact
    path = artifact / "validation" / bank_id / "student_logits.npz"
    if not path.is_file() or path.is_symlink():
        raise ValidationError(
            "STUDENT_PASS_MISSING: expected the exporter/runtime-produced logits at "
            f"{path}"
        )
    return path


def _training_ids(artifact: Path) -> set[str]:
    if artifact.is_file():
        path = artifact.with_name("training_sample_ids.json")
    else:
        path = artifact / "training_sample_ids.json"
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ValidationError(f"TRAINING_MANIFEST_INVALID: {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValidationError("TRAINING_MANIFEST_IDS_INVALID")
    if len(value) != len(set(value)):
        raise ValidationError("TRAINING_MANIFEST_DUPLICATE_IDS")
    return set(value)


def _directory_identity(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValidationError(f"ARTIFACT_SYMLINK_FORBIDDEN: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"kind": "directory", "sha256": digest, "file_count": len(rows)}


def _artifact_identity(artifact: Path) -> dict[str, Any]:
    if artifact.is_file():
        return {
            "kind": "file",
            "bytes": artifact.stat().st_size,
            "sha256": _sha256_file(artifact),
        }
    if artifact.is_dir():
        return _directory_identity(artifact)
    raise ValidationError(f"ARTIFACT_MISSING: {artifact}")


def _log_softmax(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    work = logits.astype(np.float64, copy=False)
    maximum = np.max(work, axis=-1, keepdims=True)
    shifted = work - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _kld_per_sample(
    teacher_logits: np.ndarray[Any, Any],
    student_logits: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    if teacher_logits.shape != student_logits.shape:
        raise ValidationError(
            f"TEACHER_STUDENT_SHAPE_MISMATCH: {teacher_logits.shape} != "
            f"{student_logits.shape}"
        )
    teacher_logp = _log_softmax(teacher_logits)
    student_logp = _log_softmax(student_logits)
    teacher_probability = np.exp(teacher_logp)
    values = np.sum(teacher_probability * (teacher_logp - student_logp), axis=-1)
    if values.ndim > 1:
        values = np.mean(values, axis=tuple(range(1, values.ndim)))
    if not np.isfinite(values).all():
        raise ValidationError("KLD_NONFINITE")
    return values


def validate_artifact(
    artifact: str | Path,
    *,
    bank: str | Path,
    check_exposure: bool,
    receipt_path: str | Path | None = None,
    bank_teacher_logits: str | Path | None = None,
) -> dict[str, Any]:
    """Run the banked-teacher validation ceremony and seal its provenance."""
    started = time.time()
    artifact_path = Path(artifact).expanduser().resolve()
    artifact_identity = _artifact_identity(artifact_path)
    bank_root, bank_manifest, teacher_path = _load_bank(
        bank, bank_teacher_logits=bank_teacher_logits
    )
    bank_id = str(bank_manifest.get("bank_id", bank_root.name))
    teacher = _load_logits(teacher_path, require_classes=True)

    student_path = _student_logits_path(artifact_path, bank_id)
    student = _load_logits(student_path, require_classes=False)
    teacher_ids = teacher["sample_ids"].astype(str)
    student_ids = student["sample_ids"].astype(str)
    if not np.array_equal(teacher_ids, student_ids):
        raise ValidationError("TEACHER_STUDENT_SAMPLE_ORDER_MISMATCH")
    expected_count = bank_manifest.get("sample_count")
    if expected_count != int(teacher_ids.shape[0]):
        raise ValidationError(
            f"BANK_SAMPLE_COUNT_MISMATCH: {expected_count} != {teacher_ids.shape[0]}"
        )

    exposure = {
        "checked": bool(check_exposure),
        "overlap_count": 0,
        "status": "NOT_REQUESTED",
    }
    if check_exposure:
        overlap = sorted(set(teacher_ids.tolist()) & _training_ids(artifact_path))
        if overlap:
            raise ValidationError(
                f"EXPOSURE_OVERLAP: count={len(overlap)} first={overlap[:8]}"
            )
        exposure = {"checked": True, "overlap_count": 0, "status": "PASS"}

    values = _kld_per_sample(teacher["logits"], student["logits"])
    global_kld = float(np.mean(values))
    classes = teacher["classes"].astype(str)
    per_class = {
        class_name: float(np.mean(values[classes == class_name]))
        for class_name in sorted(set(classes.tolist()))
    }
    baselines = bank_manifest.get("baselines", {})
    if not isinstance(baselines, dict) or not baselines:
        raise ValidationError("BANK_BASELINES_MISSING")
    comparison: dict[str, dict[str, float]] = {}
    for name, row in sorted(baselines.items()):
        if not isinstance(row, dict):
            raise ValidationError(f"BANK_BASELINE_INVALID: {name}")
        baseline_kld = row.get("global_kld")
        if not isinstance(baseline_kld, (int, float)) or not math.isfinite(
            float(baseline_kld)
        ):
            raise ValidationError(f"BANK_BASELINE_KLD_INVALID: {name}")
        comparison[name] = {
            "baseline_global_kld": float(baseline_kld),
            "student_global_kld": global_kld,
            "delta": global_kld - float(baseline_kld),
        }

    receipt: dict[str, Any] = {
        "schema": "bs-validation-receipt-v1",
        "status": "PASS",
        "artifact": str(artifact_path),
        "bank": bank_id,
        "sample_count": int(teacher_ids.shape[0]),
        "global_kld": global_kld,
        "per_class_kld": per_class,
        "comparison": comparison,
        "exposure": exposure,
        "student_pass": {
            "backend": "bank-bound-npz",
            "status": "PASS",
            "logits_path": str(student_path),
        },
        "provenance": {
            "artifact": artifact_identity,
            "bank_manifest_sha256": _sha256_file(bank_root / "bank.json"),
            "teacher_logits_sha256": _sha256_file(teacher_path),
            "student_logits_sha256": _sha256_file(student_path),
        },
        "started_unix": started,
        "completed_unix": time.time(),
    }
    if receipt_path is None:
        receipt_path = artifact_path.with_name("VALIDATION_RECEIPT.json")
    output = Path(receipt_path).expanduser().resolve()
    _atomic_json(output, receipt)
    result = dict(receipt)
    result["receipt"] = {"path": str(output), "sha256": _sha256_file(output)}
    return result
