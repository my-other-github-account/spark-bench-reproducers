from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .durability import (
    DurabilityError,
    atomic_json,
    atomic_npz,
    canonical_sha256,
    file_identity,
    load_json_object,
    output_lock,
    safe_relative_path,
    sha256_file,
    tree_identity,
)
from .metrics import MetricsError, teacher_support
from .real_axis import (
    RealAxisRunner,
    Window,
    WindowPopulation,
    load_instrument_profile,
)

BANK_SCHEMA = "bs-teacher-bank-v1"
BANK_MEMBER_SCHEMA = "bs-teacher-bank-member-v1"
BANK_MARKER_SCHEMA = "bs-bank-complete-v1"
BANK_RECEIPT_SCHEMA = "bs-bank-operation-receipt-v1"


class BankError(ValueError):
    """Raised when a teacher bank is incomplete or semantically invalid."""


class BankSpecMismatch(BankError):
    """Raised without mutating an output owned by a different build spec."""


def _tensor_schema(value: np.ndarray[Any, Any]) -> dict[str, Any]:
    return {"dtype": value.dtype.str, "shape": list(value.shape)}


def _member_paths(root: Path, ordinal: int) -> tuple[Path, Path]:
    stem = f"window_{ordinal:06d}"
    return root / "members" / f"{stem}.npz", root / "members" / f"{stem}.json"


def _build_spec(
    runner: RealAxisRunner,
    population: WindowPopulation,
    instrument: dict[str, Any],
) -> dict[str, Any]:
    instrument_identity = {
        key: instrument[key]
        for key in (
            "schema",
            "schema_version",
            "profile",
            "teacher_storage",
            "support",
            "cutoff",
            "direction",
            "attention",
            "estimator",
            "profile_sha256",
        )
    }
    return {
        "schema": "bs-bank-build-spec-v1",
        "model": runner.identity(),
        "corpus": {
            "corpus_id": population.corpus_id,
            "manifest_sha256": population.manifest_sha256,
            "ordered_window_ids_sha256": population.ordered_window_ids_sha256,
            "window_count": len(population.windows),
        },
        "instrument": instrument_identity,
    }


def _load_member_arrays(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected = {
                "initial_hidden",
                "teacher_indices",
                "teacher_logprob",
                "teacher_argmax",
            }
            if set(archive.files) != expected:
                raise BankError(
                    f"BANK_MEMBER_FIELDS_MISMATCH: {path}: {sorted(archive.files)}"
                )
            return {name: np.asarray(archive[name]) for name in expected}
    except BankError:
        raise
    except Exception as exc:
        raise BankError(f"BANK_MEMBER_ARCHIVE_INVALID: {path}: {exc}") from exc


def _verify_member(
    root: Path,
    row: dict[str, Any],
    *,
    expected_build_spec_sha256: str,
    expected_window: Window | None = None,
    expected_support: int | None = None,
    expected_positions: int | None = None,
) -> dict[str, Any]:
    if row.get("schema") != BANK_MEMBER_SCHEMA:
        raise BankError("BANK_MEMBER_SCHEMA_MISMATCH")
    if row.get("build_spec_sha256") != expected_build_spec_sha256:
        raise BankError("BANK_MEMBER_BUILD_SPEC_MISMATCH")
    if expected_window is not None and (
        row.get("ordinal") != expected_window.ordinal
        or row.get("window_id") != expected_window.window_id
        or row.get("class") != expected_window.class_name
    ):
        raise BankError("BANK_MEMBER_WINDOW_MISMATCH")
    if expected_window is not None:
        expected_member, expected_sidecar = _member_paths(root, expected_window.ordinal)
        if row.get("path") != expected_member.relative_to(root).as_posix():
            raise BankError("BANK_MEMBER_PATH_MISMATCH")
        if row.get("sidecar") != expected_sidecar.relative_to(root).as_posix():
            raise BankError("BANK_MEMBER_SIDECAR_PATH_MISMATCH")
    relative = row.get("path")
    if not isinstance(relative, str):
        raise BankError("BANK_MEMBER_PATH_INVALID")
    try:
        path = safe_relative_path(root, relative, label="BANK_MEMBER")
    except DurabilityError as exc:
        raise BankError(str(exc)) from exc
    if path.stat().st_size != row.get("bytes"):
        raise BankError(f"BANK_MEMBER_BYTES_MISMATCH: {relative}")
    if sha256_file(path) != row.get("sha256"):
        raise BankError(f"BANK_MEMBER_SHA256_MISMATCH: {relative}")
    arrays = _load_member_arrays(path)
    tensors = row.get("tensors")
    if not isinstance(tensors, dict) or {
        name: _tensor_schema(value) for name, value in sorted(arrays.items())
    } != tensors:
        raise BankError(f"BANK_MEMBER_TENSOR_SCHEMA_MISMATCH: {relative}")
    hidden = arrays["initial_hidden"]
    indices = arrays["teacher_indices"]
    logprob = arrays["teacher_logprob"]
    argmax = arrays["teacher_argmax"]
    if hidden.ndim != 2 or not np.issubdtype(hidden.dtype, np.floating):
        raise BankError("BANK_MEMBER_HIDDEN_INVALID")
    if indices.ndim != 2 or not np.issubdtype(indices.dtype, np.integer):
        raise BankError("BANK_MEMBER_INDICES_INVALID")
    if logprob.shape != indices.shape or not np.issubdtype(logprob.dtype, np.floating):
        raise BankError("BANK_MEMBER_LOGPROB_INVALID")
    if argmax.shape != (indices.shape[0],) or not np.issubdtype(
        argmax.dtype, np.integer
    ):
        raise BankError("BANK_MEMBER_ARGMAX_INVALID")
    if hidden.shape[0] != indices.shape[0]:
        raise BankError("BANK_MEMBER_POSITION_MISMATCH")
    if not np.isfinite(hidden).all() or not np.isfinite(logprob).all():
        raise BankError("BANK_MEMBER_NONFINITE")
    if row.get("positions") != indices.shape[0] or row.get("support") != indices.shape[1]:
        raise BankError("BANK_MEMBER_INSTRUMENT_MISMATCH")
    if expected_support is not None and indices.shape[1] != expected_support:
        raise BankError("BANK_MEMBER_SUPPORT_MISMATCH")
    if expected_positions is not None and indices.shape[0] != expected_positions:
        raise BankError("BANK_MEMBER_POSITION_MISMATCH")
    if np.any(indices < 0) or not np.array_equal(argmax, indices[:, 0]):
        raise BankError("BANK_MEMBER_INDEX_SEMANTICS_INVALID")
    return {**row, "path_obj": path, "arrays": arrays}


def _sidecar_row(root: Path, sidecar: Path) -> dict[str, Any]:
    value = load_json_object(sidecar, label="BANK_MEMBER_SIDECAR")
    if value.get("sidecar") != sidecar.relative_to(root).as_posix():
        raise BankError("BANK_MEMBER_SIDECAR_PATH_MISMATCH")
    return value


def _expected_member_names(count: int) -> set[str]:
    names: set[str] = set()
    for ordinal in range(count):
        member, sidecar = _member_paths(Path("."), ordinal)
        names.add(member.name)
        names.add(sidecar.name)
    return names


def verify_bank(bank: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    root = Path(bank).resolve()
    marker_path = root / "BANK_COMPLETE"
    if require_complete and (marker_path.is_symlink() or not marker_path.is_file()):
        raise BankError(f"BANK_INCOMPLETE: missing BANK_COMPLETE in {root}")
    marker = load_json_object(marker_path, label="BANK_MARKER")
    if marker.get("schema") != BANK_MARKER_SCHEMA or marker.get("status") != "COMPLETE":
        raise BankError("BANK_MARKER_INVALID")
    manifest_path = root / "bank.json"
    manifest = load_json_object(manifest_path, label="BANK_MANIFEST")
    if manifest.get("schema") != BANK_SCHEMA or manifest.get("status") != "COMPLETE":
        raise BankError("BANK_MANIFEST_SCHEMA_MISMATCH")
    manifest_sha256 = sha256_file(manifest_path)
    if marker.get("bank_manifest_sha256") != manifest_sha256:
        raise BankError("BANK_MANIFEST_SHA256_MISMATCH")
    for field in ("bank_id", "build_spec_sha256", "member_count"):
        if marker.get(field) != manifest.get(field):
            raise BankError(f"BANK_MARKER_{field.upper()}_MISMATCH")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != manifest.get("member_count"):
        raise BankError("BANK_MEMBER_COUNT_MISMATCH")
    model = manifest.get("model")
    corpus = manifest.get("corpus")
    instrument = manifest.get("instrument")
    population = manifest.get("population")
    if (
        not isinstance(model, dict)
        or not isinstance(corpus, dict)
        or not isinstance(instrument, dict)
        or not isinstance(population, dict)
    ):
        raise BankError("BANK_MANIFEST_METADATA_INVALID")
    reconstructed_spec = {
        "schema": "bs-bank-build-spec-v1",
        "model": model,
        "corpus": corpus,
        "instrument": instrument,
    }
    if canonical_sha256(reconstructed_spec) != manifest.get("build_spec_sha256"):
        raise BankError("BANK_BUILD_SPEC_DIGEST_MISMATCH")
    expected_ids = population.get("ordered_window_ids")
    expected_classes = population.get("classes")
    if (
        not isinstance(expected_ids, list)
        or len(expected_ids) != len(members)
        or not all(
            isinstance(value, (str, int)) and not isinstance(value, bool)
            for value in expected_ids
        )
    ):
        raise BankError("BANK_POPULATION_INVALID")
    if (
        not isinstance(expected_classes, list)
        or len(expected_classes) != len(members)
        or not all(isinstance(value, str) and value for value in expected_classes)
    ):
        raise BankError("BANK_POPULATION_CLASSES_INVALID")
    if len(set((type(value).__name__, value) for value in expected_ids)) != len(expected_ids):
        raise BankError("BANK_POPULATION_DUPLICATE_IDS")
    ordered_sha256 = canonical_sha256(expected_ids)
    if population.get("ordered_window_ids_sha256") != ordered_sha256:
        raise BankError("BANK_POPULATION_SHA256_MISMATCH")
    if (
        corpus.get("ordered_window_ids_sha256") != ordered_sha256
        or corpus.get("window_count") != len(members)
    ):
        raise BankError("BANK_CORPUS_POPULATION_MISMATCH")
    support = instrument.get("support")
    cutoff = instrument.get("cutoff")
    if (
        not isinstance(support, int)
        or isinstance(support, bool)
        or support <= 0
        or not isinstance(cutoff, int)
        or isinstance(cutoff, bool)
        or cutoff <= 0
    ):
        raise BankError("BANK_INSTRUMENT_INVALID")
    verified: list[dict[str, Any]] = []
    for ordinal, row in enumerate(members):
        if not isinstance(row, dict) or row.get("ordinal") != ordinal:
            raise BankError("BANK_MEMBER_ORDER_INVALID")
        if row.get("window_id") != expected_ids[ordinal]:
            raise BankError("BANK_MEMBER_POPULATION_MISMATCH")
        if row.get("class") != expected_classes[ordinal]:
            raise BankError("BANK_MEMBER_CLASS_MISMATCH")
        expected_member, expected_sidecar = _member_paths(root, ordinal)
        if row.get("path") != expected_member.relative_to(root).as_posix():
            raise BankError("BANK_MEMBER_PATH_MISMATCH")
        if row.get("sidecar") != expected_sidecar.relative_to(root).as_posix():
            raise BankError("BANK_MEMBER_SIDECAR_PATH_MISMATCH")
        sidecar = root / str(row.get("sidecar", ""))
        if sidecar.is_symlink() or not sidecar.is_file():
            raise BankError("BANK_MEMBER_SIDECAR_MISSING")
        sidecar_value = _sidecar_row(root, sidecar)
        if sidecar_value != row:
            raise BankError("BANK_MEMBER_SIDECAR_MANIFEST_MISMATCH")
        verified.append(
            _verify_member(
                root,
                row,
                expected_build_spec_sha256=str(manifest["build_spec_sha256"]),
                expected_support=support,
                expected_positions=cutoff,
            )
        )
    if manifest.get("total_bytes") != sum(int(row["bytes"]) for row in members):
        raise BankError("BANK_TOTAL_BYTES_MISMATCH")
    expected_bank_id = canonical_sha256(
        {
            "build_spec_sha256": manifest["build_spec_sha256"],
            "members": [row["sha256"] for row in members],
        }
    )
    if manifest.get("bank_id") != expected_bank_id:
        raise BankError("BANK_ID_MISMATCH")
    members_root = root / "members"
    actual_names = {
        path.name
        for path in members_root.iterdir()
        if not path.name.startswith(".")
    }
    if actual_names != _expected_member_names(len(members)):
        raise BankError("BANK_MEMBER_FILE_SET_MISMATCH")
    tree = tree_identity(
        root,
        excluded_names=(
            ".banana-smasher.lock",
            "BANK_COMPLETE",
            "BANK_PROGRESS.json",
            "BANK_RECEIPT.json",
        ),
    )
    return {
        "root": root,
        "bank_id": manifest["bank_id"],
        "build_spec_sha256": manifest["build_spec_sha256"],
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "members": verified,
        "tree_sha256": tree["sha256"],
    }


def build_bank(
    *,
    model_root: str | Path,
    corpus: str | Path,
    windows_manifest: str | Path,
    output: str | Path,
    instrument_profile: str | Path | None = None,
    runner: RealAxisRunner | None = None,
) -> dict[str, Any]:
    started = time.time()
    root = Path(output).resolve()
    population = WindowPopulation(corpus=corpus, manifest=windows_manifest)
    teacher = runner or RealAxisRunner(model_root)
    instrument = load_instrument_profile(instrument_profile)
    spec = _build_spec(teacher, population, instrument)
    spec_sha256 = canonical_sha256(spec)
    generated = 0
    reused = 0
    with output_lock(root):
        try:
            marker = root / "BANK_COMPLETE"
            if marker.exists():
                seal = verify_bank(root, require_complete=True)
                if seal["build_spec_sha256"] != spec_sha256:
                    raise BankSpecMismatch("BANK_BUILD_SPEC_MISMATCH")
                reused = len(population.windows)
                elapsed = time.time() - started
                receipt = {
                    "schema": BANK_RECEIPT_SCHEMA,
                    "status": "COMPLETE",
                    "operation": "bank",
                    "elapsed_seconds": elapsed,
                    "resumed": True,
                    "generated_members": 0,
                    "reused_members": reused,
                    "artifact": {
                        "path": str(root),
                        "bank_manifest_sha256": seal["manifest_sha256"],
                        "tree_sha256": seal["tree_sha256"],
                    },
                }
                receipt_path = atomic_json(root / "BANK_RECEIPT.json", receipt)
                return {
                    **receipt,
                    "receipt": {
                        "path": str(receipt_path),
                        "sha256": sha256_file(receipt_path),
                    },
                }
            progress_path = root / "BANK_PROGRESS.json"
            if progress_path.exists():
                progress = load_json_object(progress_path, label="BANK_PROGRESS")
                if progress.get("build_spec_sha256") != spec_sha256:
                    raise BankSpecMismatch("BANK_BUILD_SPEC_MISMATCH")
            atomic_json(
                progress_path,
                {
                    "schema": "bs-bank-progress-v1",
                    "status": "RUNNING",
                    "build_spec_sha256": spec_sha256,
                    "expected_member_count": len(population.windows),
                    "valid_member_count": 0,
                    "updated_unix": time.time(),
                },
            )
            member_rows: list[dict[str, Any]] = []
            for window in population.windows:
                member_path, sidecar_path = _member_paths(root, window.ordinal)
                existing: dict[str, Any] | None = None
                if sidecar_path.is_file() and member_path.is_file():
                    try:
                        candidate = _sidecar_row(root, sidecar_path)
                        existing = _verify_member(
                            root,
                            candidate,
                            expected_build_spec_sha256=spec_sha256,
                            expected_window=window,
                            expected_support=int(instrument["support"]),
                            expected_positions=int(instrument["cutoff"]),
                        )
                    except (BankError, DurabilityError):
                        existing = None
                if existing is not None:
                    row = {key: value for key, value in existing.items() if key not in {"path_obj", "arrays"}}
                    reused += 1
                else:
                    member_path.unlink(missing_ok=True)
                    sidecar_path.unlink(missing_ok=True)
                    hidden = population.load(window)
                    cutoff = int(instrument["cutoff"])
                    if hidden.shape[0] < cutoff:
                        raise BankError(
                            f"WINDOW_SHORTER_THAN_CUTOFF: {window.window_id!r}: "
                            f"{hidden.shape[0]} < {cutoff}"
                        )
                    initial_hidden = hidden[:cutoff].astype(np.float32, copy=False)
                    logits = teacher.walk(initial_hidden)
                    try:
                        indices, logprob, argmax = teacher_support(
                            logits, support=int(instrument["support"])
                        )
                    except MetricsError as exc:
                        raise BankError(str(exc)) from exc
                    arrays = {
                        "initial_hidden": initial_hidden,
                        "teacher_indices": indices,
                        "teacher_logprob": logprob.astype(np.float32),
                        "teacher_argmax": argmax,
                    }
                    atomic_npz(member_path, arrays)
                    identity = file_identity(member_path, root=root)
                    row = {
                        "schema": BANK_MEMBER_SCHEMA,
                        "build_spec_sha256": spec_sha256,
                        "ordinal": window.ordinal,
                        "window_id": window.window_id,
                        "class": window.class_name,
                        **identity,
                        "sidecar": sidecar_path.relative_to(root).as_posix(),
                        "positions": int(indices.shape[0]),
                        "support": int(indices.shape[1]),
                        "tensors": {
                            name: _tensor_schema(value)
                            for name, value in sorted(arrays.items())
                        },
                    }
                    atomic_json(sidecar_path, row)
                    _verify_member(
                        root,
                        row,
                        expected_build_spec_sha256=spec_sha256,
                        expected_window=window,
                        expected_support=int(instrument["support"]),
                        expected_positions=int(instrument["cutoff"]),
                    )
                    generated += 1
                member_rows.append(row)
                atomic_json(
                    progress_path,
                    {
                        "schema": "bs-bank-progress-v1",
                        "status": "RUNNING",
                        "build_spec_sha256": spec_sha256,
                        "expected_member_count": len(population.windows),
                        "valid_member_count": len(member_rows),
                        "updated_unix": time.time(),
                    },
                )
            actual_names = {
                path.name
                for path in (root / "members").iterdir()
                if not path.name.startswith(".")
            }
            if actual_names != _expected_member_names(len(member_rows)):
                raise BankError("BANK_MEMBER_FILE_SET_MISMATCH")
            teacher.verify_identity_unchanged()
            population.verify_identity_unchanged()
            bank_id = canonical_sha256(
                {
                    "build_spec_sha256": spec_sha256,
                    "members": [row["sha256"] for row in member_rows],
                }
            )
            manifest = {
                "schema": BANK_SCHEMA,
                "schema_version": 1,
                "status": "COMPLETE",
                "bank_id": bank_id,
                "build_spec_sha256": spec_sha256,
                "model": spec["model"],
                "corpus": spec["corpus"],
                "instrument": spec["instrument"],
                "population": {
                    "ordered_window_ids": [window.window_id for window in population.windows],
                    "classes": [window.class_name for window in population.windows],
                    "ordered_window_ids_sha256": population.ordered_window_ids_sha256,
                },
                "members": member_rows,
                "member_count": len(member_rows),
                "total_bytes": sum(int(row["bytes"]) for row in member_rows),
            }
            manifest_path = atomic_json(root / "bank.json", manifest)
            marker_value = {
                "schema": BANK_MARKER_SCHEMA,
                "status": "COMPLETE",
                "bank_id": bank_id,
                "build_spec_sha256": spec_sha256,
                "bank_manifest_sha256": sha256_file(manifest_path),
                "member_count": len(member_rows),
            }
            atomic_json(marker, marker_value)
            seal = verify_bank(root, require_complete=True)
            elapsed = time.time() - started
            receipt = {
                "schema": BANK_RECEIPT_SCHEMA,
                "status": "COMPLETE",
                "operation": "bank",
                "elapsed_seconds": elapsed,
                "resumed": reused > 0,
                "generated_members": generated,
                "reused_members": reused,
                "artifact": {
                    "path": str(root),
                    "bank_manifest_sha256": seal["manifest_sha256"],
                    "tree_sha256": seal["tree_sha256"],
                },
            }
            receipt_path = atomic_json(root / "BANK_RECEIPT.json", receipt)
            atomic_json(
                progress_path,
                {
                    "schema": "bs-bank-progress-v1",
                    "status": "COMPLETE",
                    "build_spec_sha256": spec_sha256,
                    "expected_member_count": len(population.windows),
                    "valid_member_count": len(population.windows),
                    "updated_unix": time.time(),
                },
            )
            return {
                **receipt,
                "receipt": {
                    "path": str(receipt_path),
                    "sha256": sha256_file(receipt_path),
                },
            }
        except BankSpecMismatch:
            raise
        except Exception:
            (root / "BANK_COMPLETE").unlink(missing_ok=True)
            atomic_json(
                root / "BANK_PROGRESS.json",
                {
                    "schema": "bs-bank-progress-v1",
                    "status": "FAILED",
                    "build_spec_sha256": spec_sha256,
                    "expected_member_count": len(population.windows),
                    "valid_member_count": generated + reused,
                    "updated_unix": time.time(),
                },
            )
            raise
