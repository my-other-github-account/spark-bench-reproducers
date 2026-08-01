from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .bank import verify_bank
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
from .metrics import MetricsError, aggregate_windows, paired_summary, score_candidate
from .real_axis import RealAxisRunner

EVALUATION_SCHEMA = "bs-paired-real-axis-evaluation-v1"
EVALUATION_MARKER_SCHEMA = "bs-evaluation-complete-v1"
CHECKPOINT_MANIFEST_SCHEMA = "bs-real-axis-arm-checkpoint-v1"
PAIR_MARKER_SCHEMA = "bs-real-axis-pair-checkpoint-v1"
ARM_MANIFEST_SCHEMA = "bs-real-axis-arm-artifacts-v1"


class EvaluationError(ValueError):
    """Raised when paired real-axis evaluation cannot complete fail-closed."""


class EvaluationSpecMismatch(EvaluationError):
    """Raised without mutating an output owned by a different evaluation spec."""


def _state_path(root: Path, layer: int, arm: str, ordinal: int) -> Path:
    return root / "checkpoints" / f"layer_{layer:03d}" / arm / f"window_{ordinal:06d}.npz"


def _checkpoint_manifest_path(root: Path, layer: int, arm: str) -> Path:
    return root / "checkpoints" / f"layer_{layer:03d}" / arm / "manifest.json"


def _pair_marker_path(root: Path, layer: int) -> Path:
    return root / "checkpoints" / f"layer_{layer:03d}" / "PAIR_COMPLETE"


def _tensor_schema(value: np.ndarray[Any, Any]) -> dict[str, Any]:
    return {"dtype": value.dtype.str, "shape": list(value.shape)}


def _confined_root_file(root: Path, relative: str, *, label: str) -> Path:
    try:
        return safe_relative_path(root, relative, label=label)
    except DurabilityError as exc:
        raise EvaluationError(str(exc)) from exc


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_layer_descriptor(value: Any, *, layer: int) -> None:
    if not isinstance(value, dict) or set(value) != {
        "layer",
        "activation",
        "descriptor",
        "weight",
        "bias",
        "sha256",
    }:
        raise EvaluationError("EVALUATION_TOPOLOGY_MISMATCH")
    descriptor = {key: item for key, item in value.items() if key != "sha256"}
    if (
        value.get("layer") != layer
        or value.get("activation") not in ("identity", "relu", "tanh")
        or not isinstance(value.get("descriptor"), dict)
        or not value["descriptor"]
        or not isinstance(value.get("weight"), dict)
        or (value.get("bias") is not None and not isinstance(value.get("bias"), dict))
        or canonical_sha256(descriptor) != value.get("sha256")
    ):
        raise EvaluationError("EVALUATION_TOPOLOGY_MISMATCH")


def _evaluation_spec(
    *,
    seal: dict[str, Any],
    candidate: RealAxisRunner,
    reference: RealAxisRunner,
) -> dict[str, Any]:
    return {
        "schema": "bs-paired-real-axis-evaluation-spec-v1",
        "mode": "paired_real_axis",
        "bank": {
            "bank_id": seal["bank_id"],
            "manifest_sha256": seal["manifest_sha256"],
            "build_spec_sha256": seal["build_spec_sha256"],
        },
        "candidate": candidate.identity(),
        "reference": reference.identity(),
        "population": seal["manifest"]["population"],
        "instrument": seal["manifest"]["instrument"],
    }


def _write_arm_checkpoint(
    *,
    root: Path,
    layer: int,
    arm: str,
    states: list[np.ndarray[Any, Any]],
    bank_members: list[dict[str, Any]],
    spec_sha256: str,
    input_checkpoint_sha256: str | None,
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    for ordinal, (state, bank_member) in enumerate(
        zip(states, bank_members, strict=True)
    ):
        if state.ndim != 2 or not np.issubdtype(state.dtype, np.floating):
            raise EvaluationError("CHECKPOINT_STATE_INVALID")
        if not np.isfinite(state).all():
            raise EvaluationError("CHECKPOINT_STATE_NONFINITE")
        path = _state_path(root, layer, arm, ordinal)
        atomic_npz(path, {"hidden": state})
        members.append(
            {
                "ordinal": ordinal,
                "window_id": bank_member["window_id"],
                **file_identity(path, root=root),
                "tensor": _tensor_schema(state),
            }
        )
    manifest = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "status": "COMPLETE",
        "evaluation_spec_sha256": spec_sha256,
        "arm": arm,
        "completed_layer": layer,
        "input_checkpoint_sha256": input_checkpoint_sha256,
        "layer_descriptor": descriptor,
        "members": members,
        "member_count": len(members),
    }
    path = atomic_json(_checkpoint_manifest_path(root, layer, arm), manifest)
    return {"manifest": manifest, "path": path, "sha256": sha256_file(path)}


def _verify_arm_checkpoint(
    *,
    root: Path,
    layer: int,
    arm: str,
    bank_members: list[dict[str, Any]],
    spec_sha256: str,
    expected_input_sha256: str | None,
) -> dict[str, Any]:
    manifest_path = _checkpoint_manifest_path(root, layer, arm)
    manifest = load_json_object(manifest_path, label="CHECKPOINT_MANIFEST")
    if (
        manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA
        or manifest.get("status") != "COMPLETE"
        or manifest.get("evaluation_spec_sha256") != spec_sha256
        or manifest.get("arm") != arm
        or manifest.get("completed_layer") != layer
        or manifest.get("input_checkpoint_sha256") != expected_input_sha256
    ):
        raise EvaluationError("CHECKPOINT_MANIFEST_MISMATCH")
    rows = manifest.get("members")
    if not isinstance(rows, list) or len(rows) != len(bank_members):
        raise EvaluationError("CHECKPOINT_MEMBER_COUNT_MISMATCH")
    states: list[np.ndarray[Any, Any]] = []
    for ordinal, (row, bank_member) in enumerate(zip(rows, bank_members, strict=True)):
        if (
            not isinstance(row, dict)
            or row.get("ordinal") != ordinal
            or row.get("window_id") != bank_member["window_id"]
        ):
            raise EvaluationError("CHECKPOINT_POPULATION_MISMATCH")
        try:
            path = safe_relative_path(root, str(row.get("path", "")), label="CHECKPOINT")
        except DurabilityError as exc:
            raise EvaluationError(str(exc)) from exc
        if path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get(
            "sha256"
        ):
            raise EvaluationError("CHECKPOINT_IDENTITY_MISMATCH")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if archive.files != ["hidden"]:
                    raise EvaluationError("CHECKPOINT_FIELDS_MISMATCH")
                state = np.asarray(archive["hidden"])
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"CHECKPOINT_ARCHIVE_INVALID: {path}: {exc}") from exc
        if _tensor_schema(state) != row.get("tensor"):
            raise EvaluationError("CHECKPOINT_TENSOR_SCHEMA_MISMATCH")
        if state.ndim != 2 or not np.issubdtype(state.dtype, np.floating):
            raise EvaluationError("CHECKPOINT_STATE_INVALID")
        if not np.isfinite(state).all():
            raise EvaluationError("CHECKPOINT_STATE_NONFINITE")
        states.append(state)
    actual = {
        path.name
        for path in manifest_path.parent.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    expected = {"manifest.json"} | {f"window_{index:06d}.npz" for index in range(len(rows))}
    if actual != expected:
        raise EvaluationError("CHECKPOINT_FILE_SET_MISMATCH")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "states": states,
    }


def _verify_pair_checkpoint(
    *,
    root: Path,
    layer: int,
    bank_members: list[dict[str, Any]],
    spec_sha256: str,
    expected_input_sha256: str | None,
) -> dict[str, Any]:
    marker_path = _pair_marker_path(root, layer)
    marker = load_json_object(marker_path, label="PAIR_CHECKPOINT_MARKER")
    if (
        marker.get("schema") != PAIR_MARKER_SCHEMA
        or marker.get("status") != "COMPLETE"
        or marker.get("evaluation_spec_sha256") != spec_sha256
        or marker.get("completed_layer") != layer
        or marker.get("input_pair_checkpoint_sha256") != expected_input_sha256
    ):
        raise EvaluationError("PAIR_CHECKPOINT_MARKER_MISMATCH")
    candidate = _verify_arm_checkpoint(
        root=root,
        layer=layer,
        arm="candidate",
        bank_members=bank_members,
        spec_sha256=spec_sha256,
        expected_input_sha256=expected_input_sha256,
    )
    reference = _verify_arm_checkpoint(
        root=root,
        layer=layer,
        arm="reference",
        bank_members=bank_members,
        spec_sha256=spec_sha256,
        expected_input_sha256=expected_input_sha256,
    )
    if marker.get("candidate_manifest_sha256") != candidate["manifest_sha256"]:
        raise EvaluationError("PAIR_CANDIDATE_MANIFEST_MISMATCH")
    if marker.get("reference_manifest_sha256") != reference["manifest_sha256"]:
        raise EvaluationError("PAIR_REFERENCE_MANIFEST_MISMATCH")
    marker_sha256 = sha256_file(marker_path)
    return {
        "candidate_states": candidate["states"],
        "reference_states": reference["states"],
        "marker_sha256": marker_sha256,
        "marker": marker,
    }


def _greatest_common_checkpoint(
    *,
    root: Path,
    layer_count: int,
    bank_members: list[dict[str, Any]],
    spec_sha256: str,
) -> tuple[int, dict[str, Any] | None]:
    previous_sha256: str | None = None
    last: dict[str, Any] | None = None
    for layer in range(layer_count):
        try:
            checkpoint = _verify_pair_checkpoint(
                root=root,
                layer=layer,
                bank_members=bank_members,
                spec_sha256=spec_sha256,
                expected_input_sha256=previous_sha256,
            )
        except (EvaluationError, DurabilityError, OSError, ValueError):
            break
        previous_sha256 = checkpoint["marker_sha256"]
        last = checkpoint
    return (0, None) if last is None else (int(last["marker"]["completed_layer"]) + 1, last)


def _write_arm_artifacts(
    *,
    root: Path,
    arm: str,
    runner: RealAxisRunner,
    states: list[np.ndarray[Any, Any]],
    bank_members: list[dict[str, Any]],
    spec_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    arm_root = root / "arms" / arm
    logits_batch = runner.project_logits_batch(states)
    for ordinal, (logits, bank_member) in enumerate(
        zip(logits_batch, bank_members, strict=True)
    ):
        teacher = bank_member["arrays"]
        try:
            scored = score_candidate(
                logits,
                teacher_indices=teacher["teacher_indices"],
                teacher_logprob=teacher["teacher_logprob"],
                teacher_argmax=teacher["teacher_argmax"],
            )
        except MetricsError as exc:
            raise EvaluationError(str(exc)) from exc
        path = arm_root / f"window_{ordinal:06d}.npz"
        artifact_arrays = {
            **scored,
            "teacher_logprob": np.asarray(teacher["teacher_logprob"]),
            "teacher_argmax": np.asarray(teacher["teacher_argmax"]),
        }
        atomic_npz(path, artifact_arrays)
        rows.append(
            {
                "ordinal": ordinal,
                "window_id": bank_member["window_id"],
                "class": bank_member["class"],
                "positions": int(scored["kld"].shape[0]),
                **file_identity(path, root=root),
                "tensors": {
                    name: _tensor_schema(value)
                    for name, value in sorted(artifact_arrays.items())
                },
            }
        )
        metric_rows.append(
            {
                "window_id": bank_member["window_id"],
                "class": bank_member["class"],
                "kld": scored["kld"],
                "top1_equal": scored["top1_equal"],
            }
        )
    manifest = {
        "schema": ARM_MANIFEST_SCHEMA,
        "status": "COMPLETE",
        "evaluation_spec_sha256": spec_sha256,
        "arm": arm,
        "artifact_semantics": (
            "candidate log-probability gathered at teacher support, candidate argmax, "
            "per-position KL(teacher||candidate), and teacher/candidate top-1 parity"
        ),
        "members": rows,
        "member_count": len(rows),
    }
    path = atomic_json(arm_root / "manifest.json", manifest)
    summary = aggregate_windows(metric_rows)
    return (
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "members": len(rows),
        },
        summary,
    )


def _verify_arm_artifacts(
    root: Path, row: dict[str, Any], spec_sha256: str, *, expected_arm: str
) -> dict[str, Any]:
    try:
        path = safe_relative_path(root, str(row.get("path", "")), label="ARM_MANIFEST")
    except DurabilityError as exc:
        raise EvaluationError(str(exc)) from exc
    if sha256_file(path) != row.get("sha256"):
        raise EvaluationError("ARM_MANIFEST_SHA256_MISMATCH")
    manifest = load_json_object(path, label="ARM_MANIFEST")
    expected_members = row.get("members")
    if (
        manifest.get("schema") != ARM_MANIFEST_SCHEMA
        or manifest.get("status") != "COMPLETE"
        or manifest.get("evaluation_spec_sha256") != spec_sha256
        or manifest.get("arm") != expected_arm
        or not isinstance(expected_members, int)
        or isinstance(expected_members, bool)
        or expected_members <= 0
        or manifest.get("member_count") != expected_members
    ):
        raise EvaluationError("ARM_MANIFEST_INVALID")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != expected_members:
        raise EvaluationError("ARM_MEMBER_COUNT_MISMATCH")
    metric_rows: list[dict[str, Any]] = []
    support_widths: list[int] = []
    for ordinal, member in enumerate(members):
        positions = member.get("positions") if isinstance(member, dict) else None
        if (
            not isinstance(member, dict)
            or member.get("ordinal") != ordinal
            or not isinstance(member.get("window_id"), (str, int))
            or isinstance(member.get("window_id"), bool)
            or not isinstance(member.get("class"), str)
            or not member.get("class")
            or not isinstance(positions, int)
            or isinstance(positions, bool)
            or positions <= 0
        ):
            raise EvaluationError("ARM_MEMBER_INVALID")
        expected_path = f"arms/{expected_arm}/window_{ordinal:06d}.npz"
        if member.get("path") != expected_path:
            raise EvaluationError("ARM_MEMBER_PATH_MISMATCH")
        artifact = safe_relative_path(root, str(member.get("path", "")), label="ARM_MEMBER")
        if artifact.stat().st_size != member.get("bytes") or sha256_file(
            artifact
        ) != member.get("sha256"):
            raise EvaluationError("ARM_MEMBER_IDENTITY_MISMATCH")
        try:
            with np.load(artifact, allow_pickle=False) as archive:
                if set(archive.files) != {
                    "candidate_logprob",
                    "candidate_argmax",
                    "kld",
                    "top1_equal",
                    "teacher_logprob",
                    "teacher_argmax",
                }:
                    raise EvaluationError("ARM_MEMBER_FIELDS_MISMATCH")
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
                tensors = {
                    name: _tensor_schema(arrays[name]) for name in sorted(arrays)
                }
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"ARM_MEMBER_ARCHIVE_INVALID: {artifact}: {exc}") from exc
        if tensors != member.get("tensors"):
            raise EvaluationError("ARM_MEMBER_TENSOR_SCHEMA_MISMATCH")
        logprob = arrays["candidate_logprob"]
        argmax = arrays["candidate_argmax"]
        kld = arrays["kld"]
        top1 = arrays["top1_equal"]
        teacher_logprob = arrays["teacher_logprob"]
        teacher_argmax = arrays["teacher_argmax"]
        if (
            logprob.ndim != 2
            or not np.issubdtype(logprob.dtype, np.floating)
            or not np.isfinite(logprob).all()
            or argmax.shape != (logprob.shape[0],)
            or not np.issubdtype(argmax.dtype, np.integer)
            or kld.shape != (logprob.shape[0],)
            or not np.issubdtype(kld.dtype, np.floating)
            or not np.isfinite(kld).all()
            or np.any(kld < 0)
            or top1.shape != (logprob.shape[0],)
            or not np.issubdtype(top1.dtype, np.integer)
            or not np.isin(top1, (0, 1)).all()
            or positions != logprob.shape[0]
            or teacher_logprob.shape != logprob.shape
            or not np.issubdtype(teacher_logprob.dtype, np.floating)
            or not np.isfinite(teacher_logprob).all()
            or teacher_argmax.shape != argmax.shape
            or not np.issubdtype(teacher_argmax.dtype, np.integer)
        ):
            raise EvaluationError("ARM_MEMBER_SEMANTICS_INVALID")
        teacher_support_lp = teacher_logprob.astype(np.float64, copy=False)
        teacher_support_lp -= np.log(
            np.sum(np.exp(teacher_support_lp), axis=1, keepdims=True)
        )
        candidate_support_lp = logprob.astype(np.float64, copy=False)
        candidate_support_lp -= np.log(
            np.sum(np.exp(candidate_support_lp), axis=1, keepdims=True)
        )
        expected_kld = np.maximum(
            np.sum(
                np.exp(teacher_support_lp)
                * (teacher_support_lp - candidate_support_lp),
                axis=1,
            ),
            0.0,
        )
        if not np.allclose(kld, expected_kld, rtol=1e-12, atol=1e-12):
            raise EvaluationError("ARM_MEMBER_KLD_MISMATCH")
        if not np.array_equal(top1, (argmax == teacher_argmax).astype(np.uint8)):
            raise EvaluationError("ARM_MEMBER_TOP1_MISMATCH")
        metric_rows.append(
            {
                "window_id": member["window_id"],
                "class": member["class"],
                "kld": kld,
                "top1_equal": top1,
            }
        )
        support_widths.append(int(logprob.shape[1]))
    actual_files = {
        path.name
        for path in path.parent.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    expected_files = {"manifest.json"} | {
        f"window_{ordinal:06d}.npz" for ordinal in range(len(members))
    }
    if actual_files != expected_files:
        raise EvaluationError("ARM_FILE_SET_MISMATCH")
    try:
        metrics = aggregate_windows(metric_rows)
    except MetricsError as exc:
        raise EvaluationError(str(exc)) from exc
    return {
        "manifest": manifest,
        "metrics": metrics,
        "support_widths": support_widths,
    }


def verify_evaluation(
    output: str | Path, *, expected_spec_sha256: str | None = None
) -> dict[str, Any]:
    root = Path(output).resolve()
    marker_path = _confined_root_file(
        root, "EVALUATION_COMPLETE", label="EVALUATION_MARKER"
    )
    marker = load_json_object(marker_path, label="EVALUATION_MARKER")
    receipt_path = _confined_root_file(root, "evaluation.json", label="EVALUATION_RECEIPT")
    receipt = load_json_object(receipt_path, label="EVALUATION_RECEIPT")
    if (
        set(marker)
        != {
            "schema",
            "status",
            "evaluation_id",
            "evaluation_spec_sha256",
            "evaluation_sha256",
        }
        or set(receipt)
        != {
            "schema",
            "schema_version",
            "status",
            "operation",
            "evaluation_id",
            "evaluation_spec_sha256",
            "mode",
            "bank",
            "population",
            "instrument",
            "topology",
            "arms",
            "paired",
            "performance",
            "resume",
            "elapsed_seconds",
            "artifacts",
        }
        or marker.get("schema") != EVALUATION_MARKER_SCHEMA
        or marker.get("status") != "COMPLETE"
        or receipt.get("schema") != EVALUATION_SCHEMA
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "COMPLETE"
        or receipt.get("operation") != "evaluate"
        or receipt.get("mode") != "paired_real_axis"
        or marker.get("evaluation_id") != receipt.get("evaluation_id")
        or marker.get("evaluation_spec_sha256")
        != receipt.get("evaluation_spec_sha256")
        or marker.get("evaluation_sha256") != sha256_file(receipt_path)
    ):
        raise EvaluationError("EVALUATION_COMPLETION_INVALID")
    if expected_spec_sha256 is not None and receipt.get(
        "evaluation_spec_sha256"
    ) != expected_spec_sha256:
        raise EvaluationSpecMismatch("EVALUATION_SPEC_MISMATCH")
    receipt_arms = receipt.get("arms")
    if not isinstance(receipt_arms, dict) or set(receipt_arms) != {
        "candidate",
        "reference",
    }:
        raise EvaluationError("EVALUATION_ARMS_INVALID")
    verified_arms: dict[str, dict[str, Any]] = {}
    for arm in ("candidate", "reference"):
        arm_receipt = receipt_arms.get(arm)
        if not isinstance(arm_receipt, dict):
            raise EvaluationError("EVALUATION_ARM_INVALID")
        arm_row = arm_receipt.get("logits_manifest")
        if not isinstance(arm_row, dict):
            raise EvaluationError("EVALUATION_ARM_MANIFEST_MISSING")
        if arm_receipt.get("kld_manifest") != arm_row:
            raise EvaluationError("EVALUATION_KLD_MANIFEST_MISMATCH")
        verified = _verify_arm_artifacts(
            root,
            arm_row,
            str(receipt["evaluation_spec_sha256"]),
            expected_arm=arm,
        )
        verified_arms[arm] = verified
    candidate_members = verified_arms["candidate"]["manifest"]["members"]
    reference_members = verified_arms["reference"]["manifest"]["members"]
    candidate_ids = [row["window_id"] for row in candidate_members]
    reference_ids = [row["window_id"] for row in reference_members]
    candidate_classes = [row["class"] for row in candidate_members]
    reference_classes = [row["class"] for row in reference_members]
    population = receipt.get("population")
    if not isinstance(population, dict):
        raise EvaluationError("EVALUATION_POPULATION_INVALID")
    if (
        candidate_ids != reference_ids
        or candidate_classes != reference_classes
        or canonical_sha256(candidate_ids) != population.get("ordered_window_ids_sha256")
        or len(candidate_ids) != population.get("windows")
        or sum(int(row["positions"]) for row in candidate_members)
        != population.get("positions")
        or [row["positions"] for row in candidate_members]
        != [row["positions"] for row in reference_members]
        or dict(sorted(Counter(candidate_classes).items())) != population.get("classes")
    ):
        raise EvaluationError("ARM_POPULATION_MISMATCH")
    for arm in ("candidate", "reference"):
        metrics = verified_arms[arm]["metrics"]
        arm_receipt = receipt_arms[arm]
        if (
            arm_receipt.get("kld") != metrics["kld"]
            or arm_receipt.get("top1_parity") != metrics["top1_parity"]
            or arm_receipt.get("per_window") != metrics["per_window"]
        ):
            raise EvaluationError("EVALUATION_METRICS_MISMATCH")
    instrument = receipt.get("instrument")
    instrument_keys = {
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
    }
    if (
        not isinstance(instrument, dict)
        or set(instrument) != instrument_keys
        or instrument.get("schema") != "bs-real-axis-instrument-v1"
        or instrument.get("schema_version") != 1
        or instrument.get("teacher_storage") != "top_support_logprob"
        or instrument.get("direction") != "kl_teacher_candidate"
        or not _is_sha256(instrument.get("profile_sha256"))
        or any(
            width != instrument.get("support")
            for arm in verified_arms.values()
            for width in arm["support_widths"]
        )
        or any(row["positions"] != instrument.get("cutoff") for row in candidate_members)
    ):
        raise EvaluationError("EVALUATION_INSTRUMENT_MISMATCH")
    topology = receipt.get("topology")
    topology_rows = topology.get("layers") if isinstance(topology, dict) else None
    layer_count = topology.get("layer_count") if isinstance(topology, dict) else None
    if (
        not isinstance(layer_count, int)
        or isinstance(layer_count, bool)
        or layer_count <= 0
        or not isinstance(topology_rows, list)
        or len(topology_rows) != layer_count
    ):
        raise EvaluationError("EVALUATION_TOPOLOGY_MISMATCH")
    for layer, topology_row in enumerate(topology_rows):
        if (
            not isinstance(topology_row, dict)
            or set(topology_row) != {"layer", "candidate", "reference"}
            or topology_row.get("layer") != layer
        ):
            raise EvaluationError("EVALUATION_TOPOLOGY_MISMATCH")
        _verify_layer_descriptor(topology_row.get("candidate"), layer=layer)
        _verify_layer_descriptor(topology_row.get("reference"), layer=layer)
    artifact_keys = {
        "model_id",
        "real_axis_manifest_sha256",
        "layer_count",
        "pack_manifest_sha256",
        "pack_instance_id",
    }
    for arm in ("candidate", "reference"):
        artifact = receipt_arms[arm].get("artifact")
        if (
            not isinstance(artifact, dict)
            or set(artifact) != artifact_keys
            or artifact.get("layer_count") != layer_count
            or not isinstance(artifact.get("model_id"), str)
            or not artifact["model_id"]
            or not isinstance(artifact.get("pack_instance_id"), str)
            or not artifact["pack_instance_id"]
            or not _is_sha256(artifact.get("real_axis_manifest_sha256"))
            or not _is_sha256(artifact.get("pack_manifest_sha256"))
        ):
            raise EvaluationError("EVALUATION_ARM_IDENTITY_MISMATCH")
    bank = receipt.get("bank")
    if (
        not isinstance(bank, dict)
        or set(bank)
        != {"bank_id", "manifest_sha256", "build_spec_sha256", "tree_sha256"}
        or not all(_is_sha256(value) for value in bank.values())
    ):
        raise EvaluationError("EVALUATION_SPEC_MISMATCH")
    reconstructed_spec = {
        "schema": "bs-paired-real-axis-evaluation-spec-v1",
        "mode": "paired_real_axis",
        "bank": {
            "bank_id": bank["bank_id"],
            "manifest_sha256": bank["manifest_sha256"],
            "build_spec_sha256": bank["build_spec_sha256"],
        },
        "candidate": receipt_arms["candidate"]["artifact"],
        "reference": receipt_arms["reference"]["artifact"],
        "population": {
            "ordered_window_ids": candidate_ids,
            "classes": candidate_classes,
            "ordered_window_ids_sha256": population["ordered_window_ids_sha256"],
            "ordered_classes_sha256": canonical_sha256(candidate_classes),
        },
        "instrument": instrument,
    }
    if canonical_sha256(reconstructed_spec) != receipt.get("evaluation_spec_sha256"):
        raise EvaluationError("EVALUATION_SPEC_MISMATCH")
    expected_paired = paired_summary(
        verified_arms["candidate"]["metrics"],
        verified_arms["reference"]["metrics"],
    )
    if receipt.get("paired") != expected_paired:
        raise EvaluationError("EVALUATION_METRICS_MISMATCH")
    performance = receipt.get("performance")
    elapsed_seconds = receipt.get("elapsed_seconds")
    resume = receipt.get("resume")
    expected_quality = {
        "status": "PASS",
        "candidate_global_kld": verified_arms["candidate"]["metrics"]["kld"][
            "global"
        ],
        "reference_global_kld": verified_arms["reference"]["metrics"]["kld"][
            "global"
        ],
        "paired_mean_window_delta": expected_paired["mean_window_delta"],
    }
    if (
        not isinstance(performance, dict)
        or set(performance)
        != {
            "tokens_per_second",
            "wall_seconds",
            "peak_vram_bytes",
            "quality_result",
            "kernel",
            "fallback_used",
            "fallback_status",
            "window_batch_size",
            "layer_forwards_per_arm",
            "head_forwards_per_arm",
        }
        or not isinstance(elapsed_seconds, (int, float))
        or isinstance(elapsed_seconds, bool)
        or not np.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0
        or performance.get("wall_seconds") != elapsed_seconds
        or performance.get("tokens_per_second")
        != (int(population["positions"]) * 2) / elapsed_seconds
        or performance.get("peak_vram_bytes") != 0
        or isinstance(performance.get("peak_vram_bytes"), bool)
        or performance.get("quality_result") != expected_quality
        or performance.get("kernel") != "numpy-concatenated-all-window-gemm"
        or performance.get("fallback_used") is not False
        or performance.get("fallback_status") != "none"
        or performance.get("window_batch_size") != population.get("windows")
        or not isinstance(resume, dict)
        or set(resume) != {"started_from_layer", "checkpoint_sha256"}
        or not isinstance(resume.get("started_from_layer"), int)
        or isinstance(resume.get("started_from_layer"), bool)
        or not 0 <= resume["started_from_layer"] <= layer_count
        or performance.get("layer_forwards_per_arm")
        != layer_count - resume["started_from_layer"]
        or performance.get("head_forwards_per_arm") != 1
    ):
        raise EvaluationError("EVALUATION_PERFORMANCE_INVALID")
    expected_evaluation_id = canonical_sha256(
        {
            "evaluation_spec_sha256": receipt["evaluation_spec_sha256"],
            "candidate_manifest_sha256": receipt_arms["candidate"][
                "logits_manifest"
            ]["sha256"],
            "reference_manifest_sha256": receipt_arms["reference"][
                "logits_manifest"
            ]["sha256"],
        }
    )
    if receipt.get("evaluation_id") != expected_evaluation_id:
        raise EvaluationError("EVALUATION_ID_MISMATCH")
    tree = tree_identity(
        root,
        excluded_names=(
            ".banana-smasher.lock",
            "EVALUATION_COMPLETE",
            "EVALUATION_PROGRESS.json",
            "evaluation.json",
        ),
    )
    if receipt.get("artifacts", {}).get("tree_sha256") != tree["sha256"]:
        raise EvaluationError("EVALUATION_TREE_SHA256_MISMATCH")
    return {
        "root": root,
        "receipt": receipt,
        "receipt_sha256": sha256_file(receipt_path),
        "tree_sha256": tree["sha256"],
    }


def _slim_result(seal: dict[str, Any], *, verbose: bool) -> dict[str, Any]:
    receipt = seal["receipt"]
    result = {
        "schema": EVALUATION_SCHEMA,
        "status": "COMPLETE",
        "operation": "evaluate",
        "mode": "paired_real_axis",
        "elapsed_seconds": receipt["elapsed_seconds"],
        "performance": receipt["performance"],
        "resume": receipt["resume"],
        "arms": {
            arm: {
                "global_kld": receipt["arms"][arm]["kld"]["global"],
                "top1_rate": receipt["arms"][arm]["top1_parity"]["rate"],
            }
            for arm in ("candidate", "reference")
        },
        "paired": receipt["paired"],
        "artifact": {
            "path": str(seal["root"]),
            "evaluation_sha256": seal["receipt_sha256"],
            "tree_sha256": seal["tree_sha256"],
        },
        "receipt": {
            "path": str(seal["root"] / "evaluation.json"),
            "sha256": seal["receipt_sha256"],
        },
    }
    if verbose:
        result["evaluation"] = receipt
    return result


def evaluate_paired(
    *,
    model_root: str | Path,
    candidate: str | Path,
    reference: str | Path,
    bank: str | Path,
    output: str | Path,
    resume_from_layer: int | None = None,
    verbose_receipts: bool = False,
    layer_hook: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    started = time.time()
    root = Path(output).resolve()
    bank_seal = verify_bank(bank, require_complete=True)
    native_runner = RealAxisRunner(model_root)
    if native_runner.identity() != bank_seal["manifest"].get("model"):
        raise EvaluationError("BANK_MODEL_IDENTITY_MISMATCH")
    candidate_runner = RealAxisRunner(candidate, require_pack=True)
    reference_runner = RealAxisRunner(reference, require_pack=True)
    if candidate_runner.layer_count != reference_runner.layer_count:
        raise EvaluationError("PAIRED_LAYER_COUNT_MISMATCH")
    bank_model = bank_seal["manifest"].get("model", {})
    if (
        candidate_runner.model_id != bank_model.get("model_id")
        or reference_runner.model_id != bank_model.get("model_id")
        or candidate_runner.layer_count != bank_model.get("layer_count")
    ):
        raise EvaluationError("PAIRED_BANK_MODEL_TOPOLOGY_MISMATCH")
    layer_count = candidate_runner.layer_count
    if resume_from_layer is not None and (
        not isinstance(resume_from_layer, int)
        or resume_from_layer < 0
        or resume_from_layer > layer_count
    ):
        raise EvaluationError("RESUME_LAYER_OUT_OF_RANGE")
    spec = _evaluation_spec(
        seal=bank_seal, candidate=candidate_runner, reference=reference_runner
    )
    spec_sha256 = canonical_sha256(spec)
    bank_members = bank_seal["members"]
    with output_lock(root):
        try:
            marker_path = root / "EVALUATION_COMPLETE"
            if marker_path.exists():
                seal = verify_evaluation(root, expected_spec_sha256=spec_sha256)
                return _slim_result(seal, verbose=verbose_receipts)
            progress_path = root / "EVALUATION_PROGRESS.json"
            if progress_path.exists():
                progress = load_json_object(progress_path, label="EVALUATION_PROGRESS")
                if progress.get("evaluation_spec_sha256") != spec_sha256:
                    raise EvaluationSpecMismatch("EVALUATION_SPEC_MISMATCH")
            atomic_json(
                progress_path,
                {
                    "schema": "bs-paired-evaluation-progress-v1",
                    "status": "RUNNING",
                    "evaluation_spec_sha256": spec_sha256,
                    "updated_unix": time.time(),
                },
            )
            if resume_from_layer is None:
                start_layer, checkpoint = _greatest_common_checkpoint(
                    root=root,
                    layer_count=layer_count,
                    bank_members=bank_members,
                    spec_sha256=spec_sha256,
                )
            elif resume_from_layer == 0:
                start_layer, checkpoint = 0, None
            else:
                previous_sha256: str | None = None
                checkpoint = None
                for layer in range(resume_from_layer):
                    checkpoint = _verify_pair_checkpoint(
                        root=root,
                        layer=layer,
                        bank_members=bank_members,
                        spec_sha256=spec_sha256,
                        expected_input_sha256=previous_sha256,
                    )
                    previous_sha256 = checkpoint["marker_sha256"]
                start_layer = resume_from_layer
            if checkpoint is None:
                candidate_states = [
                    np.asarray(member["arrays"]["initial_hidden"]).copy()
                    for member in bank_members
                ]
                reference_states = [state.copy() for state in candidate_states]
                previous_pair_sha256: str | None = None
            else:
                candidate_states = checkpoint["candidate_states"]
                reference_states = checkpoint["reference_states"]
                previous_pair_sha256 = checkpoint["marker_sha256"]
            topology_rows: list[dict[str, Any]] = []
            # Record every layer's independent descriptor, including divergent layers.
            for layer in range(layer_count):
                topology_rows.append(
                    {
                        "layer": layer,
                        "candidate": candidate_runner.layer_descriptor(layer),
                        "reference": reference_runner.layer_descriptor(layer),
                    }
                )
            for layer in range(start_layer, layer_count):
                candidate_states = candidate_runner.apply_layer_batch(
                    layer, candidate_states
                )
                reference_states = reference_runner.apply_layer_batch(
                    layer, reference_states
                )
                candidate_checkpoint = _write_arm_checkpoint(
                    root=root,
                    layer=layer,
                    arm="candidate",
                    states=candidate_states,
                    bank_members=bank_members,
                    spec_sha256=spec_sha256,
                    input_checkpoint_sha256=previous_pair_sha256,
                    descriptor=candidate_runner.layer_descriptor(layer),
                )
                reference_checkpoint = _write_arm_checkpoint(
                    root=root,
                    layer=layer,
                    arm="reference",
                    states=reference_states,
                    bank_members=bank_members,
                    spec_sha256=spec_sha256,
                    input_checkpoint_sha256=previous_pair_sha256,
                    descriptor=reference_runner.layer_descriptor(layer),
                )
                pair_marker = {
                    "schema": PAIR_MARKER_SCHEMA,
                    "status": "COMPLETE",
                    "evaluation_spec_sha256": spec_sha256,
                    "completed_layer": layer,
                    "input_pair_checkpoint_sha256": previous_pair_sha256,
                    "candidate_manifest_sha256": candidate_checkpoint["sha256"],
                    "reference_manifest_sha256": reference_checkpoint["sha256"],
                }
                pair_path = atomic_json(_pair_marker_path(root, layer), pair_marker)
                previous_pair_sha256 = sha256_file(pair_path)
                if layer_hook is not None:
                    layer_hook(layer)
            candidate_manifest, candidate_metrics = _write_arm_artifacts(
                root=root,
                arm="candidate",
                runner=candidate_runner,
                states=candidate_states,
                bank_members=bank_members,
                spec_sha256=spec_sha256,
            )
            reference_manifest, reference_metrics = _write_arm_artifacts(
                root=root,
                arm="reference",
                runner=reference_runner,
                states=reference_states,
                bank_members=bank_members,
                spec_sha256=spec_sha256,
            )
            candidate_runner.verify_identity_unchanged()
            reference_runner.verify_identity_unchanged()
            final_bank_seal = verify_bank(bank, require_complete=True)
            if (
                final_bank_seal["bank_id"] != bank_seal["bank_id"]
                or final_bank_seal["manifest_sha256"] != bank_seal["manifest_sha256"]
                or final_bank_seal["tree_sha256"] != bank_seal["tree_sha256"]
            ):
                raise EvaluationError("BANK_CHANGED_DURING_EVALUATION")
            paired = paired_summary(candidate_metrics, reference_metrics)
            classes = Counter(str(member["class"]) for member in bank_members)
            positions = sum(int(member["positions"]) for member in bank_members)
            artifacts = tree_identity(
                root,
                excluded_names=(
                    ".banana-smasher.lock",
                    "EVALUATION_COMPLETE",
                    "EVALUATION_PROGRESS.json",
                    "evaluation.json",
                ),
            )
            evaluation_id = canonical_sha256(
                {
                    "evaluation_spec_sha256": spec_sha256,
                    "candidate_manifest_sha256": candidate_manifest["sha256"],
                    "reference_manifest_sha256": reference_manifest["sha256"],
                }
            )
            elapsed_seconds = time.time() - started
            performance = {
                "tokens_per_second": (positions * 2) / elapsed_seconds,
                "wall_seconds": elapsed_seconds,
                "peak_vram_bytes": 0,
                "quality_result": {
                    "status": "PASS",
                    "candidate_global_kld": candidate_metrics["kld"]["global"],
                    "reference_global_kld": reference_metrics["kld"]["global"],
                    "paired_mean_window_delta": paired["mean_window_delta"],
                },
                "kernel": "numpy-concatenated-all-window-gemm",
                "fallback_used": False,
                "fallback_status": "none",
                "window_batch_size": len(bank_members),
                "layer_forwards_per_arm": layer_count - start_layer,
                "head_forwards_per_arm": 1,
            }
            receipt = {
                "schema": EVALUATION_SCHEMA,
                "schema_version": 1,
                "status": "COMPLETE",
                "operation": "evaluate",
                "evaluation_id": evaluation_id,
                "evaluation_spec_sha256": spec_sha256,
                "mode": "paired_real_axis",
                "bank": {
                    "bank_id": bank_seal["bank_id"],
                    "manifest_sha256": bank_seal["manifest_sha256"],
                    "build_spec_sha256": bank_seal["build_spec_sha256"],
                    "tree_sha256": bank_seal["tree_sha256"],
                },
                "population": {
                    "ordered_window_ids_sha256": bank_seal["manifest"]["population"][
                        "ordered_window_ids_sha256"
                    ],
                    "windows": len(bank_members),
                    "positions": positions,
                    "classes": dict(sorted(classes.items())),
                },
                "instrument": bank_seal["manifest"]["instrument"],
                "topology": {"layer_count": layer_count, "layers": topology_rows},
                "arms": {
                    "candidate": {
                        "artifact": candidate_runner.identity(),
                        "logits_manifest": candidate_manifest,
                        "kld_manifest": candidate_manifest,
                        "kld": candidate_metrics["kld"],
                        "top1_parity": candidate_metrics["top1_parity"],
                        "per_window": candidate_metrics["per_window"],
                    },
                    "reference": {
                        "artifact": reference_runner.identity(),
                        "logits_manifest": reference_manifest,
                        "kld_manifest": reference_manifest,
                        "kld": reference_metrics["kld"],
                        "top1_parity": reference_metrics["top1_parity"],
                        "per_window": reference_metrics["per_window"],
                    },
                },
                "paired": paired,
                "performance": performance,
                "resume": {
                    "started_from_layer": start_layer,
                    "checkpoint_sha256": (
                        checkpoint["marker_sha256"] if checkpoint is not None else None
                    ),
                },
                "elapsed_seconds": elapsed_seconds,
                "artifacts": {"root": str(root), "tree_sha256": artifacts["sha256"]},
            }
            receipt_path = atomic_json(root / "evaluation.json", receipt)
            atomic_json(
                marker_path,
                {
                    "schema": EVALUATION_MARKER_SCHEMA,
                    "status": "COMPLETE",
                    "evaluation_id": evaluation_id,
                    "evaluation_spec_sha256": spec_sha256,
                    "evaluation_sha256": sha256_file(receipt_path),
                },
            )
            seal = verify_evaluation(root, expected_spec_sha256=spec_sha256)
            atomic_json(
                progress_path,
                {
                    "schema": "bs-paired-evaluation-progress-v1",
                    "status": "COMPLETE",
                    "evaluation_spec_sha256": spec_sha256,
                    "updated_unix": time.time(),
                },
            )
            return _slim_result(seal, verbose=verbose_receipts)
        except EvaluationSpecMismatch:
            raise
        except Exception:
            (root / "EVALUATION_COMPLETE").unlink(missing_ok=True)
            atomic_json(
                root / "EVALUATION_PROGRESS.json",
                {
                    "schema": "bs-paired-evaluation-progress-v1",
                    "status": "FAILED",
                    "evaluation_spec_sha256": spec_sha256,
                    "updated_unix": time.time(),
                },
            )
            raise
