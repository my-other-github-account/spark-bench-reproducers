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
        atomic_npz(path, scored)
        rows.append(
            {
                "ordinal": ordinal,
                "window_id": bank_member["window_id"],
                "class": bank_member["class"],
                **file_identity(path, root=root),
                "tensors": {
                    name: _tensor_schema(value) for name, value in sorted(scored.items())
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
    if (
        manifest.get("schema") != ARM_MANIFEST_SCHEMA
        or manifest.get("status") != "COMPLETE"
        or manifest.get("evaluation_spec_sha256") != spec_sha256
        or manifest.get("arm") != expected_arm
        or manifest.get("member_count") != row.get("members")
    ):
        raise EvaluationError("ARM_MANIFEST_INVALID")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != row.get("members"):
        raise EvaluationError("ARM_MEMBER_COUNT_MISMATCH")
    for ordinal, member in enumerate(members):
        if (
            not isinstance(member, dict)
            or member.get("ordinal") != ordinal
            or not isinstance(member.get("window_id"), (str, int))
            or isinstance(member.get("window_id"), bool)
            or not isinstance(member.get("class"), str)
            or not member.get("class")
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
        ):
            raise EvaluationError("ARM_MEMBER_SEMANTICS_INVALID")
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
    return manifest


def verify_evaluation(
    output: str | Path, *, expected_spec_sha256: str | None = None
) -> dict[str, Any]:
    root = Path(output).resolve()
    marker_path = root / "EVALUATION_COMPLETE"
    marker = load_json_object(marker_path, label="EVALUATION_MARKER")
    receipt_path = root / "evaluation.json"
    receipt = load_json_object(receipt_path, label="EVALUATION_RECEIPT")
    if (
        marker.get("schema") != EVALUATION_MARKER_SCHEMA
        or marker.get("status") != "COMPLETE"
        or receipt.get("schema") != EVALUATION_SCHEMA
        or receipt.get("status") != "COMPLETE"
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
    arm_manifests: dict[str, dict[str, Any]] = {}
    for arm in ("candidate", "reference"):
        arm_receipt = receipt_arms.get(arm)
        if not isinstance(arm_receipt, dict):
            raise EvaluationError("EVALUATION_ARM_INVALID")
        arm_row = arm_receipt.get("logits_manifest")
        if not isinstance(arm_row, dict):
            raise EvaluationError("EVALUATION_ARM_MANIFEST_MISSING")
        if arm_receipt.get("kld_manifest") != arm_row:
            raise EvaluationError("EVALUATION_KLD_MANIFEST_MISMATCH")
        arm_manifests[arm] = _verify_arm_artifacts(
            root,
            arm_row,
            str(receipt["evaluation_spec_sha256"]),
            expected_arm=arm,
        )
    candidate_members = arm_manifests["candidate"]["members"]
    reference_members = arm_manifests["reference"]["members"]
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
        or dict(sorted(Counter(candidate_classes).items())) != population.get("classes")
    ):
        raise EvaluationError("ARM_POPULATION_MISMATCH")
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
                "resume": {
                    "started_from_layer": start_layer,
                    "checkpoint_sha256": (
                        checkpoint["marker_sha256"] if checkpoint is not None else None
                    ),
                },
                "elapsed_seconds": time.time() - started,
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
