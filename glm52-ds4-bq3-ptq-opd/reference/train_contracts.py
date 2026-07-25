#!/usr/bin/env python3
"""Pure fail-closed state-machine and source-seal contracts for PTQ-OPD."""
from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

MILESTONES = (4, 8, 16)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def lr_for_update(update_number: int) -> float:
    if type(update_number) is not int or update_number < 1:
        raise ValueError("update_number must be a positive exact integer")
    return 2.5e-4 if update_number <= 2 else 5.0e-4


def segment_rows(rows: Sequence[Any], max_rows: int) -> list[list[Any]]:
    """Partition an update losslessly into bounded rollout segments."""
    if type(max_rows) is not int:
        raise TypeError("max_rows must be an exact integer")
    if max_rows < 1 or max_rows > 32:
        raise ValueError("max_rows must be in [1,32]")
    values = list(rows)
    return [values[index:index + max_rows] for index in range(0, len(values), max_rows)]


def continuation_rows(
    rows: Sequence[Any],
    update_number: int,
    *,
    start_update: int,
    target_update: int,
) -> list[Any]:
    """Select one exact shard slice for a continuation without resetting update numbering.

    The shard is consumed once across ``target_update - start_update`` global
    optimizer updates, preserving the original interleaved micro-dose ordering.
    """
    for name, value in (("update_number", update_number), ("start_update", start_update),
                        ("target_update", target_update)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an exact integer")
    dose_updates = target_update - start_update
    if start_update < 0 or dose_updates < 1:
        raise ValueError("continuation update boundaries are invalid")
    if update_number <= start_update or update_number > target_update:
        raise ValueError("update_number is outside the continuation dose")
    values = list(rows)
    if not values or len(values) % dose_updates:
        raise ValueError("continuation shard must divide exactly across dose updates")
    local_ordinal = update_number - start_update
    selected = [row for index, row in enumerate(values) if index % dose_updates == local_ordinal - 1]
    if len(selected) != len(values) // dose_updates:
        raise RuntimeError("continuation selection coverage drift")
    return selected


def deep_dose_rows(rows: Sequence[Any], update_number: int) -> list[Any]:
    """Select four rows for each update in the exact 8→12→16 dose curve.

    The same sealed 16-row bank is consumed once in updates 9..12 and once
    again, in the identical order, in updates 13..16.
    """
    if type(update_number) is not int:
        raise TypeError("update_number must be an exact integer")
    if update_number < 9 or update_number > 16:
        raise ValueError("deep-dose update_number must be in [9,16]")
    values = list(rows)
    if len(values) != 16:
        raise ValueError("deep-dose bank must contain exactly 16 rows")
    block_start = 8 if update_number <= 12 else 12
    return continuation_rows(
        values, update_number, start_update=block_start, target_update=block_start + 4
    )


def campaign_continuation_rows(
    rows: Sequence[Any], update_number: int, *, start_update: int, target_update: int
) -> list[Any]:
    """Select rows for one of the two source-sealed continuation shapes."""
    if (start_update, target_update) == (4, 8):
        return continuation_rows(
            rows, update_number, start_update=start_update, target_update=target_update
        )
    if (start_update, target_update) == (8, 16):
        return deep_dose_rows(rows, update_number)
    raise ValueError("sealed continuation must be exactly 4→8 or 8→16")


def _sha(value: object, name: str) -> str:
    if type(value) is not str or not _SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _validate_passed_gate(state: Mapping[str, Any], milestone: int) -> None:
    if state.get("last_passed_milestone") != milestone:
        raise ValueError("last_passed_milestone mismatch")
    candidate_sha = _sha(state.get("candidate_sha256"), "candidate_sha256")
    gate = state.get("gate")
    if type(gate) is not dict or gate.get("passed") is not True:
        raise ValueError("passed gate receipt missing")
    if gate.get("milestone") != milestone or gate.get("candidate_sha256") != candidate_sha:
        raise ValueError("gate receipt candidate/milestone mismatch")
    receipt_sha = _sha(state.get("gate_receipt_sha256"), "gate_receipt_sha256")
    if receipt_sha != canonical_sha256(gate):
        raise ValueError("gate receipt digest mismatch")


def validate_transition(checkpoint: Optional[Mapping[str, Any]], target: int) -> str:
    """Return ``train``, ``gate``, or ``done`` for the exact 0→4→8→16 ladder."""
    if target not in MILESTONES:
        raise ValueError("target must be one exact milestone: 4, 8, or 16")
    if checkpoint is None:
        next_update = 0
        passed = 0
        gate = None
    else:
        next_update = checkpoint.get("next_update")
        passed = checkpoint.get("last_passed_milestone")
        gate = checkpoint.get("gate")
        if type(next_update) is not int or type(passed) is not int:
            raise ValueError("checkpoint ladder fields must be exact integers")
        if next_update < 0 or next_update > 16 or passed not in (0, 4, 8, 16) or passed > next_update:
            raise ValueError("invalid checkpoint ladder state")
        if passed:
            _validate_passed_gate(checkpoint, passed)
    required_target = 4 if passed == 0 else 8 if passed == 4 else 16 if passed == 8 else 16
    if target != required_target:
        raise ValueError(f"exact ladder requires target={required_target}, not {target}")
    if next_update > target:
        raise ValueError("checkpoint is beyond requested target")
    if next_update < passed:
        raise ValueError("checkpoint precedes passed milestone")
    if next_update < target:
        # A step-N emergency checkpoint cannot jump over its pending milestone.
        prior_boundary = 0 if target == 4 else target // 2
        if passed != prior_boundary:
            raise ValueError("training cannot continue without prior passed gate")
        return "train"
    if next_update == target and passed < target:
        return "gate"
    if next_update == target and passed == target:
        return "done"
    raise ValueError("unreachable ladder state")


def gated_state(milestone: int) -> Dict[str, Any]:
    if milestone not in MILESTONES:
        raise ValueError("invalid milestone")
    candidate_sha = hashlib.sha256(f"candidate-{milestone}".encode()).hexdigest()
    gate = {
        "format": "ptq-opd-static-gate-receipt-v2",
        "milestone": milestone,
        "candidate_sha256": candidate_sha,
        "passed": True,
    }
    return {
        "next_update": milestone,
        "last_passed_milestone": milestone,
        "candidate_sha256": candidate_sha,
        "gate": gate,
        "gate_receipt_sha256": canonical_sha256(gate),
    }


def _regular_relative(root: Path, relative: str) -> Path:
    if type(relative) is not str or not relative:
        raise ValueError("source path must be a nonempty relative string")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
        raise ValueError("source path must be relative and traversal-free")
    root_resolved = root.resolve(strict=True)
    path = root / rel
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise ValueError(f"sealed source missing: {relative}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"sealed source must be nonsymlinked regular file: {relative}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"sealed source escapes root: {relative}")
    return path


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_source_seal(root: Path, relative_paths: Sequence[str]) -> Dict[str, Any]:
    if not relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise ValueError("source seal paths must be nonempty and unique")
    files = {}
    for relative in sorted(relative_paths):
        path = _regular_relative(Path(root), relative)
        files[relative] = {"sha256": _sha_file(path), "bytes": path.stat().st_size}
    body = {"format": "ptq-opd-source-seal-v2", "files": files}
    body["seal_sha256"] = canonical_sha256(body)
    return body


def verify_source_seal(root: Path, seal: Mapping[str, Any]) -> None:
    if type(seal) is not dict or seal.get("format") != "ptq-opd-source-seal-v2" or type(seal.get("files")) is not dict:
        raise ValueError("source seal format mismatch")
    declared_sha = _sha(seal.get("seal_sha256"), "seal_sha256")
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    if canonical_sha256(body) != declared_sha:
        raise ValueError("source seal self-digest mismatch")
    for relative, receipt in seal["files"].items():
        path = _regular_relative(Path(root), relative)
        if type(receipt) is not dict or type(receipt.get("bytes")) is not int:
            raise ValueError("source seal receipt malformed")
        if path.stat().st_size != receipt["bytes"] or _sha_file(path) != receipt.get("sha256"):
            raise ValueError(f"source seal drift: {relative}")
