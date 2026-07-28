#!/usr/bin/env python3
"""Pure checkpoint contracts for allocation re-adaptation evaluation."""
from __future__ import annotations

from typing import Mapping

RAW_COMPOSITE_SHA256 = "f9470fb05946f7068bdc87871985cf9ee8c04d020e7ed4408457b666fcf3e7c8"
READOUT_STEPS = (4, 8)


def overlay_format_ok(value: object) -> bool:
    return value == "multi-kpi-allocation-readapt-v1"


def validate_candidate_header(
    checkpoint: Mapping[str, object], *, step: int, identity_sha256: str, arm: str
) -> None:
    if step not in READOUT_STEPS:
        raise ValueError(f"step must be one of {READOUT_STEPS}")
    expected = {
        "format": "multi-kpi-allocation-readapt-v1",
        "task_id": "PUBLIC_TASK",
        "arm": arm,
        "next_step": step,
        "identity_sha256": identity_sha256,
    }
    bad = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    identity = checkpoint.get("identity")
    if not isinstance(identity, Mapping):
        bad["identity"] = (type(identity).__name__, "mapping")
    else:
        if identity.get("raw_composite_sha256") != RAW_COMPOSITE_SHA256:
            bad["raw_composite_sha256"] = (
                identity.get("raw_composite_sha256"), RAW_COMPOSITE_SHA256
            )
        if identity.get("load_repaired_checkpoint") is not False:
            bad["load_repaired_checkpoint"] = (
                identity.get("load_repaired_checkpoint"), False
            )
    state = checkpoint.get("state")
    if not isinstance(state, Mapping) or set(state) != {"codebooks", "norms", "outputs"}:
        bad["state"] = ("invalid", "codebooks,norms,outputs")
    elif set(state["codebooks"]) != {f"L{i}" for i in range(43)}:
        bad["codebook_layers"] = (sorted(state["codebooks"]), "L0..L42")
    if bad:
        raise ValueError(f"fresh raw candidate header mismatch: {bad}")
