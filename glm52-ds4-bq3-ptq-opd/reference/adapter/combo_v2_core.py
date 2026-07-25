#!/usr/bin/env python3
"""Pure contracts for the disjoint-corpus COMBO-V2 arm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

FORMAT = "combo-repair-v1"
MECHANISM = "codebooks-plus-all-rmsnorms-plus-attention-output-gains"


def _unique(name: str, values: Sequence[int]) -> list[int]:
    result = list(values)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate windows in {name}")
    return result


def validate_layout(
    *,
    eval_count: int,
    train_count: int,
    train_wins: Sequence[int],
    eval_probes: Sequence[int],
    purity_wins: Sequence[int],
    steps: int,
    batch: int,
) -> dict[str, object]:
    """Validate the combined [EVAL, TRAIN] corpus index contract."""
    if eval_count <= 0 or train_count <= 0:
        raise ValueError("partition counts must be positive")
    total = eval_count + train_count
    train = _unique("train_wins", train_wins)
    eval_holdout = _unique("eval_probes", eval_probes)
    purity = _unique("purity_wins", purity_wins)

    if not 192 <= len(train) <= 512:
        raise ValueError(f"COMBO-V4 needs 192..512 train windows, got {len(train)}")
    if len(eval_holdout) != 8:
        raise ValueError(f"COMBO-V2 needs exactly 8 eval probes, got {len(eval_holdout)}")
    if len(purity) != 8:
        raise ValueError(f"COMBO-V2 needs exactly 8 purity probes, got {len(purity)}")
    if any(index < eval_count or index >= total for index in train):
        raise ValueError("train_wins must all belong to the TRAIN partition")
    if any(index < eval_count or index >= total for index in purity):
        raise ValueError("purity_wins must all belong to the TRAIN partition")
    if any(index < 0 or index >= eval_count for index in eval_holdout):
        raise ValueError("eval_probes must all belong to the EVAL partition")
    leaked = set(train) & set(purity)
    if leaked:
        raise ValueError(f"purity leakage: {sorted(leaked)}")
    if batch != 4:
        raise ValueError(f"COMBO-V2 batch must be 4, got {batch}")
    if not 48 <= steps <= 128:
        raise ValueError(f"COMBO-V4 steps must be 48..128, got {steps}")
    epochs = steps * batch / len(train)
    if not 1.0 <= epochs <= 1.3:
        raise ValueError(f"COMBO-V2 exposure must be 1.0..1.3 epochs, got {epochs:.4f}")

    return {
        "eval_count": eval_count,
        "train_count": train_count,
        "combined_count": total,
        "train_window_count": len(train),
        "eval_probe_count": len(eval_holdout),
        "purity_probe_count": len(purity),
        "steps": steps,
        "batch": batch,
        "epochs": epochs,
        "disjoint": True,
    }


def validate_warm_start_header(
    checkpoint: Mapping[str, object], *, expected_manifest: str
) -> dict[str, object]:
    """Accept only an exact arm-B COMBO checkpoint as V2 initialization."""
    expected = {
        "format": FORMAT,
        "mechanism": MECHANISM,
        "manifest_md5": expected_manifest,
        "pool_role": "V2",
    }
    bad = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if bad:
        raise ValueError(f"warm-start header mismatch: {sorted(bad)}")
    state = checkpoint.get("state")
    if not isinstance(state, Mapping) or set(state) != {"codebooks", "norms", "outputs"}:
        raise ValueError("warm-start state surface mismatch")
    next_step = checkpoint.get("next_step")
    if not isinstance(next_step, int) or next_step <= 0:
        raise ValueError("warm-start next_step must be a positive integer")
    return {
        "state": state,
        "source_pool_role": checkpoint["pool_role"],
        "source_next_step": next_step,
    }


def purity_metrics(
    *, warm_start: Mapping[int, float], final: Mapping[int, float]
) -> dict[str, object]:
    """Compare held-out TRAIN KLD at V2 warm start and sealed best state."""
    if not warm_start or set(warm_start) != set(final):
        raise ValueError("purity probe sets must be equal and non-empty")
    wins = sorted(warm_start)
    before = sum(float(warm_start[win]) for win in wins) / len(wins)
    after = sum(float(final[win]) for win in wins) / len(wins)
    if before <= 0.0:
        raise ValueError("warm-start purity mean must be positive")
    return {
        "wins": wins,
        "warm_start": {win: float(warm_start[win]) for win in wins},
        "final": {win: float(final[win]) for win in wins},
        "warm_start_mean": before,
        "final_mean": after,
        "delta_pct": (before - after) / before * 100.0,
    }
