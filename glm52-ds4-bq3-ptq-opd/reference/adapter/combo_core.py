#!/usr/bin/env python3
"""Pure validation/math helpers for COMBO-REPAIR."""
from __future__ import annotations

import math
from collections.abc import Iterable


def validate_pools(
    arm_a: Iterable[int],
    arm_b: Iterable[int],
    probes: Iterable[int],
    *,
    min_size: int = 64,
) -> dict[str, object]:
    a = list(arm_a)
    b = list(arm_b)
    p = list(probes)
    if len(a) < min_size or len(b) < min_size:
        raise ValueError(
            f"each arm needs at least {min_size} windows: {len(a)}, {len(b)}"
        )
    for name, values in (("arm_a", a), ("arm_b", b), ("probes", p)):
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate windows in {name}")
        if any(value < 0 or value > 511 for value in values):
            raise ValueError(f"window outside 0..511 in {name}")
    leaked = (set(a) | set(b)) & set(p)
    if leaked:
        raise ValueError(f"probe leakage: {sorted(leaked)}")
    overlap = set(a) & set(b)
    if overlap:
        raise ValueError(f"arm overlap: {sorted(overlap)}")
    return {
        "arm_a_count": len(a),
        "arm_b_count": len(b),
        "probe_count": len(p),
        "disjoint": True,
    }


def partition_seed_keys(
    rms_keys: Iterable[str], attention_keys: Iterable[str]
) -> dict[str, object]:
    rms = set(rms_keys)
    attention = set(attention_keys)
    attention_norms = {
        key[:-7]: key for key in attention if key.endswith(".weight")
    }
    output_gains = {
        key for key in attention if key.endswith(".output_log_gain")
    }
    recognized = set(attention_norms.values()) | output_gains
    overlap = {
        semantic: attention_norms[semantic]
        for semantic in sorted(rms & set(attention_norms))
    }
    return {
        "overlap": overlap,
        "output_gains": output_gains,
        "unexpected_attention": attention - recognized,
        "rms_only": rms - set(overlap),
    }


def cosine_multiplier(step: int, total_steps: int, min_ratio: float) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be in [0, 1]")
    clamped = min(max(step, 0), total_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * clamped / total_steps))
    return min_ratio + (1.0 - min_ratio) * cosine


def floor_label(delta_pct: float, floor_pct: float = 2.6) -> str:
    if delta_pct > floor_pct:
        return "ABOVE_FLOOR_POSITIVE"
    if delta_pct < -floor_pct:
        return "ABOVE_FLOOR_NEGATIVE"
    return "SUB_FLOOR_ZERO"


def hypothesis_band(delta_pct: float) -> str:
    if delta_pct >= 20.0:
        return "FULL_ADDITIVITY"
    if delta_pct >= 15.0:
        return "PARTIAL_OVERLAP"
    return "INTERFERENCE"
