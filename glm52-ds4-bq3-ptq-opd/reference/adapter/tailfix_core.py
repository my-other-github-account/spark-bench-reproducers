#!/usr/bin/env python3
"""Pure contracts for Combo-V4 step-32 tail-repair experiments."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from pathlib import Path
from typing import Mapping, Sequence

ARMS = ("control", "tail", "tail_class40", "trajectory_micro", "trajectory_full")
CLASSES = ("chat", "reasoning", "code", "prose")
START_SHA256 = "fae41d519193269aec4b2221c97a1dc00e0b00d3d66074d917a78489fac2149c"
BASELINE_START_SHA256 = START_SHA256
START_NEXT_STEP = 32
START_FORMAT = "combo-repair-v1"
START_MECHANISM = "codebooks-plus-all-rmsnorms-plus-attention-output-gains"
START_POOL_ROLE = "V2"
START_MANIFEST_MD5 = "5a622d5139e73452b719d5d2cfeb2571"


def canonical_sha256(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _interleave_counts(counts: Mapping[str, int]) -> list[str]:
    """Deterministic deficit-round-robin order with exact requested counts."""
    if set(counts) != set(CLASSES) or any(v < 0 for v in counts.values()):
        raise ValueError("counts must contain the four non-negative sealed classes")
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("empty schedule")
    remaining = dict(counts)
    emitted = Counter()
    out: list[str] = []
    for slot in range(total):
        choices = [c for c in CLASSES if remaining[c] > 0]
        # Choose the class farthest below its ideal cumulative count.
        chosen = max(
            choices,
            key=lambda c: ((slot + 1) * counts[c] / total - emitted[c], -CLASSES.index(c)),
        )
        out.append(chosen)
        remaining[chosen] -= 1
        emitted[chosen] += 1
    return out


def build_plan(
    rows: Sequence[Mapping[str, object]],
    arm: str,
    *,
    per_class_train: int = 12,
    per_class_holdout: int = 4,
    batch: int = 4,
) -> dict[str, object]:
    """Build a matched 48-window-exposure plan from the sealed fresh-200 corpus.

    The first 16 stable-ranked rows/class are selected. Twelve/class form the
    train pool and four/class are held out from every arm. Control and tail use
    25% exposure/class. tail_class40 uses 5/12 code steps, 5/12 reasoning,
    1/12 chat, and 1/12 prose, guaranteeing >=40% target gradient exposure to
    each of code and reasoning while preserving the same 48 total exposures.

    ``trajectory_micro`` is the separately-labelled fast-signal arm ordered
    after the behavioral-drift reprioritization.  It is intentionally a
    shortest-meaningful four-step dose, with two code and two reasoning
    standard-KL batches plus a hard-NLL term over sixteen frozen FP-visible
    HumanEval sequences.  It is not part of the matched 12-step tail ladder.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if batch != 4 or per_class_train != 12 or per_class_holdout != 4:
        raise ValueError("sealed pilot contract is batch=4, train=12/class, holdout=4/class")
    grouped: dict[str, list[Mapping[str, object]]] = {c: [] for c in CLASSES}
    seen: set[int] = set()
    for row in rows:
        cls = str(row["class"])
        win = int(row["local_win"])
        if cls not in grouped:
            raise ValueError(f"unexpected class: {cls}")
        if win in seen:
            raise ValueError(f"duplicate local_win: {win}")
        seen.add(win)
        grouped[cls].append(row)
    for cls in CLASSES:
        grouped[cls].sort(key=lambda row: (str(row["selection_rank_sha256"]), int(row["local_win"])))
        if len(grouped[cls]) != 50:
            raise ValueError(f"sealed fresh-200 class count drift for {cls}: {len(grouped[cls])}")

    train_by_class: dict[str, list[int]] = {}
    holdout_by_class: dict[str, list[int]] = {}
    for cls in CLASSES:
        chosen = grouped[cls][: per_class_train + per_class_holdout]
        train_by_class[cls] = [int(row["local_win"]) for row in chosen[:per_class_train]]
        holdout_by_class[cls] = [int(row["local_win"]) for row in chosen[per_class_train:]]

    if arm in {"control", "tail"}:
        step_counts = {cls: 3 for cls in CLASSES}
    elif arm == "tail_class40":
        step_counts = {"chat": 1, "reasoning": 5, "code": 5, "prose": 1}
    else:
        step_counts = {"chat": 0, "reasoning": 2, "code": 2, "prose": 0}
    queues = {
        cls: deque(train_by_class[cls])
        for cls in CLASSES
    }
    schedule: list[dict[str, object]] = []
    for step, cls in enumerate(_interleave_counts(step_counts), start=1):
        wins: list[int] = []
        for _ in range(batch):
            if not queues[cls]:
                queues[cls].extend(train_by_class[cls])
            wins.append(queues[cls].popleft())
        schedule.append({"step": step, "class": cls, "wins": wins})

    exposures = Counter()
    for item in schedule:
        exposures[str(item["class"])] += len(item["wins"])
    total_exposures = sum(exposures.values())
    mass = {cls: exposures[cls] / total_exposures for cls in CLASSES}
    train_wins = sorted({win for values in train_by_class.values() for win in values})
    probe_wins = sorted({win for values in holdout_by_class.values() for win in values})
    if set(train_wins) & set(probe_wins):
        raise AssertionError("train/held-out overlap")
    expected_steps = 4 if arm == "trajectory_micro" else 20 if arm == "trajectory_full" else 12
    expected_exposures = expected_steps * batch
    if len(schedule) != expected_steps or total_exposures != expected_exposures or len(probe_wins) != 16:
        raise AssertionError("pilot budget drift")
    if arm == "tail_class40" and (mass["code"] < 0.40 or mass["reasoning"] < 0.40):
        raise AssertionError("40% class-mass contract failed")

    objective = (
        "mean_kld"
        if arm == "control"
        else "mean_kld_plus_fp_visible_hard_nll"
        if arm == "trajectory_micro"
        else "tail_weighted_kld"
    )
    trajectory_task_ids = [
        "HumanEval/116", "HumanEval/132", "HumanEval/134", "HumanEval/13",
        "HumanEval/93", "HumanEval/73", "HumanEval/31", "HumanEval/99",
        "HumanEval/146", "HumanEval/26", "HumanEval/106", "HumanEval/9",
        "HumanEval/122", "HumanEval/114", "HumanEval/3", "HumanEval/57",
    ] if arm == "trajectory_micro" else []
    plan: dict[str, object] = {
        "format": "combo-v4-tailfix-plan-v1",
        "arm": arm,
        "objective": objective,
        "start_checkpoint_sha256": START_SHA256,
        "start_checkpoint_next_step": START_NEXT_STEP,
        "batch": batch,
        "steps": len(schedule),
        "total_window_exposures": total_exposures,
        "tail_token_quantile": None if arm in {"control", "trajectory_micro"} else 0.80,
        "tail_token_boost": None if arm in {"control", "trajectory_micro"} else 4.0,
        "tail_window_quantile": None if arm in {"control", "trajectory_micro"} else 0.75,
        "tail_window_boost": None if arm in {"control", "trajectory_micro"} else 4.0,
        "trajectory_weight": 0.25 if arm == "trajectory_micro" else None,
        "trajectory_batch": 4 if arm == "trajectory_micro" else None,
        "trajectory_task_ids": trajectory_task_ids,
        "trajectory_caveat": (
            "FP artifact preserves visible answer tokens only; upstream reasoning tokens were discarded. "
            "The paired standard-KL schedule supplies generic reasoning windows, so this micro-dose is a "
            "trajectory/code plus reasoning-class tractability probe, not a sealed reasoning-span corpus."
            if arm == "trajectory_micro" else None
        ),
        "train_wins": train_wins,
        "probe_wins": probe_wins,
        "train_by_class": train_by_class,
        "probe_by_class": holdout_by_class,
        "step_counts": step_counts,
        "exposures_by_class": dict(exposures),
        "target_gradient_mass_by_class": mass,
        "schedule": schedule,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_plan(plan: Mapping[str, object]) -> None:
    expected_hash = plan.get("plan_sha256")
    without_hash = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if expected_hash != canonical_sha256(without_hash):
        raise ValueError("plan digest mismatch")
    if plan.get("format") != "combo-v4-tailfix-plan-v1":
        raise ValueError("plan format mismatch")
    if plan.get("start_checkpoint_sha256") != START_SHA256:
        raise ValueError("start checkpoint digest drift")
    arm = str(plan.get("arm"))
    expected_steps = 4 if arm == "trajectory_micro" else 20 if arm == "trajectory_full" else 12
    if int(plan.get("batch", 0)) != 4 or int(plan.get("steps", 0)) != expected_steps:
        raise ValueError("budget drift")
    train = list(map(int, plan["train_wins"]))
    probes = list(map(int, plan["probe_wins"]))
    if len(train) != 48 or len(probes) != 16 or set(train) & set(probes):
        raise ValueError("partition drift")
    schedule = list(plan["schedule"])
    if len(schedule) != expected_steps or any(len(item["wins"]) != 4 for item in schedule):
        raise ValueError("schedule shape drift")
    if arm in {"trajectory_micro", "trajectory_full"}:
        tasks = list(map(str, plan.get("trajectory_task_ids", [])))
        expected_tasks = expected_steps * int(plan.get("trajectory_batch", 0))
        if len(tasks) != expected_tasks:
            raise ValueError("trajectory task contract drift")
        if float(plan.get("trajectory_weight", 0.0)) != 0.25 or int(plan.get("trajectory_batch", 0)) != 4:
            raise ValueError("trajectory objective contract drift")
        if arm == "trajectory_micro" and {"HumanEval/116", "HumanEval/132"} - set(tasks):
            raise ValueError("sealed micro trajectory task contract drift")
        if arm == "trajectory_full":
            if plan.get("bin") != "PTQ-OPD":
                raise ValueError("trajectory_full must be labelled PTQ-OPD")
            holdout = list(map(str, plan.get("trajectory_holdout_task_ids", [])))
            if len(holdout) != 18 or len(set(holdout)) != 18:
                raise ValueError("clean trajectory holdout must contain 18 unique tasks")
            if {"HumanEval/116", "HumanEval/132"} - set(holdout):
                raise ValueError("clean trajectory holdout must include 116/132")
            if set(tasks) & set(holdout):
                raise ValueError("trajectory train/held-out task contamination")
            split_identity = {
                "seed": int(plan.get("trajectory_holdout_seed", -1)),
                "task_ids": holdout,
            }
            if plan.get("trajectory_holdout_sha256") != canonical_sha256(split_identity):
                raise ValueError("trajectory holdout split digest mismatch")
            source_sha = str(plan.get("trajectory_source_sha256", ""))
            if len(source_sha) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha):
                raise ValueError("clean trajectory source digest missing")


def validate_start_header(checkpoint: Mapping[str, object], actual_sha256: str) -> None:
    expected = {
        "format": START_FORMAT,
        "mechanism": START_MECHANISM,
        "manifest_md5": START_MANIFEST_MD5,
        "pool_role": START_POOL_ROLE,
        "next_step": START_NEXT_STEP,
    }
    bad = {key: (checkpoint.get(key), value) for key, value in expected.items() if checkpoint.get(key) != value}
    if actual_sha256 != START_SHA256:
        bad["sha256"] = (actual_sha256, START_SHA256)
    state = checkpoint.get("state")
    if not isinstance(state, Mapping) or set(state) != {"codebooks", "norms", "outputs"}:
        bad["state"] = ("invalid", "codebooks,norms,outputs")
    if bad:
        raise ValueError(f"frozen step-32 lineage mismatch: {sorted(bad)}")


def normalized_tail_weights(values: Sequence[float], quantile: float, boost: float) -> list[float]:
    """Pure reference for detached quantile weights used by the GPU trainer."""
    if not values or not 0.0 <= quantile <= 1.0 or boost < 1.0:
        raise ValueError("invalid tail-weight request")
    ordered = sorted(float(v) for v in values)
    rank = min(len(ordered) - 1, int(quantile * len(ordered)))
    threshold = ordered[rank]
    raw = [boost if float(v) >= threshold else 1.0 for v in values]
    scale = len(raw) / sum(raw)
    return [weight * scale for weight in raw]


def tail_candidate_decision(
    baseline_by_class: Mapping[str, Mapping[str, float]],
    candidate_by_class: Mapping[str, Mapping[str, float]],
    *,
    primary_class: str = "code",
    mean_tolerance: float = 0.01,
) -> dict[str, object]:
    """Apply the held-out tail-repair promotion rule.

    A tail arm is promotable only when its primary-class p99 improves and no
    represented held-out class regresses by more than ``mean_tolerance``. The
    full six-class rail remains the ship gate; this local rule prevents the
    trainer from discarding a useful tail candidate merely because its global
    mean is microscopically above the warm start.
    """
    if not 0.0 <= mean_tolerance < 1.0:
        raise ValueError("mean_tolerance must be in [0, 1)")
    if set(baseline_by_class) != set(candidate_by_class):
        raise ValueError("baseline/candidate class sets differ")
    if primary_class not in baseline_by_class:
        raise ValueError(f"missing primary class: {primary_class}")

    mean_delta_pct: dict[str, float] = {}
    violations: dict[str, float] = {}
    for cls in sorted(baseline_by_class):
        base_mean = float(baseline_by_class[cls]["mean"])
        cand_mean = float(candidate_by_class[cls]["mean"])
        if base_mean <= 0.0:
            raise ValueError(f"non-positive baseline mean for {cls}")
        delta_pct = (cand_mean / base_mean - 1.0) * 100.0
        mean_delta_pct[cls] = delta_pct
        if cand_mean > base_mean * (1.0 + mean_tolerance):
            violations[cls] = delta_pct

    base_p99 = float(baseline_by_class[primary_class]["p99"])
    cand_p99 = float(candidate_by_class[primary_class]["p99"])
    if base_p99 <= 0.0:
        raise ValueError(f"non-positive baseline p99 for {primary_class}")
    p99_reduction_pct = (base_p99 - cand_p99) / base_p99 * 100.0
    selected = cand_p99 < base_p99 and not violations
    return {
        "selected": selected,
        "primary_class": primary_class,
        "baseline_p99": base_p99,
        "candidate_p99": cand_p99,
        "p99_reduction_pct": p99_reduction_pct,
        "mean_tolerance_pct": mean_tolerance * 100.0,
        "mean_delta_pct_by_class": mean_delta_pct,
        "mean_guard_violations": violations,
        "reason": (
            "primary_p99_improved_and_all_class_means_within_guard"
            if selected
            else "primary_p99_not_improved"
            if cand_p99 >= base_p99
            else "all_class_mean_guard_failed"
        ),
    }


def load_manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
