from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


class MetricsError(ValueError):
    """Raised when real-axis numerical artifacts are malformed."""


_T_CRITICAL_975 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _student_t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise MetricsError("PAIRED_DEGREES_OF_FREEDOM_INVALID")
    if degrees_of_freedom <= len(_T_CRITICAL_975):
        return _T_CRITICAL_975[degrees_of_freedom - 1]
    # Cornish-Fisher expansion around the standard-normal 97.5th percentile.
    # The degrees of freedom are derived from the actual paired population.
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def require_finite_float(
    value: np.ndarray[Any, Any], *, label: str, rank: int | None = None
) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if rank is not None and array.ndim != rank:
        raise MetricsError(f"{label}_RANK_MISMATCH: expected {rank}, got {array.ndim}")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise MetricsError(f"{label}_NONFINITE_OR_NONFLOAT")
    return array


def log_softmax(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    logits = require_finite_float(value, label="LOGITS")
    work = logits.astype(np.float64, copy=False)
    maximum = np.max(work, axis=-1, keepdims=True)
    shifted = work - maximum
    result = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    if not np.isfinite(result).all():
        raise MetricsError("LOG_SOFTMAX_NONFINITE")
    return result


def teacher_support(
    logits: np.ndarray[Any, Any], *, support: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    values = require_finite_float(logits, label="TEACHER_LOGITS", rank=2)
    if not isinstance(support, int) or support <= 0 or support > values.shape[1]:
        raise MetricsError(
            f"SUPPORT_INVALID: expected 1..{values.shape[1]}, got {support!r}"
        )
    logprob = log_softmax(values)
    indices = np.empty((values.shape[0], support), dtype=np.int32)
    selected = np.empty((values.shape[0], support), dtype=np.float64)
    token_ids = np.arange(values.shape[1], dtype=np.int64)
    for row in range(values.shape[0]):
        # Highest log-probability first, with the lowest token ID winning ties.
        order = np.lexsort((token_ids, -logprob[row]))[:support]
        indices[row] = order.astype(np.int32, copy=False)
        selected[row] = logprob[row, order]
    argmax = indices[:, 0].copy()
    return indices, selected, argmax


def score_candidate(
    logits: np.ndarray[Any, Any],
    *,
    teacher_indices: np.ndarray[Any, Any],
    teacher_logprob: np.ndarray[Any, Any],
    teacher_argmax: np.ndarray[Any, Any],
) -> dict[str, np.ndarray[Any, Any]]:
    values = require_finite_float(logits, label="CANDIDATE_LOGITS", rank=2)
    indices = np.asarray(teacher_indices)
    teacher_lp = require_finite_float(
        teacher_logprob, label="TEACHER_LOGPROB", rank=2
    ).astype(np.float64, copy=False)
    teacher_top1 = np.asarray(teacher_argmax)
    if indices.ndim != 2 or not np.issubdtype(indices.dtype, np.integer):
        raise MetricsError("TEACHER_INDICES_INVALID")
    if indices.shape != teacher_lp.shape or indices.shape[0] != values.shape[0]:
        raise MetricsError("TEACHER_SUPPORT_SHAPE_MISMATCH")
    if teacher_top1.shape != (values.shape[0],):
        raise MetricsError("TEACHER_ARGMAX_SHAPE_MISMATCH")
    if np.any(indices < 0) or np.any(indices >= values.shape[1]):
        raise MetricsError("TEACHER_INDICES_OUT_OF_RANGE")
    candidate_lp = log_softmax(values)
    gathered = np.take_along_axis(candidate_lp, indices.astype(np.int64), axis=1)
    # The instrument evaluates the declared teacher support. Renormalize both
    # distributions on exactly that support so this remains a true KL
    # divergence rather than a signed partial-vocabulary contribution.
    teacher_support_lp = teacher_lp - np.log(
        np.sum(np.exp(teacher_lp), axis=1, keepdims=True)
    )
    candidate_support_lp = gathered - np.log(
        np.sum(np.exp(gathered), axis=1, keepdims=True)
    )
    probability = np.exp(teacher_support_lp)
    kld = np.sum(
        probability * (teacher_support_lp - candidate_support_lp), axis=1
    )
    kld = np.maximum(kld, 0.0)
    if not np.isfinite(kld).all():
        raise MetricsError("KLD_NONFINITE")
    candidate_argmax = np.argmax(values, axis=1).astype(np.int32)
    top1_equal = (candidate_argmax == teacher_top1).astype(np.uint8)
    return {
        "candidate_logprob": gathered,
        "candidate_argmax": candidate_argmax,
        "kld": kld.astype(np.float64),
        "top1_equal": top1_equal,
    }


def aggregate_windows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise MetricsError("METRIC_ROWS_EMPTY")
    class_kld: dict[str, list[np.ndarray[Any, Any]]] = defaultdict(list)
    class_parity: dict[str, list[np.ndarray[Any, Any]]] = defaultdict(list)
    all_kld: list[np.ndarray[Any, Any]] = []
    all_parity: list[np.ndarray[Any, Any]] = []
    per_window: list[dict[str, Any]] = []
    for row in rows:
        kld = require_finite_float(np.asarray(row["kld"]), label="KLD", rank=1)
        parity = np.asarray(row["top1_equal"])
        if parity.shape != kld.shape or not np.isin(parity, (0, 1)).all():
            raise MetricsError("TOP1_PARITY_INVALID")
        class_name = str(row["class"])
        all_kld.append(kld)
        all_parity.append(parity)
        class_kld[class_name].append(kld)
        class_parity[class_name].append(parity)
        per_window.append(
            {
                "window_id": row["window_id"],
                "class": class_name,
                "positions": int(kld.size),
                "mean_kld": float(np.mean(kld)),
                "top1_agreement_count": int(np.sum(parity)),
                "top1_rate": float(np.mean(parity)),
            }
        )
    joined_kld = np.concatenate(all_kld)
    joined_parity = np.concatenate(all_parity)
    return {
        "kld": {
            "global": float(np.mean(joined_kld)),
            "per_class": {
                name: float(np.mean(np.concatenate(values)))
                for name, values in sorted(class_kld.items())
            },
        },
        "top1_parity": {
            "agreement_count": int(np.sum(joined_parity)),
            "positions": int(joined_parity.size),
            "rate": float(np.mean(joined_parity)),
            "per_class": {
                name: {
                    "agreement_count": int(np.sum(np.concatenate(values))),
                    "positions": int(np.concatenate(values).size),
                    "rate": float(np.mean(np.concatenate(values))),
                }
                for name, values in sorted(class_parity.items())
            },
        },
        "per_window": per_window,
    }


def paired_summary(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    candidate_rows = candidate["per_window"]
    reference_rows = reference["per_window"]
    if [row["window_id"] for row in candidate_rows] != [
        row["window_id"] for row in reference_rows
    ]:
        raise MetricsError("PAIRED_WINDOW_ORDER_MISMATCH")
    deltas = [
        float(left["mean_kld"]) - float(right["mean_kld"])
        for left, right in zip(candidate_rows, reference_rows, strict=True)
    ]
    if not deltas or not np.isfinite(deltas).all():
        raise MetricsError("PAIRED_DELTAS_INVALID")
    mean_delta = float(np.mean(deltas))
    if len(deltas) == 1:
        paired_ci95 = [mean_delta, mean_delta]
    else:
        standard_error = float(np.std(deltas, ddof=1)) / math.sqrt(len(deltas))
        margin = _student_t_critical_975(len(deltas) - 1) * standard_error
        paired_ci95 = [mean_delta - margin, mean_delta + margin]
    candidate_global = float(candidate["kld"]["global"])
    reference_global = float(reference["kld"]["global"])
    ratio: float | None
    if candidate_global == 0.0:
        ratio = None
    else:
        ratio = reference_global / candidate_global
        if not math.isfinite(ratio):
            ratio = None
    return {
        "delta_definition": (
            "candidate window-mean KLD minus reference window-mean KLD"
        ),
        "window_deltas": deltas,
        "mean_window_delta": mean_delta,
        "paired_ci95": paired_ci95,
        "improvement_ratio_definition": (
            "reference global KLD divided by candidate global KLD"
        ),
        "improvement_ratio": ratio,
    }
