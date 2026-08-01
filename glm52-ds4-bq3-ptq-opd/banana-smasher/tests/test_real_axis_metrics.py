from __future__ import annotations

import math

import numpy as np
import pytest

from banana_smasher.metrics import paired_summary, score_candidate, teacher_support
from banana_smasher.real_axis import RealAxisRunner
from real_axis_fixtures import real_axis_fixture


def test_known_support_kld_direction_and_top1_parity() -> None:
    teacher_logprob = np.log(np.asarray([[0.75, 0.25]], dtype=np.float64))
    scored = score_candidate(
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        teacher_indices=np.asarray([[0, 1]], dtype=np.int32),
        teacher_logprob=teacher_logprob,
        teacher_argmax=np.asarray([0], dtype=np.int32),
    )
    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert scored["kld"][0] == pytest.approx(expected, rel=1e-12)
    assert scored["candidate_argmax"].tolist() == [0]
    assert scored["top1_equal"].tolist() == [1]


def test_teacher_support_uses_lowest_token_id_for_ties() -> None:
    indices, logprob, argmax = teacher_support(
        np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32), support=2
    )
    assert indices.tolist() == [[0, 1]]
    assert argmax.tolist() == [0]
    assert np.isfinite(logprob).all()


def test_paired_summary_derives_ci95_from_actual_window_count() -> None:
    def arm(values: list[float]) -> dict:
        return {
            "kld": {"global": float(np.mean(values))},
            "per_window": [
                {"window_id": index, "mean_kld": value}
                for index, value in enumerate(values)
            ],
        }

    paired = paired_summary(arm([2.0, 4.0, 8.0]), arm([1.0, 2.0, 4.0]))
    assert paired["paired_ci95"][0] < paired["mean_window_delta"]
    assert paired["paired_ci95"][1] > paired["mean_window_delta"]

    singleton = paired_summary(arm([2.0]), arm([1.0]))
    assert singleton["paired_ci95"] == [1.0, 1.0]


def test_real_axis_batches_all_windows_into_one_layer_and_head_forward(
    tmp_path, monkeypatch
) -> None:
    paths = real_axis_fixture(tmp_path)
    runner = RealAxisRunner(paths["candidate"], require_pack=True)
    states = [
        np.full((4, 4), float(ordinal), dtype=np.float32)
        for ordinal in range(64)
    ]
    layer_calls = 0
    head_calls = 0
    apply_layer_arrays = runner._apply_layer_arrays
    project_logits_arrays = runner._project_logits_arrays

    def count_layer(*args, **kwargs):
        nonlocal layer_calls
        layer_calls += 1
        return apply_layer_arrays(*args, **kwargs)

    def count_head(*args, **kwargs):
        nonlocal head_calls
        head_calls += 1
        return project_logits_arrays(*args, **kwargs)

    monkeypatch.setattr(runner, "_apply_layer_arrays", count_layer)
    monkeypatch.setattr(runner, "_project_logits_arrays", count_head)

    outputs = runner.apply_layer_batch(0, states)
    logits = runner.project_logits_batch(outputs)

    assert len(outputs) == len(logits) == 64
    assert layer_calls == 1
    assert head_calls == 1
    assert all(output.shape == (4, 4) for output in outputs)
    assert all(value.shape == (4, 8) for value in logits)
