from __future__ import annotations

import math

import numpy as np
import pytest

from banana_smasher.metrics import score_candidate, teacher_support


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
