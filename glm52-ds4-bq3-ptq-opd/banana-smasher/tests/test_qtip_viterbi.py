from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from banana_smasher import qtip_viterbi  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_persistent_viterbi_matches_launch_by_step_over_one_million_decisions() -> None:
    torch.manual_seed(20260731)
    batch = 128
    cb = SimpleNamespace(
        L=16,
        K=3,
        V=2,
        lut=torch.randn((2, 65536), device="cuda", dtype=torch.float16),
    )
    x = torch.randn((256, batch), device="cuda", dtype=torch.float16)

    reference = qtip_viterbi.exact_prefix_viterbi_reference(cb, x)
    candidate = qtip_viterbi.exact_prefix_viterbi(cb, x)
    torch.cuda.synchronize()

    transition_decisions = batch * 127 * 64
    assert transition_decisions >= 1_000_000
    assert torch.equal(candidate, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_persistent_viterbi_preserves_overlap_and_tie_order() -> None:
    batch = 16
    cb = SimpleNamespace(
        L=16,
        K=3,
        V=2,
        lut=torch.zeros((2, 65536), device="cuda", dtype=torch.float16),
    )
    x = torch.zeros((256, batch), device="cuda", dtype=torch.float16)
    overlap = torch.arange(batch, device="cuda", dtype=torch.int32)

    reference = qtip_viterbi.exact_prefix_viterbi_reference(cb, x, overlap)
    candidate = qtip_viterbi.exact_prefix_viterbi(cb, x, overlap)
    torch.cuda.synchronize()

    assert torch.equal(candidate, reference)
