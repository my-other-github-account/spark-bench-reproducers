from __future__ import annotations

import math

import pytest
import torch

from banana_smasher.fwht import bounded_fwht, fwht_stats


def reference_fwht(x: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    y = x.contiguous()
    n = y.shape[-1]
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, 2, h)
        left = y[..., 0, :]
        right = y[..., 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(*x.shape)
        h *= 2
    return y / math.sqrt(n) if normalize else y


@pytest.mark.parametrize("shape", [(8,), (3, 16), (2, 5, 32)])
@pytest.mark.parametrize("normalize", [False, True])
def test_bounded_fwht_matches_reference(
    shape: tuple[int, ...], normalize: bool
) -> None:
    torch.manual_seed(7)
    value = torch.randn(shape, dtype=torch.float64)
    assert torch.equal(
        bounded_fwht(value, normalize=normalize),
        reference_fwht(value, normalize=normalize),
    )


@pytest.mark.parametrize("shape", [(8,), (3, 16), (2, 5, 32)])
@pytest.mark.parametrize("normalize", [False, True])
def test_bounded_fwht_self_adjoint_backward(
    shape: tuple[int, ...], normalize: bool
) -> None:
    torch.manual_seed(11)
    value = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    weight = torch.randn_like(value)
    (bounded_fwht(value, normalize=normalize) * weight).sum().backward()
    assert torch.allclose(
        value.grad,
        reference_fwht(weight, normalize=normalize),
        atol=0.0,
        rtol=0.0,
    )


def test_inplace_decode_path_uses_half_tensor_scratch() -> None:
    value = torch.randn(32, 128)
    original_pointer = value.data_ptr()
    fwht_stats(reset=True)
    transformed = bounded_fwht(value, inplace=True)
    stats = fwht_stats()
    assert transformed.data_ptr() == original_pointer
    assert stats["inplace_calls"] == 1
    assert stats["max_scratch_bytes"] == value.numel() * value.element_size() // 2


def test_bounded_fwht_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="power of two"):
        bounded_fwht(torch.randn(7))
    with pytest.raises(ValueError, match="without gradients"):
        bounded_fwht(torch.randn(8, requires_grad=True), inplace=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocator proof")
def test_cuda_peak_is_below_concat_reference() -> None:
    value = torch.randn(256, 4096, device="cuda")

    def peak(function) -> int:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = function(value)
        torch.cuda.synchronize()
        measured = torch.cuda.max_memory_allocated() - baseline
        del output
        return int(measured)

    reference_peak = peak(reference_fwht)
    bounded_peak = peak(lambda item: bounded_fwht(item, inplace=False))
    tensor_bytes = value.numel() * value.element_size()
    assert bounded_peak + tensor_bytes <= reference_peak
