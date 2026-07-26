"""Whole-matrix batched LDLQ for independent QTIP units.

The unit axis is preserved through the LDL feedback matmuls while every
codebook call flattens all units into one larger sequence batch.  This keeps
the exact serial recurrence for each unit but amortizes Python and Triton
launches across the batch.
"""
from __future__ import annotations

import torch


_PERMUTE = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0, 4).flatten()
_INV_PERMUTE = torch.empty(256, dtype=torch.int64)
_INV_PERMUTE[_PERMUTE] = torch.arange(256)


def block_LDL_batch(hessian: torch.Tensor, block: int) -> torch.Tensor:
    """Batched equivalent of QTIP math_utils.block_LDL's normalized L."""
    if hessian.ndim != 3 or hessian.shape[-1] != hessian.shape[-2]:
        raise ValueError("batched block LDL expects [batch,n,n]")
    units, n, _ = hessian.shape
    if n % block:
        raise ValueError(f"matrix width {n} is not divisible by block {block}")
    blocks = n // block
    lower = torch.linalg.cholesky(hessian)
    view = lower.reshape(units, blocks, block, blocks, block)
    index = torch.arange(blocks, device=hessian.device)
    diagonal_blocks = view.permute(0, 1, 3, 2, 4)
    diagonal = diagonal_blocks[:, index, index]
    inv = torch.linalg.inv(diagonal)
    normalized = torch.einsum(
        "unib,uibc->unic", lower.view(units, n, blocks, block), inv
    ).reshape(units, n, n).contiguous()
    block_view = normalized.view(units, blocks, block, blocks, block).permute(
        0, 1, 3, 2, 4
    )
    block_view[:, index, index] = torch.eye(
        block, dtype=hessian.dtype, device=hessian.device
    )
    return normalized


def pack_kernel_layout_batch(cb, states: torch.Tensor, m: int, k: int):
    """Pack an independent unit batch for any integer-rate QTIP codebook.

    QTIP emits ``K*V`` transition bits for every ``V`` reconstructed scalar
    values, so the trellis payload is exactly ``K`` bits per source value.
    The CUDA-tensor decoder's swizzle is identical for K=1..4; only the final
    per-tile word count (16*K uint16 words) changes.
    """
    vector = int(cb.V)
    rate_bits = int(cb.K)
    if vector != 2:
        raise ValueError(f"kernel pack is sealed only for V=2, got V={vector}")
    if rate_bits not in (1, 2, 3, 4):
        raise ValueError(f"kernel pack requires integer K in 1..4, got K={rate_bits}")
    if states.ndim != 3 or tuple(states.shape[1:]) != (m, k // vector):
        raise ValueError(
            f"batched state shape mismatch: {tuple(states.shape)} != [U,{m},{k // vector}]"
        )
    units = states.shape[0]
    tiles = (m // 16) * (k // 16)
    words_per_tile = 16 * rate_bits
    tiled = (
        states.reshape(units, m // 16, 16, k // 16, 16 // vector)
        .transpose(2, 3)
        .reshape(units, tiles, 256 // vector)
    )
    packed = cb.pack_trellis(tiled.reshape(units * tiles, 256 // vector)).contiguous()
    if tuple(packed.shape) != (units * tiles, words_per_tile) or packed.dtype != torch.uint16:
        raise RuntimeError(f"canonical batched pack mismatch: {tuple(packed.shape)} {packed.dtype}")
    packed = packed.reshape(units, tiles, words_per_tile)
    unpacked = cb.unpack_trellis(
        packed.reshape(units * tiles, words_per_tile), 256
    ).reshape(units, tiles, 256 // vector)
    roundtrip = unpacked.to(tiled.dtype).eq(tiled)
    kernel = (
        packed.view(torch.uint8)
        .reshape(units, -1, 2)
        .flip((-1,))
        .reshape(units, m // 32, 2, k // 32, 2, 32, rate_bits)
        .permute(0, 1, 3, 5, 4, 2, 6)
        .flip((-1,))
        .contiguous()
        .flatten(1)
        .view(torch.int16)
        .reshape(units, tiles, words_per_tile)
    )
    expected_bytes = rate_bits * m * k // 8
    receipts = []
    for unit in range(units):
        actual_bytes = kernel[unit].numel() * kernel[unit].element_size()
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"packed byte mismatch unit={unit}: {actual_bytes} != {expected_bytes}"
            )
        receipts.append({
            "tile_states_shape": list(tiled[unit].shape),
            "canonical_packed_shape": list(packed[unit].shape),
            "canonical_packed_dtype": str(packed.dtype),
            "canonical_pack_roundtrip_fraction": float(roundtrip[unit].float().mean()),
            "canonical_pack_roundtrip_exact": bool(roundtrip[unit].all()),
            "kernel_packed_shape": list(kernel[unit].shape),
            "kernel_packed_bytes": actual_bytes,
            "batch_units": units,
            "rate_bits_per_value": rate_bits,
        })
    return kernel, receipts


def LDLQ_batch(Wr, L, cb, args, buf_cols=128, for_kernel=True):
    if Wr.ndim != 3 or L.ndim != 3:
        raise ValueError("batched LDLQ expects Wr [batch,m,n] and L [batch,n,n]")
    units, m, n = Wr.shape
    if L.shape != (units, n, n):
        raise ValueError(
            f"batch/shape mismatch: Wr={tuple(Wr.shape)} L={tuple(L.shape)}"
        )
    if for_kernel and (args.td_x, args.td_y) != (16, 16):
        raise ValueError("kernel layout requires td_x=td_y=16")
    buf_cols = max(buf_cols, args.td_y)
    trellis_size = args.td_x * args.td_y
    if buf_cols % args.td_y or n % buf_cols or args.td_y % args.V:
        raise ValueError("incompatible LDLQ tile geometry")
    buf_size = buf_cols // args.td_y

    hat_t = torch.zeros((units, n, m), dtype=L.dtype, device=L.device)
    idxs_t = torch.zeros(
        (units, n // args.V, m), dtype=cb.idx_dtype, device=L.device
    )
    wr_t = Wr.transpose(1, 2).contiguous().to(L.device)
    prod = torch.zeros_like(wr_t)
    permute = _PERMUTE.to(L.device)
    inv_permute = _INV_PERMUTE.to(L.device)

    for cur_col in range(n // args.td_y, 0, -buf_size):
        lo = args.td_y * (cur_col - buf_size)
        hi = args.td_y * cur_col
        b_wr = wr_t[:, lo:hi]
        b_hat = hat_t[:, lo:hi]
        b_l = L[:, lo:hi].contiguous()
        b_prod = prod[:, lo:hi]
        for i in reversed(range(buf_size)):
            il = args.td_y * i
            ih = args.td_y * (i + 1)
            correction = torch.bmm(
                b_l[:, ih:, lo + il : lo + ih].transpose(1, 2),
                b_wr[:, ih:] - b_hat[:, ih:],
            )
            wx = b_wr[:, il:ih] + correction + b_prod[:, il:ih]
            thing = wx.transpose(1, 2).reshape(units, -1, trellis_size)
            if for_kernel:
                thing = thing[..., permute]
            sequence_rows = thing.shape[1]
            q_values, q_indices = cb.quantize(
                thing.reshape(units * sequence_rows, trellis_size)
            )
            if for_kernel:
                q_values = q_values[..., inv_permute]
            q_values = q_values.reshape(units, m, args.td_y)
            b_hat[:, il:ih] = q_values.transpose(1, 2)
            index_rows = args.td_y // args.V
            q_indices = q_indices.reshape(units, m, index_rows)
            idx_lo = lo // args.V + index_rows * i
            idx_hi = idx_lo + index_rows
            idxs_t[:, idx_lo:idx_hi] = q_indices.transpose(1, 2)
        prod.add_(torch.bmm(b_l.transpose(1, 2), b_wr - b_hat))
        hat_t[:, lo:hi] = b_hat

    return hat_t.transpose(1, 2).contiguous(), idxs_t.transpose(1, 2).contiguous()
