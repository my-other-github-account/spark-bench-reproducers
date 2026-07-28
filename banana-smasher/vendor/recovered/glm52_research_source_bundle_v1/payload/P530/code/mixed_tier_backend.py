"""Compact heterogeneous expert backend for the mixed-tier serving gate.

Each model layer owns independent compact templates for four genuinely different
wire/kernel classes: packed QTIP, trueVQ d4, trueVQ d8, and native MXFP4. Values
are uncalibrated serving templates, but every routed expert is dispatched by a
static per-expert tier map to its tier's distinct Triton kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Any

import torch
import triton
import triton.language as tl

TIER_NAMES = ("qtip", "truevq_d4", "truevq_d8", "native_mxfp4")
TIER_CODES = {name: index for index, name in enumerate(TIER_NAMES)}


@triton.jit
def _qtip_gemv(x_ptr, codes_ptr, scales_ptr, cb_ptr, y_ptr,
                R: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        xv = tl.load(x_ptr + r * K + k, mask=k < K, other=0.0).to(tl.float32)
        code = tl.load(codes_ptr + n[:, None] * (K // 8) + (k[None, :] // 8),
                       mask=n_mask[:, None] & (k[None, :] < K), other=0).to(tl.int32)
        centroid = tl.load(cb_ptr + code * 8 + (k[None, :] % 8)).to(tl.float32)
        scale = tl.load(scales_ptr + n[:, None] * (K // 32) + (k[None, :] // 32),
                        mask=n_mask[:, None] & (k[None, :] < K), other=127).to(tl.float32)
        acc += tl.sum(centroid * tl.exp2(scale - 127.0) * xv[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


@triton.jit
def _truevq_d4_gemv(x_ptr, codes_ptr, scales_ptr, cb_ptr, y_ptr,
                     R: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        xv = tl.load(x_ptr + r * K + k, mask=k < K, other=0.0).to(tl.float32)
        code = tl.load(codes_ptr + n[:, None] * (K // 4) + (k[None, :] // 4),
                       mask=n_mask[:, None] & (k[None, :] < K), other=0).to(tl.int32)
        centroid = tl.load(cb_ptr + code * 4 + (k[None, :] % 4)).to(tl.float32)
        scale = tl.load(scales_ptr + n[:, None] * (K // 32) + (k[None, :] // 32),
                        mask=n_mask[:, None] & (k[None, :] < K), other=127).to(tl.float32)
        acc += tl.sum(centroid * tl.exp2(scale - 127.0) * xv[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


@triton.jit
def _truevq_d8_gemv(x_ptr, codes_ptr, scales_ptr, cb_ptr, y_ptr,
                     R: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        xv = tl.load(x_ptr + r * K + k, mask=k < K, other=0.0).to(tl.float32)
        code = tl.load(codes_ptr + n[:, None] * (K // 8) + (k[None, :] // 8),
                       mask=n_mask[:, None] & (k[None, :] < K), other=0).to(tl.int32)
        centroid = tl.load(cb_ptr + code * 8 + (k[None, :] % 8)).to(tl.float32)
        scale = tl.load(scales_ptr + n[:, None] * (K // 32) + (k[None, :] // 32),
                        mask=n_mask[:, None] & (k[None, :] < K), other=127).to(tl.float32)
        acc += tl.sum(centroid * tl.exp2(scale - 127.0) * xv[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


@triton.jit
def _e2m1(code):
    mag = code & 7
    value = tl.where(mag == 0, 0.0,
            tl.where(mag == 1, 0.5,
            tl.where(mag == 2, 1.0,
            tl.where(mag == 3, 1.5,
            tl.where(mag == 4, 2.0,
            tl.where(mag == 5, 3.0,
            tl.where(mag == 6, 4.0, 6.0)))))))
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _native_mxfp4_gemv(x_ptr, packed_ptr, scales_ptr, y_ptr,
                         R: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        xv = tl.load(x_ptr + r * K + k, mask=k < K, other=0.0).to(tl.float32)
        byte = tl.load(packed_ptr + n[:, None] * (K // 2) + (k[None, :] // 2),
                       mask=n_mask[:, None] & (k[None, :] < K), other=0).to(tl.int32)
        nibble = tl.where((k[None, :] & 1) == 0, byte & 15, byte >> 4)
        scale = tl.load(scales_ptr + n[:, None] * (K // 32) + (k[None, :] // 32),
                        mask=n_mask[:, None] & (k[None, :] < K), other=127).to(tl.float32)
        acc += tl.sum(_e2m1(nibble) * tl.exp2(scale - 127.0) * xv[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


# P526 candidate SHA 655634773e941f6fa310235fe1adfbd1803eaa8d9207c9b51640b93d947e98a9,
# integrated at the actual active-tier projection site. This is the same
# explicit-M gather/dequant/tl.dot grid, specialized for the single compact
# template resident for each product tier. No dense weight is retained.
@triton.jit
def _vq_gemm_mbatched(x_ptr, codes_ptr, scales_ptr, cb_ptr, y_ptr,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      BK: tl.constexpr):
    pid_mn = tl.program_id(0)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn - pid_m * grid_n
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    m_mask = m < M
    n_mask = n < N
    acc = tl.zeros((BM, BN), tl.float32)
    codes_k = K // D
    scales_k = K // 32
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        xv = tl.load(
            x_ptr + m[:, None] * K + k[None, :],
            mask=m_mask[:, None] & (k[None, :] < K), other=0.0)
        code = tl.load(
            codes_ptr + n[:, None] * codes_k + (k[None, :] // D),
            mask=n_mask[:, None] & (k[None, :] < K), other=0).to(tl.int32)
        weight = tl.load(cb_ptr + code * D + (k[None, :] % D)).to(tl.float32)
        scale_u8 = tl.load(
            scales_ptr + n[:, None] * scales_k + (k[None, :] // 32),
            mask=n_mask[:, None] & (k[None, :] < K), other=127).to(tl.float32)
        weight *= tl.exp2(scale_u8 - 127.0)
        acc += tl.dot(xv.to(tl.bfloat16), tl.trans(weight.to(tl.bfloat16)))
    tl.store(y_ptr + m[:, None] * N + n[None, :], acc,
             mask=m_mask[:, None] & n_mask[None, :])


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


@dataclass
class Projection:
    tier: str
    codes: torch.Tensor | None
    scales: torch.Tensor
    codebook: torch.Tensor | None
    packed: torch.Tensor | None
    n: int
    k: int
    dense_threshold: int = field(
        default_factory=lambda: int(os.environ.get("P525_DENSE_THRESHOLD", "64")))
    dense_chunk_rows: int = field(
        default_factory=lambda: int(os.environ.get("P525_DENSE_CHUNK_ROWS", "1024")))
    _dispatch: dict[str, int] = field(default_factory=lambda: {
        "triton_calls": 0,
        "mbatched_prefill_calls": 0,
        "mbatched_rows": 0,
        "mbatched_peak_scratch_bytes": 0,
        "dense_prefill_calls": 0,
        "dequantizations": 0,
        "dense_gemm_chunks": 0,
        "dense_rows": 0,
        "dense_weight_bytes": 0,
    }, repr=False)

    @property
    def resident_bytes(self) -> int:
        tensors = [self.scales]
        if self.codes is not None:
            tensors.append(self.codes)
        if self.codebook is not None:
            tensors.append(self.codebook)
        if self.packed is not None:
            tensors.append(self.packed)
        return sum(tensor_bytes(value) for value in tensors)

    @staticmethod
    def _mem_available_bytes() -> int:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("MemAvailable unavailable")

    def dispatch_counters(self) -> dict[str, int]:
        return dict(self._dispatch)

    def dequantize_weight_bf16(self) -> torch.Tensor:
        """Materialize this exact compact projection once as a BF16 [N,K] matrix."""
        scales = torch.exp2(self.scales.to(torch.float32) - 127.0)
        scales = scales.repeat_interleave(32, dim=1)
        if tuple(scales.shape) != (self.n, self.k):
            raise ValueError(f"{self.tier}: bad scale shape {tuple(scales.shape)}")
        if self.tier == "native_mxfp4":
            if self.packed is None:
                raise ValueError("native_mxfp4 projection missing packed tensor")
            low = self.packed.bitwise_and(15)
            high = self.packed.bitwise_right_shift(4)
            nibbles = torch.stack((low, high), dim=-1).reshape(self.n, self.k)
            table = torch.tensor(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                device=self.scales.device, dtype=torch.float32)
            values = table[nibbles.to(torch.long)]
        else:
            if self.codes is None or self.codebook is None:
                raise ValueError(f"{self.tier} projection missing codes/codebook")
            d = int(self.codebook.shape[1])
            values = self.codebook.to(torch.float32).index_select(
                0, self.codes.to(torch.long).reshape(-1))
            values = values.reshape(self.n, self.k // d, d).reshape(self.n, self.k)
        weight = (values * scales).to(torch.bfloat16).contiguous()
        self._dispatch["dequantizations"] += 1
        self._dispatch["dense_weight_bytes"] += tensor_bytes(weight)
        return weight

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.k:
            raise ValueError(f"{self.tier}: expected [R,{self.k}], got {tuple(x.shape)}")
        x = x.to(torch.bfloat16).contiguous()
        r = int(x.shape[0])
        y = torch.empty((r, self.n), device=x.device, dtype=torch.bfloat16)
        grid = (triton.cdiv(self.n, 8), r)
        args = (x, self.codes, self.scales, self.codebook, y)
        if self.tier == "qtip":
            _qtip_gemv[grid](*args, R=r, N=self.n, K=self.k, BN=8, BK=256,
                              num_warps=4, num_stages=2)
        elif self.tier == "truevq_d4":
            _truevq_d4_gemv[grid](*args, R=r, N=self.n, K=self.k, BN=8, BK=256,
                                   num_warps=4, num_stages=2)
        elif self.tier == "truevq_d8":
            _truevq_d8_gemv[grid](*args, R=r, N=self.n, K=self.k, BN=8, BK=256,
                                   num_warps=4, num_stages=2)
        elif self.tier == "native_mxfp4":
            _native_mxfp4_gemv[grid](x, self.packed, self.scales, y,
                                      R=r, N=self.n, K=self.k, BN=8, BK=256,
                                      num_warps=4, num_stages=2)
        else:
            raise ValueError(self.tier)
        self._dispatch["triton_calls"] += 1
        return y

    def forward_mbatched(self, x: torch.Tensor) -> torch.Tensor:
        """P526 explicit-M grid for QTIP/trueVQ prefill; dense copies forbidden."""
        if self.tier == "native_mxfp4" or self.codes is None or self.codebook is None:
            raise ValueError(f"{self.tier}: explicit-M VQ path unavailable")
        x = x.to(torch.bfloat16).contiguous()
        m = int(x.shape[0])
        y = torch.empty((m, self.n), device=x.device, dtype=torch.bfloat16)
        d = int(self.codebook.shape[1])
        bm, bn, bk = 16, 32, 32
        grid = (triton.cdiv(m, bm) * triton.cdiv(self.n, bn),)
        _vq_gemm_mbatched[grid](
            x, self.codes, self.scales, self.codebook, y,
            M=m, N=self.n, K=self.k, D=d, BM=bm, BN=bn, BK=bk,
            num_warps=4, num_stages=2)
        self._dispatch["mbatched_prefill_calls"] += 1
        self._dispatch["mbatched_rows"] += m
        self._dispatch["mbatched_peak_scratch_bytes"] = max(
            self._dispatch["mbatched_peak_scratch_bytes"], 8_388_608)
        return y

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        available = self._mem_available_bytes()
        if available < (8 << 30):
            raise MemoryError(
                f"P525 safety floor: MemAvailable={available} below 8 GiB before dense dispatch")
        x = x.to(torch.bfloat16).contiguous()
        r = int(x.shape[0])
        weight = self.dequantize_weight_bf16()
        y = torch.empty((r, self.n), device=x.device, dtype=torch.bfloat16)
        chunks = 0
        weight_t = weight.t()
        for start in range(0, r, self.dense_chunk_rows):
            stop = min(start + self.dense_chunk_rows, r)
            torch.mm(x[start:stop], weight_t, out=y[start:stop])
            chunks += 1
        self._dispatch["dense_prefill_calls"] += 1
        self._dispatch["dense_gemm_chunks"] += chunks
        self._dispatch["dense_rows"] += r
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.k:
            raise ValueError(f"{self.tier}: expected [R,{self.k}], got {tuple(x.shape)}")
        if int(x.shape[0]) >= self.dense_threshold:
            # Rung selection is fixed at server launch; no persistent dense copy.
            if os.environ.get("P530_PREFILL_MODE") == "dense_all":
                return self.forward_dense(x)
            # Smallest product integration: P526 explicit-M for packed VQ tiers,
            # one-projection-at-a-time streaming dequant+dense for native MXFP4.
            if self.tier == "native_mxfp4":
                return self.forward_dense(x)
            return self.forward_mbatched(x)
        return self.forward_triton(x)


class MixedTierLayer:
    def __init__(self, projections: dict[str, dict[str, Projection]], tier_map: torch.Tensor,
                 layer_index: int, source: str):
        self.projections = projections
        self.tier_map = tier_map
        self.layer_index = int(layer_index)
        self.source = source
        self._counters = {
            tier: {projection: {"expert_projection_operations": 0, "kernel_launches": 0}
                   for projection in ("fused13", "down")}
            for tier in TIER_NAMES
        }
        self.route_calls = 0
        self.heterogeneous_route_calls = 0

    @classmethod
    def from_file(cls, path: str | Path, layer_index: int, device: str = "cuda") -> "MixedTierLayer":
        path = Path(path)
        raw = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        tier_map = raw["tier_map"].to(device=device, dtype=torch.int8).contiguous()
        if tuple(tier_map.shape) != (256,):
            raise ValueError(f"tier_map expected [256], got {tuple(tier_map.shape)}")
        projections: dict[str, dict[str, Projection]] = {}
        for projection, (n, k) in {"fused13": (4096, 4096), "down": (4096, 2048)}.items():
            projections[projection] = {}
            suffix = "13" if projection == "fused13" else "2"
            for tier in TIER_NAMES:
                row = raw[tier]
                scales = row[f"sc{suffix}"].to(device=device).contiguous().clone()
                codes = codebook = packed = None
                if tier == "native_mxfp4":
                    packed = row[f"packed{suffix}"].to(device=device).contiguous().clone()
                    if layer_index:
                        packed[0, 0].bitwise_xor_(layer_index & 15)
                else:
                    codes = row[f"codes{suffix}"].to(device=device).contiguous().clone()
                    codebook = row[f"cb{suffix}"].to(device=device).contiguous().clone()
                    modulus = int(codebook.shape[0])
                    codes[0, 0] = (codes[0, 0].to(torch.int32) + layer_index) % modulus
                projections[projection][tier] = Projection(
                    tier=tier, codes=codes, scales=scales, codebook=codebook,
                    packed=packed, n=n, k=k)
        return cls(projections, tier_map, layer_index, str(path))

    @property
    def resident_bytes(self) -> int:
        return tensor_bytes(self.tier_map) + sum(
            projection.resident_bytes
            for by_tier in self.projections.values()
            for projection in by_tier.values()
        )

    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor, projection: str,
                source_positions: torch.Tensor | None = None) -> torch.Tensor:
        if projection not in self.projections:
            raise ValueError(projection)
        expert_ids = expert_ids.to(device=x.device, dtype=torch.long).reshape(-1)
        x = x.reshape(-1, x.shape[-1])
        if source_positions is None:
            if x.shape[0] != expert_ids.numel():
                raise ValueError(
                    f"{projection}: x rows {x.shape[0]} != expert pairs {expert_ids.numel()}")
            source_positions = torch.arange(
                expert_ids.numel(), device=x.device, dtype=torch.long)
        else:
            source_positions = source_positions.to(
                device=x.device, dtype=torch.long).reshape(-1)
            if source_positions.numel() != expert_ids.numel():
                raise ValueError(
                    f"{projection}: source positions {source_positions.numel()} "
                    f"!= expert pairs {expert_ids.numel()}")
            if source_positions.numel() and int(source_positions.max().item()) >= x.shape[0]:
                raise ValueError(f"{projection}: source position exceeds x rows {x.shape[0]}")
        tier_codes = self.tier_map[expert_ids].to(torch.int64)
        out = torch.empty((expert_ids.numel(), 4096), device=x.device, dtype=torch.bfloat16)
        active = 0
        for tier in TIER_NAMES:
            positions = torch.nonzero(tier_codes == TIER_CODES[tier], as_tuple=False).flatten()
            rows = int(positions.numel())
            if rows == 0:
                continue
            active += 1
            selected = x.index_select(0, positions)
            result = self.projections[projection][tier].forward(selected)
            out.index_copy_(0, positions, result)
            counter = self._counters[tier][projection]
            counter["expert_projection_operations"] += rows
            counter["kernel_launches"] += 1
        self.route_calls += 1
        if active >= 2:
            self.heterogeneous_route_calls += 1
        return out

    def counters(self) -> dict[str, Any]:
        return {
            "route_calls": self.route_calls,
            "heterogeneous_route_calls": self.heterogeneous_route_calls,
            "tiers": {
                tier: {
                    projection: {
                        **dict(values),
                        **self.projections[projection][tier].dispatch_counters(),
                    }
                    for projection, values in by_projection.items()
                }
                for tier, by_projection in self._counters.items()
            },
        }

    def sentinel(self) -> dict[str, Any]:
        values = []
        for projection in ("fused13", "down"):
            for tier in TIER_NAMES:
                row = self.projections[projection][tier]
                tensor = row.packed if tier == "native_mxfp4" else row.codes
                assert tensor is not None
                values.append(int(tensor[0, 0].item()))
        return {"layer": self.layer_index, "sentinel": values,
                "resident_bytes": self.resident_bytes}

    def pointers(self) -> list[int]:
        pointers = [int(self.tier_map.data_ptr())]
        for projection in ("fused13", "down"):
            for tier in TIER_NAMES:
                row = self.projections[projection][tier]
                for tensor in (row.codes, row.scales, row.codebook, row.packed):
                    if tensor is not None:
                        pointers.append(int(tensor.data_ptr()))
        return pointers
