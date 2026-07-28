"""Packed d8/k256 VQ kernels for DS4-Flash MoE layers.

The loader keeps uint8 code/scales and fp16 codebooks resident.  It never
materializes a persistent dense/upcast weight.  API shape:

    layer = PackedQTIPLayer.from_file(path, device="cuda")
    y = layer.forward(x, expert_ids, projection="fused13")

x is [R, M, K] and expert_ids is [R].  Output is [R, M, 4096].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import triton
import triton.language as tl


@triton.jit
def _vq_gemv(
    x_ptr, codes_ptr, scales_ptr, cb_ptr, expert_ptr, y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_rm = tl.program_id(1)
    r = pid_rm // M
    m = pid_rm - r * M
    e = tl.load(expert_ptr + r).to(tl.int64)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    acc = tl.zeros((BN,), tl.float32)
    codes_k = K // 8
    scales_k = K // 32
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        xval = tl.load(x_ptr + (r * M + m) * K + k, mask=k < K, other=0.0).to(tl.float32)
        code = tl.load(
            codes_ptr + e * N * codes_k + n[:, None] * codes_k + (k[None, :] // 8),
            mask=n_mask[:, None] & (k[None, :] < K), other=0,
        ).to(tl.int32)
        lane = k[None, :] % 8
        weight = tl.load(cb_ptr + code * 8 + lane).to(tl.float32)
        scale_u8 = tl.load(
            scales_ptr + e * N * scales_k + n[:, None] * scales_k + (k[None, :] // 32),
            mask=n_mask[:, None] & (k[None, :] < K), other=127,
        ).to(tl.float32)
        weight *= tl.exp2(scale_u8 - 127.0)
        acc += tl.sum(weight * xval[None, :], axis=1)
    tl.store(y_ptr + (r * M + m) * N + n, acc, mask=n_mask)


@triton.jit
def _vq_gemm(
    x_ptr, codes_ptr, scales_ptr, cb_ptr, expert_ptr, y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    r = tl.program_id(1)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn - pid_m * grid_n
    e = tl.load(expert_ptr + r).to(tl.int64)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    m_mask = m < M
    n_mask = n < N
    acc = tl.zeros((BM, BN), tl.float32)
    codes_k = K // 8
    scales_k = K // 32
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        xv = tl.load(
            x_ptr + (r * M + m[:, None]) * K + k[None, :],
            mask=m_mask[:, None] & (k[None, :] < K), other=0.0,
        )
        code = tl.load(
            codes_ptr + e * N * codes_k + n[:, None] * codes_k + (k[None, :] // 8),
            mask=n_mask[:, None] & (k[None, :] < K), other=0,
        ).to(tl.int32)
        weight = tl.load(cb_ptr + code * 8 + (k[None, :] % 8)).to(tl.float32)
        scale_u8 = tl.load(
            scales_ptr + e * N * scales_k + n[:, None] * scales_k + (k[None, :] // 32),
            mask=n_mask[:, None] & (k[None, :] < K), other=127,
        ).to(tl.float32)
        weight *= tl.exp2(scale_u8 - 127.0)
        # Tensor-core dot: reconstructed tile is ephemeral register data only.
        acc += tl.dot(xv.to(tl.bfloat16), tl.trans(weight.to(tl.bfloat16)))
    tl.store(
        y_ptr + (r * M + m[:, None]) * N + n[None, :],
        acc, mask=m_mask[:, None] & n_mask[None, :],
    )


@dataclass
class PackedProjection:
    codes: torch.Tensor
    scales: torch.Tensor
    codebook: torch.Tensor
    n: int
    k: int

    @property
    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.codes, self.scales, self.codebook))

    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != expert_ids.numel() or x.shape[2] != self.k:
            raise ValueError(f"expected x [R,M,{self.k}] and expert_ids [R], got {tuple(x.shape)} {tuple(expert_ids.shape)}")
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = x.contiguous()
        expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
        r, m, _ = x.shape
        y = torch.empty((r, m, self.n), device=x.device, dtype=torch.bfloat16)
        if m <= 4:
            # SM121 tuning on real DS4 shapes: BN=8/BK=256/4 warps won the
            # 12-point launch race for both 4096x4096 and 4096x2048.
            bn, bk = 8, 256
            _vq_gemv[(triton.cdiv(self.n, bn), r * m)](
                x, self.codes, self.scales, self.codebook, expert_ids, y,
                M=m, N=self.n, K=self.k, BN=bn, BK=bk,
                num_warps=4, num_stages=2,
            )
        else:
            bm, bn, bk = 16, 32, 32
            _vq_gemm[(triton.cdiv(m, bm) * triton.cdiv(self.n, bn), r)](
                x, self.codes, self.scales, self.codebook, expert_ids, y,
                M=m, N=self.n, K=self.k, BM=bm, BN=bn, BK=bk,
                num_warps=4, num_stages=2,
            )
        return y


@dataclass
class PackedQTIPLayer:
    fused13: PackedProjection
    down: PackedProjection
    source: str

    @classmethod
    def from_file(cls, path: str | Path, device: str = "cuda") -> "PackedQTIPLayer":
        path = Path(path)
        data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        expected = {
            "codes13": ((256, 4096, 512), torch.uint8),
            "sc13": ((256, 4096, 128), torch.uint8),
            "cb13": ((256, 8), torch.float16),
            "codes2": ((256, 4096, 256), torch.uint8),
            "sc2": ((256, 4096, 64), torch.uint8),
            "cb2": ((256, 8), torch.float16),
        }
        for key, (shape, dtype) in expected.items():
            if tuple(data[key].shape) != shape or data[key].dtype != dtype:
                raise ValueError(f"{key}: expected {shape}/{dtype}, got {tuple(data[key].shape)}/{data[key].dtype}")
        def move(key: str) -> torch.Tensor:
            return data[key].to(device=device, non_blocking=False).contiguous()
        return cls(
            fused13=PackedProjection(move("codes13"), move("sc13"), move("cb13"), 4096, 4096),
            down=PackedProjection(move("codes2"), move("sc2"), move("cb2"), 4096, 2048),
            source=str(path),
        )

    @property
    def resident_bytes(self) -> int:
        return self.fused13.resident_bytes + self.down.resident_bytes

    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor,
                projection: Literal["fused13", "down"] = "fused13") -> torch.Tensor:
        return getattr(self, projection).forward(x, expert_ids)
