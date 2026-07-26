#!/usr/bin/env python3
"""Trainable codebook surface over the sealed GENESIS physical wire.

The physical package remains immutable. VQ codebooks are copied into fp32 master
parameters and cast through their original fp16 wire representation on every
forward. Codes, E8M0 scales, expert assignments, and native MXFP4 rows remain
frozen. The forward layout matches genesis_stream_eval.GenesisTierSource.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fused_expert_linear import fused_native_linear, fused_vq_linear

EXPERTS = 256
ROWS = 4096
PROJECTION_LAYOUT = {
    "13": {"assignment": "fused13", "file": "fused13", "out_cols": 4096},
    "2": {"assignment": "down", "file": "down", "out_cols": 2048},
}
_E2M1 = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def tier_params(tier: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"d(\d+)_k(\d+)", tier)
    if not match:
        raise ValueError(f"unsupported VQ tier: {tier}")
    d, k = map(int, match.groups())
    bits = int(math.log2(k))
    if (1 << bits) != k:
        raise ValueError(f"codebook size is not a power of two: {tier}")
    return d, k, bits


def unpack_packed_codes(
    path: str | os.PathLike[str],
    bits: int,
    *,
    offset_values: int,
    count_values: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if bits <= 0 or bits > 16:
        raise ValueError(f"unsupported packed width: {bits}")
    if (offset_values * bits) % 8 or (count_values * bits) % 8:
        raise ValueError("packed-code range must be byte-aligned")
    byte_offset = offset_values * bits // 8
    byte_count = count_values * bits // 8
    with Path(path).open("rb") as handle:
        handle.seek(byte_offset)
        raw = handle.read(byte_count)
    if len(raw) != byte_count:
        raise ValueError(
            f"short packed-code read: {path} offset={byte_offset} "
            f"expected={byte_count} actual={len(raw)}"
        )
    # Decode on the consumer device.  The old np.unpackbits path expanded every
    # packed bit to a byte and then formed a [count,bits] uint32 temporary.  A
    # production chunk can therefore allocate multiple GiB and spend minutes
    # in CPU reduction before the GPU sees any work.  Three-byte little-endian
    # gather handles every supported width (<=16 bits, maximum shift 7) without
    # the bit matrix and can execute directly on CUDA.
    packed = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).to(device)
    if bits == 8:
        values = packed
    else:
        positions = torch.arange(count_values, device=packed.device, dtype=torch.int64) * bits
        byte = torch.bitwise_right_shift(positions, 3)
        shift = torch.bitwise_and(positions, 7)
        padded = torch.nn.functional.pad(packed, (0, 2))
        words = (
            padded[byte].to(torch.int32)
            | (padded[byte + 1].to(torch.int32) << 8)
            | (padded[byte + 2].to(torch.int32) << 16)
        )
        values = torch.bitwise_and(
            torch.bitwise_right_shift(words, shift), (1 << bits) - 1
        )
    if values.numel() != count_values:
        raise AssertionError(
            f"packed-code decode count drift: {values.numel()} != {count_values}"
        )
    # CUDA keeps immutable indices int32 until the actual gather/reduction,
    # halving the resident saved-tensor footprint.  Preserve the historical
    # CPU int64 API used by inspection tools and tests.
    return values.long() if packed.device.type == "cpu" else values.to(torch.int32)


class BatchedGenericVQDeqFn(torch.autograd.Function):
    """Exact fp16-wire VQ forward with STE gradient to one shared codebook."""

    @staticmethod
    def forward(ctx, codebook32, codes, scales):
        wire = codebook32.detach().to(torch.float16).float()
        scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
        out = (
            wire[codes.long()].reshape(codes.shape[0], codes.shape[1], -1)
            * scale_columns
        ).to(torch.bfloat16)
        ctx.save_for_backward(codes, scales)
        ctx.k = int(codebook32.shape[0])
        ctx.d = int(codebook32.shape[1])
        return out

    @staticmethod
    def backward(ctx, grad):
        codes, scales = ctx.saved_tensors
        require_deterministic = (
            os.environ.get("GENESIS_REPAIR_REQUIRE_DETERMINISTIC_REDUCTION", "0") == "1"
        )
        if grad.is_cuda and require_deterministic and not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("deterministic GENESIS codebook reduction was explicitly required")
        scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
        grouped = (grad.float() * scale_columns).reshape(
            codes.shape[0], codes.shape[1], codes.shape[2], ctx.d
        )
        grad_codebook = torch.zeros(
            ctx.k, ctx.d, device=grad.device, dtype=torch.float32
        )
        # With deterministic algorithms enabled, CUDA index_add_ uses a fixed
        # reduction order for duplicate centroid indices. Accumulation remains
        # fp32; fail closed above rather than falling back to atomic order.
        grad_codebook.index_add_(
            0, codes.reshape(-1).long(), grouped.reshape(-1, ctx.d)
        )
        return grad_codebook, None, None


def _generic_vq_dequant_eager(
    codebook32: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    return BatchedGenericVQDeqFn.apply(codebook32, codes, scales)


# This bounded graph covers only the VQ gather/scale/cast custom autograd op.
# It leaves file IO, routing, expert GEMMs, and the transformer layer graph in
# eager mode, so layer-level activation checkpointing remains memory-safe.
# Fixed-size chunk padding below removes the 1..7 tail-batch guard family.
# Keep a larger hard limit as a fail-closed safety margin for the d4/d8 and
# fused13/down families rather than falling back to eager mid-update.
torch._dynamo.config.recompile_limit = max(
    int(torch._dynamo.config.recompile_limit), 32
)
_generic_vq_dequant_compiled = torch.compile(
    _generic_vq_dequant_eager,
    fullgraph=True,
    dynamic=True,
    mode="default",
)


def generic_vq_dequant(
    codebook32: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    if os.environ.get("GENESIS_REPAIR_COMPILE_VQ", "0") == "1":
        return _generic_vq_dequant_compiled(codebook32, codes, scales)
    return _generic_vq_dequant_eager(codebook32, codes, scales)


def warm_compiled_vq_dequant() -> dict[str, object]:
    """Compile both projection-width graphs outside the timed update."""
    enabled = os.environ.get("GENESIS_REPAIR_COMPILE_VQ", "0") == "1"
    if not enabled:
        return {"enabled": False, "status": "disabled"}
    started = time.perf_counter()
    rows = []
    # Warm the exact fixed batch/row shapes used by the real update before its
    # timer starts.  d4 covers all production VQ tiers; d8 covers the small
    # auxiliary tier.  Dynamic k then spans 256/1024/2048/4096 codebooks.
    shapes = (
        (8, 4096, 2048, 256, 4),
        (8, 4096, 4096, 1024, 4),
        (8, 4096, 2048, 256, 8),
        (8, 4096, 4096, 256, 8),
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for chunk, nrows, out_cols, k, d in shapes:
            codebook = torch.randn(k, d, device="cuda", dtype=torch.float32,
                                   requires_grad=True)
            codes = torch.randint(
                0, k, (chunk, nrows, out_cols // d),
                device="cuda", dtype=torch.int32,
            )
            scales = torch.full(
                (chunk, nrows, out_cols // 32), 127,
                device="cuda", dtype=torch.uint8,
            )
            grad = torch.randn(
                chunk, nrows, out_cols, device="cuda", dtype=torch.bfloat16,
            )
            output = _generic_vq_dequant_compiled(codebook, codes, scales)
            output.backward(grad)
            torch.cuda.synchronize()
            if not torch.isfinite(output).all() or codebook.grad is None \
                    or not torch.isfinite(codebook.grad).all():
                raise RuntimeError(f"compiled VQ warmup non-finite: {out_cols}")
            rows.append({"shape": [chunk, nrows, out_cols, k, d],
                         "output_finite": True, "gradient_finite": True})
            del codebook, codes, scales, grad, output
    torch.cuda.empty_cache()
    return {
        "enabled": True,
        "status": "PASS",
        "mode": "default-no-cudagraphs",
        "fullgraph": True,
        "dynamic": True,
        "warmup_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def dequant_native_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 3 or scales.ndim != 3:
        raise ValueError("native MXFP4 tensors must be [batch,rows,columns]")
    table = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    values = table[nibbles.long()]
    scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
    if values.shape != scale_columns.shape:
        raise ValueError(
            f"native MXFP4 shape mismatch: values={tuple(values.shape)} "
            f"scales={tuple(scale_columns.shape)}"
        )
    return (values * scale_columns).to(torch.bfloat16)


def parameter_key(tier: str, projection: str) -> str:
    return f"{tier}__{projection}"


def surface_state(student, *, layers: Iterable[int] = range(43)) -> dict[str, dict[str, torch.Tensor]]:
    return {
        f"L{layer}": {
            name: parameter.detach().cpu().clone()
            for name, parameter in student.experts[layer].named_codebooks()
        }
        for layer in layers
    }


def load_surface_state(
    student,
    state: dict[str, dict[str, torch.Tensor]],
    *,
    layers: Iterable[int] = range(43),
    device: str | torch.device = "cuda",
) -> None:
    for layer in layers:
        live = dict(student.experts[layer].named_codebooks())
        saved = state[f"L{layer}"]
        if set(live) != set(saved):
            raise RuntimeError(
                f"GENESIS codebook state keys mismatch L{layer}: "
                f"live={sorted(live)} saved={sorted(saved)}"
            )
        for name, parameter in live.items():
            parameter.data.copy_(saved[name].to(device))


def surface_parameters(student, *, layers: Iterable[int] = range(43)) -> list[nn.Parameter]:
    return [
        parameter
        for layer in layers
        for _name, parameter in student.experts[layer].named_codebooks()
    ]


def _assignment_tier(entry: object, projection: str) -> str:
    if not isinstance(entry, dict):
        return str(entry)
    return str(entry[PROJECTION_LAYOUT[projection]["assignment"]])


def _evict_file(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable")


def _memory_floor_guard(where: str) -> None:
    floor = int(os.environ.get("GENESIS_REPAIR_MEM_FLOOR_BYTES", str(8 * 1024**3)))
    available = _mem_available_bytes()
    if available >= floor:
        return
    root = Path(os.environ["GENESIS_REPAIR_ROOT"])
    destination = root / "run/MEMORY_FLOOR_STOP.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": "repair-reduced-resident-memory-floor-stop-v1",
        "task_id": os.environ.get("GENESIS_TASK_ID"),
        "where": where,
        "mem_floor_bytes": floor,
        "mem_available_bytes": available,
        "preserved_checkpoint": os.environ.get("GENESIS_REPAIR_CANARY_SEED"),
        "stopped_unix": time.time(),
    }
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    raise RuntimeError(
        f"MEMORY_FLOOR_STOP at {where}: MemAvailable={available} floor={floor}"
    )


class GenesisPhysicalExperts(nn.Module):
    """Drop-in lp4_train expert module backed by immutable physical wire bytes."""

    def __init__(self, layer: int, pilot: bool):
        super().__init__()
        if not pilot:
            raise AssertionError(f"GENESIS BASIC repair requires all layers trainable: L{layer}")
        self.L = int(layer)
        self.limit = 10.0
        self.act = F.silu
        self.device = torch.device(os.environ.get("GENESIS_REPAIR_DEVICE", "cuda"))
        self.evict_after_use = os.environ.get("GENESIS_REPAIR_EVICT", "1") == "1"
        self.root = Path(os.environ["GENESIS_PHYSICAL_PACKAGE"]).expanduser().resolve()
        assignment_path = Path(os.environ["GENESIS_ASSIGNMENT"]).expanduser().resolve()
        assignment = json.loads(assignment_path.read_text())["assignment"]
        self.layer_map = assignment[str(layer)]
        self.layer_dir = self.root / f"layer_{layer:03d}"
        if not self.layer_dir.is_dir():
            raise FileNotFoundError(self.layer_dir)

        self.groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
        # P662 builds the routed expert -> (tier, physical row) metadata once.
        # The P649 path rescanned every tier's full expert list for each
        # resident chunk, even though this mapping is immutable for a layer.
        self.expert_lookup: dict[str, dict[int, tuple[str, int]]] = {
            projection: {} for projection in PROJECTION_LAYOUT
        }
        self.group_files: dict[tuple[str, str], dict[str, object]] = {}
        self.codebooks = nn.ParameterDict()
        for projection, layout in PROJECTION_LAYOUT.items():
            tiers = sorted(
                {_assignment_tier(self.layer_map[str(expert)], projection) for expert in range(EXPERTS)}
            )
            for tier in tiers:
                suffix = str(layout["file"])
                expected = [
                    expert for expert in range(EXPERTS)
                    if _assignment_tier(self.layer_map[str(expert)], projection) == tier
                ]
                ids_path = self.layer_dir / f"{tier}.{suffix}.expert_ids.i16.bin"
                if not ids_path.is_file():
                    raise FileNotFoundError(ids_path)
                ids = np.fromfile(ids_path, dtype="<i2").astype(np.int64).tolist()
                if sorted(ids) != expected or len(ids) != len(set(ids)):
                    raise RuntimeError(
                        f"GENESIS expert-id drift L{layer} {tier} p{projection}: "
                        f"expected={expected[:8]} actual={ids[:8]}"
                    )
                pairs = [(expert, row) for row, expert in enumerate(ids)]
                self.groups[(projection, tier)] = pairs
                for expert, row in pairs:
                    self.expert_lookup[projection][expert] = (tier, row)
                if tier == "native_mxfp4":
                    self.group_files[(projection, tier)] = {
                        "ids": ids,
                        "weights": self.layer_dir / f"{tier}.{suffix}.weights.mxfp4.bin",
                        "scales": self.layer_dir / f"{tier}.{suffix}.scales.e8m0.bin",
                    }
                    continue
                d, k, bits = tier_params(tier)
                codebook_path = self.layer_dir / f"{tier}.{suffix}.codebook.fp16.bin"
                codes_path = self.layer_dir / f"{tier}.{suffix}.codes.le{bits}.bin"
                scales_path = self.layer_dir / f"{tier}.{suffix}.scales.e8m0.bin"
                codebook = np.memmap(codebook_path, dtype="<f2", mode="r", shape=(k, d))
                self.codebooks[parameter_key(tier, projection)] = nn.Parameter(
                    torch.from_numpy(np.array(codebook, copy=True)).float().to(self.device)
                )
                del codebook
                self.group_files[(projection, tier)] = {
                    "ids": ids,
                    "d": d,
                    "k": k,
                    "bits": bits,
                    "codes": codes_path,
                    "scales": scales_path,
                }

    def named_codebooks(self):
        return self.codebooks.items()

    def _vq_payloads(
        self, projection: str, tier: str, selected: list[tuple[int, int]]
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Load/decode only routed experts, but do not form dense weights."""
        info = self.group_files[(projection, tier)]
        d = int(info["d"])
        bits = int(info["bits"])
        out_cols = int(PROJECTION_LAYOUT[projection]["out_cols"])
        code_width = out_cols // d
        scale_cols = out_cols // 32
        nrows = len(info["ids"])
        scales_mm = np.memmap(
            info["scales"], dtype=np.uint8, mode="r",
            shape=(nrows, ROWS, scale_cols),
        )
        payloads: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        selected = sorted(selected, key=lambda pair: pair[1])
        # P649 selected resident scope 4 / payload chunk 1 after the exact
        # exhaustive canonical B32 sweep plus full43 memory-floor eligibility:
        # the faster r16 and r8 families both fell below 32 GiB.
        # Use a P649-specific diagnostic override so legacy launchers exporting
        # the P643 value cannot silently disable the adopted postimage.
        chunk_size = int(
            os.environ.get("P662_P13_DEQ_CHUNK", "4")
            if projection == "13"
            else os.environ.get("P649_DEQ_CHUNK", "1")
        )
        if chunk_size <= 0 or chunk_size > 256:
            raise ValueError(f"invalid P662/P649 deq chunk={chunk_size}")
        values_per_row = ROWS * code_width
        for start in range(0, len(selected), chunk_size):
            chunk = selected[start : start + chunk_size]
            experts = [expert for expert, _row in chunk]
            rows = [row for _expert, row in chunk]
            if rows == list(range(rows[0], rows[0] + len(rows))):
                codes = unpack_packed_codes(
                    info["codes"], bits,
                    offset_values=rows[0] * values_per_row,
                    count_values=len(rows) * values_per_row,
                    device=self.device,
                ).reshape(len(rows), ROWS, code_width)
            else:
                codes = torch.stack([
                    unpack_packed_codes(
                        info["codes"], bits,
                        offset_values=row * values_per_row,
                        count_values=values_per_row,
                        device=self.device,
                    ).reshape(ROWS, code_width)
                    for row in rows
                ])
            scales = torch.from_numpy(np.array(scales_mm[rows], copy=True)).to(self.device)
            for index, expert in enumerate(experts):
                payloads[expert] = (codes[index], scales[index])
            del codes, scales
        del scales_mm
        if self.evict_after_use:
            _evict_file(Path(info["codes"]))
            _evict_file(Path(info["scales"]))
        return payloads

    def _native_payloads(
        self, projection: str, tier: str, selected: list[tuple[int, int]]
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Load only routed packed MXFP4 experts; dequantization stays fused."""
        info = self.group_files[(projection, tier)]
        out_cols = int(PROJECTION_LAYOUT[projection]["out_cols"])
        packed_cols = out_cols // 2
        scale_cols = out_cols // 32
        nrows = len(info["ids"])
        weights_mm = np.memmap(
            info["weights"], dtype=np.uint8, mode="r",
            shape=(nrows, ROWS, packed_cols),
        )
        scales_mm = np.memmap(
            info["scales"], dtype=np.uint8, mode="r",
            shape=(nrows, ROWS, scale_cols),
        )
        payloads: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        selected = sorted(selected, key=lambda pair: pair[1])
        chunk_size = int(
            os.environ.get("P662_P13_NATIVE_CHUNK", "4")
            if projection == "13"
            else os.environ.get("P649_NATIVE_CHUNK", "1")
        )
        if chunk_size <= 0 or chunk_size > 256:
            raise ValueError(f"invalid P662/P649 native chunk={chunk_size}")
        for start in range(0, len(selected), chunk_size):
            chunk = selected[start : start + chunk_size]
            experts = [expert for expert, _row in chunk]
            rows = [row for _expert, row in chunk]
            packed = torch.from_numpy(np.array(weights_mm[rows], copy=True)).to(self.device)
            scales = torch.from_numpy(np.array(scales_mm[rows], copy=True)).to(self.device)
            for index, expert in enumerate(experts):
                payloads[expert] = (packed[index], scales[index])
            del packed, scales
        del weights_mm, scales_mm
        if self.evict_after_use:
            _evict_file(Path(info["weights"]))
            _evict_file(Path(info["scales"]))
        return payloads

    def _payloads_for(
        self, projection: str, hit_ids: list[int]
    ) -> dict[int, tuple[str, str, torch.Tensor, torch.Tensor]]:
        payloads: dict[int, tuple[str, str, torch.Tensor, torch.Tensor]] = {}
        selected_by_tier: dict[str, list[tuple[int, int]]] = {}
        lookup = self.expert_lookup[projection]
        for expert in hit_ids:
            tier, row = lookup[expert]
            selected_by_tier.setdefault(tier, []).append((expert, row))
        for tier, selected in selected_by_tier.items():
            if tier == "native_mxfp4":
                loaded = self._native_payloads(projection, tier, selected)
            else:
                loaded = self._vq_payloads(projection, tier, selected)
            for expert, (encoded, scales) in loaded.items():
                payloads[expert] = (projection, tier, encoded, scales)
        hit_set = set(hit_ids)
        if set(payloads) != hit_set:
            raise RuntimeError(
                f"GENESIS fused payload coverage drift L{self.L} p{projection}: "
                f"missing={sorted(hit_set - set(payloads))[:8]}"
            )
        return payloads

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=EXPERTS).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        hit_ids = [int(expert[0]) for expert in hit]
        intermediates: dict[int, torch.Tensor] = {}
        positions: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        resident_scope = int(os.environ.get("P649_EXPERT_RESIDENT_SCOPE", "4"))
        if not 1 <= resident_scope <= 16:
            raise ValueError(
                f"invalid P649_EXPERT_RESIDENT_SCOPE={resident_scope}"
            )
        # P13 experts are independent until the later ordered p2 index_add.
        # Grouping by immutable (tier,row) preserves each expert's math while
        # making the resident-four mmap/decode microbatch contiguous.
        p13_hit_ids = sorted(
            hit_ids,
            key=lambda expert: self.expert_lookup["13"][expert],
        )
        for start in range(0, len(p13_hit_ids), resident_scope):
            expert_chunk = p13_hit_ids[start : start + resident_scope]
            _memory_floor_guard(
                f"L{self.L}:projection13:chunk{start // resident_scope}:before"
            )
            payloads13 = self._payloads_for("13", expert_chunk)
            for expert in expert_chunk:
                top_k_pos, token_idx = torch.where(mask[expert])
                projection, tier, encoded, scales = payloads13[expert]
                if tier == "native_mxfp4":
                    current = fused_native_linear(hidden_states[token_idx], encoded, scales)
                else:
                    codebook = self.codebooks[parameter_key(tier, projection)]
                    current = fused_vq_linear(
                        hidden_states[token_idx], codebook, encoded, scales
                    )
                gate, up = current.chunk(2, dim=-1)
                intermediates[expert] = F.silu(gate.clamp(max=self.limit)) * up.clamp(
                    min=-self.limit, max=self.limit
                )
                positions[expert] = (top_k_pos, token_idx)
            del payloads13
            _memory_floor_guard(
                f"L{self.L}:projection13:chunk{start // resident_scope}:after"
            )
        for start in range(0, len(hit_ids), resident_scope):
            expert_chunk = hit_ids[start : start + resident_scope]
            _memory_floor_guard(
                f"L{self.L}:projection2:chunk{start // resident_scope}:before"
            )
            payloads2 = self._payloads_for("2", expert_chunk)
            for expert in expert_chunk:
                top_k_pos, token_idx = positions[expert]
                projection, tier, encoded, scales = payloads2[expert]
                if tier == "native_mxfp4":
                    current = fused_native_linear(intermediates[expert], encoded, scales)
                else:
                    codebook = self.codebooks[parameter_key(tier, projection)]
                    current = fused_vq_linear(
                        intermediates[expert], codebook, encoded, scales
                    )
                current = current * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, current.to(final.dtype))
            del payloads2
            _memory_floor_guard(
                f"L{self.L}:projection2:chunk{start // resident_scope}:after"
            )
        del intermediates, positions
        return final
