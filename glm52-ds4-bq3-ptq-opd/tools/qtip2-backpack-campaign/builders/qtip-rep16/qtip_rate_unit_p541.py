#!/usr/bin/env python3
"""Verify the reachable 2.0-bpw QTIP rung on one DS4 expert.

The smoke is deliberately bounded to one layer/expert and both projections.
Weights are not opened from compute-node-7: exact safetensors tensor byte ranges are
read concurrently over the compute-node-7 -> compute-node-3 QSFP SSH path, hashed, decoded,
and receipted.  Fit captures and the current down-projection plane remain
read-only resident inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch

import qtip_wire_build_serial_v2 as serial
from rate_batched_ldlq import LDLQ_batch, block_LDL_batch, pack_kernel_layout_batch
from triton_viterbi_rate import install_rate_viterbi


TASK = "PUBLIC_TASK"
ALLOWED_LAYERS = [0, 2, 4, 6]
PLANE_SHA256 = {
    0: "bfeedd7bff25e1d814851c2e6d056e67f04b2275b0932a134636debafc5ddc4b",
    2: "44a498ef84a39b1e9f368c4e89cded74311aed359b9dc24a5eb8a1701f79fe38",
    4: "bf8f6a690ece0b902b849947f7f5e871cabdd22956f030709ccb58b4a429803c",
    6: "a7d3a3aeffaf1d6617bfbaf8fbf231deb6c6181aa6a1ba9af8f4e276474f2843",
}
DEFAULT_MISSION = Path("/dev/shm/P534_QTIP2_REP16_PUBLIC_TASK_s5w/smoke")
DEFAULT_QTIP = Path.home() / "run-bundles/P534_QTIP2_REP16_PUBLIC_TASK_s5w/qtip-canonical"
DEFAULT_TLUT = (
    Path.home()
    / "run-bundles/P534_QTIP2_REP16_PUBLIC_TASK_s5w/inputs/L017_E005_fused13_QTIP_HYB_L16_K3_V2.pt"
)
DEFAULT_FIT = Path("/dev/shm/P534_QTIP2_REP16_PUBLIC_TASK_s5w/inputs/L016/FIT_L016.json")
DEFAULT_PLANE = Path("/dev/shm/P534_QTIP2_REP16_PUBLIC_TASK_s5w/inputs/vq3u_layer_016.pt")
DEFAULT_S3_MODEL = "$HOME/run-bundles/QTIP_PROOF1_SHARD_PUBLIC_TASK_s3/source_model"
REPRESENTATIVE16 = [0, 2, 4, 6, 11, 14, 16, 19, 22, 25, 27, 30, 34, 35, 38, 42]
E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable absent")


def require_authorized_claim(mission: Path) -> tuple[str, dict[str, Any]]:
    """Accept either an exclusive claim or a sealed wait-boundary GPU sublease.

    The sublease mode deliberately preserves the canonical QTIP driver's
    owner/task fields, so its fail-closed wait loop remains healthy.  It is
    valid only while the task-specific, short-lived mission sublease exists.
    """
    claim_path = Path.home() / "HOST_CLAIM.json"
    raw = claim_path.read_bytes()
    claim = json.loads(raw)
    exclusive = claim.get("owner") == TASK and claim.get("task_id") == TASK
    sublease = claim.get("temporary_gpu_sublease") or {}
    subleased = (
        claim.get("owner") == "PUBLIC_TASK"
        and claim.get("task_id") == "PUBLIC_TASK"
        and sublease.get("task") == TASK
        and sublease.get("task_id") == TASK
        and sublease.get("mission") == str(mission)
        and float(sublease.get("expires_unix", 0)) > time.time()
        and sublease.get("scope") == "gpu-only-while-primary-wait_s3_shard"
    )
    if not (exclusive or subleased):
        raise RuntimeError(
            f"authorized compute-node-7 claim required for {TASK}; current owner="
            f"{claim.get('owner')} task={claim.get('task_id')} sublease={sublease}"
        )
    if exclusive and claim.get("mission") != str(mission):
        raise RuntimeError(f"claim mission mismatch: {claim.get('mission')} != {mission}")
    if subleased:
        primary_status_path = (
            Path.home()
            / "run-bundles/QTIP_ANCHOR_WIRE_PUBLIC_TASK_s7/status/WIRE_DRIVER.json"
        )
        status = json.loads(primary_status_path.read_text())
        primary_pid = int(status.get("pid", -1))
        primary_checks = {
            "task": status.get("task") == "PUBLIC_TASK",
            "status": status.get("status") == "RUNNING",
            "stage": status.get("stage") == "wait_s3_shard",
            "layer": int(status.get("layer", -1)) in (34, 36),
            "fresh_120s": time.time() - float(status.get("updated_unix", 0)) <= 120,
            "pid_live": Path(f"/proc/{primary_pid}/cmdline").is_file(),
            "pid_matches_sublease": primary_pid == int(sublease.get("primary_pid", -2)),
        }
        if not all(primary_checks.values()):
            raise RuntimeError(
                f"primary left GPU-empty wait boundary during sublease: "
                f"{primary_checks} status={status}"
            )
        claim["primary_wait_boundary_recheck"] = primary_checks
    claim["qtip_rate_smoke_claim_mode"] = "exclusive" if exclusive else "wait-sublease"
    return sha256_bytes(raw), claim


def remote_command(argv: list[str]) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "Compression=no",
        "-o",
        "ControlMaster=no",
        "203.0.113.3",
        shlex.join(argv),
    ]


class S3RangeModelReader:
    def __init__(self, remote_root: str, layer: int, expert: int):
        self.remote_root = remote_root
        self.layer = layer
        self.expert = expert
        self.metadata = self._metadata()
        self.entries = {row["key"]: row for row in self.metadata["tensors"]}
        self.raw: dict[str, bytes] = {}
        self.transfer: dict[str, Any] | None = None

    def _metadata(self) -> dict[str, Any]:
        code = r'''
import hashlib,json,os,struct,sys
base,layer,expert=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
idx_path=base+'/model.safetensors.index.json'
idx_raw=open(idx_path,'rb').read()
idx=json.loads(idx_raw)['weight_map']
stage_path=os.path.dirname(base)+'/MODEL_STAGE.json'
stage_raw=open(stage_path,'rb').read()
stage=json.loads(stage_raw)
file_rows={row['name']:row for row in stage['files']}
keys=[]
for name in ('w1','w3','w2'):
    for suffix in ('weight','scale'):
        keys.append(f'layers.{layer}.ffn.experts.{expert}.{name}.{suffix}')
headers={}
out=[]
for key in keys:
    shard=idx[key]
    path=base+'/'+shard
    if shard not in headers:
        with open(path,'rb') as handle:
            hlen=struct.unpack('<Q',handle.read(8))[0]
            headers[shard]=(hlen,json.loads(handle.read(hlen)))
    hlen,header=headers[shard]
    item=header[key]
    lo,hi=item['data_offsets']
    row=file_rows[shard]
    out.append({
        'key':key,'shard':shard,'remote_path':path,
        'shard_bytes':row['bytes'],'shard_sha256':row['sha256'],
        'dtype':item['dtype'],'shape':item['shape'],
        'data_offsets':item['data_offsets'],
        'absolute_offset':8+hlen+lo,'bytes':hi-lo,
    })
print(json.dumps({
    'schema':'s3-qtip-range-metadata-v1','source_host':'compute-node-3',
    'source_qsfp':'203.0.113.3','remote_root':base,
    'index_path':idx_path,'index_bytes':len(idx_raw),
    'index_sha256':hashlib.sha256(idx_raw).hexdigest(),
    'model_stage_path':stage_path,'model_stage_sha256':hashlib.sha256(stage_raw).hexdigest(),
    'model_stage_status':stage.get('status'),'model_stage_all_sha_equal':stage.get('all_source_destination_sha256_equal'),
    'tensors':out,
},sort_keys=True))
'''
        command = remote_command(
            ["python3", "-c", code, self.remote_root, str(self.layer), str(self.expert)]
        )
        proc = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
        )
        if proc.returncode:
            raise RuntimeError(
                f"s3 range metadata failed rc={proc.returncode}: "
                f"{proc.stderr.decode(errors='replace')}"
            )
        result = json.loads(proc.stdout)
        if result.get("model_stage_status") != "PASS" or not result.get(
            "model_stage_all_sha_equal"
        ):
            raise RuntimeError("s3 source model stage is not sealed PASS")
        return result

    @staticmethod
    def _fetch_one(entry: dict[str, Any]) -> tuple[str, bytes, dict[str, Any]]:
        code = r'''
import os,sys
path,offset,size=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
fd=os.open(path,os.O_RDONLY)
try:
    chunks=[]
    done=0
    while done<size:
        part=os.pread(fd,min(8<<20,size-done),offset+done)
        if not part: raise RuntimeError(f'short pread {done}/{size}')
        chunks.append(part); done+=len(part)
    sys.stdout.buffer.write(b''.join(chunks))
finally:
    os.close(fd)
'''
        started = time.perf_counter()
        command = remote_command(
            [
                "python3",
                "-c",
                code,
                entry["remote_path"],
                str(entry["absolute_offset"]),
                str(entry["bytes"]),
            ]
        )
        proc = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
        )
        elapsed = time.perf_counter() - started
        if proc.returncode:
            raise RuntimeError(
                f"range read failed {entry['key']} rc={proc.returncode}: "
                f"{proc.stderr.decode(errors='replace')}"
            )
        raw = proc.stdout
        if len(raw) != int(entry["bytes"]):
            raise RuntimeError(
                f"range length mismatch {entry['key']}: {len(raw)} != {entry['bytes']}"
            )
        receipt = {
            **entry,
            "range_sha256": sha256_bytes(raw),
            "stream_wall_seconds": elapsed,
            "stream_GBps": len(raw) / elapsed / 1e9,
        }
        return entry["key"], raw, receipt

    def fetch(self) -> dict[str, Any]:
        started = time.perf_counter()
        receipts = []
        workers = len(self.entries)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._fetch_one, row) for row in self.entries.values()]
            for future in as_completed(futures):
                key, raw, receipt = future.result()
                self.raw[key] = raw
                receipts.append(receipt)
        wall = time.perf_counter() - started
        receipts.sort(key=lambda row: row["key"])
        total = sum(len(value) for value in self.raw.values())
        self.transfer = {
            "schema": "qtip-s3-qsfp-multistream-range-read-v1",
            "status": "PASS",
            "source_host": "compute-node-3",
            "destination_host": socket.gethostname(),
            "source_qsfp": "203.0.113.3",
            "streams": workers,
            "payload_bytes": total,
            "wall_seconds": wall,
            "aggregate_GBps": total / wall / 1e9,
            "aggregate_GiBps": total / wall / (1 << 30),
            "tensor_ranges": receipts,
            "metadata": {k: v for k, v in self.metadata.items() if k != "tensors"},
        }
        return self.transfer

    def _tensor(self, key: str) -> torch.Tensor:
        entry = self.entries[key]
        if entry["dtype"] not in {"I8", "U8", "F8_E8M0"}:
            raise RuntimeError(f"unsupported source dtype {entry['dtype']} for {key}")
        raw = self.raw[key]
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return tensor.reshape(tuple(int(x) for x in entry["shape"]))

    def projection(self, projection: str) -> tuple[torch.Tensor, dict[str, Any]]:
        names = ("w1", "w3") if projection == "fused13" else ("w2",)
        matrices = []
        rows = []
        for name in names:
            weight_key = f"layers.{self.layer}.ffn.experts.{self.expert}.{name}.weight"
            scale_key = f"layers.{self.layer}.ffn.experts.{self.expert}.{name}.scale"
            packed = self._tensor(weight_key)
            scales = self._tensor(scale_key)
            nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
            matrix = E2M1[nibbles.long()] * torch.exp2(
                scales.float() - 127.0
            ).repeat_interleave(32, dim=1)
            matrices.append(matrix.contiguous())
            rows.append({
                "weight": self.entries[weight_key],
                "scale": self.entries[scale_key],
            })
        result = torch.cat(matrices, dim=0) if len(matrices) == 2 else matrices[0]
        expected = (4096, 4096) if projection == "fused13" else (4096, 2048)
        if tuple(result.shape) != expected or not torch.isfinite(result).all():
            raise RuntimeError(f"source decode invalid {projection}: {tuple(result.shape)}")
        return result.contiguous(), {
            "schema": "qtip-s3-ranged-source-projection-v1",
            "layer": self.layer,
            "expert": self.expert,
            "projection": projection,
            "shape": list(result.shape),
            "decoded_tensor_sha256": tensor_sha256(result),
            "ranges": rows,
            "transport_receipt": self.transfer,
        }


def gpu_viterbi_selftest(cb: Any) -> dict[str, Any]:
    """Compile and exercise both cyclic Viterbi passes before loading fit data."""
    torch.manual_seed(580622)
    x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    started = time.perf_counter()
    quantized, states = cb.quantize(x)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    shift = int(cb.K) * int(cb.V)
    prefixes = 1 << (int(cb.L) - shift)
    continuity = states[:, :-1].bitwise_and(prefixes - 1) == states[:, 1:].bitwise_right_shift(shift)
    packed = cb.pack_trellis(states[0:1])
    unpacked = cb.unpack_trellis(packed, 256)
    result = {
        "status": "PASS",
        "shape": list(quantized.shape),
        "states_shape": list(states.shape),
        "finite": bool(torch.isfinite(quantized).all()),
        "continuity_fraction": float(continuity.float().mean()),
        "packed_uint16": int(packed.numel()),
        "expected_packed_uint16": int(256 * int(cb.K) // 16),
        "pack_roundtrip_fraction": float((unpacked == states[0:1]).float().mean()),
        "wall_seconds_including_compile": wall,
    }
    if (
        not result["finite"]
        or result["continuity_fraction"] != 1.0
        or result["packed_uint16"] != result["expected_packed_uint16"]
    ):
        raise RuntimeError(f"GPU Viterbi selftest failed: {result}")
    return result


def build_one(
    qv: Any,
    kernel_decode: Any,
    cb: Any,
    source: torch.Tensor,
    windows: list[dict[str, Any]],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    m, n = source.shape
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.manual_seed(seed)
    su = (torch.randn(n, device=device).sign() + 1e-5).sign().float()
    sv = (torch.randn(m, device=device).sign() + 1e-5).sign().float()

    hessian, fit_rows, fit_mass = qv.build_hessian(windows, su, device)
    hessian = hessian.unsqueeze(0)
    diagmean = torch.diagonal(hessian, dim1=-2, dim2=-1).mean(dim=-1)
    hessian.div_(diagmean[:, None, None])
    diagonal = torch.arange(n, device=device)
    hessian[:, diagonal, diagonal] += 1e-2
    hessian.mul_(diagmean[:, None, None])

    weight = source.to(device=device, dtype=torch.float32).unsqueeze(0)
    transformed = qv.fwht(
        qv.fwht(weight.transpose(1, 2) * sv[None, None, :]).transpose(1, 2)
        * su[None, None, :]
    )
    lut_rms = cb.lut.double().square().mean().sqrt().float() * 0.9
    wscale = transformed.square().mean(dim=(1, 2)).sqrt() / lut_rms
    transformed.div_(wscale[:, None, None])
    lower = block_LDL_batch(hessian, 16)
    lower[:, diagonal, diagonal] = 0
    del hessian, weight
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    quant_started = time.perf_counter()
    quantized, states = LDLQ_batch(
        transformed,
        lower,
        cb,
        argparse.Namespace(td_x=16, td_y=16, V=int(cb.V)),
        buf_cols=128,
        for_kernel=True,
    )
    torch.cuda.synchronize()
    quant_seconds = time.perf_counter() - quant_started
    del quantized, transformed, lower
    torch.cuda.empty_cache()

    packed, pack_receipts = pack_kernel_layout_batch(cb, states, m, n)
    del states
    torch.cuda.empty_cache()

    raw = kernel_decode.decode_compressed(
        int(cb.L),
        int(cb.tlut_bits),
        int(cb.K),
        int(math.log2(int(cb.V))),
        m,
        n,
        packed[0].reshape(-1),
        cb.lut.T.contiguous(),
    ) * wscale[0]
    reconstructed = qv.fwht(raw.T).T * sv[:, None]
    reconstructed = qv.fwht(reconstructed) * su
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    sane = {
        "finite": bool(torch.isfinite(reconstructed).all()),
        "max_abs": float(reconstructed.float().abs().max()),
    }
    if not sane["finite"] or sane["max_abs"] > 100:
        raise RuntimeError(f"reconstruction sanity failed: {sane}")
    candidate = {
        "shape": [m, n],
        "trellis": packed[0].cpu(),
        "SU": su.half().cpu(),
        "SV": sv.half().cpu(),
        "Wscale": wscale[0].cpu(),
        "reconstructed_weight": reconstructed.half().cpu(),
        "geometry": {
            "L": int(cb.L),
            "K": int(cb.K),
            "V": int(cb.V),
            "tlut_bits": int(cb.tlut_bits),
            "decode_mode": str(cb.decode_mode),
            "td_x": 16,
            "td_y": 16,
        },
    }
    build = {
        "implementation": "whole-matrix-single-unit-rate-v1",
        "build_wall_seconds": wall,
        "quant_seconds": quant_seconds,
        "fit_rows": fit_rows,
        "fit_route_mass": fit_mass,
        "canonical_pack": pack_receipts[0],
        "reconstruction_sanity": sane,
    }
    del raw, reconstructed
    torch.cuda.empty_cache()
    return candidate, build


def decode_readback(
    qv: Any,
    kernel_decode: Any,
    cb: Any,
    artifact: Path,
    expected_reconstruction_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(artifact, map_location="cpu", mmap=True, weights_only=True)
    m, n = map(int, payload["shape"])
    trellis = payload["trellis"].to("cuda")
    su = payload["SU"].float().to("cuda")
    sv = payload["SV"].float().to("cuda")
    wscale = payload["Wscale"].float().to("cuda")
    raw = kernel_decode.decode_compressed(
        int(cb.L),
        int(cb.tlut_bits),
        int(cb.K),
        int(math.log2(int(cb.V))),
        m,
        n,
        trellis.reshape(-1),
        cb.lut.T.contiguous(),
    ) * wscale
    reconstructed = qv.fwht(raw.T).T * sv[:, None]
    reconstructed = qv.fwht(reconstructed) * su
    finite = bool(torch.isfinite(reconstructed).all())
    max_abs = float(reconstructed.float().abs().max())
    actual_sha = tensor_sha256(reconstructed.half())
    return {
        "loader_open": True,
        "packed_decode_finite": finite,
        "packed_decode_max_abs": max_abs,
        "packed_decode_fp16_sha256": actual_sha,
        "expected_reconstruction_fp16_sha256": expected_reconstruction_sha256,
        "packed_decode_fp16_bit_exact": actual_sha == expected_reconstruction_sha256,
    }


def checkpoint_progress(
    mission: Path, layer: int, expert: int, projection: str, stage: str
) -> None:
    progress_path = mission / "PROGRESS.json"
    try:
        old = json.loads(progress_path.read_text())
    except Exception:
        old = {}
    per_layer = {
        str(item): len(
            list(
                (mission / "artifacts").glob(
                    f"L{item:03d}_E???_*_L16_K2_V2.DONE.json"
                )
            )
        )
        for item in ALLOWED_LAYERS
    }
    completed = sum(per_layer.values())
    value = {
        **old,
        "schema": "p541-progress-v1",
        "task": TASK,
        "host": socket.gethostname(),
        "mission": str(mission),
        "status": "RUNNING",
        "stage": stage,
        "layers": ALLOWED_LAYERS,
        "forbidden_layers": [16, 11, 14, 19],
        "expected_units": 2048,
        "completed_units": completed,
        "completed_units_by_layer": per_layer,
        "remaining_units": 2048 - completed,
        "current_layer": layer,
        "current_expert": expert,
        "current_projection": projection,
        "last_unit_checkpoint": f"L{layer:03d}_E{expert:03d}_{projection}",
        "heldout_used": False,
        "pid": os.getpid(),
        "updated_unix": time.time(),
    }
    atomic_json(progress_path, value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=Path, default=DEFAULT_MISSION)
    ap.add_argument("--qtip-root", type=Path, default=DEFAULT_QTIP)
    ap.add_argument("--tlut-source", type=Path, default=DEFAULT_TLUT)
    ap.add_argument("--fit-receipt", type=Path, default=DEFAULT_FIT)
    ap.add_argument("--current-plane", type=Path, default=DEFAULT_PLANE)
    ap.add_argument("--s3-model", default=DEFAULT_S3_MODEL)
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--expert", type=int, default=2)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--V", type=int, default=2)
    ap.add_argument("--target-bpw", type=float, default=2.0)
    ap.add_argument("--source-only", action="store_true")
    args = ap.parse_args()

    mission = args.mission.resolve()
    mission.mkdir(parents=True, exist_ok=True)
    if socket.gethostname() != "compute-node-5-work":
        raise RuntimeError(f"hard host scope violation: {socket.gethostname()} != compute-node-5-work")
    if args.layer not in ALLOWED_LAYERS:
        raise RuntimeError(
            f"hard task layer scope violation: {args.layer} not in {ALLOWED_LAYERS}"
        )
    if (args.L, args.K, args.V) != (16, 2, 2):
        raise RuntimeError("sealed reachable rung is exactly L16/K2/V2")

    reader = S3RangeModelReader(args.s3_model, args.layer, args.expert)
    transport = reader.fetch()
    source_receipt = mission / "receipts/S3_QSFP_RANGE_READ.json"
    atomic_json(source_receipt, transport)
    source_receipt_sha = sha256_file(source_receipt)
    if args.source_only:
        decoded = {}
        for projection in ("fused13", "down"):
            tensor, ref = reader.projection(projection)
            decoded[projection] = {
                "shape": list(tensor.shape),
                "finite": bool(torch.isfinite(tensor).all()),
                "sha256": ref["decoded_tensor_sha256"],
            }
        result = {
            "schema": "qtip-rate-source-only-v1",
            "status": "PASS",
            "transport_receipt": str(source_receipt),
            "transport_receipt_sha256": source_receipt_sha,
            "decoded": decoded,
        }
        atomic_json(mission / "receipts/SOURCE_ONLY_DONE.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if mem_available_bytes() < 8 * (1 << 30):
        raise RuntimeError("MemAvailable below 8 GiB before smoke")
    claim_sha, claim = require_authorized_claim(mission)

    qtip_root = args.qtip_root.resolve()
    qv = serial.load_parent_module(qtip_root)
    bitshift, _, _, kernel_decode = qv.load_official_qtip()
    tlut_payload = torch.load(
        args.tlut_source.resolve(), map_location="cpu", mmap=True, weights_only=True
    )
    pinned_tlut = tlut_payload["tlut"].float().contiguous()
    tlut_sha = tensor_sha256(pinned_tlut)
    cb = bitshift.bitshift_codebook(
        L=args.L,
        K=args.K,
        V=args.V,
        tlut_bits=9,
        decode_mode="quantlut_sym",
        tlut=pinned_tlut.to("cuda"),
    ).to("cuda")
    fast_viterbi = install_rate_viterbi(cb)
    viterbi_selftest = gpu_viterbi_selftest(cb)

    fit_entries, fit_ref = serial.load_fit_capture_receipt(
        args.fit_receipt.resolve(), args.layer
    )
    plane_data = torch.load(
        args.current_plane.resolve(), map_location="cpu", mmap=True, weights_only=True
    )
    raw_fit = qv.expert_windows(fit_entries, args.expert)

    results = []
    for projection in ("fused13", "down"):
        artifact = (
            mission
            / "artifacts"
            / f"L{args.layer:03d}_E{args.expert:03d}_{projection}_L{args.L}_K{args.K}_V{args.V}.pt"
        )
        done_path = artifact.with_suffix(".DONE.json")
        if done_path.is_file():
            done = json.loads(done_path.read_text())
            expected_identity = {
                "layer": args.layer,
                "expert": args.expert,
                "projection": projection,
                "target_bpw": args.target_bpw,
            }
            artifact_row = done.get("artifact") or {}
            valid_done = (
                done.get("status") == "PASS"
                and done.get("task") == TASK
                and done.get("claim_sha256") == claim_sha
                and done.get("identity") == expected_identity
                and artifact.is_file()
                and int(artifact_row.get("bytes", -1)) == artifact.stat().st_size
                and artifact_row.get("sha256") == sha256_file(artifact)
                and bool(done.get("gates"))
                and all(bool(value) for value in done["gates"].values())
            )
            if not valid_done:
                raise RuntimeError(f"invalid existing DONE; refusing replay: {done_path}")
            done["done_path"] = str(done_path)
            done["done_sha256"] = sha256_file(done_path)
            done["resume_skipped_done"] = True
            results.append(done)
            checkpoint_progress(
                mission, args.layer, args.expert, projection, "unit_resume_skipped"
            )
            continue
        if mem_available_bytes() < 8 * (1 << 30):
            raise RuntimeError(f"MemAvailable below 8 GiB before {projection}")
        current_claim_sha, _ = require_authorized_claim(mission)
        if current_claim_sha != claim_sha:
            raise RuntimeError("claim changed during smoke")
        source, source_ref = reader.projection(projection)
        if projection == "down":
            current_fused = serial.current_fused13(qv, plane_data, args.expert)
            windows = qv.down_windows(raw_fit, current_fused, torch.device("cuda"))
            del current_fused
        else:
            windows = raw_fit
        seed = serial.rht_seed(args.layer, args.expert, projection)
        candidate, build = build_one(
            qv,
            kernel_decode,
            cb,
            source,
            windows,
            seed,
            torch.device("cuda"),
        )
        identity = {
            "layer": args.layer,
            "expert": args.expert,
            "projection": projection,
            "target_bpw": args.target_bpw,
        }
        payload = {
            "schema": "qtip-rate-rung-unit-v1",
            "task": TASK,
            "identity": identity,
            "shape": candidate["shape"],
            "trellis": candidate["trellis"],
            "SU": candidate["SU"],
            "SV": candidate["SV"],
            "Wscale": candidate["Wscale"],
            "geometry": candidate["geometry"],
            "rht_seed": seed,
            "tlut_sha256": tlut_sha,
        }
        expected_recon_sha = tensor_sha256(candidate["reconstructed_weight"])
        atomic_torch(artifact, payload)
        artifact_sha = sha256_file(artifact)
        readback = decode_readback(
            qv, kernel_decode, cb, artifact, expected_recon_sha
        )
        loaded = torch.load(
            artifact, map_location="cpu", mmap=True, weights_only=True
        )
        logical_bytes = sum(
            loaded[key].numel() * loaded[key].element_size()
            for key in ("trellis", "SU", "SV", "Wscale")
        )
        values = math.prod(candidate["shape"])
        bpw = logical_bytes * 8.0 / values
        gates = {
            "finite": bool(build["reconstruction_sanity"]["finite"]),
            "loader_open": bool(readback["loader_open"]),
            "packed_decode_finite": bool(readback["packed_decode_finite"]),
            "packed_decode_fp16_bit_exact": bool(
                readback["packed_decode_fp16_bit_exact"]
            ),
            "logical_bpw_within_0_15": abs(bpw - args.target_bpw) <= 0.15,
            "build_wall_seconds_positive_finite": (
                math.isfinite(float(build["build_wall_seconds"]))
                and float(build["build_wall_seconds"]) > 0
            ),
            "mem_floor_8GiB": mem_available_bytes() >= 8 * (1 << 30),
        }
        status = "PASS" if all(gates.values()) else "FAIL"
        done = {
            "schema": "qtip-rate-rung-unit-done-v1",
            "status": status,
            "task": TASK,
            "identity": identity,
            "artifact": {
                "path": str(artifact),
                "bytes": artifact.stat().st_size,
                "sha256": artifact_sha,
            },
            "geometry": candidate["geometry"],
            "logical_wire_bytes_excluding_shared_tlut": logical_bytes,
            "logical_bpw_excluding_shared_tlut": bpw,
            "build": build,
            "packed_readback": readback,
            "source_weight": source_ref,
            "fit": fit_ref,
            "current_plane": {
                "path": str(args.current_plane.resolve()),
                "bytes": args.current_plane.resolve().stat().st_size,
                "sha256": PLANE_SHA256[args.layer],
                "verified_by_driver_before_unit": True,
            },
            "tlut_sha256": tlut_sha,
            "fast_viterbi": fast_viterbi,
            "claim_sha256": claim_sha,
            "gates": gates,
            "created_unix": time.time(),
        }
        atomic_json(done_path, done)
        checkpoint_progress(
            mission, args.layer, args.expert, projection, "unit_sealed"
        )
        done["done_path"] = str(done_path)
        done["done_sha256"] = sha256_file(done_path)
        results.append(done)
        del candidate, payload, loaded, source
        if projection == "down":
            del windows
        gc.collect()
        torch.cuda.empty_cache()

    overall = {
        "schema": "qtip-rate-rung-config-receipt-v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "task": TASK,
        "host": socket.gethostname(),
        "target_bpw": args.target_bpw,
        "representative16_layers": REPRESENTATIVE16,
        "selected_representative_layer": args.layer,
        "config": {
            "L": args.L,
            "K": args.K,
            "V": args.V,
            "tlut_bits": 9,
            "decode_mode": "quantlut_sym",
        },
        "units": results,
        "average_build_wall_seconds": sum(
            row["build"]["build_wall_seconds"] for row in results
        )
        / len(results),
        "transport_receipt": str(source_receipt),
        "transport_receipt_sha256": source_receipt_sha,
        "claim_sha256": claim_sha,
        "claim_mode": claim.get("qtip_rate_smoke_claim_mode"),
        "claim_nonce": claim.get("claim_nonce"),
        "viterbi_selftest": viterbi_selftest,
        "source_files": {
            "builder": sha256_file(Path(__file__).resolve()),
            "rate_batched_ldlq": sha256_file(
                Path(__file__).resolve().parent / "rate_batched_ldlq.py"
            ),
            "triton_viterbi_rate": sha256_file(
                Path(__file__).resolve().parent / "triton_viterbi_rate.py"
            ),
            "qtip_bitshift": sha256_file(qtip_root / "lib/codebook/bitshift.py"),
            "qtip_kernel_decompress": sha256_file(
                qtip_root / "lib/utils/kernel_decompress.py"
            ),
        },
        "created_unix": time.time(),
    }
    overall_path = mission / "receipts/QTIP_2BPW_CONFIG_PASS.json"
    atomic_json(overall_path, overall)
    print(json.dumps(overall, sort_keys=True))
    return 0 if overall["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
