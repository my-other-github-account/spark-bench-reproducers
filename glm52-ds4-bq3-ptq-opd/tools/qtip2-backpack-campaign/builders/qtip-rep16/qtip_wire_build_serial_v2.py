#!/usr/bin/env python3
"""Resumable packed-only builder for the fixed QTIP HYB L16/K3/V2 anchor wire."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any

import torch
from safetensors import safe_open

from triton_viterbi_prefix32 import install_prefix_viterbi

TASK = "PUBLIC_TASK"
PARENT = Path("$HOME/run-bundles/QTIP_VALIDATE_PUBLIC_TASK_s7")
PARENT_RUNNER = Path(__file__).resolve().parent / "qtip_validate_prefix.py"
PARENT_MANIFEST = PARENT / "SELECTION_MANIFEST.json"
SEED_MATERIAL = (
    "4fa7b1213db1d6a4670b534785edb1681d1538bb6d12a90222e33c30251c2462"
    "|PUBLIC_TASK|heldout-experts-v1"
)
RHT_DOMAIN = "qtip-rht-bounded36-v1"
PATCHED_VQ_BUILDER_SHA256 = "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e"
PARENT_FINAL_SHA256 = "c4033dac1c754d8e9611a5f41fc1ba38b2d539699f32e97390d28c1d2bfb1197"
QTIP_COMMIT = "e90c6688c8dfae326a3a81b5eb032db7c6680ec0"
PROJECTIONS = ("fused13", "down")


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    x = t.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
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
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def load_parent_module(qtip_root: Path):
    spec = importlib.util.spec_from_file_location("qtip_wire_parent", PARENT_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PARENT_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.QTIP = qtip_root
    return module


def require_claim() -> tuple[str, dict[str, Any]]:
    path = Path("$HOME/HOST_CLAIM.json")
    raw = path.read_bytes()
    claim = json.loads(raw)
    if claim.get("owner") != TASK or claim.get("task_id") != TASK:
        raise RuntimeError(
            f"host claim not owned by {TASK}: {claim.get('owner')} {claim.get('task_id')}"
        )
    return hashlib.sha256(raw).hexdigest(), claim


def parse_int_set(text: str, lo: int, hi: int) -> list[int]:
    result: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = map(int, item.split("-", 1))
            result.update(range(a, b + 1))
        else:
            result.add(int(item))
    if not result or min(result) < lo or max(result) > hi:
        raise ValueError(f"range outside [{lo},{hi}]: {text}")
    return sorted(result)


def unit_name(layer: int, expert: int, projection: str) -> str:
    return f"L{layer:03d}_E{expert:03d}_{projection}"


def rht_seed(layer: int, expert: int, projection: str) -> int:
    identity = unit_name(layer, expert, projection)
    digest = hashlib.sha256(
        f"{RHT_DOMAIN}|{SEED_MATERIAL}|{identity}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def validate_seed_compatibility() -> None:
    manifest = json.loads(PARENT_MANIFEST.read_text())
    expected = manifest["qtip_package"]["rht_seed_map"]
    actual = {
        name: rht_seed(
            int(name[1:4]), int(name[6:9]),
            "fused13" if name.endswith("fused13") else "down",
        )
        for name in expected
    }
    if actual != expected:
        raise RuntimeError("full-wire seed derivation does not reproduce bounded-36 map")


def load_fit_capture_receipt(path: Path, layer: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if receipt.get("status") != "PASS" or int(receipt.get("layer", -1)) != layer:
        raise RuntimeError(f"fit receipt identity/status mismatch: {path}")
    files = receipt.get("files", [])
    windows = [int(row["window"]) for row in files]
    if windows != list(range(128)):
        raise RuntimeError(f"fit receipt window closure mismatch L{layer}: {windows[:3]}..{windows[-3:]}")
    entries = []
    for row in files:
        capture = Path(row["path"])
        done = Path(row["done_path"])
        if not capture.is_file() or capture.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"fit capture absent/size drift: {capture}")
        done_obj = json.loads(done.read_text())
        if done_obj.get("md5") != row["md5"]:
            raise RuntimeError(f"fit DONE digest drift: {done}")
        data = torch.load(capture, map_location="cpu", mmap=True, weights_only=True)
        if int(data["layer"]) != layer or int(data["win"]) != int(row["window"]):
            raise RuntimeError(f"fit tensor identity drift: {capture}")
        if str(data["corpus_md5"]) != receipt["corpus_md5"]:
            raise RuntimeError(f"fit corpus identity drift: {capture}")
        entries.append({
            "window": int(row["window"]),
            "x": data["x"].to(torch.bfloat16).contiguous(),
            "topk": data["topk"].to(torch.int64).contiguous(),
            "route": data["w"].float().contiguous(),
        })
    return entries, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "corpus_md5": receipt["corpus_md5"],
        "files": len(files),
        "source_receipt_sha256": receipt.get("source_receipt_sha256"),
    }


class ModelReader:
    def __init__(self, model: Path, shard_hash_manifest: Path):
        self.model = model
        self.index_path = model / "model.safetensors.index.json"
        self.mapping = json.loads(self.index_path.read_text())["weight_map"]
        self.index_sha256 = sha256(self.index_path)
        manifest = json.loads(shard_hash_manifest.read_text())
        if manifest.get("status") != "PASS":
            raise RuntimeError("model shard hash manifest is not PASS")
        self.shards = {str(Path(k).resolve()): v for k, v in manifest["shards"].items()}
        self.shard_manifest = {
            "path": str(shard_hash_manifest.resolve()),
            "sha256": sha256(shard_hash_manifest),
        }
        self.lut = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
        )

    def projection(self, layer: int, expert: int, projection: str):
        names = ("w1", "w3") if projection == "fused13" else ("w2",)
        matrices = []
        source = []
        for name in names:
            weight_key = f"layers.{layer}.ffn.experts.{expert}.{name}.weight"
            scale_key = f"layers.{layer}.ffn.experts.{expert}.{name}.scale"
            shard = (self.model / self.mapping[weight_key]).resolve()
            if self.mapping[scale_key] != self.mapping[weight_key]:
                raise RuntimeError(f"weight/scale shard split: {weight_key}")
            digest = self.shards.get(str(shard))
            if digest is None:
                raise RuntimeError(f"model shard absent from preflight: {shard}")
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                packed = handle.get_tensor(weight_key).view(torch.uint8)
                scales = handle.get_tensor(scale_key).view(torch.uint8)
            nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
            matrix = self.lut[nibbles.long()] * torch.exp2(
                scales.float() - 127.0
            ).repeat_interleave(32, dim=1)
            matrices.append(matrix.contiguous())
            source.append({
                "weight_key": weight_key,
                "scale_key": scale_key,
                "shard": str(shard),
                "shard_sha256": digest,
            })
        result = torch.cat(matrices, dim=0) if len(matrices) == 2 else matrices[0]
        expected = (4096, 4096) if projection == "fused13" else (4096, 2048)
        if tuple(result.shape) != expected:
            raise RuntimeError(f"source shape drift: {tuple(result.shape)} != {expected}")
        return result.contiguous(), {
            "index_path": str(self.index_path),
            "index_sha256": self.index_sha256,
            "model_shard_hash_manifest": self.shard_manifest,
            "shards": source,
        }


def validate_plane_receipt(path: Path, plane: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if receipt.get("status") != "PASS" or Path(receipt["path"]).resolve() != plane.resolve():
        raise RuntimeError(f"current-plane receipt identity/status mismatch: {path}")
    if plane.stat().st_size != int(receipt["bytes"]):
        raise RuntimeError(f"current-plane size drift: {plane}")
    return {
        "path": str(plane.resolve()),
        "sha256": receipt["sha256"],
        "bytes": int(receipt["bytes"]),
        "receipt": str(path.resolve()),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }


def current_fused13(qv, plane_data: dict[str, Any], expert: int) -> torch.Tensor:
    return qv.decode_vq(
        plane_data["codes13"][expert].contiguous(),
        plane_data["sc13"][expert].contiguous(),
        plane_data["cb13"].contiguous(),
    )


def resume_ok(artifact: Path, done_path: Path, identity: dict[str, Any]) -> bool:
    if not artifact.is_file() or not done_path.is_file():
        return False
    try:
        done = json.loads(done_path.read_text())
        if done.get("status") != "PASS" or done.get("identity") != identity:
            return False
        if artifact.stat().st_size != int(done["artifact"]["bytes"]):
            return False
        if sha256(artifact) != done["artifact"]["sha256"]:
            return False
        data = torch.load(artifact, map_location="cpu", mmap=True, weights_only=True)
        return (
            data.get("schema") == "qtip-hyb-wire-unit-v1"
            and data.get("identity") == identity
            and data.get("geometry") == {
                "L": 16, "K": 3, "V": 2, "tlut_bits": 9,
                "decode_mode": "quantlut_sym", "td_x": 16, "td_y": 16,
            }
            and tensor_sha256(data["trellis"]) == done["trellis_sha256"]
        )
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--experts", default="0-255")
    ap.add_argument("--projections", default="fused13,down")
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fit-receipt", type=Path, required=True)
    ap.add_argument("--current-plane", type=Path, required=True)
    ap.add_argument("--current-plane-receipt", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("$HOME/models/hf/DeepSeek-V4-Flash"))
    ap.add_argument("--model-shard-hashes", type=Path, required=True)
    ap.add_argument("--qtip-root", type=Path, required=True)
    ap.add_argument("--tlut-source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not (0 <= args.layer <= 42):
        raise ValueError(args.layer)
    if not (args.workers >= 1 and 0 <= args.worker_index < args.workers):
        raise ValueError("invalid worker partition")
    experts = parse_int_set(args.experts, 0, 255)
    experts = [e for e in experts if e % args.workers == args.worker_index]
    projections = [p.strip() for p in args.projections.split(",") if p.strip()]
    if not projections or any(p not in PROJECTIONS for p in projections):
        raise ValueError(f"invalid projections: {projections}")
    claim_sha, claim = require_claim()
    if socket.gethostname() not in {"compute-node-1", "compute-node-2", "compute-node-3", "compute-node-4", "compute-node-6", "compute-node-7", "compute-node-8", "compute-node-5-work"}:
        raise RuntimeError(f"unexpected host {socket.gethostname()}")
    validate_seed_compatibility()
    qtip_root = args.qtip_root.resolve()
    qv = load_parent_module(qtip_root)
    bitshift, ldlq, math_utils, kernel_decode = qv.load_official_qtip()
    tlut_payload = torch.load(
        args.tlut_source, map_location="cpu", mmap=True, weights_only=True
    )
    pinned_tlut = tlut_payload["tlut"].float().contiguous()
    tlut_sha = tensor_sha256(pinned_tlut)
    expected_tlut_sha = json.loads(PARENT_MANIFEST.read_text())["qtip_package"]["tlut_tensor_sha256"]
    if tlut_sha != expected_tlut_sha:
        raise RuntimeError(f"TLUT drift {tlut_sha} != {expected_tlut_sha}")
    cb = bitshift.bitshift_codebook(
        L=16, K=3, V=2, tlut_bits=9, decode_mode="quantlut_sym",
        tlut=pinned_tlut.to("cuda"),
    ).to("cuda")
    fast_viterbi = install_prefix_viterbi(cb)
    fit_entries, fit_ref = load_fit_capture_receipt(args.fit_receipt, args.layer)
    model = ModelReader(args.model.resolve(), args.model_shard_hashes.resolve())
    plane = args.current_plane.resolve()
    plane_ref = validate_plane_receipt(args.current_plane_receipt.resolve(), plane)
    plane_data = torch.load(plane, map_location="cpu", mmap=True, weights_only=True)
    out = args.output.resolve()
    units = out / "units" / f"L{args.layer:03d}"
    status_path = out / "status" / f"L{args.layer:03d}.W{args.worker_index:02d}.json"
    builder_path = Path(__file__).resolve()
    builder_sha = sha256(builder_path)
    qtip_sources = {
        "qtip_commit": QTIP_COMMIT,
        "bitshift_sha256": sha256(qtip_root / "lib/codebook/bitshift.py"),
        "ldlq_sha256": sha256(qtip_root / "lib/algo/ldlq.py"),
        "math_utils_sha256": sha256(qtip_root / "lib/utils/math_utils.py"),
        "kernel_decompress_sha256": sha256(qtip_root / "lib/utils/kernel_decompress.py"),
        "parent_runner_sha256": sha256(PARENT_RUNNER),
        "builder_sha256": builder_sha,
        "fast_viterbi": fast_viterbi,
        "fast_viterbi_sha256": sha256(builder_path.parent / "triton_viterbi_prefix32.py"),
        "patched_vq_builder_sha256_context_pin": PATCHED_VQ_BUILDER_SHA256,
    }
    expected_units = [
        (expert, projection) for expert in experts for projection in projections
    ]
    completed = 0
    atomic_json(status_path, {
        "schema": "qtip-wire-worker-status-v1", "status": "RUNNING", "task": TASK,
        "host": socket.gethostname(), "layer": args.layer,
        "worker_index": args.worker_index, "workers": args.workers,
        "expected_units": len(expected_units), "completed_units": 0,
        "claim_sha256": claim_sha, "claim_nonce": claim.get("claim_nonce"),
        "fit": fit_ref, "current_plane": plane_ref, "sources": qtip_sources,
        "started_unix": time.time(),
    })
    for expert, projection in expected_units:
        identity = {"layer": args.layer, "expert": expert, "projection": projection}
        name = unit_name(args.layer, expert, projection)
        artifact = units / f"{name}.pt"
        done_path = units / f"{name}.DONE.json"
        if resume_ok(artifact, done_path, identity):
            completed += 1
            print(f"RESUME {name}", flush=True)
            continue
        require_claim()
        raw_fit = qv.expert_windows(fit_entries, expert)
        if projection == "down":
            fit_windows = qv.down_windows(
                raw_fit, current_fused13(qv, plane_data, expert), torch.device("cuda")
            )
        else:
            fit_windows = raw_fit
        source_weight, source_ref = model.projection(args.layer, expert, projection)
        seed = rht_seed(args.layer, expert, projection)
        started = time.time()
        candidate, build = qv.build_qtip(
            source_weight, fit_windows, cb, ldlq, math_utils, kernel_decode,
            torch.device("cuda"), seed,
        )
        reconstruction_sha = tensor_sha256(candidate["reconstructed_weight"])
        packed = {
            "schema": "qtip-hyb-wire-unit-v1",
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
        logical_bytes = sum(
            packed[key].numel() * packed[key].element_size()
            for key in ("trellis", "SU", "SV", "Wscale")
        )
        source_values = math.prod(candidate["shape"])
        atomic_torch(artifact, packed)
        artifact_ref = {
            "path": str(artifact), "sha256": sha256(artifact),
            "bytes": artifact.stat().st_size,
        }
        readback = torch.load(artifact, map_location="cpu", mmap=True, weights_only=True)
        trellis_sha = tensor_sha256(readback["trellis"])
        if trellis_sha != tensor_sha256(candidate["trellis"]):
            raise RuntimeError(f"artifact readback trellis mismatch: {artifact}")
        done = {
            "schema": "qtip-hyb-wire-unit-done-v1", "status": "PASS", "task": TASK,
            "identity": identity, "artifact": artifact_ref,
            "trellis_sha256": trellis_sha,
            "reconstructed_weight_sha256": reconstruction_sha,
            "logical_wire_bytes_excluding_shared_tlut": logical_bytes,
            "logical_bpw_excluding_shared_tlut": logical_bytes * 8.0 / source_values,
            "rht_seed": seed, "tlut_sha256": tlut_sha,
            "fit": fit_ref, "current_plane": plane_ref,
            "source_weight": source_ref, "sources": qtip_sources,
            "build": build,
            "build_wall_seconds": time.time() - started,
            "exact_command": " ".join(sys.argv),
            "created_unix": time.time(),
        }
        atomic_json(done_path, done)
        completed += 1
        atomic_json(status_path, {
            "schema": "qtip-wire-worker-status-v1", "status": "RUNNING", "task": TASK,
            "host": socket.gethostname(), "layer": args.layer,
            "worker_index": args.worker_index, "workers": args.workers,
            "expected_units": len(expected_units), "completed_units": completed,
            "last_unit": identity, "last_done_sha256": sha256(done_path),
            "claim_sha256": require_claim()[0], "epoch": time.time(),
        })
        print(
            f"PASS {name} quant={build['quant_seconds']:.3f}s "
            f"logical_bytes={logical_bytes} artifact={artifact_ref['bytes']}", flush=True,
        )
        del candidate, packed, readback, source_weight, raw_fit, fit_windows
        gc.collect()
        torch.cuda.empty_cache()
    final = {
        "schema": "qtip-wire-worker-done-v1", "status": "PASS", "task": TASK,
        "host": socket.gethostname(), "layer": args.layer,
        "worker_index": args.worker_index, "workers": args.workers,
        "expected_units": len(expected_units), "completed_units": completed,
        "experts": experts, "projections": projections,
        "fit": fit_ref, "current_plane": plane_ref, "sources": qtip_sources,
        "claim_sha256": require_claim()[0], "finished_unix": time.time(),
    }
    done_worker = out / "done" / f"L{args.layer:03d}.W{args.worker_index:02d}.DONE.json"
    atomic_json(done_worker, final)
    final["done_sha256"] = sha256(done_worker)
    atomic_json(status_path, final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
