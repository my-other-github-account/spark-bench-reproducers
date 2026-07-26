#!/usr/bin/env python3
"""Production-compatible two-unit whole-matrix QTIP builder.

Keeps the canonical worker/artifact/DONE schemas while batching independent
experts through Hessian LDL, LDLQ/Viterbi, packing, and reconstruction.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import socket
import sys
import time

import torch

import qtip_wire_build_serial_v2 as serial
from batched_ldlq_v3 import LDLQ_batch, block_LDL_batch, pack_kernel_layout_batch
from triton_viterbi_prefix32 import install_prefix_viterbi

TASK = serial.TASK
PROJECTIONS = serial.PROJECTIONS
BATCH_UNITS = 2


def tensor_sane(t: torch.Tensor) -> dict:
    return {
        "finite": bool(torch.isfinite(t).all()),
        "max_abs": float(t.float().abs().max()),
    }


def build_batch(qv, kernel_decode, cb, sources, windows, seeds, device):
    units = len(sources)
    if units != BATCH_UNITS:
        raise ValueError(f"batch-v3 requires exactly {BATCH_UNITS} units, got {units}")
    shapes = {tuple(x.shape) for x in sources}
    if len(shapes) != 1:
        raise ValueError(f"mixed source geometry: {shapes}")
    m, k = next(iter(shapes))

    started = time.perf_counter()
    su_rows, sv_rows = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        su_rows.append((torch.randn(k, device=device).sign() + 1e-5).sign().float())
        sv_rows.append((torch.randn(m, device=device).sign() + 1e-5).sign().float())
    su = torch.stack(su_rows)
    sv = torch.stack(sv_rows)

    hessians, fit_rows, fit_mass = [], [], []
    for unit in range(units):
        h, rows, mass = qv.build_hessian(windows[unit], su[unit], device)
        hessians.append(h)
        fit_rows.append(rows)
        fit_mass.append(mass)
    hessian = torch.stack(hessians)
    diagmean = torch.diagonal(hessian, dim1=-2, dim2=-1).mean(dim=-1)
    hessian.div_(diagmean[:, None, None])
    diagonal = torch.arange(k, device=device)
    hessian[:, diagonal, diagonal] += 1e-2
    hessian.mul_(diagmean[:, None, None])

    weight = torch.stack(sources).to(device=device, dtype=torch.float32)
    transformed = qv.fwht(
        qv.fwht(weight.transpose(1, 2) * sv[:, None, :]).transpose(1, 2)
        * su[:, None, :]
    )
    lut_rms = cb.lut.double().square().mean().sqrt().float() * 0.9
    wscale = transformed.square().mean(dim=(1, 2)).sqrt() / lut_rms
    transformed.div_(wscale[:, None, None])
    lower = block_LDL_batch(hessian, 16)
    lower[:, diagonal, diagonal] = 0
    del hessian, hessians, weight
    torch.cuda.empty_cache()

    quant_started = time.perf_counter()
    quantized, states = LDLQ_batch(
        transformed, lower, cb,
        argparse.Namespace(td_x=16, td_y=16, V=2),
        buf_cols=128, for_kernel=True,
    )
    quant_seconds = time.perf_counter() - quant_started
    del quantized, transformed, lower
    torch.cuda.empty_cache()

    packed, pack_receipts = pack_kernel_layout_batch(cb, states, m, k)
    del states
    torch.cuda.empty_cache()

    lut_index = torch.arange(1 << 16, device=device)
    quadratic = (lut_index + 1) * lut_index
    sign_flip = 1 - ((quadratic >> 15) & 1) * 2
    lookup = (quadratic >> (16 - 9 - 1)) & 511
    expanded = cb.tlut.float().to(device)[lookup]
    expanded[:, 0] *= sign_flip

    candidates = []
    builds = []
    for unit in range(units):
        raw = kernel_decode.decode_compressed(
            16, 9, 3, 1, m, k, packed[unit].reshape(-1), expanded
        ) * wscale[unit]
        reconstructed = qv.fwht(raw.T).T * sv[unit, :, None]
        reconstructed = qv.fwht(reconstructed) * su[unit]
        sane = tensor_sane(reconstructed)
        if not sane["finite"] or sane["max_abs"] > 100:
            raise RuntimeError(f"batch-v3 reconstruction sanity failed unit={unit}: {sane}")
        candidates.append({
            "schema": "ds4-qtip-hyb-bounded36-unit-v1",
            "shape": [m, k],
            "trellis": packed[unit].cpu(),
            "SU": su[unit].half().cpu(),
            "SV": sv[unit].half().cpu(),
            "Wscale": wscale[unit].cpu(),
            "tlut": cb.tlut.cpu(),
            "reconstructed_weight": reconstructed.half().cpu(),
            "geometry": {
                "L": 16, "K": 3, "V": 2, "tlut_bits": 9,
                "decode_mode": "quantlut_sym", "td_x": 16, "td_y": 16,
            },
        })
        builds.append({
            "rht_seed": seeds[unit],
            "quant_seconds": quant_seconds,
            "batch_units": units,
            "batch_quant_seconds_total": quant_seconds,
            "batch_wall_seconds_total": time.perf_counter() - started,
            "fit_rows": fit_rows[unit],
            "fit_route_mass": fit_mass[unit],
            "canonical_pack": pack_receipts[unit],
            "reconstruction_sanity": sane,
            "implementation": "whole-matrix-cross-unit-batch-v3",
        })
        del raw, reconstructed
    torch.cuda.empty_cache()
    return candidates, builds


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

    if not 0 <= args.layer <= 42:
        raise ValueError(args.layer)
    if not (args.workers >= 1 and 0 <= args.worker_index < args.workers):
        raise ValueError("invalid worker partition")
    experts = serial.parse_int_set(args.experts, 0, 255)
    experts = [e for e in experts if e % args.workers == args.worker_index]
    projections = [p.strip() for p in args.projections.split(",") if p.strip()]
    if projections != list(PROJECTIONS):
        raise ValueError(f"batch-v3 production path requires {list(PROJECTIONS)}, got {projections}")
    if len(experts) % BATCH_UNITS:
        raise ValueError(f"expert partition must be divisible by {BATCH_UNITS}: {len(experts)}")

    claim_sha, claim = serial.require_claim()
    if socket.gethostname() not in {
        "compute-node-1", "compute-node-2", "compute-node-3", "compute-node-4", "compute-node-6", "compute-node-7", "compute-node-8", "compute-node-5-work"
    }:
        raise RuntimeError(f"unexpected host {socket.gethostname()}")
    serial.validate_seed_compatibility()
    qtip_root = args.qtip_root.resolve()
    qv = serial.load_parent_module(qtip_root)
    bitshift, _, _, kernel_decode = qv.load_official_qtip()
    tlut_payload = torch.load(args.tlut_source, map_location="cpu", mmap=True, weights_only=True)
    pinned_tlut = tlut_payload["tlut"].float().contiguous()
    tlut_sha = serial.tensor_sha256(pinned_tlut)
    expected_tlut_sha = json.loads(serial.PARENT_MANIFEST.read_text())["qtip_package"]["tlut_tensor_sha256"]
    if tlut_sha != expected_tlut_sha:
        raise RuntimeError(f"TLUT drift {tlut_sha} != {expected_tlut_sha}")
    cb = bitshift.bitshift_codebook(
        L=16, K=3, V=2, tlut_bits=9, decode_mode="quantlut_sym", tlut=pinned_tlut.to("cuda")
    ).to("cuda")
    fast_viterbi = install_prefix_viterbi(cb)
    fit_entries, fit_ref = serial.load_fit_capture_receipt(args.fit_receipt, args.layer)
    model = serial.ModelReader(args.model.resolve(), args.model_shard_hashes.resolve())
    plane = args.current_plane.resolve()
    plane_ref = serial.validate_plane_receipt(args.current_plane_receipt.resolve(), plane)
    plane_data = torch.load(plane, map_location="cpu", mmap=True, weights_only=True)
    device = torch.device("cuda")

    out = args.output.resolve()
    units_dir = out / "units" / f"L{args.layer:03d}"
    status_path = out / "status" / f"L{args.layer:03d}.W{args.worker_index:02d}.json"
    builder_path = Path(__file__).resolve()
    qtip_sources = {
        "qtip_commit": serial.QTIP_COMMIT,
        "bitshift_sha256": serial.sha256(qtip_root / "lib/codebook/bitshift.py"),
        "ldlq_sha256": serial.sha256(qtip_root / "lib/algo/ldlq.py"),
        "math_utils_sha256": serial.sha256(qtip_root / "lib/utils/math_utils.py"),
        "kernel_decompress_sha256": serial.sha256(qtip_root / "lib/utils/kernel_decompress.py"),
        "parent_runner_sha256": serial.sha256(serial.PARENT_RUNNER),
        "builder_sha256": serial.sha256(builder_path),
        "fast_viterbi": fast_viterbi,
        "fast_viterbi_sha256": serial.sha256(builder_path.parent / "triton_viterbi_prefix32.py"),
        "batched_ldlq_sha256": serial.sha256(builder_path.parent / "batched_ldlq_v3.py"),
        "implementation": "whole-matrix-cross-unit-batch-v3",
    }
    expected_units = [(expert, projection) for expert in experts for projection in projections]
    completed = 0
    serial.atomic_json(status_path, {
        "schema": "qtip-wire-worker-status-v1", "status": "RUNNING", "task": TASK,
        "host": socket.gethostname(), "layer": args.layer,
        "worker_index": args.worker_index, "workers": args.workers,
        "expected_units": len(expected_units), "completed_units": 0,
        "claim_sha256": claim_sha, "claim_nonce": claim.get("claim_nonce"),
        "fit": fit_ref, "current_plane": plane_ref, "sources": qtip_sources,
        "started_unix": time.time(),
    })

    for projection in projections:
        for group_start in range(0, len(experts), BATCH_UNITS):
            group = experts[group_start:group_start + BATCH_UNITS]
            identities = [
                {"layer": args.layer, "expert": expert, "projection": projection}
                for expert in group
            ]
            artifacts = [units_dir / f"{serial.unit_name(args.layer, expert, projection)}.pt" for expert in group]
            done_paths = [x.with_name(x.stem + ".DONE.json") for x in artifacts]
            resume = [serial.resume_ok(a, d, i) for a, d, i in zip(artifacts, done_paths, identities)]
            if all(resume):
                completed += len(group)
                print(f"RESUME_BATCH projection={projection} experts={group}", flush=True)
                continue
            if any(resume):
                print(
                    f"REBUILD_PARTIAL_BATCH projection={projection} experts={group} resume={resume}",
                    flush=True,
                )

            serial.require_claim()
            raw_windows, fit_windows, sources, source_refs, seeds = [], [], [], [], []
            for expert in group:
                raw = qv.expert_windows(fit_entries, expert)
                if projection == "down":
                    fit = qv.down_windows(raw, serial.current_fused13(qv, plane_data, expert), device)
                else:
                    fit = raw
                source, source_ref = model.projection(args.layer, expert, projection)
                raw_windows.append(raw)
                fit_windows.append(fit)
                sources.append(source)
                source_refs.append(source_ref)
                seeds.append(serial.rht_seed(args.layer, expert, projection))

            batch_started = time.time()
            candidates, builds = build_batch(qv, kernel_decode, cb, sources, fit_windows, seeds, device)
            batch_wall = time.time() - batch_started
            for unit, (expert, identity, artifact, done_path, candidate, build, source_ref, seed) in enumerate(
                zip(group, identities, artifacts, done_paths, candidates, builds, source_refs, seeds)
            ):
                reconstruction_sha = serial.tensor_sha256(candidate["reconstructed_weight"])
                packed = {
                    "schema": "qtip-hyb-wire-unit-v1", "task": TASK,
                    "identity": identity, "shape": candidate["shape"],
                    "trellis": candidate["trellis"], "SU": candidate["SU"],
                    "SV": candidate["SV"], "Wscale": candidate["Wscale"],
                    "geometry": candidate["geometry"], "rht_seed": seed,
                    "tlut_sha256": tlut_sha,
                }
                logical_bytes = sum(
                    packed[key].numel() * packed[key].element_size()
                    for key in ("trellis", "SU", "SV", "Wscale")
                )
                source_values = math.prod(candidate["shape"])
                serial.atomic_torch(artifact, packed)
                artifact_ref = {
                    "path": str(artifact), "sha256": serial.sha256(artifact),
                    "bytes": artifact.stat().st_size,
                }
                readback = torch.load(artifact, map_location="cpu", mmap=True, weights_only=True)
                trellis_sha = serial.tensor_sha256(readback["trellis"])
                if trellis_sha != serial.tensor_sha256(candidate["trellis"]):
                    raise RuntimeError(f"artifact readback trellis mismatch: {artifact}")
                build["batch_wall_seconds_total"] = batch_wall
                build["average_full_unit_wall_seconds"] = batch_wall / len(group)
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
                    "build": build, "build_wall_seconds": batch_wall / len(group),
                    "batch": {"units": len(group), "experts": group, "projection": projection},
                    "exact_command": " ".join(sys.argv), "created_unix": time.time(),
                }
                serial.atomic_json(done_path, done)
                completed += 1
                print(
                    f"PASS_BATCH {serial.unit_name(args.layer, expert, projection)} "
                    f"pair_wall={batch_wall:.3f}s avg={batch_wall/len(group):.3f}s",
                    flush=True,
                )
                del candidate, packed, readback
            del raw_windows, fit_windows, sources, source_refs, candidates, builds
            gc.collect()
            torch.cuda.empty_cache()
            serial.atomic_json(status_path, {
                "schema": "qtip-wire-worker-status-v1", "status": "RUNNING", "task": TASK,
                "host": socket.gethostname(), "layer": args.layer,
                "worker_index": args.worker_index, "workers": args.workers,
                "expected_units": len(expected_units), "completed_units": completed,
                "last_batch": {"experts": group, "projection": projection},
                "claim_sha256": serial.require_claim()[0], "epoch": time.time(),
                "sources": qtip_sources,
            })

    final = {
        "schema": "qtip-wire-worker-done-v1", "status": "PASS", "task": TASK,
        "host": socket.gethostname(), "layer": args.layer,
        "worker_index": args.worker_index, "workers": args.workers,
        "expected_units": len(expected_units), "completed_units": completed,
        "experts": experts, "projections": projections,
        "fit": fit_ref, "current_plane": plane_ref, "sources": qtip_sources,
        "claim_sha256": serial.require_claim()[0], "finished_unix": time.time(),
    }
    done_worker = out / "done" / f"L{args.layer:03d}.W{args.worker_index:02d}.DONE.json"
    serial.atomic_json(done_worker, final)
    final["done_sha256"] = serial.sha256(done_worker)
    serial.atomic_json(status_path, final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
