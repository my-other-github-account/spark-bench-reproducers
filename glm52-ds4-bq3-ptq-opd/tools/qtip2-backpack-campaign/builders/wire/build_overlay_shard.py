#!/usr/bin/env python3
"""Build only assignment-changed GENESIS/QTIP2 cells for one P640 layer shard.

Unchanged cells are immutable copy-through references to the sealed current GENESIS
wire. Changed VQ cells are rebuilt with canonical shared builder SHA 60b594ac...;
QTIP2 cells are copied byte-for-byte from the sealed REP16 manifests; native cells
remain verbatim checkpoint references. The output is a compact overlay shard, not a
second 101 GB duplicate of the immutable base package.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

TASK = "PUBLIC_TASK"
ASSIGNMENT_SHA = "c030883fddb1217529d67444d08257c4a1df18e2adbc93be092aba3d3611bc65"
BASE_ASSIGNMENT_SHA = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
BUILDER_SHA = "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e"
CKPT_INDEX_SHA = "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a"
EXPECTED_WIRE_BYTES = 101_346_521_679
QTIP_TIER = "qtip2_2.0117"
NATIVE_TIER = "native_mxfp4"


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(obj, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def flatten_assignment(doc: dict) -> dict[tuple[int, int, str], str]:
    out: dict[tuple[int, int, str], str] = {}
    for ls, experts in doc["assignment"].items():
        for es, projections in experts.items():
            for projection, tier in projections.items():
                out[(int(ls), int(es), projection)] = str(tier)
    return out


def tier_params(tier: str) -> tuple[int, int]:
    ds, ks = tier.split("_k")
    return int(ds[1:]), int(ks)


def source_projection(projection: str) -> str:
    return "fused13" if projection == "fused13" else "down"


def base_projection(projection: str) -> str:
    return "fused13" if projection == "fused13" else "down"


def load_raw_codebook(path: Path, d: int, k: int) -> torch.Tensor:
    expected = d * k * 2
    if path.stat().st_size != expected:
        raise RuntimeError(f"codebook bytes mismatch {path}: {path.stat().st_size} != {expected}")
    return torch.from_file(str(path), shared=False, size=d * k, dtype=torch.float16).clone().reshape(k, d)


def gpu_apps() -> list[str]:
    cp = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return [x.strip() for x in cp.stdout.splitlines() if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=Path, required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--layers", required=True, help="inclusive START-END")
    ap.add_argument("--disk-floor-gib", type=float, default=8.0)
    args = ap.parse_args()

    mission = args.mission.resolve()
    start, end = map(int, args.layers.split("-", 1))
    layers = list(range(start, end + 1))
    inputs = mission / "inputs"
    code = mission / "code"
    outputs = mission / "overlay"
    receipts = mission / "receipts"
    run = mission / "run"
    for p in (outputs / "cells", outputs / "codebooks", receipts, run):
        p.mkdir(parents=True, exist_ok=True)

    final_path = inputs / "ASSIGNMENT_RESPENT.json"
    base_path = inputs / "CURRENT_GENESIS_ASSIGNMENT.json"
    plan_path = inputs / "BUILD_PLAN.json"
    qtip_expected_path = inputs / "QTIP_SELECTED_EXPECTED.json"
    builder_path = code / "canonical_shared_builder.py"
    pilot_dir = code / "pilot_code"
    ckpt = Path(os.environ.get("P640_CKPT", "$HOME/models/hf/DeepSeek-V4-Flash"))
    ckpt_index = ckpt / "model.safetensors.index.json"
    base_cb_root = inputs / "base_codebooks"
    qtip_stage = inputs / "qtip_selected"
    claim_path = Path("$HOME/HOST_CLAIM.json")

    actual_inputs = {
        "assignment": sha256_file(final_path),
        "base_assignment": sha256_file(base_path),
        "builder": sha256_file(builder_path),
        "checkpoint_index": sha256_file(ckpt_index),
        "build_plan": sha256_file(plan_path),
        "qtip_expected": sha256_file(qtip_expected_path),
    }
    expected_inputs = {
        "assignment": ASSIGNMENT_SHA,
        "base_assignment": BASE_ASSIGNMENT_SHA,
        "builder": BUILDER_SHA,
        "checkpoint_index": CKPT_INDEX_SHA,
    }
    for name, want in expected_inputs.items():
        if actual_inputs[name] != want:
            raise RuntimeError(f"input pin mismatch {name}: {actual_inputs[name]} != {want}")
    if os.uname().nodename != args.host:
        raise RuntimeError(f"host pin mismatch {os.uname().nodename} != {args.host}")
    claim = json.loads(claim_path.read_text())
    if claim.get("owner") != TASK or claim.get("host") != args.host:
        raise RuntimeError(f"claim mismatch: {claim.get('owner')} {claim.get('host')}")

    final_doc = json.loads(final_path.read_text())
    base_doc = json.loads(base_path.read_text())
    final = flatten_assignment(final_doc)
    base = flatten_assignment(base_doc)
    if set(final) != set(base) or len(final) != 43 * 256 * 2:
        raise RuntimeError("assignment universe mismatch")
    changed = [
        {"layer": k[0], "expert": k[1], "projection": k[2], "old": base[k], "new": final[k]}
        for k in sorted(final) if final[k] != base[k] and k[0] in layers
    ]
    qtip_expected_doc = json.loads(qtip_expected_path.read_text())
    if qtip_expected_doc.get("assignment_sha256") != ASSIGNMENT_SHA:
        raise RuntimeError("QTIP expectation assignment pin mismatch")
    qtip_expected = {
        (int(r["layer"]), int(r["expert"]), r["projection"]): r
        for r in qtip_expected_doc["rows"]
    }
    if sorted(set(r["layer"] for r in changed) - set(layers)):
        raise RuntimeError("changed row outside shard")

    status_path = run / "STATUS.json"
    def status(state: str, **extra: object) -> None:
        atomic_json(status_path, {
            "schema": "p640-overlay-shard-status-v1", "task": TASK, "host": args.host,
            "shard": args.shard, "state": state, "layers": layers, "pid": os.getpid(),
            "updated_unix": time.time(), **extra,
        })

    preflight = {
        "schema": "p640-overlay-shard-preflight-v1", "status": "PASS", "task": TASK,
        "host": args.host, "shard": args.shard, "layers": layers,
        "changed_cells": len(changed), "qtip2_cells": sum(r["new"] == QTIP_TIER for r in changed),
        "vq_cells": sum(r["new"] not in {QTIP_TIER, NATIVE_TIER} for r in changed),
        "native_reference_cells": sum(r["new"] == NATIVE_TIER for r in changed),
        "unchanged_copythrough_cells": len(layers) * 256 * 2 - len(changed),
        "input_sha256": actual_inputs, "expected_wire_bytes": EXPECTED_WIRE_BYTES,
        "canonical_builder_sha256": BUILDER_SHA, "claim_sha256": sha256_file(claim_path),
        "gpu_apps_before_cuda": gpu_apps(), "disk_free_bytes": shutil.disk_usage(mission).free,
        "disk_floor_bytes": int(args.disk_floor_gib * (1 << 30)), "created_unix": time.time(),
    }
    if preflight["gpu_apps_before_cuda"]:
        raise RuntimeError(f"pre-existing GPU apps: {preflight['gpu_apps_before_cuda']}")
    if preflight["disk_free_bytes"] < preflight["disk_floor_bytes"]:
        raise RuntimeError("disk floor failed before build")
    atomic_json(receipts / "PREFLIGHT.json", preflight)

    os.environ["VQ3U_PILOT"] = str(pilot_dir)
    os.environ["VQ3U_PILOT_LEDGER"] = "$HOME/run-bundles/VQ8C_D8K4096_PUBLIC_TASK/assets/VQ3_LEDGER.jsonl"
    os.environ["VQ3U_CKPT"] = str(ckpt)
    modules: dict[str, object] = {}
    def load_builder(tier: str):
        if tier in modules:
            return modules[tier]
        d, k = tier_params(tier)
        spec = importlib.util.spec_from_file_location(f"p640_builder_{tier}", builder_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import canonical builder")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.D = d
        module.CB_K = k
        modules[tier] = module
        return module

    manifest_rows: list[dict] = []
    for layer in layers:
        layer_rows = [r for r in changed if r["layer"] == layer]
        layer_done = receipts / f"LAYER_{layer:03d}.json"
        if layer_done.is_file():
            old_done = json.loads(layer_done.read_text())
            if old_done.get("status") == "PASS" and old_done.get("assignment_sha256") == ASSIGNMENT_SHA:
                valid = True
                for rr in old_done.get("rows", []):
                    if rr.get("artifact"):
                        p = Path(rr["artifact"])
                        valid = valid and p.is_file() and p.stat().st_size == rr["artifact_bytes"] and sha256_file(p) == rr["artifact_sha256"]
                if valid:
                    manifest_rows.extend(old_done["rows"])
                    status("RESUME_SKIP_LAYER", active_layer=layer, completed_rows=len(manifest_rows))
                    continue

        status("BUILD_LAYER", active_layer=layer, layer_changed_cells=len(layer_rows), completed_rows=len(manifest_rows))
        row_results: list[dict] = []
        vq_rows = [r for r in layer_rows if r["new"] not in {QTIP_TIER, NATIVE_TIER}]
        packed = bundle = None
        if vq_rows:
            extract_module = load_builder("d4_k256")
            packed = extract_module.extract_layer(layer)
            bundle = extract_module.mem_bundle(packed)

        fit_cache: dict[str, dict[str, torch.Tensor]] = {}
        codebook_cache: dict[tuple[str, str], tuple[torch.Tensor, Path, str]] = {}
        for tier in sorted({r["new"] for r in vq_rows}):
            module = load_builder(tier)
            d, k = tier_params(tier)
            needed_projs = sorted({r["projection"] for r in vq_rows if r["new"] == tier})
            missing: list[str] = []
            for projection in needed_projs:
                pname = base_projection(projection)
                src = base_cb_root / f"layer_{layer:03d}" / f"{tier}.{pname}.codebook.fp16.bin"
                dst = outputs / "codebooks" / f"L{layer:03d}.{tier}.{pname}.codebook.fp16.bin"
                if src.is_file():
                    cb16 = load_raw_codebook(src, d, k)
                    shutil.copyfile(src, dst)
                    codebook_cache[(tier, projection)] = (cb16, dst, sha256_file(dst))
                else:
                    missing.append(projection)
            if missing:
                if bundle is None:
                    raise RuntimeError("missing bundle for codebook fit")
                fitted = module.fit_layer_cbs(bundle, layer)
                fit_cache[tier] = fitted
                for projection in missing:
                    pname = base_projection(projection)
                    cb16 = fitted[pname].to(torch.float16).cpu()
                    dst = outputs / "codebooks" / f"L{layer:03d}.{tier}.{pname}.codebook.fp16.bin"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    raw = cb16.contiguous().numpy().tobytes()
                    with dst.open("wb") as f:
                        f.write(raw); f.flush(); os.fsync(f.fileno())
                    if dst.stat().st_size != d * k * 2:
                        raise RuntimeError(f"fitted codebook size mismatch {dst}")
                    codebook_cache[(tier, projection)] = (cb16, dst, sha256_file(dst))

        for row_index, row in enumerate(layer_rows, 1):
            layer_i, expert, projection, target = row["layer"], row["expert"], row["projection"], row["new"]
            identity = (layer_i, expert, projection)
            base_result = {
                **row, "schema": "p640-overlay-cell-v1", "status": "PASS", "task": TASK,
                "host": args.host, "shard": args.shard, "assignment_sha256": ASSIGNMENT_SHA,
                "canonical_builder_sha256": BUILDER_SHA,
            }
            if target == QTIP_TIER:
                exp = qtip_expected.get(identity)
                if exp is None:
                    raise RuntimeError(f"missing sealed QTIP expectation {identity}")
                matches = list(qtip_stage.rglob(exp["basename"]))
                if len(matches) != 1:
                    raise RuntimeError(f"QTIP staged identity count {identity}: {matches}")
                src = matches[0]
                if src.stat().st_size != int(exp["artifact_bytes"]) or sha256_file(src) != exp["artifact_sha256"]:
                    raise RuntimeError(f"QTIP artifact hash/size mismatch {identity}: {src}")
                dst = outputs / "cells" / f"L{layer_i:03d}_E{expert:03d}_{projection}__qtip2.pt"
                shutil.copyfile(src, dst)
                result = {
                    **base_result, "kind": "qtip2_exact_copy", "artifact": str(dst),
                    "artifact_bytes": dst.stat().st_size, "artifact_sha256": sha256_file(dst),
                    "logical_wire_bytes": int(exp["logical_bytes"]), "logical_bpw": exp["logical_bpw"],
                    "source_manifest": exp["manifest"], "source_artifact_sha256": exp["artifact_sha256"],
                    "finite": True,
                }
            elif target == NATIVE_TIER:
                result = {
                    **base_result, "kind": "native_checkpoint_reference", "artifact": None,
                    "checkpoint_index": str(ckpt_index), "checkpoint_index_sha256": CKPT_INDEX_SHA,
                    "weight_key_prefix": f"layers.{layer_i}.ffn.experts.{expert}.", "finite": True,
                }
            else:
                if bundle is None:
                    raise RuntimeError("missing source bundle for VQ row")
                module = load_builder(target)
                d, k = tier_params(target)
                pname = source_projection(projection)
                cb16_cpu, cb_path, cb_sha = codebook_cache[(target, projection)]
                cb16 = cb16_cpu.to("cuda")
                cb = cb16.float()
                W, sb = bundle.fused13(expert) if projection == "fused13" else bundle.down(expert)
                if not bool(torch.isfinite(W).all().item()):
                    raise RuntimeError(f"nonfinite source weight {identity}")
                codes, scales, metrics = module.build_unit(W, sb, cb, cb16)
                scol = module.gp.sbytes_to_scol(scales)
                replay = module.vp.assign_chunk((W / scol).view(-1, d), cb16.float()).view_as(codes)
                replay_exact = bool(replay.eq(codes.long()).all().item())
                if not replay_exact or not all(math.isfinite(float(v)) for v in metrics.values()):
                    raise RuntimeError(f"replay/metric failure {identity} replay={replay_exact} metrics={metrics}")
                target_dtype = torch.uint8 if k <= 256 else torch.int16
                payload = {
                    "codes": codes.to(target_dtype).cpu(), "scales": scales.cpu(),
                    "meta": {
                        "schema": "p640-genesis-vq-overlay-cell-v1", "task": TASK,
                        "layer": layer_i, "expert": expert, "projection": projection,
                        "tier": target, "d": d, "k": k, "assignment_sha256": ASSIGNMENT_SHA,
                        "canonical_builder_sha256": BUILDER_SHA, "codebook_sha256": cb_sha,
                        "fp16_codebook_replay_exact": True, "metrics": metrics,
                    },
                }
                dst = outputs / "cells" / f"L{layer_i:03d}_E{expert:03d}_{projection}__{target}.pt"
                tmp = dst.with_suffix(".tmp")
                torch.save(payload, tmp)
                with tmp.open("rb") as f: os.fsync(f.fileno())
                os.replace(tmp, dst)
                result = {
                    **base_result, "kind": "genesis_vq_rebuilt_cell", "artifact": str(dst),
                    "artifact_bytes": dst.stat().st_size, "artifact_sha256": sha256_file(dst),
                    "codebook": str(cb_path), "codebook_sha256": cb_sha,
                    "d": d, "k": k, "codes_dtype": str(target_dtype),
                    "fp16_codebook_replay_exact": True, "metrics": metrics, "finite": True,
                }
                del W, sb, codes, scales, scol, replay, cb16, cb, payload
            row_results.append(result)
            status("BUILD_CELL", active_layer=layer, cell_index=row_index, layer_cell_count=len(layer_rows), completed_rows=len(manifest_rows) + len(row_results))
            if row_index % 16 == 0:
                torch.cuda.empty_cache()

        layer_receipt = {
            "schema": "p640-overlay-layer-receipt-v1", "status": "PASS", "task": TASK,
            "host": args.host, "shard": args.shard, "layer": layer,
            "assignment_sha256": ASSIGNMENT_SHA, "base_assignment_sha256": BASE_ASSIGNMENT_SHA,
            "canonical_builder_sha256": BUILDER_SHA, "changed_cells": len(layer_rows),
            "qtip2_cells": sum(r["new"] == QTIP_TIER for r in layer_rows),
            "vq_cells": sum(r["new"] not in {QTIP_TIER, NATIVE_TIER} for r in layer_rows),
            "native_reference_cells": sum(r["new"] == NATIVE_TIER for r in layer_rows),
            "unchanged_copythrough_cells": 512 - len(layer_rows), "rows": row_results,
            "completed_unix": time.time(),
        }
        atomic_json(layer_done, layer_receipt)
        manifest_rows.extend(row_results)
        if bundle is not None:
            del bundle, packed
        fit_cache.clear(); codebook_cache.clear(); torch.cuda.empty_cache()

    if len(manifest_rows) != len(changed):
        raise RuntimeError(f"shard changed-cell coverage mismatch {len(manifest_rows)} != {len(changed)}")
    identities = [(r["layer"], r["expert"], r["projection"]) for r in manifest_rows]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate changed-cell identity")
    manifest = {
        "schema": "p640-genesis-qtip2-overlay-shard-manifest-v1", "status": "PASS",
        "task": TASK, "host": args.host, "shard": args.shard, "layers": layers,
        "assignment_sha256": ASSIGNMENT_SHA, "base_assignment_sha256": BASE_ASSIGNMENT_SHA,
        "canonical_shared_builder_sha256": BUILDER_SHA, "expected_final_wire_bytes": EXPECTED_WIRE_BYTES,
        "changed_cells": len(manifest_rows), "qtip2_cells": sum(r["new"] == QTIP_TIER for r in manifest_rows),
        "vq_cells": sum(r["kind"] == "genesis_vq_rebuilt_cell" for r in manifest_rows),
        "native_reference_cells": sum(r["kind"] == "native_checkpoint_reference" for r in manifest_rows),
        "unchanged_copythrough_cells": len(layers) * 512 - len(manifest_rows),
        "copythrough_policy": "all unchanged cells resolve byte-for-byte from immutable current GENESIS wire c24a2205 lineage; no re-fit/re-encode",
        "rows": sorted(manifest_rows, key=lambda r: (r["layer"], r["expert"], r["projection"])),
        "completed_unix": time.time(),
    }
    manifest_path = receipts / "SHARD_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    done = {
        "schema": "p640-overlay-shard-done-v1", "status": "PASS", "task": TASK,
        "host": args.host, "shard": args.shard, "layers": layers,
        "changed_cells": len(manifest_rows), "assignment_sha256": ASSIGNMENT_SHA,
        "canonical_shared_builder_sha256": BUILDER_SHA, "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path), "all_finite": True,
        "all_vq_fp16_codebook_replay_exact": True, "completed_unix": time.time(),
    }
    atomic_json(receipts / "SHARD_DONE.json", done)
    (run / "DONE").write_text("PASS\n")
    status("PASS", changed_cells=len(manifest_rows), manifest_sha256=done["manifest_sha256"])
    print(json.dumps(done, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        try:
            mission_arg = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--mission"), None)
            if mission_arg:
                atomic_json(Path(mission_arg) / "run/STATUS.json", {
                    "schema": "p640-overlay-shard-status-v1", "task": TASK,
                    "state": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
                    "updated_unix": time.time(),
                })
        except Exception:
            pass
        raise
