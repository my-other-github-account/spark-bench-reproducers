#!/usr/bin/env python3
from __future__ import annotations

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
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

TASK = "PUBLIC_TASK"
HOST = "compute-node-6"
SHARD = "L00_L11"
LAYERS = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11]
SIBLINGS = {"A": list(range(0, 15)), "B": list(range(15, 29)), "C": list(range(29, 43))}
MISSION = Path("$HOME/run-bundles/P640_GENESIS_QTIP2_WIRE_PUBLIC_TASK_s6")
ASSIGNMENT = MISSION / "inputs/ASSIGNMENT_WITH.json"
INPUT_MANIFEST = MISSION / "inputs/INPUT_MANIFEST.json"
REPRO_GATE = MISSION / "inputs/REPRODUCTION_GATE.json"
FIX_RECEIPT = MISSION / "inputs/SHARED_REBUILD_IDENTITY_FIX.json"
BUILDER = MISSION / "code/canonical_shared_builder.py"
PILOT_DIR = MISSION / "code/pilot_code"
PILOT_GP = PILOT_DIR / "gptqv2_pilot.py"
PILOT_VP = PILOT_DIR / "vqw2_pilot.py"
CKPT = Path("$HOME/models/hf/DeepSeek-V4-Flash")
CKPT_INDEX = CKPT / "model.safetensors.index.json"
CLAIM = Path("$HOME/HOST_CLAIM.json")
OUT = MISSION / "outputs"
RECEIPTS = MISSION / "receipts"
RUN = MISSION / "run"
LOGS = MISSION / "logs"
PROGRESS = MISSION / "SHARD_L00_L11_PROGRESS.json"
PREFLIGHT = RECEIPTS / "PREFLIGHT.json"
MANIFEST = RECEIPTS / "SHARD_L00_L11_MANIFEST.json"
DONE = RECEIPTS / "SHARD_L00_L11_DONE.json"
WIRE_STREAM_IN = MISSION / "WIRE_STREAM_IN"
ASSIGNMENT_MAP_SHA256 = "26d0cd3bc7dfcb0cca3ffc37c628fd31342451d18386912842a40b2eb9243900"
EXPECTED = {
    ASSIGNMENT: "0ffe9e67d9ebda23295b15189785bf22297459e7ba3d2b33efd100a8c45021d2",
    INPUT_MANIFEST: "88ebfc21a7134088cad0a9a4f09821410db29ac13cd754d4d46f8902bebecb42",
    REPRO_GATE: "0877d42c61cbb0b22e404b93bfd0d1a5cd621c21ae7db4ecf2f4bc5195715536",
    FIX_RECEIPT: "f4ec0d70f95e81b7f2b445591aeb33ea1d73931a5885e6bf42d872e4c63b0fde",
    BUILDER: "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e",
    PILOT_GP: "3be4c3bf8704150104fdb6c426f0ea042a5a9775f7627891d3082e7c53a77e5a",
    PILOT_VP: "dd1ac4aeaaaa997a588e28124ff31f75d0185de4d8d3aa160d4a768dc3a7aceb",
    CKPT_INDEX: "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a",
}
TIERS = ["d4_k256", "d4_k512", "d4_k1024", "d4_k2048", "d4_k4096", "d8_k256", "d8_k512"]
DISK_FLOOR = 40 * (1 << 30)


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def set_status(state: str, **extra: object) -> None:
    atomic_json(RUN / "STATUS.json", {"task": TASK, "host": HOST, "state": state, "epoch": time.time(), **extra})


def assert_claim() -> dict:
    d = json.loads(CLAIM.read_text())
    if d.get("owner") != TASK or d.get("host") != HOST:
        raise RuntimeError(f"claim lost: owner={d.get('owner')} host={d.get('host')}")
    return d


def gpu_apps() -> list[str]:
    p = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    return [x.strip() for x in p.stdout.splitlines() if x.strip()]


def free_bytes() -> int:
    return shutil.disk_usage(MISSION).free


def tier_of(entry: object, proj: str) -> str:
    if isinstance(entry, dict):
        return str(entry["fused13" if proj == "13" else "down"])
    return str(entry)


def ids_for(assignment: dict, layer: int, tier: str, proj: str) -> list[int]:
    amap = assignment["assignment"][str(layer)]
    return [e for e in range(256) if tier_of(amap[str(e)], proj) == tier]


def tier_params(tier: str) -> tuple[int, int]:
    ds, ks = tier.split("_k")
    return int(ds[1:]), int(ks)


def row_weights(proj: str) -> int:
    return 4096 * (4096 if proj == "13" else 2048)


def wire_row_bytes(tier: str, proj: str) -> int:
    n = row_weights(proj)
    if tier == "native_mxfp4":
        return n * 3 // 32  # 4-bit value + 8-bit block32 scale = .75 bpw
    if tier.startswith("qtip2_"):
        return (n * 20117 + 79999) // 80000  # accounting-only; immutable QTIP2 planes are supplied separately
    d, k = tier_params(tier)
    return n * int(math.log2(k)) // d // 8 + n // 32


def intermediate_row_bytes(tier: str, proj: str) -> int:
    n = row_weights(proj)
    if tier == "native_mxfp4" or tier.startswith("qtip2_"):
        return 0  # source is referenced verbatim, not duplicated
    d, k = tier_params(tier)
    code_bytes = 1 if k <= 256 else 2
    return n // d * code_bytes + n // 32


def projections(assignment: dict, layer: int) -> dict:
    counts: dict[str, int] = {}
    wire = 0
    intermediate = 0
    represented: set[str] = set()
    for entry in assignment["assignment"][str(layer)].values():
        for proj in ("13", "2"):
            tier = tier_of(entry, proj)
            counts[tier] = counts.get(tier, 0) + 1
            represented.add(tier)
            wire += wire_row_bytes(tier, proj)
            intermediate += intermediate_row_bytes(tier, proj)
    for tier in represented:
        if tier != "native_mxfp4" and not tier.startswith("qtip2_"):
            d, k = tier_params(tier)
            wire += 2 * k * d * 2
            intermediate += 2 * k * d * 2
    return {"tier_projection_counts": dict(sorted(counts.items())), "projected_wire_bytes": wire, "projected_intermediate_bytes": intermediate}


def preflight(assignment: dict) -> dict:
    assert_claim()
    actual = {str(p): sha256_file(p) for p in EXPECTED}
    for p, want in EXPECTED.items():
        if actual[str(p)] != want:
            raise RuntimeError(f"sha mismatch {p}: {actual[str(p)]} != {want}")
    union = sorted(set().union(*(set(x) for x in SIBLINGS.values())))
    overlaps = {f"{a}_{b}": sorted(set(SIBLINGS[a]) & set(SIBLINGS[b])) for a, b in (("A", "B"), ("A", "C"), ("B", "C"))}
    if union != list(range(43)) or any(overlaps.values()):
        raise RuntimeError(f"shard partition invalid union={union} overlaps={overlaps}")
    if sorted(int(x) for x in assignment["assignment"]) != list(range(43)):
        raise RuntimeError("assignment layer set is not exactly 0..42")
    layer_rows = {str(L): projections(assignment, L) for L in LAYERS}
    projected = sum(x["projected_intermediate_bytes"] for x in layer_rows.values())
    free = free_bytes()
    if free - projected < DISK_FLOOR:
        raise RuntimeError(f"disk gate fail free={free} projected={projected} floor={DISK_FLOOR}")
    obj = {
        "schema": "genesis-build-shard-preflight-v1",
        "status": "PASS",
        "task": TASK,
        "host": HOST,
        "shard": SHARD,
        "layers": LAYERS,
        "layer_count": len(LAYERS),
        "sibling_layer_sets": SIBLINGS,
        "partition_union": union,
        "pairwise_overlaps": overlaps,
        "input_sha256": actual,
        "assignment_schema": assignment.get("schema"),
        "assignment_input_manifest_sha256": assignment.get("input_manifest_sha256"),
        "canonical_builder_runtime_parameterization": "import exact source SHA then set module.D/module.CB_K before fit/build; build_unit remains exact patched fp16-serialized assignment implementation",
        "layer_projections": layer_rows,
        "projected_intermediate_bytes": projected,
        "projected_wire_bytes": sum(x["projected_wire_bytes"] for x in layer_rows.values()),
        "disk_free_bytes": free,
        "disk_floor_bytes": DISK_FLOOR,
        "projected_free_after_intermediate": free - projected,
        "gpu_apps_before_launch": gpu_apps(),
        "completed_epoch": time.time(),
    }
    if obj["gpu_apps_before_launch"]:
        raise RuntimeError(f"pre-existing GPU apps: {obj['gpu_apps_before_launch']}")
    atomic_json(PREFLIGHT, obj)
    return obj


def load_builder(tier: str):
    d, k = tier_params(tier)
    os.environ["VQ3U_PILOT"] = str(PILOT_DIR)
    os.environ["VQ3U_PILOT_LEDGER"] = "$HOME/run-bundles/VQ8C_D8K4096_PUBLIC_TASK/assets/VQ3_LEDGER.jsonl"
    os.environ["VQ3U_CKPT"] = str(CKPT)
    spec = importlib.util.spec_from_file_location(f"canonical_builder_{tier}", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.D = d
    module.CB_K = k
    if int(module.D) != d or int(module.CB_K) != k:
        raise RuntimeError("builder runtime parameterization failed")
    return module, d, k


def file_valid(path: Path, marker: Path, layer: int, tier: str, ids13: list[int], ids2: list[int], command_sha: str) -> bool:
    if not path.is_file() or not marker.is_file():
        return False
    try:
        m = json.loads(marker.read_text())
        if m.get("status") != "PASS" or m.get("bytes") != path.stat().st_size or m.get("sha256") != sha256_file(path):
            return False
        if m.get("builder_sha256") != EXPECTED[BUILDER] or m.get("command_sha256") != command_sha or m.get("fp16_codebook_replay_exact_fraction") != 1.0:
            return False
        d = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        return (
            list(map(int, d["expert_ids13"].tolist())) == ids13
            and list(map(int, d["expert_ids2"].tolist())) == ids2
            and int(d["meta"]["layer"]) == layer
            and d["meta"]["tier"] == tier
        )
    except Exception:
        return False


def metric_summary(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r})
    out: dict[str, dict[str, float | int]] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r]
        if not vals or not all(math.isfinite(x) for x in vals):
            raise RuntimeError(f"non-finite metric {key}")
        out[key] = {"count": len(vals), "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
    return out


def build_tier_layer(assignment: dict, layer: int, tier: str, command_sha: str) -> dict:
    assert_claim()
    ids13 = ids_for(assignment, layer, tier, "13")
    ids2 = ids_for(assignment, layer, tier, "2")
    if not ids13 and not ids2:
        raise RuntimeError(f"empty represented tier {tier} L{layer}")
    out_dir = OUT / tier
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"layer_{layer:03d}.pt"
    marker = out.with_suffix(".DONE.json")
    if file_valid(out, marker, layer, tier, ids13, ids2, command_sha):
        print(f"L{layer:03d} {tier} skip verified", flush=True)
        return json.loads(marker.read_text())

    projected = projections(assignment, layer)["projected_intermediate_bytes"]
    if free_bytes() < DISK_FLOOR + min(projected, 6 * (1 << 30)):
        raise RuntimeError(f"disk floor before L{layer} {tier}: free={free_bytes()}")
    set_status("BUILDING", layer=layer, tier=tier, completed_layers=completed_layers())
    t0 = time.time()
    module, d, k = load_builder(tier)
    packed = module.extract_layer(layer)
    bundle = module.mem_bundle(packed)
    codebooks = module.fit_layer_cbs(bundle, layer)
    cb16 = {name: value.to(torch.float16) for name, value in codebooks.items()}
    if not all(torch.isfinite(v).all().item() for v in cb16.values()):
        raise RuntimeError(f"non-finite codebook L{layer} {tier}")

    data: dict[str, object] = {
        "expert_ids13": torch.tensor(ids13, dtype=torch.int16),
        "expert_ids2": torch.tensor(ids2, dtype=torch.int16),
        "cb13": cb16["fused13"].cpu(),
        "cb2": cb16["down"].cpu(),
    }
    replay_exact = 0
    replay_total = 0
    metrics_by_proj: dict[str, dict] = {}
    for proj, ids, pname in (("13", ids13, "fused13"), ("2", ids2, "down")):
        codes: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        metrics: list[dict] = []
        for n, expert in enumerate(ids, 1):
            W, sb = bundle.fused13(expert) if proj == "13" else bundle.down(expert)
            if not torch.isfinite(W).all().item():
                raise RuntimeError(f"non-finite source W L{layer} {tier} {proj} e{expert}")
            c, s, m = module.build_unit(W, sb, codebooks[pname], cb16[pname])
            scol = module.gp.sbytes_to_scol(s)
            vectors = (W / scol).view(-1, d)
            replay = module.vp.assign_chunk(vectors, cb16[pname].float()).view_as(c)
            same = replay.eq(c.long())
            replay_exact += int(same.sum().item())
            replay_total += int(same.numel())
            if not bool(same.all().item()):
                bad = int((~same).reshape(-1).nonzero()[0].item())
                raise RuntimeError(f"fp16 replay mismatch L{layer} {tier} {proj} e{expert} flat={bad}")
            target_dtype = torch.uint8 if k <= 256 else torch.int16
            codes.append(c.to(target_dtype).cpu())
            scales.append(s.cpu())
            metrics.append(m)
            del W, sb, c, s, scol, vectors, replay, same
            if n % 16 == 0:
                torch.cuda.empty_cache()
        code_shape = (4096, (4096 if proj == "13" else 2048) // d)
        scale_shape = (4096, 128 if proj == "13" else 64)
        target_dtype = torch.uint8 if k <= 256 else torch.int16
        data[f"codes{proj}"] = torch.stack(codes) if codes else torch.empty((0, *code_shape), dtype=target_dtype)
        data[f"sc{proj}"] = torch.stack(scales) if scales else torch.empty((0, *scale_shape), dtype=torch.uint8)
        metrics_by_proj[proj] = metric_summary(metrics)

    exact_fraction = replay_exact / replay_total if replay_total else 1.0
    if exact_fraction != 1.0:
        raise RuntimeError(f"fp16 replay fraction {exact_fraction}")
    data["meta"] = {
        "schema": "genesis-nominated-selected-vq-tier-v1",
        "task": TASK,
        "host": HOST,
        "shard": SHARD,
        "layer": layer,
        "tier": tier,
        "d": d,
        "k": k,
        "assignment_sha256": ASSIGNMENT_MAP_SHA256,
        "input_manifest_sha256": EXPECTED[INPUT_MANIFEST],
        "checkpoint_index_sha256": EXPECTED[CKPT_INDEX],
        "builder": str(BUILDER),
        "builder_sha256": EXPECTED[BUILDER],
        "builder_runtime_parameters": {"D": d, "CB_K": k},
        "command_sha256": command_sha,
        "selected_counts": {"fused13": len(ids13), "down": len(ids2)},
        "fp16_codebook_replay": {"exact": replay_exact, "total": replay_total, "exact_fraction": exact_fraction, "first_mismatch": None},
        "local_fit_metrics": metrics_by_proj,
        "finite_checks": {"source_weights": True, "codebooks": True, "metrics": True},
    }
    tmp = out.with_suffix(".tmp")
    torch.save(data, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, out)
    digest = sha256_file(out)
    mark = {
        "schema": "genesis-tier-layer-build-receipt-v1",
        "status": "PASS",
        "task": TASK,
        "host": HOST,
        "shard": SHARD,
        "layer": layer,
        "tier": tier,
        "output_path": str(out),
        "sha256": digest,
        "bytes": out.stat().st_size,
        "tier_counts": {"fused13": len(ids13), "down": len(ids2)},
        "input_sha256": {"assignment_file": EXPECTED[ASSIGNMENT], "assignment_map": ASSIGNMENT_MAP_SHA256, "input_manifest": EXPECTED[INPUT_MANIFEST], "checkpoint_index": EXPECTED[CKPT_INDEX], "shared_fix_receipt": EXPECTED[FIX_RECEIPT]},
        "builder_sha256": EXPECTED[BUILDER],
        "command_sha256": command_sha,
        "builder_runtime_parameters": {"D": d, "CB_K": k},
        "fp16_codebook_replay_exact": replay_exact,
        "fp16_codebook_replay_total": replay_total,
        "fp16_codebook_replay_exact_fraction": exact_fraction,
        "fp16_codebook_replay_first_mismatch": None,
        "local_fit_metrics": metrics_by_proj,
        "finite_checks": {"source_weights": True, "codebooks": True, "metrics": True},
        "elapsed_seconds": time.time() - t0,
        "completed_epoch": time.time(),
    }
    atomic_json(marker, mark)
    del data, packed, bundle, codebooks, cb16
    torch.cuda.empty_cache()
    print(f"L{layer:03d} {tier} PASS bytes={mark['bytes']} sha={digest[:12]} replay=1.0 secs={mark['elapsed_seconds']:.1f}", flush=True)
    return mark


def source_shard(layer: int, index: dict) -> tuple[Path, list[str]]:
    prefix = f"layers.{layer}.ffn.experts."
    keys = [k for k in index["weight_map"] if k.startswith(prefix) and (k.endswith(".weight") or k.endswith(".scale"))]
    shards = sorted({index["weight_map"][k] for k in keys})
    if len(shards) != 1:
        raise RuntimeError(f"L{layer} source shards {shards}")
    return CKPT / shards[0], keys


def layer_receipt_valid(layer: int) -> bool:
    p = RECEIPTS / f"LAYER_{layer:03d}.json"
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
        if d.get("status") != "PASS" or d.get("layer") != layer or d.get("fp16_codebook_replay_exact_fraction") != 1.0:
            return False
        for x in d["vq_outputs"]:
            q = Path(x["output_path"])
            if not q.is_file() or q.stat().st_size != x["bytes"] or sha256_file(q) != x["sha256"]:
                return False
        s = Path(d["native_source"]["source_shard"])
        return s.is_file() and s.stat().st_size == d["native_source"]["bytes"] and sha256_file(s) == d["native_source"]["sha256"]
    except Exception:
        return False


def completed_layers() -> list[int]:
    return [L for L in LAYERS if (RECEIPTS / f"LAYER_{L:03d}.json").is_file()]


def update_progress(active_layer: int | None = None, active_tier: str | None = None) -> None:
    rows = []
    for L in LAYERS:
        p = RECEIPTS / f"LAYER_{L:03d}.json"
        if p.is_file():
            try:
                d = json.loads(p.read_text())
                rows.append({"layer": L, "status": d.get("status"), "receipt": str(p), "receipt_sha256": sha256_file(p)})
            except Exception:
                pass
    atomic_json(PROGRESS, {
        "schema": "genesis-shard-progress-v1", "task": TASK, "host": HOST, "shard": SHARD,
        "expected_layers": LAYERS, "completed_layers": [x["layer"] for x in rows if x["status"] == "PASS"],
        "completed_count": sum(x["status"] == "PASS" for x in rows), "active_layer": active_layer,
        "active_tier": active_tier, "layers": rows, "updated_epoch": time.time(),
    })
    atomic_json(RUN / "RESUME.json", {"task": TASK, "expected_layers": LAYERS, "completed_layers": [x["layer"] for x in rows if x["status"] == "PASS"], "next_layer": next((L for L in LAYERS if L not in {x["layer"] for x in rows if x["status"] == "PASS"}), None), "updated_epoch": time.time()})


def finalize(assignment: dict, command_sha: str) -> None:
    assert_claim()
    receipts = []
    all_files = []
    for L in LAYERS:
        p = RECEIPTS / f"LAYER_{L:03d}.json"
        if not layer_receipt_valid(L):
            raise RuntimeError(f"layer receipt invalid L{L}")
        d = json.loads(p.read_text())
        receipts.append({"layer": L, "path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
        all_files.extend(d["vq_outputs"])
    layers = [x["layer"] for x in receipts]
    if layers != LAYERS or len(set(layers)) != len(LAYERS):
        raise RuntimeError(f"final exact layer set fail {layers}")
    manifest = {
        "schema": "genesis-build-shard-manifest-v1", "status": "PASS", "task": TASK,
        "host": HOST, "shard": SHARD, "layers": layers, "layer_count": len(layers),
        "missing_layers": sorted(set(LAYERS) - set(layers)), "extra_layers": sorted(set(layers) - set(LAYERS)),
        "placeholders": [], "assignment_sha256": ASSIGNMENT_MAP_SHA256, "assignment_file_sha256": EXPECTED[ASSIGNMENT], "input_manifest_sha256": EXPECTED[INPUT_MANIFEST],
        "canonical_shared_builder_sha256": EXPECTED[BUILDER], "command_sha256": command_sha,
        "layer_receipts": receipts, "vq_files": all_files,
        "vq_file_count": len(all_files), "vq_bytes": sum(x["bytes"] for x in all_files),
        "projected_wire_bytes": sum(projections(assignment, L)["projected_wire_bytes"] for L in LAYERS),
        "completed_epoch": time.time(),
    }
    atomic_json(MANIFEST, manifest)
    done = {
        "schema": "genesis-build-shard-done-v1", "status": "PASS", "task": TASK, "host": HOST,
        "shard": SHARD, "layers": layers, "layer_count": len(layers), "manifest": str(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST), "assignment_sha256": ASSIGNMENT_MAP_SHA256,
        "canonical_shared_builder_sha256": EXPECTED[BUILDER], "all_fp16_codebook_replay_exact_fraction": 1.0,
        "all_finite": True, "consumer": "PUBLIC_TASK P0 GENESIS FAN-IN", "completed_epoch": time.time(),
    }
    atomic_json(DONE, done)
    (RUN / "DONE").write_text("PASS\n")
    update_progress()
    set_status("BUILT", completed_layers=LAYERS, done=str(DONE), done_sha256=sha256_file(DONE))


def stage_layer(layer: int, tier_outputs: list[dict]) -> dict:
    """QSFP-stream a completed layer to the rail host using >=4 rsync streams."""
    assert_claim()
    stage = WIRE_STREAM_IN / f"layer_{layer:03d}"
    subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "203.0.113.6", "mkdir", "-p", str(stage)
    ], check=True)

    def push(mark: dict) -> dict:
        tier = str(mark["tier"])
        source = Path(str(mark["output_path"]))
        marker = source.with_suffix(".DONE.json")
        remote_dir = stage / tier
        subprocess.run([
            "ssh", "-o", "BatchMode=yes", "203.0.113.6",
            "mkdir", "-p", str(remote_dir)
        ], check=True)
        command = [
            "rsync", "-a", "--partial", "--append-verify", "--remove-source-files",
            "--timeout=180", "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4",
            str(source), str(marker), f"203.0.113.6:{remote_dir}/",
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"QSFP rsync failed tier={tier} rc={result.returncode}: {result.stderr.strip()}")
        dest = remote_dir / source.name
        dest_marker = remote_dir / marker.name
        if sha256_file(dest) != mark["sha256"]:
            raise RuntimeError(f"QSFP staged SHA drift L{layer} {tier}")
        source.symlink_to(dest)
        marker.symlink_to(dest_marker)
        return {"tier": tier, "path": str(dest), "sha256": mark["sha256"], "bytes": mark["bytes"]}

    if len(tier_outputs) < 4:
        raise RuntimeError(f"stream-count gate L{layer}: outputs={len(tier_outputs)} < 4")
    rows = []
    with ThreadPoolExecutor(max_workers=max(4, len(tier_outputs))) as pool:
        futures = [pool.submit(push, mark) for mark in tier_outputs]
        for future in as_completed(futures):
            rows.append(future.result())
    layer_receipt = RECEIPTS / f"LAYER_{layer:03d}.json"
    subprocess.run([
        "rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
        str(layer_receipt), f"203.0.113.6:{stage}/"
    ], check=True)
    done = {
        "schema": "p645-plane-done-v1", "status": "PLANE_DONE", "task": TASK,
        "host": HOST, "layer": layer, "assignment_map_sha256": ASSIGNMENT_MAP_SHA256,
        "builder_sha256": EXPECTED[BUILDER], "streams": len(tier_outputs),
        "transport": "QSFP_203.0.113.6", "outputs": sorted(rows, key=lambda x: x["tier"]),
        "layer_receipt_sha256": sha256_file(layer_receipt), "completed_epoch": time.time(),
    }
    atomic_json(stage / "PLANE_DONE.json", done)
    atomic_json(RECEIPTS / f"PLANE_DONE_L{layer:03d}.json", done)
    print(f"L{layer:03d} PLANE_DONE streams={len(tier_outputs)} -> {stage}", flush=True)
    return done


def run() -> int:
    for d in (OUT, RECEIPTS, RUN, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    script_sha = sha256_file(Path(__file__).resolve())
    launcher = MISSION / "code/launch_shard.sh"
    launcher_sha = sha256_file(launcher) if launcher.exists() else None
    command_spec = {"argv": [str(Path(__file__).resolve())], "script_sha256": script_sha, "launcher_sha256": launcher_sha, "task": TASK, "layers": LAYERS}
    command_sha = sha256_json(command_spec)
    set_status("PREFLIGHT", command=command_spec, command_sha256=command_sha)
    assignment = json.loads(ASSIGNMENT.read_text())
    preflight(assignment)
    index = json.loads(CKPT_INDEX.read_text())
    update_progress(active_layer=LAYERS[0])
    for layer in LAYERS:
        assert_claim()
        if layer_receipt_valid(layer):
            print(f"L{layer:03d} complete skip verified", flush=True)
            plane_done = RECEIPTS / f"PLANE_DONE_L{layer:03d}.json"
            if not plane_done.is_file():
                old = json.loads((RECEIPTS / f"LAYER_{layer:03d}.json").read_text())
                stage_layer(layer, old["vq_outputs"])
            update_progress(active_layer=next((x for x in LAYERS if x > layer), None))
            continue
        layer_t0 = time.time()
        tier_outputs = []
        represented = [t for t in TIERS if ids_for(assignment, layer, t, "13") or ids_for(assignment, layer, t, "2")]
        for tier in represented:
            update_progress(active_layer=layer, active_tier=tier)
            tier_outputs.append(build_tier_layer(assignment, layer, tier, command_sha))
        shard, source_keys = source_shard(layer, index)
        source_sha = sha256_file(shard)
        native = {
            "tier": "native_mxfp4", "source_shard": str(shard), "sha256": source_sha,
            "bytes": shard.stat().st_size, "checkpoint_index": str(CKPT_INDEX),
            "checkpoint_index_sha256": EXPECTED[CKPT_INDEX], "source_key_count": len(source_keys),
            "fused13_expert_ids": ids_for(assignment, layer, "native_mxfp4", "13"),
            "down_expert_ids": ids_for(assignment, layer, "native_mxfp4", "2"),
            "copy_policy": "verbatim resident checkpoint MXFP4 weight+scale bytes; not duplicated in shard intermediate",
        }
        replay_exact = sum(x["fp16_codebook_replay_exact"] for x in tier_outputs)
        replay_total = sum(x["fp16_codebook_replay_total"] for x in tier_outputs)
        frac = replay_exact / replay_total if replay_total else 1.0
        if frac != 1.0:
            raise RuntimeError(f"layer aggregate replay fail {layer} {frac}")
        proj = projections(assignment, layer)
        receipt = {
            "schema": "genesis-shard-layer-receipt-v1", "status": "PASS", "task": TASK,
            "host": HOST, "shard": SHARD, "layer": layer, "assignment_sha256": ASSIGNMENT_MAP_SHA256,
            "input_manifest_sha256": EXPECTED[INPUT_MANIFEST], "checkpoint_index_sha256": EXPECTED[CKPT_INDEX],
            "source_shard_sha256": source_sha, "shared_fix_receipt_sha256": EXPECTED[FIX_RECEIPT],
            "canonical_shared_builder_sha256": EXPECTED[BUILDER], "command_sha256": command_sha,
            "tier_projection_counts": proj["tier_projection_counts"], "projected_wire_bytes": proj["projected_wire_bytes"],
            "projected_intermediate_bytes": proj["projected_intermediate_bytes"], "vq_outputs": tier_outputs,
            "native_source": native, "output_bytes": sum(x["bytes"] for x in tier_outputs),
            "fp16_codebook_replay_exact": replay_exact, "fp16_codebook_replay_total": replay_total,
            "fp16_codebook_replay_exact_fraction": frac, "fp16_codebook_replay_first_mismatch": None,
            "local_fit_metrics": {x["tier"]: x["local_fit_metrics"] for x in tier_outputs},
            "finite_checks": {"source_weights": True, "codebooks": True, "metrics": True},
            "elapsed_seconds": time.time() - layer_t0, "completed_epoch": time.time(),
        }
        atomic_json(RECEIPTS / f"LAYER_{layer:03d}.json", receipt)
        stage_layer(layer, tier_outputs)
        update_progress(active_layer=next((x for x in LAYERS if x > layer), None))
        print(f"L{layer:03d} RECEIPT PASS outputs={len(tier_outputs)} bytes={receipt['output_bytes']} replay=1.0", flush=True)
    finalize(assignment, command_sha)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except BaseException as exc:
        try:
            set_status("FAILED", error=repr(exc), traceback=traceback.format_exc())
        except Exception:
            pass
        raise
