#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import torch

TASK = os.environ.get("BANANA_SMASHER_TASK_ID", "standalone")
MISSION = Path(
    os.environ.get(
        "BANANA_SMASHER_DEPTH_MISSION",
        str(Path.cwd() / ".banana-smasher-update"),
    )
).resolve()
EXPECTED_HOST = os.environ.get("BANANA_SMASHER_EXPECTED_HOST", socket.gethostname())
CLAIM = Path(
    os.environ.get("GENESIS_HOST_CLAIM", str(MISSION / "HOST_CLAIM.json"))
).resolve()
EXPECTED_CLAIM_SHA = os.environ.get("P1436_EXPECTED_CLAIM_SHA256", "")
HARNESS_PATH = MISSION / "harness/bench/arm_sweep.py"
STAGE = Path(
    os.environ.get("BANANA_SMASHER_DEPTH_STAGE", f"/dev/shm/{MISSION.name}")
)
RESULT = MISSION / "results/P1436_LAYER_GRAPH_FRESH43L_1024_UPDATE.json"
PROGRESS = MISSION / "receipts/P1436_LAYER_GRAPH_PROGRESS.json"
IO_GUARD = MISSION / "code/external_io_guard.py"
LAYERS = 43
SEED = 1436
MICROBATCH = 1
REFERENCE_LOSS = 0.000393650378100574
REFERENCE_THRESHOLD = 0.10
P1376B_PACKAGE_SHA = "61c26e2b9a39e4c3256142626dd5d8bc948e6ea1ac0b1c13242ea404a83fd208"
P1395_PACKAGE_SHA = "2a13b5f3dd124bda8cc30947ed70a67a55fefd9dae270eae86543c7fb4e80a83"
P1395_RESULT_SHA = "4eb2f547f3f3c08596beb85bb4de2659946858f909427012cc5fb6cd40092e6b"
INTEGRATION_PATCH_SHA = "eef9596cd746744897c3a68e7932c71582f8b5649d944facb318e85150323991"
AOT_SHA = "1f5a78ec847bb33a6d10fa3512e2b788fefe56368df58110e3fb256d0c80773a"
EXPECTED_UNIQUE_TILES = 512
EXPECTED_KMAJOR_CALLS = LAYERS * EXPECTED_UNIQUE_TILES
EXPECTED_VJP_GROUPS = LAYERS * 2
EXPECTED_VJP_BATCH_SIZE = 16
EXPECTED_VJP_BATCH_FLUSHES = EXPECTED_KMAJOR_CALLS // EXPECTED_VJP_BATCH_SIZE
EXPECTED_GRAPH_FORWARD_NODES = EXPECTED_VJP_GROUPS * 2
EXPECTED_GRAPH_BACKWARD_NODES = EXPECTED_VJP_GROUPS
EXPECTED_GRAPH_GROUPED_EXPERTS = EXPECTED_GRAPH_FORWARD_NODES * 256
MAX_PERSISTENT_TILE_BYTES = 13 * 1024**3
HISTORICAL_UPDATE_BASELINE_S = 890.0
REQUIRED_SPEEDUP = 10.0
FIRST_WINDOW_MIN_S = 90.0
FIRST_WINDOW_MAX_S = 200.0


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def io_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path(f"/proc/{os.getpid()}/io").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("rchar", "read_bytes"):
            values[key] = int(value.strip())
    return values


def atomic_json(path: Path, value: object) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(raw).hexdigest()


def mem_info() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable"):
            values[key] = int(rest.split()[0]) * 1024
    return values


def execute_with_external_io_guard(label: str, callback):
    """Measure target-process IO externally so the measurement cannot self-contaminate."""
    root = Path(f"/dev/shm/P1429_IO_GUARD_{os.getpid()}_{label}")
    paths = {name: root.with_suffix(f".{name}.json") for name in ("ready", "done", "result")}
    for path in paths.values():
        path.unlink(missing_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    helper = subprocess.Popen(
        [sys.executable, "-u", str(IO_GUARD), str(os.getpid()), str(paths["ready"]), str(paths["done"]), str(paths["result"])],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    deadline = time.monotonic() + 10.0
    while not paths["ready"].exists():
        if helper.poll() is not None:
            raise RuntimeError(f"external IO helper exited before READY rc={helper.returncode}")
        if time.monotonic() >= deadline:
            helper.kill()
            raise TimeoutError(paths["ready"])
        time.sleep(0.005)
    try:
        value = callback()
    finally:
        fd = os.open(paths["done"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, b"1")
        os.fsync(fd)
        os.close(fd)
    deadline = time.monotonic() + 10.0
    while not paths["result"].exists():
        if helper.poll() is not None and helper.returncode:
            raise RuntimeError(f"external IO helper failed rc={helper.returncode}")
        if time.monotonic() >= deadline:
            helper.kill()
            raise TimeoutError(paths["result"])
        time.sleep(0.005)
    guard = json.loads(paths["result"].read_text())
    helper_rc = helper.wait(timeout=5.0)
    if helper_rc != 0:
        raise RuntimeError(f"external IO helper failed rc={helper_rc}")
    for path in paths.values():
        path.unlink(missing_ok=True)
    return value, guard


def progress(event: str, **fields: object) -> None:
    row = {
        "schema": "p1436-layer-graph-43l-update-progress-v1",
        "task_id": TASK,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "sid": os.getsid(0),
        "event": event,
        "updated_unix": time.time(),
        **fields,
    }
    atomic_json(PROGRESS, row)
    log = MISSION / "logs/P1436_LAYER_GRAPH43L_PROGRESS.jsonl"
    with log.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(row, sort_keys=True), flush=True)


def verify_claim() -> str:
    raw = CLAIM.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw)
    exclusive = bool(
        doc.get("owner") == TASK
        and doc.get("task_id") == TASK
        and doc.get("mission_root") == str(MISSION)
    )
    cohabitant = any(
        row.get("task_id") == TASK and row.get("mission") == str(MISSION)
        for row in doc.get("cohabitants", [])
    )
    if not (
        digest == EXPECTED_CLAIM_SHA
        and doc.get("state") == "CLAIMED"
        and doc.get("host") == EXPECTED_HOST
        and (exclusive or cohabitant)
    ):
        raise RuntimeError(f"claim drift digest={digest} doc={doc}")
    return digest


def load_harness():
    spec = importlib.util.spec_from_file_location("p1404_arm_sweep", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.LAYERS = LAYERS
    return module


def install_committed_bounded_fwht() -> None:
    import f521_repair_overlay as f521
    from banana_smasher.fwht import bounded_fwht

    original = f521.CorrectQtipDecoder.__init__
    if getattr(original, "_banana_smasher_bounded_fwht", False):
        return

    def bounded_init(self, device):
        original(self, device)
        self.fwht = lambda value: bounded_fwht(
            value, inplace=not bool(value.requires_grad)
        )

    bounded_init._banana_smasher_bounded_fwht = True
    f521.CorrectQtipDecoder.__init__ = bounded_init


class MemorySampler:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.rows: list[dict[str, int | float]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            self.rows.append({"unix": time.time(), **mem_info()})
            self.stop.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        self.rows.append({"unix": time.time(), **mem_info()})

    def summary(self) -> dict[str, int]:
        total = max(int(row["MemTotal"]) for row in self.rows)
        minimum = min(int(row["MemAvailable"]) for row in self.rows)
        return {
            "mem_total_bytes": total,
            "mem_available_min_bytes": minimum,
            "peak_uma_used_bytes": total - minimum,
            "samples": len(self.rows),
        }


def instrument_planes(layers) -> dict[str, int]:
    metrics = {
        "payload_calls": 0,
        "integer_tensors_checked": 0,
        "requires_grad_count": 0,
        "grad_present_count": 0,
    }
    for module in layers:
        original = module._payloads_for

        def wrapped(projection, hit_ids, _original=original):
            rows = _original(projection, hit_ids)
            metrics["payload_calls"] += 1
            for row in rows.values():
                for value in row:
                    if torch.is_tensor(value) and not value.dtype.is_floating_point:
                        metrics["integer_tensors_checked"] += 1
                        metrics["requires_grad_count"] += int(value.requires_grad)
                        metrics["grad_present_count"] += int(value.grad is not None)
            return rows

        module._payloads_for = wrapped
    return metrics


def preload_all_planes(layers) -> tuple[dict[str, int], dict[tuple[int, str], object]]:
    """Load once, retain, and bind every decoded payload object through timing."""
    tensors: dict[int, torch.Tensor] = {}
    runtime_slab: dict[tuple[int, str], object] = {}
    payload_rows = 0
    shared_projection_slab: dict[str, dict[int, object]] | None = None
    for layer_index, module in enumerate(layers):
        projection_slab: dict[str, dict[int, object]] = {}
        original_loader = module._payloads_for
        if shared_projection_slab is None:
            for projection in ("13", "2"):
                rows = original_loader(projection, list(range(256)))
                if set(rows) != set(range(256)):
                    raise RuntimeError(f"incomplete runtime slab layer={layer_index} projection={projection}")
                projection_slab[projection] = rows
                payload_rows += len(rows)
                for row in rows.values():
                    for value in row:
                        if torch.is_tensor(value):
                            tensors[value.data_ptr()] = value
            shared_projection_slab = projection_slab
        else:
            projection_slab = shared_projection_slab
        for projection in ("13", "2"):
            runtime_slab[(layer_index, projection)] = projection_slab[projection]

        def slabbed_payloads(projection, hit_ids, _slab=projection_slab):
            requested = list(dict.fromkeys(map(int, hit_ids)))
            return {expert: _slab[projection][expert] for expert in requested}

        module._payloads_for = slabbed_payloads
    torch.cuda.synchronize()
    return {
        "layers": len(layers),
        "payload_rows": payload_rows,
        "runtime_slab_groups": len(runtime_slab),
        "unique_cuda_tensors": len(tensors),
        "resident_plane_bytes": sum(value.numel() * value.element_size() for value in tensors.values()),
    }, runtime_slab


def grad_profile(layers, windows: torch.Tensor) -> dict[str, Any]:
    rows = []
    vectors = []
    for layer_index, module in enumerate(layers):
        layer_vectors = []
        for name, parameter in module.codebooks.items():
            grad = parameter.grad
            if grad is None:
                continue
            value = grad.detach().float()
            layer_vectors.append(value.reshape(-1))
            vectors.append(value.reshape(-1))
            rows.append({
                "layer": layer_index,
                "name": name,
                "l2": float(value.norm()),
                "max_abs": float(value.abs().max()),
                "finite": bool(torch.isfinite(value).all()),
                "nonzero": bool(torch.count_nonzero(value)),
            })
    by_layer = []
    for layer_index in range(LAYERS):
        selected = [row for row in rows if row["layer"] == layer_index]
        by_layer.append({
            "layer": layer_index,
            "parameter_grads": len(selected),
            "all_finite": bool(selected) and all(row["finite"] for row in selected),
            "all_nonzero": bool(selected) and all(row["nonzero"] for row in selected),
            "l2_quadrature": math.sqrt(sum(float(row["l2"]) ** 2 for row in selected)),
        })
    vector = torch.cat(vectors) if vectors else torch.empty(0, device="cuda")
    input_grad = windows.grad
    return {
        "parameter_grad_count": len(rows),
        "aggregate_l2": float(vector.norm()) if vector.numel() else 0.0,
        "aggregate_finite": bool(vector.numel() and torch.isfinite(vector).all()),
        "aggregate_nonzero": bool(vector.numel() and torch.count_nonzero(vector)),
        "per_layer": by_layer,
        "rows": rows,
        "input_grad_present": input_grad is not None,
        "input_grad_l2": float(input_grad.float().norm()) if input_grad is not None else 0.0,
        "input_grad_finite": bool(input_grad is not None and torch.isfinite(input_grad).all()),
        "input_grad_nonzero": bool(input_grad is not None and torch.count_nonzero(input_grad)),
    }


def execute_step_phased(harness, layers, optimizer, tensors, microbatch: int) -> dict[str, Any]:
    """Execute exactly one update with synchronized per-phase wall receipts."""
    optimizer.zero_grad(set_to_none=True)
    batch = int(tensors["windows"].shape[0])
    if batch != 1 or int(microbatch) != 1:
        raise RuntimeError(f"P1434 exact shape drift batch={batch} microbatch={microbatch}")
    torch.cuda.synchronize()
    total_started = time.perf_counter()
    forward_started = time.perf_counter()
    output = harness.layer_subset_forward(
        layers,
        tensors["windows"],
        tensors["top_k_index"],
        tensors["top_k_weights"],
    )
    loss = output.float().square().mean()
    torch.cuda.synchronize()
    forward_s = time.perf_counter() - forward_started
    backward_started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_s = time.perf_counter() - backward_started
    optimizer_started = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    optimizer_s = time.perf_counter() - optimizer_started
    total_s = time.perf_counter() - total_started
    value = float(loss.detach())
    del output, loss
    return {
        "wall_s": total_s,
        "cuda_stream_span_s": total_s,
        "loss": value,
        "phase_wall_s": {
            "forward": forward_s,
            "backward": backward_s,
            "optimizer": optimizer_s,
            "total": total_s,
        },
    }


def parameter_sha256(layers) -> str:
    digest = hashlib.sha256()
    for parameter in layers.parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_arm(
    harness,
    stage_receipt: dict[str, Any],
    full_tensors: dict[str, torch.Tensor],
    *,
    candidate: bool,
    use_layer_graph: bool,
) -> dict[str, Any]:
    os.environ["GENESIS_REPAIR_KEEP_PLANES_RESIDENT"] = "1"
    os.environ["GENESIS_REPAIR_EXPERT_RESIDENT_SCOPE"] = "1"
    os.environ["GENESIS_REPAIR_EVICT"] = "0"
    os.environ["GENESIS_REPAIR_KMAJOR_WINDOWED"] = "1"
    os.environ["GENESIS_REPAIR_KMAJOR_10X"] = "1" if candidate else "0"
    os.environ["GENESIS_REPAIR_P1363_M16"] = "0"
    os.environ["GENESIS_REPAIR_MEM_FLOOR_BYTES"] = str(16 * 1024**3)
    os.environ["GENESIS_REPAIR_RELIEF_MARGIN_BYTES"] = str(8 * 1024**3)
    os.environ["GENESIS_REPAIR_ROOT"] = str(MISSION)
    os.environ["GENESIS_TASK_ID"] = TASK
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    harness.physical_surface._RESIDENT_MODULES.clear()
    layers = harness.build_layers(STAGE, torch.device("cuda"), SEED)
    if len(layers) != LAYERS:
        raise RuntimeError(f"layer count drift {len(layers)} != {LAYERS}")
    plane_preload, plane_runtime_slab = preload_all_planes(layers)
    plane_metrics = instrument_planes(layers)
    tensors = {
        "windows": full_tensors["windows"][:1].detach().clone().requires_grad_(True),
        "top_k_index": full_tensors["top_k_index"][:1],
        "top_k_weights": full_tensors["top_k_weights"][:1],
    }

    # Compile the production M=24 forward/backward/optimizer kernels on one
    # restored layer outside the measured interval. This is compilation only:
    # the main 43-layer optimizer is not created until after the clone state is
    # restored and the disposable warm optimizer is deleted.
    warm_snapshots = [parameter.detach().clone() for parameter in layers[0].parameters()]
    if candidate:
        layers[0].prepare_kmajor_identity_manifest()
    warm_layers = torch.nn.ModuleList([layers[0]])
    warm_optimizer = harness.make_optimizer(warm_layers, "fused_grouped_adam")
    harness.prime_optimizer(warm_optimizer, warm_layers)
    harness.execute_step(warm_layers, warm_optimizer, tensors, MICROBATCH)
    del warm_optimizer, warm_layers
    with torch.no_grad():
        for parameter, snapshot in zip(layers[0].parameters(), warm_snapshots):
            parameter.copy_(snapshot)
    del warm_snapshots
    for parameter in layers.parameters():
        parameter.grad = None
    tensors["windows"].grad = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    optimizer = harness.make_optimizer(layers, "fused_grouped_adam")
    harness.prime_optimizer(optimizer, layers)
    prefill = None
    if candidate:
        import kmajor_autograd
        kmajor_autograd.reset_kmajor_sentinel(clear_cache=True)
        manifests = [module.prepare_kmajor_identity_manifest() for module in layers]
        reference_manifest = manifests[0]
        if len(reference_manifest) != EXPECTED_UNIQUE_TILES:
            raise RuntimeError(
                f"P1404 logical tile count drift {len(reference_manifest)} != {EXPECTED_UNIQUE_TILES}"
            )
        if any(manifest != reference_manifest for manifest in manifests[1:]):
            raise RuntimeError("P1404 bounded persistent cache cannot cover divergent layer identities")
        visited = layers[0].prefill_kmajor_tiles()
        torch.cuda.synchronize()
        graph_prepare = None
        if use_layer_graph:
            from .kmajor_graph import prepare_layer_graph_vjp

            graph_prepare = prepare_layer_graph_vjp(layers, kmajor_autograd)
            torch.cuda.synchronize()
        sealed = kmajor_autograd.seal_kmajor_update(
            expected_entries=EXPECTED_UNIQUE_TILES,
            max_bytes=MAX_PERSISTENT_TILE_BYTES,
        )
        prefill = {
            **visited,
            **sealed,
            "outside_timed_window": True,
            "unique_layer_manifests": 1,
            "layers_covered": LAYERS,
            "sentinel": kmajor_autograd.kmajor_sentinel(),
        }
        if graph_prepare is not None:
            prefill["layer_graph"] = graph_prepare
        progress("CANDIDATE_PREFILL_DONE", **{k: v for k, v in prefill.items() if k != "sentinel"})

        if use_layer_graph:
            from .kmajor_graph import reset_layer_graph_vjp

            # Compile the exact balanced layer/projection graph after its two
            # dense slabs exist but before the measured update. Keep the
            # official prefill counters/cache intact; reset only graph runtime
            # counts afterward.
            graph_warm_snapshots = [
                parameter.detach().clone() for parameter in layers[0].parameters()
            ]
            graph_warm_layers = torch.nn.ModuleList([layers[0]])
            graph_warm_optimizer = harness.make_optimizer(
                graph_warm_layers, "fused_grouped_adam"
            )
            harness.prime_optimizer(graph_warm_optimizer, graph_warm_layers)
            harness.execute_step(
                graph_warm_layers, graph_warm_optimizer, tensors, MICROBATCH
            )
            del graph_warm_optimizer, graph_warm_layers
            with torch.no_grad():
                for parameter, snapshot in zip(
                    layers[0].parameters(), graph_warm_snapshots
                ):
                    parameter.copy_(snapshot)
            del graph_warm_snapshots
            for parameter in layers.parameters():
                parameter.grad = None
            tensors["windows"].grad = None
            reset_layer_graph_vjp()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    tensors["windows"].grad = None
    parameter_before_sha256 = parameter_sha256(layers)
    optimizer_state_entries_before = len(optimizer.state)
    verify_claim()
    progress(
        "CANDIDATE_UPDATE_START" if candidate else "BASE_UPDATE_START",
        layers=LAYERS,
        microbatch=MICROBATCH,
        mem_available_bytes=mem_info()["MemAvailable"],
    )
    before = mem_info()
    # Full slab closure is already proven. The old per-chunk guard read
    # /proc/meminfo thousands of times inside forward, so move monitoring to
    # the independent no-CUDA helper for the measured interval.
    for module in layers:
        module._resource_floor_relief = lambda _where, _active: None
    torch.cuda.reset_peak_memory_stats()
    step, io_guard = execute_with_external_io_guard(
        "candidate" if candidate else "base",
        lambda: execute_step_phased(harness, layers, optimizer, tensors, MICROBATCH),
    )
    parameter_after_sha256 = parameter_sha256(layers)
    after = mem_info()
    grads = grad_profile(layers, tensors["windows"])
    sentinel = None
    if candidate:
        import kmajor_autograd
        sentinel = kmajor_autograd.kmajor_sentinel()
        release = kmajor_autograd.end_kmajor_update()
        sentinel["terminal_cache_release"] = release
    packed = {
        **plane_metrics,
        "pass": (
            plane_metrics["integer_tensors_checked"] > 0
            and plane_metrics["requires_grad_count"] == 0
            and plane_metrics["grad_present_count"] == 0
        ),
    }
    row = {
        "mode": "kmajor_windowed_autograd" if candidate else "current_fused_vq_windowed_control",
        "step": step,
        "optimizer_update": {
            "parameter_sha256_before": parameter_before_sha256,
            "parameter_sha256_after": parameter_after_sha256,
            "parameter_changed": parameter_before_sha256 != parameter_after_sha256,
            "state_entries_before": optimizer_state_entries_before,
            "state_entries_after": len(optimizer.state),
            "checkpoint_loaded": False,
            "complete_steps": 1,
        },
        "finite_loss": math.isfinite(float(step["loss"])),
        "gradients": grads,
        "packed_planes": packed,
        "sentinel": sentinel,
        "prefill": prefill,
        "window_policy": {
            "layers_per_window": 1,
            "keep_planes_resident_within_active_layer": True,
            "persistent_logical_tiles_across_update": bool(candidate),
            "prefill_outside_timed_window": bool(candidate),
            "evict_derived_tiles_at_layer_boundary": not candidate,
            "backward_tile_rematerialization": False if candidate else None,
            "microbatch": MICROBATCH,
        },
        "memory": {
            "mem_total_bytes": before["MemTotal"],
            "peak_uma_used_bytes": before["MemTotal"] - min(before["MemAvailable"], after["MemAvailable"]),
            "mem_available_before_bytes": before["MemAvailable"],
            "mem_available_after_bytes": after["MemAvailable"],
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "external_memory_map": {
                "before": io_guard["before"]["smaps_rollup"],
                "after": io_guard["after"]["smaps_rollup"],
            },
        },
        "full_residency": {
            "activation_windows_preloaded": True,
            "teacher_references_preloaded": True,
            "plane_layers_preloaded": plane_preload["layers"],
            "resident_plane_bytes": plane_preload["resident_plane_bytes"],
            "plane_payload_rows": plane_preload["payload_rows"],
            "unique_plane_tensors": plane_preload["unique_cuda_tensors"],
            "kmajor_prefilled": bool(candidate),
        },
        "timed_io": {
            "measurement": "external no-CUDA helper",
            "helper_pid": io_guard["helper_pid"],
            "rchar_before": io_guard["before"]["io"]["rchar"],
            "rchar_after": io_guard["after"]["io"]["rchar"],
            "rchar_delta": io_guard["deltas"]["rchar"],
            "read_bytes_before": io_guard["before"]["io"]["read_bytes"],
            "read_bytes_after": io_guard["after"]["io"]["read_bytes"],
            "read_bytes_delta": io_guard["deltas"]["read_bytes"],
        },
        "stage_contract": stage_receipt["production_contract"],
    }
    progress(
        "CANDIDATE_UPDATE_DONE" if candidate else "BASE_UPDATE_DONE",
        wall_s=step["wall_s"],
        loss=step["loss"],
        peak_uma_used_bytes=row["memory"]["peak_uma_used_bytes"],
    )
    del optimizer, layers, tensors, plane_runtime_slab
    harness.physical_surface._RESIDENT_MODULES.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return row


def main(
    *,
    legacy_backward: bool = False,
    result_path: str | Path | None = None,
) -> int:
    if socket.gethostname() != EXPECTED_HOST:
        raise RuntimeError(socket.gethostname())
    selected_result = RESULT if result_path is None else Path(result_path)
    claim_sha = verify_claim()
    for directory in (MISSION / "results", MISSION / "receipts", MISSION / "logs", MISSION / "pids"):
        directory.mkdir(parents=True, exist_ok=True)
    (MISSION / "pids/P1436_LAYER_GRAPH43L_RUNNER.pid").write_text(f"{os.getpid()}\n")
    progress("START", claim_sha256=claim_sha)
    lineage_files = {
        "p1376b_package": (
            MISSION / "upstream/P1376B_PERSISTENT_DENSE_TC_TRANSFERABLE.tar.gz",
            P1376B_PACKAGE_SHA,
        ),
        "p1395_package": (
            MISSION / "upstream/P1395_S4_BOUNDARY_ADOPTION_PACKAGE_NO_TAKE.tar.gz",
            P1395_PACKAGE_SHA,
        ),
        "p1395_result": (
            MISSION / "upstream/P1395_43L_WINDOWED_AUTOGRAD_AB.json",
            P1395_RESULT_SHA,
        ),
        "aot": (
            MISSION / "upstream/P1378_KMAJOR_AOT_C.cpython-312-aarch64-linux-gnu.so",
            AOT_SHA,
        ),
    }
    lineage = {}
    for name, (path, expected) in lineage_files.items():
        observed = sha(path)
        if observed != expected:
            raise RuntimeError(f"lineage drift {name}: {observed} != {expected}")
        lineage[name] = {"path": str(path), "sha256": observed}
    progress("LINEAGE_VERIFIED", files=len(lineage), lineage=lineage)
    harness = load_harness()
    install_committed_bounded_fwht()
    import kmajor_autograd
    from .kmajor_batch import install_batched_kmajor_vjp

    batched_vjp_install = install_batched_kmajor_vjp(kmajor_autograd)
    graph_vjp_install = None
    if not legacy_backward:
        from .kmajor_graph import install_layer_graph_vjp

        graph_vjp_install = install_layer_graph_vjp(
            harness.physical_surface, kmajor_autograd
        )
    progress(
        "KMAJOR_VJP_PATHS_INSTALLED",
        batched=batched_vjp_install,
        layer_graph=graph_vjp_install,
    )
    if int(harness.PRODUCTION_LAYERS) != 43:
        raise RuntimeError("production layer contract drift")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    try:
        stage_receipt = harness.stage_synthetic(STAGE, SEED)
        package_layers = sorted((STAGE / "wire/physical_package").glob("layer_*"))
        if len(package_layers) != LAYERS:
            raise RuntimeError(f"staged {len(package_layers)} layers")
        full_tensors = harness.load_windows(STAGE, torch.device("cuda"))
        fresh_input_identity = {
            "fresh_seed": SEED,
            "warm_start_used": False,
            "windows_sha256": tensor_sha(full_tensors["windows"]),
            "top_k_index_sha256": tensor_sha(full_tensors["top_k_index"]),
            "top_k_weights_sha256": tensor_sha(full_tensors["top_k_weights"]),
            "identical_base_candidate_inputs": True,
            "all_inputs_loaded_before_timing": True,
        }
        progress(
            "FRESH43L_INPUTS_PRELOADED",
            layer_count=len(package_layers),
            layer_subset=stage_receipt["production_contract"]["layer_subset"],
            full_model_layers=stage_receipt["production_contract"]["full_model_layers"],
            input_identity=fresh_input_identity,
        )
        candidate = run_arm(
            harness,
            stage_receipt,
            full_tensors,
            candidate=True,
            use_layer_graph=not legacy_backward,
        )
        candidate_loss = float(candidate["step"]["loss"])
        trajectory = {
            "sealed_u007_family_reference_loss": REFERENCE_LOSS,
            "candidate_loss": candidate_loss,
            "candidate_relative_to_reference": abs(candidate_loss - REFERENCE_LOSS) / max(abs(candidate_loss), abs(REFERENCE_LOSS), 1e-12),
            "in_family_threshold": REFERENCE_THRESHOLD,
            "numeric_or_bit_exactness_computed": False,
        }
        trajectory["in_family_pass"] = bool(
            trajectory["candidate_relative_to_reference"] <= REFERENCE_THRESHOLD
        )
        grads = candidate["gradients"]
        sentinel = candidate["sentinel"]
        batched_vjp = sentinel["batched_vjp"] if sentinel is not None else None
        layer_graph_vjp = (
            sentinel.get("layer_graph_vjp") if sentinel is not None else None
        )
        if legacy_backward:
            vjp_path_pass = bool(
                batched_vjp is not None
                and batched_vjp["forward_calls"] == 2 * EXPECTED_KMAJOR_CALLS
                and batched_vjp["backward_calls"] == EXPECTED_KMAJOR_CALLS
                and batched_vjp["unique_groups"] == EXPECTED_VJP_GROUPS
                and batched_vjp["batch_size"] == EXPECTED_VJP_BATCH_SIZE
                and batched_vjp["batch_flushes"] == EXPECTED_VJP_BATCH_FLUSHES
                and batched_vjp["active_groups"] == 0
                and layer_graph_vjp is None
            )
        else:
            vjp_path_pass = bool(
                batched_vjp is not None
                and batched_vjp["backward_calls"] == 0
                and batched_vjp["active_groups"] == 0
                and layer_graph_vjp is not None
                and layer_graph_vjp["forward_calls"]
                == EXPECTED_GRAPH_FORWARD_NODES
                and layer_graph_vjp["backward_calls"]
                == EXPECTED_GRAPH_BACKWARD_NODES
                and layer_graph_vjp["grouped_experts"]
                == EXPECTED_GRAPH_GROUPED_EXPERTS
                and layer_graph_vjp["max_nodes_per_projection"] == 1
                and layer_graph_vjp["grad_weight_bmm_launches"]
                == EXPECTED_GRAPH_BACKWARD_NODES
                and layer_graph_vjp["reduction_kernel_launches"]
                == EXPECTED_GRAPH_BACKWARD_NODES
            )
        timed_io_pass = abs(int(candidate["timed_io"]["rchar_delta"])) <= 4096 and int(candidate["timed_io"]["read_bytes_delta"]) == 0
        memory_pass = bool(
            int(candidate["memory"]["mem_available_after_bytes"]) >= 4 * 1024**3
            and int(candidate["memory"]["cuda_peak_reserved_bytes"]) < 60 * 1024**3
        )
        pass_gate = bool(
            candidate["finite_loss"]
            and grads["parameter_grad_count"] == 2 * LAYERS
            and grads["aggregate_finite"]
            and grads["aggregate_nonzero"]
            and all(row["all_finite"] and row["all_nonzero"] for row in grads["per_layer"])
            and grads["input_grad_finite"]
            and grads["input_grad_nonzero"]
            and candidate["packed_planes"]["pass"]
            and candidate["optimizer_update"]["parameter_changed"]
            and candidate["optimizer_update"]["complete_steps"] == 1
            and sentinel is not None
            and sentinel["on_path_pass"]
            and sentinel["materializations"] == EXPECTED_UNIQUE_TILES
            and sentinel["prefill_materializations"] == EXPECTED_UNIQUE_TILES
            and sentinel["backward_rematerializations"] == 0
            and sentinel["timed_cache_misses"] == 0
            and sentinel["window_boundary_evicted_entries"] == 0
            and sentinel["cache_hits"] >= sentinel["bmm_launches"]
            and vjp_path_pass
            and timed_io_pass
            and memory_pass
            and trajectory["in_family_pass"]
        )
        result = {
            "schema": (
                "p1436-legacy-batched-fused-cuda-43l-1024-depth-proof-v1"
                if legacy_backward
                else "p1436-layer-graph-fused-cuda-43l-1024-depth-proof-v1"
            ),
            "status": "PASS_FRESH_43L_1024_DEPTH" if pass_gate else "FAIL_FRESH_43L_1024_DEPTH",
            "task_id": TASK,
            "host": socket.gethostname(),
            "scope": "INTERMEDIATE depth-scaling receipt: fresh 43/43-layer x one logical 1024-token forward/backward/optimizer update; never a final 10x claim",
            "public_api_command": os.environ.get("BANANA_SMASHER_PUBLIC_API_COMMAND"),
            "backward_mode": "legacy_batched" if legacy_backward else "layer_graph",
            "fresh_input_identity": fresh_input_identity,
            "candidate": candidate,
            "measurement": {
                "candidate_wall_s": candidate["step"]["wall_s"],
                "timed_io_pass": timed_io_pass,
                "memory_pass": memory_pass,
            },
            "trajectory": trajectory,
            "lineage": {
                "claim_sha256": claim_sha,
                "p1376b_persistent_package_sha256": P1376B_PACKAGE_SHA,
                "p1395_boundary_package_sha256": P1395_PACKAGE_SHA,
                "p1395_result_sha256": P1395_RESULT_SHA,
                "integration_patch_sha256": INTEGRATION_PATCH_SHA,
                "aot_sha256": AOT_SHA,
                "surface_sha256": sha(MISSION / "harness/source/genesis_physical_surface.py"),
                "helper_sha256": sha(MISSION / "harness/source/kmajor_autograd.py"),
                "harness_sha256": sha(HARNESS_PATH),
                "runner_sha256": sha(Path(__file__)),
                "batched_kmajor_vjp_sha256": sha(Path(__file__).with_name("kmajor_batch.py")),
                "fused_kmajor_vjp_sha256": sha(Path(__file__).with_name("kmajor_fused.py")),
                "layer_graph_kmajor_vjp_sha256": sha(Path(__file__).with_name("kmajor_graph.py")),
                "external_io_guard_sha256": sha(IO_GUARD),
                "bounded_fwht_source_commit": "9ff07a6be0f6a8080b1c64043e0c674c2d7096e4",
            },
            "completed_unix": time.time(),
        }
        result_sha = atomic_json(selected_result, result)
        progress("COMPLETE", status=result["status"], result_sha256=result_sha)
        print(json.dumps({"result": str(selected_result), "result_sha256": result_sha, "status": result["status"]}, indent=2), flush=True)
        return 0 if pass_gate else 2
    finally:
        shutil.rmtree(STAGE, ignore_errors=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
