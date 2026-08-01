from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BASELINE_WINDOW_SECONDS = 890.0
PRODUCTION_LAYERS = 43
MINIMUM_MEM_AVAILABLE_BYTES = 4 * 1024**3
ONE_LAYER_TARGET_MEM_AVAILABLE_BYTES = 50 * 1024**3
ONE_LAYER_MAX_DEVICE_USED_BYTES = 60 * 1024**3
FULL_DEPTH_MINIMUM_MEM_AVAILABLE_BYTES = 4 * 1024**3
FULL_DEPTH_MAX_DEVICE_USED_BYTES = 112 * 1024**3
FULL_DEPTH_DEVICE_TARGET_BYTES = 60 * 1024**3
UPDATE_RUNTIME_COMPONENTS = (
    "base_binrepair_e2e.py",
    "f521_repair_overlay.py",
    "banana_smasher_physical_surface.py",
    "kmajor_autograd.py",
    "lp4_train.py",
)


@contextmanager
def _backend_environment(backend: str):
    """Select one update backend without leaking process-global runtime flags."""
    if backend not in {"accelerated", "reference"}:
        raise ValueError(f"unsupported update backend {backend!r}")
    values = {
        "GENESIS_REPAIR_KMAJOR_10X": "1" if backend == "accelerated" else "0",
        "GENESIS_REPAIR_KMAJOR_WINDOWED": "1" if backend == "accelerated" else "0",
    }
    managed = set(values) | {
        "GENESIS_REPAIR_KEEP_PLANES_RESIDENT",
        "GENESIS_REPAIR_PIN_PLANES",
        "GENESIS_REPAIR_EVICT",
        "GENESIS_REPAIR_CHECKPOINT",
        "GENESIS_REPAIR_KMAJOR_CACHE_MAX_BYTES",
        "GENESIS_REPAIR_KMAJOR_PREFETCH_EXPERTS",
    }
    previous = {name: os.environ.get(name) for name in managed}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _validate_runtime_components(runtime_root: str | Path) -> None:
    root = Path(runtime_root).resolve()
    missing = [name for name in UPDATE_RUNTIME_COMPONENTS if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            "banana-smasher update runtime is incomplete; missing "
            + ", ".join(missing)
            + "; install the update extra and provide a complete versioned runtime root."
        )


def _runtime_memory_acceptance(
    layers: int, minimum_mem_available_bytes: int, maximum_device_used_bytes: int
) -> dict[str, Any]:
    """Apply the corrected hard guard while retaining 60 GiB as a flag-only target."""
    full_depth = int(layers) == PRODUCTION_LAYERS
    minimum_required = (
        FULL_DEPTH_MINIMUM_MEM_AVAILABLE_BYTES
        if full_depth
        else MINIMUM_MEM_AVAILABLE_BYTES
    )
    maximum_allowed = (
        FULL_DEPTH_MAX_DEVICE_USED_BYTES
        if full_depth
        else ONE_LAYER_MAX_DEVICE_USED_BYTES
    )
    minimum_pass = int(minimum_mem_available_bytes) >= minimum_required
    maximum_pass = (
        int(maximum_device_used_bytes) <= maximum_allowed
        if full_depth
        else int(maximum_device_used_bytes) < maximum_allowed
    )
    target_delta = int(maximum_device_used_bytes) - FULL_DEPTH_DEVICE_TARGET_BYTES
    return {
        "guard_policy": (
            "full-depth-hard-4gib-os-law-floor-and-112gib-device-ceiling"
            if full_depth
            else "legacy-one-layer-memory-envelope"
        ),
        "minimum_mem_available_hard_floor_bytes": minimum_required,
        "minimum_mem_available_hard_floor_pass": minimum_pass,
        "maximum_device_used_hard_ceiling_bytes": maximum_allowed,
        "maximum_device_used_hard_ceiling_pass": maximum_pass,
        "device_used_target_bytes": FULL_DEPTH_DEVICE_TARGET_BYTES,
        "device_used_target_pass": target_delta < 0,
        "device_used_target_delta_bytes": target_delta,
        "device_used_target_flag": (
            "PASS_BELOW_60_GIB_TARGET"
            if target_delta < 0
            else "FLAG_AT_OR_ABOVE_60_GIB_TARGET"
        ),
        "device_used_target_is_flag_only": full_depth,
        "hard_pass": minimum_pass and maximum_pass,
    }


def _resident_prefill_policy(layers: int, accumulation_segments: int) -> str:
    if int(layers) == PRODUCTION_LAYERS and int(accumulation_segments) == 8:
        return "sealed-eight-segment-full-depth"
    if int(layers) == 1:
        return "manual-one-layer"
    return "layer-window-eviction"


def _runtime_memory_policy(layers: int, accumulation_segments: int) -> dict[str, str]:
    resident = _resident_prefill_policy(layers, accumulation_segments) != (
        "layer-window-eviction"
    )
    return {
        "keep_planes_resident": "1" if resident else "0",
        "pin_planes": "1" if resident else "0",
        "evict": "0" if resident else "1",
        "checkpoint": "0" if int(layers) == 1 else "1",
    }


def _seal_prefilled_planes(surface: Any, policy: str) -> dict[str, Any]:
    if policy == "sealed-eight-segment-full-depth":
        return surface.seal_resident_planes()
    if policy != "manual-one-layer":
        raise RuntimeError(f"resident-plane seal is invalid for policy {policy!r}")
    inventory = surface.resident_plane_inventory()
    if int(inventory["layers"]) != 1 or int(inventory["entries"]) <= 0:
        raise RuntimeError(f"one-layer plane preload failed: {inventory}")
    surface._RESIDENT_PLANES_PREFILLING = False
    surface._RESIDENT_PLANES_SEALED = True
    return inventory


def _begin_timed_segment(surface: Any, policy: str, segment_index: int) -> Any:
    if policy == "sealed-eight-segment-full-depth":
        return surface.begin_kmajor_timed_segment(segment_index)
    return None


def _end_timed_segment(surface: Any, policy: str, segment_index: int) -> Any:
    if policy == "sealed-eight-segment-full-depth":
        return surface.end_kmajor_timed_segment(segment_index)
    return None


def _logical_segment_bounds(
    physical_tokens: int, accumulation_segments: int
) -> list[tuple[int, int]]:
    physical_tokens = int(physical_tokens)
    accumulation_segments = int(accumulation_segments)
    if not 1 <= physical_tokens <= 1024:
        raise ValueError(f"physical_tokens must be in [1,1024], got {physical_tokens}")
    if not 1 <= accumulation_segments <= 8:
        raise ValueError(
            f"accumulation_segments must be in [1,8], got {accumulation_segments}"
        )
    return [
        (index * physical_tokens, (index + 1) * physical_tokens)
        for index in range(accumulation_segments)
    ]


def _set_logical_training_extent(training_module: Any, logical_items: int) -> int:
    """Expand the legacy 1024-token loader to this exact logical window."""
    previous = int(training_module.T_TRAIN)
    logical_items = int(logical_items)
    if logical_items <= 0:
        raise ValueError(f"logical_items must be positive, got {logical_items}")
    training_module.T_TRAIN = logical_items
    return previous


def _logical_source_plan(
    corpus: list[dict[str, Any]], source_windows: Sequence[int], logical_items: int
) -> list[tuple[int, int]]:
    """Take an exact logical extent from explicit aligned corpus/teacher windows."""
    windows = tuple(int(window) for window in source_windows)
    if not windows:
        raise ValueError("source_windows must not be empty")
    if len(set(windows)) != len(windows):
        raise ValueError(f"source_windows must be unique, got {windows}")
    remaining = int(logical_items)
    if remaining <= 0:
        raise ValueError(f"logical_items must be positive, got {logical_items}")
    plan: list[tuple[int, int]] = []
    for position, source_window in enumerate(windows):
        if not 0 <= source_window < len(corpus):
            raise RuntimeError(
                f"source window {source_window} is outside corpus range [0,{len(corpus)})"
            )
        if remaining <= 0:
            raise RuntimeError(
                f"source windows {windows[position:]} are unused after exact logical "
                f"extent {logical_items}"
            )
        row = corpus[source_window]
        available = min(int(row["real_len"]), len(row["token_ids"]))
        take = min(remaining, available)
        if take > 0:
            plan.append((source_window, take))
            remaining -= take
    if remaining > 0:
        raise RuntimeError(
            f"explicit source windows provide {logical_items - remaining} tokens; "
            f"needs exact logical extent {logical_items}"
        )
    return plan


def _progress(phase: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"phase": phase, "unix": time.time(), **fields},
            sort_keys=True,
        ),
        flush=True,
    )


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _proc_identity(pid: int | None = None) -> dict[str, Any]:
    pid = os.getpid() if pid is None else int(pid)
    root = Path(f"/proc/{pid}")
    stat = (root / "stat").read_text()
    close = stat.rfind(")")
    if close < 0:
        raise RuntimeError(f"malformed process stat for {pid}")
    tail = stat[close + 2 :].split()
    return {
        "pid": pid,
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
        "argv": [
            part.decode(errors="replace")
            for part in (root / "cmdline").read_bytes().split(b"\0")
            if part
        ],
    }


def _proc_io() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1)
        result[key] = int(value.strip())
    return result


def _memory_snapshot(torch: Any, phase: str) -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree"}:
            meminfo[key] = int(value.split()[0]) * 1024
    free_cuda, total_cuda = torch.cuda.mem_get_info()
    return {
        "phase": phase,
        "unix": time.time(),
        "meminfo_bytes": meminfo,
        "cuda_mem_get_info": {"free_bytes": int(free_cuda), "total_bytes": int(total_cuda)},
        "torch": {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(data).hexdigest()


def _seal_segment_progress(
    receipt: Path,
    segment_phases: list[dict[str, Any]],
    *,
    logical_items: int,
    segments: int,
) -> Path:
    progress = receipt.with_name(f"{receipt.stem}.progress.json")
    value = {
        "schema": "banana-smasher-segment-progress-v1",
        "status": "RUNNING",
        "updated_unix": time.time(),
        "logical_items": int(logical_items),
        "completed_segments": len(segment_phases),
        "expected_segments": int(segments),
        "segment_phases": segment_phases,
    }
    _atomic_json(progress, value)
    return progress


def _load_aot(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_C", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load AOT extension {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _install_bounded_qtip(f521: Any) -> None:
    from .fwht import bounded_fwht

    original = f521.CorrectQtipDecoder.__init__
    if getattr(original, "_banana_smasher_bounded_fwht", False):
        return

    def bounded_init(self: Any, device: Any) -> None:
        original(self, device)
        self.fwht = lambda value: bounded_fwht(
            value,
            inplace=not bool(value.requires_grad),
        )

    bounded_init._banana_smasher_bounded_fwht = True  # type: ignore[attr-defined]
    f521.CorrectQtipDecoder.__init__ = bounded_init


def _build_student(
    torch: Any,
    runtime: Any,
    surface: Any,
    model_root: Path,
    layers: int,
) -> Any:
    from safetensors import safe_open
    from torch import nn
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4RotaryEmbedding,
    )

    config = AutoConfig.from_pretrained(model_root, local_files_only=True)
    original_layers = int(config.num_hidden_layers)
    if original_layers != PRODUCTION_LAYERS:
        raise RuntimeError(
            f"update expects the {PRODUCTION_LAYERS}-layer production config, got {original_layers}"
        )
    if layers not in (1, PRODUCTION_LAYERS):
        raise ValueError(f"layers must be 1 or {PRODUCTION_LAYERS}, got {layers}")
    config.num_hidden_layers = int(layers)
    weight_map = json.loads((model_root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.eval()
    handles: dict[str, Any] = {}

    def get_tensor(name: str) -> Any:
        shard = weight_map[name]
        if shard not in handles:
            while len(handles) >= 3:
                handles.pop(next(iter(handles)))
            handles[shard] = safe_open(model_root / shard, framework="pt")
        return handles[shard].get_tensor(name)

    model.model.embed_tokens.weight = nn.Parameter(
        get_tensor("embed.weight").to("cuda").to(torch.bfloat16), requires_grad=False
    )
    model.lm_head.weight = nn.Parameter(
        get_tensor("head.weight").to("cuda").to(torch.bfloat16), requires_grad=False
    )
    model.model.norm.weight = nn.Parameter(
        get_tensor("norm.weight").to("cuda").to(torch.bfloat16), requires_grad=False
    )
    model.model.hc_head.hc_fn = nn.Parameter(
        get_tensor("hc_head_fn").to("cuda").float(), requires_grad=False
    )
    model.model.hc_head.hc_base = nn.Parameter(
        get_tensor("hc_head_base").to("cuda").float(), requires_grad=False
    )
    model.model.hc_head.hc_scale = nn.Parameter(
        get_tensor("hc_head_scale").to("cuda").float(), requires_grad=False
    )
    model.model.rotary_emb = DeepseekV4RotaryEmbedding(config).to("cuda")

    experts: dict[int, Any] = {}
    for layer in range(int(layers)):
        owner = surface.GenesisPhysicalExperts(layer, pilot=True)
        model.model.layers[layer].mlp.experts = owner
        experts[layer] = owner
        state = runtime.build_nonexpert_sd(layer, weight_map, get_tensor)
        runtime.v3.materialize_layer(model, layer, state, config)
        del state
        if layer == 0 or (layer + 1) % 4 == 0 or layer + 1 == layers:
            _progress(
                "model_layer_materialized",
                layer=layer,
                layers=int(layers),
                torch_allocated_bytes=int(torch.cuda.memory_allocated()),
            )
    train_ids = {
        id(parameter)
        for owner in experts.values()
        for parameter in owner.parameters()
    }
    for parameter in model.parameters():
        if id(parameter) not in train_ids:
            parameter.requires_grad_(False)

    class FreshStudent:
        pass

    student = FreshStudent()
    student.config = config
    student.model = model
    student.experts = experts
    student.original_num_hidden_layers = original_layers
    return student


def _activate_local_preflight(
    f521: Any,
    preflight_runtime: Any,
    *,
    receipt_path: str | Path | None,
    expected_receipt_sha256: str | None,
    expected_manifest_sha256: str | None,
    migration_receipt_path: str | Path | None,
    expected_migration_receipt_sha256: str | None,
    expected_task_id: str | None,
) -> dict[str, Any]:
    bindings = {
        "receipt_path": receipt_path,
        "expected_receipt_sha256": expected_receipt_sha256,
        "expected_manifest_sha256": expected_manifest_sha256,
        "migration_receipt_path": migration_receipt_path,
        "expected_migration_receipt_sha256": expected_migration_receipt_sha256,
        "expected_task_id": expected_task_id,
    }
    missing = sorted(name for name, value in bindings.items() if value is None)
    if missing:
        raise RuntimeError(
            "sealed local-input preflight bindings are required before model materialization: "
            + ", ".join(missing)
        )
    receipt = Path(receipt_path).resolve()
    migration = Path(migration_receipt_path).resolve()
    authority = preflight_runtime.assert_preflight_sealed_structural(
        receipt,
        expected_receipt_sha256=str(expected_receipt_sha256),
        expected_manifest_sha256=str(expected_manifest_sha256),
        migration_receipt_path=migration,
        expected_migration_receipt_sha256=str(expected_migration_receipt_sha256),
        expected_task_id=str(expected_task_id),
    )
    f521.activate_local_only(receipt, authority)
    return {
        "receipt": str(receipt),
        "receipt_sha256": authority.receipt_sha256,
        "manifest_sha256": authority.manifest_sha256,
        "migration_receipt": str(migration),
        "migration_receipt_sha256": str(expected_migration_receipt_sha256),
        "cache_root": str(authority["cache_root"]),
        "ordered_window_ids": list(map(int, authority["ordered_window_ids"])),
        "network_forbidden_during_update": authority["network_forbidden_during_update"],
    }


def _run_minimal_update_impl(
    *,
    runtime_root: str | Path,
    model_root: str | Path,
    aot: str | Path,
    receipt: str | Path,
    window: int = 27,
    source_windows: Sequence[int] | None = None,
    tokens: int = 1024,
    learning_rate: float = 1e-4,
    layers: int = PRODUCTION_LAYERS,
    accumulation_segments: int = 8,
    output: str | Path | None = None,
    backend: str = "accelerated",
    resume: bool = True,
    restart: bool = False,
    verbose_receipts: bool = False,
) -> dict[str, Any]:
    missing_dependencies = [
        name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None
    ]
    if missing_dependencies:
        raise RuntimeError(
            "smash update requires the update extra; missing Python components: "
            + ", ".join(missing_dependencies)
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("smash update requires CUDA")
    runtime_root = Path(runtime_root).resolve()
    model_root = Path(model_root).resolve()
    aot = Path(aot).resolve()
    receipt = Path(receipt).resolve()
    if not runtime_root.is_dir() or not model_root.is_dir() or not aot.is_file():
        raise FileNotFoundError(
            f"invalid update inputs runtime={runtime_root} model={model_root} aot={aot}"
        )
    _validate_runtime_components(runtime_root)
    layers = int(layers)
    if layers not in (1, PRODUCTION_LAYERS):
        raise ValueError(f"layers must be 1 or {PRODUCTION_LAYERS}, got {layers}")
    segment_bounds = _logical_segment_bounds(tokens, accumulation_segments)
    accumulation_segments = len(segment_bounds)
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    process = _proc_identity()
    aot_sha = _sha256(aot)
    _load_aot(aot)
    _progress(
        "boot_aot_loaded",
        aot=str(aot),
        aot_sha256=aot_sha,
        pid=process["pid"],
        start_ticks=process["start_ticks"],
    )

    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    import base_binrepair_e2e as base
    import f521_repair_overlay as f521
    import banana_smasher_physical_surface as surface
    import lp4_train as runtime

    _install_bounded_qtip(f521)
    teacher_root = os.environ.get("BANANA_SMASHER_TEACHER_ROOT")
    if teacher_root:
        base.T.TEACH = Path(teacher_root).resolve()
    input_configuration = {
        "mode": "runtime-configured",
        "teacher_root": str(Path(base.T.TEACH).resolve()),
    }
    surface.reset_real10x_dispatch_trace()
    resident_policy = (
        _resident_prefill_policy(layers, accumulation_segments)
        if backend == "accelerated"
        else "layer-window-eviction"
    )
    resident_mode = resident_policy != "layer-window-eviction"
    memory_policy = (
        _runtime_memory_policy(layers, accumulation_segments)
        if backend == "accelerated"
        else {
            "keep_planes_resident": "0",
            "pin_planes": "0",
            "evict": "1",
            "checkpoint": "1" if layers != 1 else "0",
        }
    )
    os.environ["GENESIS_REPAIR_KEEP_PLANES_RESIDENT"] = memory_policy[
        "keep_planes_resident"
    ]
    os.environ["GENESIS_REPAIR_PIN_PLANES"] = memory_policy["pin_planes"]
    os.environ["GENESIS_REPAIR_EVICT"] = memory_policy["evict"]
    os.environ["GENESIS_REPAIR_CHECKPOINT"] = memory_policy["checkpoint"]
    os.environ["GENESIS_REPAIR_KMAJOR_10X"] = "1" if backend == "accelerated" else "0"
    os.environ["GENESIS_REPAIR_KMAJOR_WINDOWED"] = "1" if backend == "accelerated" else "0"
    os.environ.setdefault("GENESIS_REPAIR_KMAJOR_CACHE_MAX_BYTES", str(3 * 1024**3))
    os.environ.setdefault("GENESIS_REPAIR_KMAJOR_PREFETCH_EXPERTS", "16")
    # The timed interval is forbidden from reading /proc. Safety headroom is
    # sampled immediately around it instead of inside each expert chunk.
    surface._memory_floor_guard = lambda _where: None

    snapshots = [_memory_snapshot(torch, "boot")]
    student = _build_student(torch, runtime, surface, model_root, layers)
    base.FIRST_TRAIN = 0
    snapshots.append(_memory_snapshot(torch, "fresh_model_loaded"))
    _progress(
        "fresh_model_loaded",
        layers=layers,
        mem_available_bytes=snapshots[-1]["meminfo_bytes"]["MemAvailable"],
        torch_allocated_bytes=snapshots[-1]["torch"]["allocated_bytes"],
    )

    corpus_path = Path(os.environ["BR_CORPUS"]).resolve()
    assignment = Path(os.environ["GENESIS_ASSIGNMENT"]).resolve()
    base_assignment = Path(os.environ["GENESIS_BASE_ASSIGNMENT"]).resolve()

    corpus = base.T.load_corpus()
    segment_tokens = int(tokens)
    logical_items = segment_tokens * accumulation_segments
    _set_logical_training_extent(base.T, logical_items)
    selected_windows = (int(window),) if source_windows is None else tuple(source_windows)
    source_plan = _logical_source_plan(corpus, selected_windows, logical_items)
    ids_chunks = []
    teacher_idx_chunks = []
    teacher_lp_chunks = []
    teacher_prob_chunks = []
    teacher_paths: list[Path] = []
    for source_window, take in source_plan:
        ids_one, valid_one = base.T.window_ids(corpus, source_window)
        teacher_idx_one, teacher_lp_one, teacher_prob_one = base.teacher_rows_mmap(source_window)
        available = min(
            int(valid_one),
            int(ids_one.shape[0]),
            int(teacher_idx_one.shape[0]),
            int(teacher_lp_one.shape[0]),
            int(teacher_prob_one.shape[0]),
        )
        if available < take:
            raise RuntimeError(
                f"source window {source_window} has {available} aligned input/teacher rows, needs {take}"
            )
        ids_chunks.append(ids_one[:take])
        teacher_idx_chunks.append(teacher_idx_one[:take])
        teacher_lp_chunks.append(teacher_lp_one[:take])
        teacher_prob_chunks.append(teacher_prob_one[:take])
        teacher_paths.append(Path(base.T.TEACH).resolve() / f"t8192_win{source_window}.pt")
    ids_cpu = torch.cat(ids_chunks, dim=0).contiguous()
    ids = ids_cpu.unsqueeze(0).to("cuda")
    teacher_idx = torch.cat(teacher_idx_chunks, dim=0).contiguous()
    teacher_lp = torch.cat(teacher_lp_chunks, dim=0).contiguous()
    teacher_prob = torch.cat(teacher_prob_chunks, dim=0).contiguous()
    segments = [
        {
            "index": index,
            "token_start": start,
            "token_stop": stop,
            "ids": ids[:, start:stop],
            "teacher_idx": teacher_idx[start:stop],
            "teacher_lp": teacher_lp[start:stop],
            "teacher_prob": teacher_prob[start:stop],
        }
        for index, (start, stop) in enumerate(segment_bounds)
    ]
    input_hashes = {
        "corpus": _sha256(corpus_path),
        "teacher_file_by_window": {
            str(source_window): _sha256(path)
            for (source_window, _take), path in zip(source_plan, teacher_paths)
        },
        "input_ids_tensor": _tensor_sha256(ids_cpu),
        "teacher_index_tensor": _tensor_sha256(teacher_idx),
        "teacher_logprob_tensor": _tensor_sha256(teacher_lp),
        "assignment": _sha256(assignment),
        "base_assignment": _sha256(base_assignment),
        "model_config": _sha256(model_root / "config.json"),
        "aot": aot_sha,
        "input_configuration": input_configuration,
    }
    snapshots.append(_memory_snapshot(torch, "inputs_and_teacher_preloaded"))

    def make_hidden(segment: dict[str, Any]) -> Any:
        embeddings = student.model.model.embed_tokens(segment["ids"])
        return embeddings.unsqueeze(2).expand(
            -1, -1, student.config.hc_mult, -1
        ).contiguous()

    def forward_loss(segment: dict[str, Any], *, requires_grad: bool, reduction: str) -> Any:
        hidden = make_hidden(segment)
        output = base.fast_forward(
            student, hidden, segment["ids"], requires_grad
        )[0, :segment_tokens]
        logits = student.model.lm_head(output.to(torch.bfloat16))
        selected = logits.gather(1, segment["teacher_idx"]).float()
        student_lp = selected - selected.logsumexp(-1, keepdim=True)
        item_loss = (
            segment["teacher_prob"] * (segment["teacher_lp"] - student_lp)
        ).sum(-1)
        if reduction == "sum":
            return item_loss.sum()
        if reduction == "mean":
            return item_loss.mean()
        raise ValueError(f"unsupported loss reduction {reduction!r}")

    if resident_mode:
        surface.begin_resident_plane_prefill()
        with torch.no_grad():
            preload_losses = []
            for segment in segments:
                current = forward_loss(
                    segment, requires_grad=False, reduction="mean"
                )
                torch.cuda.synchronize()
                preload_losses.append(current)
                _progress(
                    "resident_prefill_segment_complete",
                    segment_index=segment["index"],
                    segments=accumulation_segments,
                    finite=bool(torch.isfinite(current)),
                )
        torch.cuda.synchronize()
        preload_inventory = _seal_prefilled_planes(surface, resident_policy)
        preload_mode = (
            resident_policy
            if resident_policy == "sealed-eight-segment-full-depth"
            else "exact-routed-resident"
        )
    else:
        # Full depth cannot retain every decoded QTIP row inside the GB10 safety
        # envelope. An untimed traversal warms every routed immutable source page
        # while the layer-window policy evicts decoded rows after each layer.
        with torch.no_grad():
            preload_losses = []
            for segment in segments:
                current = forward_loss(
                    segment, requires_grad=False, reduction="mean"
                )
                torch.cuda.synchronize()
                preload_losses.append(current)
                _progress(
                    "source_preload_segment_complete",
                    segment_index=segment["index"],
                    segments=accumulation_segments,
                    finite=bool(torch.isfinite(current)),
                )
        torch.cuda.synchronize()
        preload_inventory = surface.resident_plane_inventory()
        preload_mode = "full-depth-source-page-warm-with-layer-window-eviction"
    if not preload_losses or not all(bool(torch.isfinite(value)) for value in preload_losses):
        raise RuntimeError(f"non-finite preload loss: {preload_losses}")
    surface.reset_real10x_dispatch_trace()
    snapshots.append(_memory_snapshot(torch, "one_window_source_preload_complete"))
    _progress(
        "one_window_source_preload_complete",
        preload_mode=preload_mode,
        resident_plane_bytes=preload_inventory["bytes"],
        resident_plane_entries=preload_inventory["entries"],
        mem_available_bytes=snapshots[-1]["meminfo_bytes"]["MemAvailable"],
    )

    parameters = surface.surface_parameters(student, layers=range(layers))
    if not parameters:
        raise RuntimeError("one-layer update has no trainable surface parameters")
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    if optimizer.state:
        raise RuntimeError("fresh optimizer unexpectedly contains warm-start state")
    if output is None:
        raise ValueError("smash update requires an output artifact path")

    from .update_engine import run_segmented_update

    def validate_operational_memory() -> dict[str, Any]:
        snapshots.append(_memory_snapshot(torch, "optimizer_complete"))
        minimum_available = min(
            int(row["meminfo_bytes"]["MemAvailable"]) for row in snapshots
        )
        maximum_device_used = max(
            int(row["cuda_mem_get_info"]["total_bytes"])
            - int(row["cuda_mem_get_info"]["free_bytes"])
            for row in snapshots
        )
        acceptance = _runtime_memory_acceptance(
            layers, minimum_available, maximum_device_used
        )
        if not acceptance["hard_pass"]:
            raise RuntimeError(f"update memory safety guard failed: {acceptance}")
        return {"memory": acceptance} if verbose_receipts else {}

    return run_segmented_update(
        parameters=parameters,
        optimizer=optimizer,
        segments=segments,
        item_count=lambda segment: int(segment["token_stop"] - segment["token_start"]),
        loss_sum=lambda segment: forward_loss(
            segment, requires_grad=True, reduction="sum"
        ),
        output=output,
        receipt=receipt,
        identity={
            "backend": backend,
            "layers": layers,
            "logical_items": logical_items,
            "source_plan": source_plan,
            "input_sha256": input_hashes,
            "learning_rate": float(learning_rate),
        },
        backend=backend,
        resume=resume,
        restart=restart,
        verbose_receipts=verbose_receipts,
        synchronize=torch.cuda.synchronize,
        receipt_fields={
            "layers": layers,
            "tokens_per_segment": segment_tokens,
        },
        post_step_validate=validate_operational_memory,
    )


def run_minimal_update(
    *,
    runtime_root: str | Path,
    model_root: str | Path,
    aot: str | Path,
    receipt: str | Path,
    output: str | Path,
    window: int = 27,
    source_windows: Sequence[int] | None = None,
    tokens: int = 1024,
    learning_rate: float = 1e-4,
    layers: int = PRODUCTION_LAYERS,
    accumulation_segments: int = 8,
    backend: str = "accelerated",
    resume: bool = True,
    restart: bool = False,
    verbose_receipts: bool = False,
) -> dict[str, Any]:
    """Run the shipped backend under scoped runtime flags; reference is explicit only."""
    with _backend_environment(backend):
        return _run_minimal_update_impl(
            runtime_root=runtime_root,
            model_root=model_root,
            aot=aot,
            receipt=receipt,
            output=output,
            window=window,
            source_windows=source_windows,
            tokens=tokens,
            learning_rate=learning_rate,
            layers=layers,
            accumulation_segments=accumulation_segments,
            backend=backend,
            resume=resume,
            restart=restart,
            verbose_receipts=verbose_receipts,
        )
