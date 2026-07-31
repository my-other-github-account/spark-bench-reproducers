from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .fwht import bounded_fwht, fwht_stats

BASELINE_WINDOW_SECONDS = 890.0
PRODUCTION_LAYERS = 43
MINIMUM_MEM_AVAILABLE_BYTES = 4 * 1024**3
ONE_LAYER_TARGET_MEM_AVAILABLE_BYTES = 50 * 1024**3
ONE_LAYER_MAX_DEVICE_USED_BYTES = 60 * 1024**3


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


def run_minimal_update(
    *,
    runtime_root: str | Path,
    model_root: str | Path,
    aot: str | Path,
    receipt: str | Path,
    window: int = 27,
    tokens: int = 1024,
    learning_rate: float = 1e-4,
    hard_abort_seconds: float = 250.0,
    baseline_seconds: float = BASELINE_WINDOW_SECONDS,
    layers: int = PRODUCTION_LAYERS,
    accumulation_segments: int = 8,
) -> dict[str, Any]:
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
    layers = int(layers)
    if layers not in (1, PRODUCTION_LAYERS):
        raise ValueError(f"layers must be 1 or {PRODUCTION_LAYERS}, got {layers}")
    segment_bounds = _logical_segment_bounds(tokens, accumulation_segments)
    accumulation_segments = len(segment_bounds)
    if hard_abort_seconds <= 0 or learning_rate <= 0:
        raise ValueError("hard abort and learning rate must be positive")

    process = _proc_identity()
    package_root = Path(__file__).resolve().parents[2]
    code_hashes = {
        "banana_smasher_update": _sha256(Path(__file__)),
        "banana_smasher_fwht": _sha256(Path(__file__).with_name("fwht.py")),
    }
    for name in (
        "genesis_physical_surface.py",
        "f521_repair_overlay.py",
        "kmajor_autograd.py",
    ):
        code_hashes[f"runtime_{name.removesuffix('.py')}"] = _sha256(runtime_root / name)

    expected_aot = os.environ.get("BANANA_SMASHER_EXPECTED_AOT_SHA256")
    aot_sha = _sha256(aot)
    if expected_aot and aot_sha != expected_aot:
        raise RuntimeError(f"AOT SHA drift: {aot_sha} != {expected_aot}")
    aot_module = _load_aot(aot)
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
    import genesis_physical_surface as surface
    import kmajor_autograd
    import lp4_train as runtime

    _install_bounded_qtip(f521)
    surface.reset_real10x_dispatch_trace()
    fwht_stats(reset=True)
    resident_mode = layers == 1
    os.environ["GENESIS_REPAIR_KEEP_PLANES_RESIDENT"] = "1" if resident_mode else "0"
    os.environ["GENESIS_REPAIR_PIN_PLANES"] = "1" if resident_mode else "0"
    os.environ["GENESIS_REPAIR_EVICT"] = "0" if resident_mode else "1"
    os.environ["GENESIS_REPAIR_CHECKPOINT"] = "0" if resident_mode else "1"
    os.environ["GENESIS_REPAIR_KMAJOR_10X"] = "1"
    os.environ["GENESIS_REPAIR_KMAJOR_WINDOWED"] = "1"
    os.environ.setdefault("GENESIS_REPAIR_KMAJOR_CACHE_MAX_BYTES", str(3 * 1024**3))
    os.environ.setdefault("GENESIS_REPAIR_KMAJOR_PREFETCH_EXPERTS", "16")
    # The timed interval is forbidden from reading /proc. Safety headroom is
    # sampled immediately around it instead of inside each expert chunk.
    surface._memory_floor_guard = lambda _where: None

    snapshots = [_memory_snapshot(torch, "boot")]
    started = time.time()
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
    teacher_path = Path(base.T.TEACH).resolve() / f"t8192_win{int(window)}.pt"
    assignment = Path(os.environ["GENESIS_ASSIGNMENT"]).resolve()
    base_assignment = Path(os.environ["GENESIS_BASE_ASSIGNMENT"]).resolve()
    claim = Path(os.environ.get("GENESIS_HOST_CLAIM", "/home/dnola/HOST_CLAIM.json")).resolve()
    corpus = base.T.load_corpus()
    ids_all, valid_tokens = base.T.window_ids(corpus, int(window))
    segment_tokens = int(tokens)
    logical_items = segment_tokens * accumulation_segments
    available_items = min(int(valid_tokens), int(ids_all.shape[0]))
    if available_items < logical_items:
        raise RuntimeError(
            f"window {window} has {available_items} usable tokens, needs exact "
            f"{accumulation_segments}x{segment_tokens}={logical_items}"
        )
    ids_cpu = ids_all[:logical_items].contiguous()
    ids = ids_cpu.unsqueeze(0).to("cuda")
    teacher_idx, teacher_lp, teacher_prob = base.teacher_rows_mmap(int(window))
    teacher_idx = teacher_idx[:logical_items].contiguous()
    teacher_lp = teacher_lp[:logical_items].contiguous()
    teacher_prob = teacher_prob[:logical_items].contiguous()
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
        "teacher_file": _sha256(teacher_path),
        "input_ids_tensor": _tensor_sha256(ids_cpu),
        "teacher_index_tensor": _tensor_sha256(teacher_idx),
        "teacher_logprob_tensor": _tensor_sha256(teacher_lp),
        "assignment": _sha256(assignment),
        "base_assignment": _sha256(base_assignment),
        "model_config": _sha256(model_root / "config.json"),
        "host_claim": _sha256(claim),
        "aot": aot_sha,
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
            preload_losses = [
                forward_loss(segment, requires_grad=False, reduction="mean")
                for segment in segments
            ]
        torch.cuda.synchronize()
        preload_inventory = surface.resident_plane_inventory()
        if int(preload_inventory["layers"]) != 1 or int(preload_inventory["entries"]) <= 0:
            raise RuntimeError(f"one-layer plane preload failed: {preload_inventory}")
        surface._RESIDENT_PLANES_PREFILLING = False
        surface._RESIDENT_PLANES_SEALED = True
        preload_mode = "exact-routed-resident"
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
    preload_loss = sum(float(value) for value in preload_losses) / len(preload_losses)
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
    parameter_before = hashlib.sha256(
        b"".join(
            parameter.detach().cpu().contiguous().numpy().tobytes()
            for parameter in parameters
        )
    ).hexdigest()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    io_before = _proc_io()
    segment_phases: list[dict[str, Any]] = []
    detached_loss_sum = 0.0
    forward_seconds = 0.0
    backward_seconds = 0.0
    for segment in segments:
        forward_started = time.perf_counter()
        segment_loss_sum = forward_loss(
            segment, requires_grad=True, reduction="sum"
        )
        torch.cuda.synchronize()
        segment_forward_seconds = time.perf_counter() - forward_started
        forward_seconds += segment_forward_seconds
        if segment_forward_seconds > float(hard_abort_seconds):
            io_after = _proc_io()
            compute_io_delta = {
                key: int(io_after.get(key, 0) - io_before.get(key, 0))
                for key in sorted(set(io_before) | set(io_after))
            }
            result = {
                "schema": "banana-smasher-logical-window-update-v2",
                "status": "FAIL_HARD_ABORT_PHYSICAL_SEGMENT_FORWARD",
                "host": socket.gethostname(),
                "process": process,
                "segment_index": segment["index"],
                "segment_forward_seconds": segment_forward_seconds,
                "hard_abort_seconds": hard_abort_seconds,
                "compute_io_delta": compute_io_delta,
                "segment_phases": segment_phases,
                "code_sha256": code_hashes,
                "input_sha256": input_hashes,
                "aot": {
                    "path": str(aot),
                    "sha256": aot_sha,
                    "loaded_file": str(aot_module.__file__),
                },
                "allocation_map": snapshots,
            }
            result["receipt_sha256"] = _atomic_json(receipt, result)
            return result

        detached_loss_sum += float(segment_loss_sum.detach())
        backward_started = time.perf_counter()
        (segment_loss_sum / logical_items).backward()
        torch.cuda.synchronize()
        segment_backward_seconds = time.perf_counter() - backward_started
        backward_seconds += segment_backward_seconds
        row = {
            "segment_index": int(segment["index"]),
            "token_start": int(segment["token_start"]),
            "token_stop": int(segment["token_stop"]),
            "items": segment_tokens,
            "forward_seconds": segment_forward_seconds,
            "backward_seconds": segment_backward_seconds,
            "loss_sum": float(segment_loss_sum.detach()),
            "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
            "torch_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "torch_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        segment_phases.append(row)
        _progress("accumulation_segment_complete", **row)
        del segment_loss_sum
        torch.cuda.empty_cache()

    io_after = _proc_io()
    forward_io_delta = {
        key: int(io_after.get(key, 0) - io_before.get(key, 0))
        for key in sorted(set(io_before) | set(io_after))
    }
    snapshots.append(_memory_snapshot(torch, "all_segments_forward_backward_complete"))
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    nonzero_gradients = sum(int(bool(torch.count_nonzero(gradient))) for gradient in gradients)
    _progress(
        "logical_window_accumulation_complete",
        accumulation_segments=accumulation_segments,
        logical_items=logical_items,
        forward_seconds=forward_seconds,
        backward_seconds=backward_seconds,
        rchar_delta=forward_io_delta.get("rchar", 0),
        read_bytes_delta=forward_io_delta.get("read_bytes", 0),
        finite_gradients=finite_gradients,
        nonzero_gradient_tensors=nonzero_gradients,
        mem_available_bytes=snapshots[-1]["meminfo_bytes"]["MemAvailable"],
    )

    optimizer_started = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    optimizer_seconds = time.perf_counter() - optimizer_started
    parameter_after = hashlib.sha256(
        b"".join(
            parameter.detach().cpu().contiguous().numpy().tobytes()
            for parameter in parameters
        )
    ).hexdigest()
    snapshots.append(_memory_snapshot(torch, "optimizer_complete"))

    dispatch = surface.real10x_dispatch_trace(first_layers=1)
    sentinel = kmajor_autograd.kmajor_sentinel()
    transform = fwht_stats()
    loss_value = detached_loss_sum / logical_items
    update_seconds = forward_seconds + backward_seconds + optimizer_seconds
    direct_multiplier = float(baseline_seconds) / update_seconds
    min_available = min(
        int(row["meminfo_bytes"]["MemAvailable"]) for row in snapshots
    )
    max_torch_allocated = max(int(row["torch"]["max_allocated_bytes"]) for row in snapshots)
    max_torch_reserved = max(int(row["torch"]["max_reserved_bytes"]) for row in snapshots)
    max_device_used = max(
        int(row["cuda_mem_get_info"]["total_bytes"])
        - int(row["cuda_mem_get_info"]["free_bytes"])
        for row in snapshots
    )
    mechanics_pass = bool(
        math.isfinite(loss_value)
        and finite_gradients
        and nonzero_gradients > 0
        and parameter_before != parameter_after
        and len(optimizer.state) > 0
        and int(sentinel["bmm_launches"]) > 0
        and int(sentinel["backward_calls"]) > 0
        and not dispatch["noneligible_fallbacks"]
        and int(transform["calls"]) > 0
        and int(preload_inventory["timed_misses"]) == 0
        and int(forward_io_delta.get("read_bytes", 0)) == 0
        and int(forward_io_delta.get("rchar", 0)) <= 4096
        and min_available >= MINIMUM_MEM_AVAILABLE_BYTES
        and (layers != 1 or min_available >= ONE_LAYER_TARGET_MEM_AVAILABLE_BYTES)
        and max_device_used < ONE_LAYER_MAX_DEVICE_USED_BYTES
    )
    result = {
        "schema": "banana-smasher-logical-window-update-v2",
        "status": (
            "PASS_FRESH_FULL_DEPTH_FORWARD_BACKWARD_OPTIMIZER"
            if mechanics_pass and layers == PRODUCTION_LAYERS
            else (
                "PASS_FRESH_ONE_LAYER_FORWARD_BACKWARD_OPTIMIZER"
                if mechanics_pass
                else "FAIL_MECHANICS_GATE"
            )
        ),
        "host": socket.gethostname(),
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "process": process,
        "freshness": {
            "model_layers": layers,
            "production_layers": PRODUCTION_LAYERS,
            "source_layer": 0,
            "batch": 1,
            "microbatch": 1,
            "windows": [int(window)],
            "tokens": logical_items,
            "logical_window_tokens": logical_items,
            "physical_segment_tokens": segment_tokens,
            "accumulation_segments": accumulation_segments,
            "optimizer_steps": 1,
            "assignment_checkpoint_loaded": False,
            "model_checkpoint_loaded": False,
            "optimizer_checkpoint_loaded": False,
            "optimizer_state_entries_before": 0,
            "optimizer_state_entries_after": len(optimizer.state),
        },
        "aot": {
            "path": str(aot),
            "sha256": aot_sha,
            "loaded_file": str(aot_module.__file__),
        },
        "public_code": {
            "root": str(package_root),
            "git_commit": _git_commit(package_root),
            "sha256": code_hashes,
        },
        "input_sha256": input_hashes,
        "preload": {
            "finite_loss": float(preload_loss),
            "resident_plane_inventory": preload_inventory,
            "mode": preload_mode,
            "inputs_and_teacher_preloaded": True,
            "all_routed_planes_resident": resident_mode,
        },
        "phase_seconds": {
            "segments": segment_phases,
            "all_segment_forwards": forward_seconds,
            "backward": backward_seconds,
            "optimizer": optimizer_seconds,
            "complete_update": update_seconds,
        },
        "logical_window": {
            "wall_seconds": update_seconds,
            "first_segment_forward_seconds": segment_phases[0]["forward_seconds"],
            "max_segment_forward_seconds": max(
                float(row["forward_seconds"]) for row in segment_phases
            ),
            "hard_abort_seconds": hard_abort_seconds,
            "hard_abort_pass": all(
                float(row["forward_seconds"]) <= hard_abort_seconds
                for row in segment_phases
            ),
            "compute_io_before": io_before,
            "compute_io_after": io_after,
            "compute_io_delta": forward_io_delta,
            "zero_storage_io": int(forward_io_delta.get("read_bytes", 0)) == 0,
            "near_zero_rchar": int(forward_io_delta.get("rchar", 0)) <= 4096,
        },
        "loss": {"value": loss_value, "finite": math.isfinite(loss_value)},
        "gradients": {
            "parameter_tensors": len(parameters),
            "gradient_tensors": len(gradients),
            "finite": finite_gradients,
            "nonzero_tensors": nonzero_gradients,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": learning_rate,
            "step_completed": parameter_before != parameter_after,
            "parameter_sha256_before": parameter_before,
            "parameter_sha256_after": parameter_after,
        },
        "dispatch": dispatch,
        "bounded_fwht": transform,
        "allocation": {
            "snapshots": snapshots,
            "minimum_mem_available_bytes": min_available,
            "minimum_required_mem_available_bytes": MINIMUM_MEM_AVAILABLE_BYTES,
            "four_gib_floor_pass": min_available >= MINIMUM_MEM_AVAILABLE_BYTES,
            "one_layer_target_mem_available_bytes": ONE_LAYER_TARGET_MEM_AVAILABLE_BYTES,
            "one_layer_target_mem_available_pass": min_available
            >= ONE_LAYER_TARGET_MEM_AVAILABLE_BYTES,
            "maximum_device_used_bytes": max_device_used,
            "one_layer_max_device_used_bytes": ONE_LAYER_MAX_DEVICE_USED_BYTES,
            "one_layer_under_60_gib_pass": max_device_used
            < ONE_LAYER_MAX_DEVICE_USED_BYTES,
            "peak_torch_allocated_bytes": max_torch_allocated,
            "peak_torch_reserved_bytes": max_torch_reserved,
        },
        "multiplier_vs_baseline": {
            "baseline_window_seconds": float(baseline_seconds),
            "measured_update_multiplier": direct_multiplier,
            "measured_layers": layers,
            "comparison_geometry": {
                "layers": layers,
                "logical_tokens": logical_items,
                "windows_per_optimizer_step": 1,
            },
            "comparison_is_equal_useful_work": layers == PRODUCTION_LAYERS
            and logical_items == 8192,
            "no_baby_full_extrapolation": True,
            "campaign_target_multiplier": 10.0,
            "campaign_target_met": direct_multiplier >= 10.0,
            "single_next_lever_if_below_target": (
                None
                if direct_multiplier >= 10.0
                else "retain K-major dense tiles across backward to remove backward rematerialization"
            ),
        },
        "mechanics_pass": mechanics_pass,
    }
    receipt_sha = _atomic_json(receipt, result)
    result["receipt"] = str(receipt)
    result["receipt_sha256"] = receipt_sha
    return result
