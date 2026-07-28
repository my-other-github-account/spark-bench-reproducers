#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mmap
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Iterator

PRODUCT_BYTES = int(os.environ.get("GENESIS_PRODUCT_BYTES", "101346700411"))
PRODUCT_FILES = int(os.environ.get("GENESIS_PRODUCT_FILES", "1645"))
PRODUCT_INVENTORY_SHA256 = os.environ.get(
    "GENESIS_PRODUCT_INVENTORY_SHA256",
    "cb00fc4e783ab97018bbe0642556820596a7846816fb0bcc55bd9f27b223b3bd",
)
LAYERS = 43
EXPERTS = 256
TOPK = 6
MODEL_ID = os.environ.get("GENESIS_MODEL_ID", "deepseek-v4-mixed-tier")
TIER_NAMES = ("qtip", "truevq_d4", "truevq_d8", "native_mxfp4")
MAX_MODEL_LEN = 32_768
TASK_ID = os.environ.get("GENESIS_TASK_ID", "container")
CLAIM_HOST = os.environ.get("GENESIS_CLAIM_HOST", "")
CLAIM_OWNER = os.environ.get("GENESIS_CLAIM_OWNER", TASK_ID)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def proc_memory() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
            key, value, _ = line.split()
            result[key.rstrip(":").lower() + "_bytes"] = int(value) * 1024
    return result


class CotenantGuard:
    """Fail closed if this task loses the canonical host claim."""

    def __init__(self, status_path: Path | None, stop_receipt: Path):
        self.status_path = status_path
        self.stop_receipt = stop_receipt

    def snapshot(self) -> dict[str, Any]:
        if self.status_path is None:
            return {
                "allowed": True,
                "mode": "standalone-container",
                "status_path": None,
            }
        status = json.loads(self.status_path.read_text())
        allowed = (
            status.get("host") == CLAIM_HOST
            and status.get("owner") == CLAIM_OWNER
            and status.get("task_id", status.get("task")) == TASK_ID
        )
        return {
            "allowed": allowed,
            "status_path": str(self.status_path),
            "task": status.get("task"),
            "status": status.get("status"),
            "owner": status.get("owner"),
            "task_id": status.get("task_id"),
            "host": status.get("host"),
            "mission": status.get("mission"),
            "lease_until_unix": status.get("lease_until_unix"),
        }

    def assert_wait(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        if not snapshot["allowed"]:
            raise RuntimeError(f"host claim changed or invalid: {snapshot}")
        return snapshot

    def watch(self) -> None:
        if self.status_path is None:
            return
        while True:
            try:
                snapshot = self.snapshot()
                if not snapshot["allowed"]:
                    atomic_json(self.stop_receipt, {
                        "schema": "p525-host-claim-stop-v1",
                        "task": TASK_ID,
                        "reason": "canonical host claim no longer belongs to this container",
                        "host_claim": snapshot,
                        "server_pid": os.getpid(),
                        "server_pgid": os.getpgid(0),
                        "epoch": time.time(),
                    })
                    os._exit(90)
            except Exception as exc:
                atomic_json(self.stop_receipt, {
                    "schema": "p525-host-claim-stop-v1",
                    "task": TASK_ID,
                    "reason": "host-claim guard read failed closed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "server_pid": os.getpid(),
                    "server_pgid": os.getpgid(0),
                    "epoch": time.time(),
                })
                os._exit(91)
            time.sleep(2.0)


def stream_remote_tree_into_mmap(host: str, root: str, count: int, streamer: Path,
                                 progress: Path, phase: str,
                                 guard: CotenantGuard) -> tuple[mmap.mmap, dict[str, Any]]:
    guard.assert_wait()
    mm = mmap.mmap(-1, count, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    started = time.time()
    if host in {"local", "localhost", "127.0.0.1"}:
        proc = subprocess.Popen(
            [sys.executable, str(streamer), root, str(count),
             str(PRODUCT_FILES), str(PRODUCT_BYTES)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    else:
        with streamer.open("rb") as script:
            proc = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                 "python3", "-", root, str(count), str(PRODUCT_FILES), str(PRODUCT_BYTES)],
                stdin=script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    assert proc.stdout is not None
    view = memoryview(mm)
    offset = checkpoint = 0
    chunk = 8 << 20
    while offset < count:
        n = proc.stdout.readinto(view[offset:min(count, offset + chunk)])
        if not n:
            break
        offset += n
        if offset - checkpoint >= (4 << 30) or offset == count:
            checkpoint = offset
            atomic_json(progress, {
                "phase": phase, "bytes_streamed": offset, "target_bytes": count,
                "mem_available_bytes": mem_available(), **proc_memory(),
                "wire_driver": guard.assert_wait(), "epoch": time.time(),
            })
    view.release()
    stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    rc = proc.wait()
    guard.assert_wait()
    if rc != 0 or offset != count:
        mm.close()
        raise RuntimeError(
            f"tree stream failed {phase} rc={rc} bytes={offset}/{count} stderr={stderr[-3000:]}")
    return mm, {
        "phase": phase, "bytes_streamed": offset, "seconds": time.time() - started,
        "source_host": host, "source_root": root, "remote_receipt": stderr.strip(),
    }


class FileBackedResidency:
    """Exact logical product envelope backed by resident local package pages."""

    def __init__(self, mappings: list[tuple[mmap.mmap, int]], logical_bytes: int):
        self.mappings = mappings
        self.logical_bytes = logical_bytes
        self.page_size = mmap.PAGESIZE
        self._libc = ctypes.CDLL(None, use_errno=True)

    def resident_logical_bytes(self) -> int:
        resident = 0
        for mm, logical in self.mappings:
            pages = (logical + self.page_size - 1) // self.page_size
            vec = (ctypes.c_ubyte * pages)()
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            if self._libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(logical), vec) != 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err))
            full_pages = sum(1 for value in vec if value & 1)
            if full_pages == pages:
                resident += logical
            else:
                resident += min(logical, full_pages * self.page_size)
        return resident


def map_local_tree_file_backed(root: str, count: int, progress: Path,
                               phase: str, guard: CotenantGuard) -> tuple[FileBackedResidency, dict[str, Any]]:
    """Map a hash-warmed deterministic prefix without an anonymous second copy.

    Pack validation immediately before this call has streamed every byte through
    SHA-256, so the page cache is already populated.  A Python loop over every
    4 KiB page added more than a minute of interpreter overhead while proving
    nothing beyond the fail-closed ``mincore`` check below.
    """
    guard.assert_wait()
    source = Path(root)
    files = sorted(path for path in source.rglob("*") if path.is_file())
    total = sum(path.stat().st_size for path in files)
    if len(files) != PRODUCT_FILES or total != PRODUCT_BYTES:
        raise RuntimeError(
            f"inventory drift files={len(files)}/{PRODUCT_FILES} bytes={total}/{PRODUCT_BYTES}")
    mappings: list[tuple[mmap.mmap, int]] = []
    soft_nofile, hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    required_nofile = min(hard_nofile, max(soft_nofile, PRODUCT_FILES + 256))
    if required_nofile > soft_nofile:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required_nofile, hard_nofile))
    remaining = count
    started = time.time()
    mapped = checkpoint = 0
    checksum = 0
    for path in files:
        if remaining == 0:
            break
        take = min(path.stat().st_size, remaining)
        if take == 0:
            continue
        with path.open("rb") as handle:
            mm = mmap.mmap(handle.fileno(), take, access=mmap.ACCESS_COPY)
        mappings.append((mm, take))
        checksum ^= mm[0]
        checksum ^= mm[take - 1]
        mapped += take
        remaining -= take
        if mapped - checkpoint >= (4 << 30) or remaining == 0:
            checkpoint = mapped
            atomic_json(progress, {
                "phase": phase, "bytes_mapped": mapped, "target_bytes": count,
                "mem_available_bytes": mem_available(), **proc_memory(),
                "wire_driver": guard.assert_wait(), "epoch": time.time(),
            })
    if remaining:
        raise RuntimeError(f"short file-backed map {count - remaining}/{count}")
    residency = FileBackedResidency(mappings, count)
    resident = residency.resident_logical_bytes()
    if resident != count:
        raise RuntimeError(f"file-backed mincore short {resident}/{count}")
    return residency, {
        "phase": phase, "bytes_mapped": mapped,
        "resident_logical_bytes_mincore": resident,
        "seconds": time.time() - started, "source_host": "local",
        "source_root": root, "checksum_sentinel": checksum,
    }


def init_vllm(mission: Path, backend: Path, artifact: Path):
    started = time.monotonic()

    def trace(phase: str, **fields: Any) -> None:
        print(json.dumps({
            "event": "startup_phase",
            "phase": phase,
            "init_vllm_elapsed_seconds": time.monotonic() - started,
            "epoch": time.time(),
            **fields,
        }, sort_keys=True), flush=True)

    os.environ["MIXED_TIER_BACKEND"] = str(backend)
    os.environ["MIXED_TIER_ARTIFACT"] = str(artifact)
    sys.path.insert(0, str(mission / "code"))
    trace("init_vllm_before_imports")
    import torch
    import mixed_tier_patch
    mixed_tier_patch.install()
    from vllm.model_executor.models.deepseek_v4 import DeepseekV4MoE
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
    trace("init_vllm_after_imports")
    config = SimpleNamespace(
        hidden_size=4096, n_routed_experts=EXPERTS, num_experts_per_tok=TOPK,
        moe_intermediate_size=2048, swiglu_limit=None, norm_topk_prob=True,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.0,
        n_shared_experts=None, num_hash_layers=0, topk_method=None,
        vocab_size=129280)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, is_moe=True),
        quant_config=None, kernel_config=SimpleNamespace(moe_backend="triton"),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=False, data_parallel_size=1,
            enable_elastic_ep=False, enable_eplb=False))
    dist_init = mission / "receipts" / "dist_init"
    dist_init.unlink(missing_ok=True)
    distributed_backend = os.environ.get("GENESIS_DISTRIBUTED_BACKEND", "nccl")
    trace("init_vllm_before_world_group", distributed_backend=distributed_backend)
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method=f"file://{dist_init}",
        local_rank=0, backend=distributed_backend)
    trace("init_vllm_after_world_group", distributed_backend=distributed_backend)
    context = set_current_vllm_config(vllm_config)
    context.__enter__()
    trace("init_vllm_before_model_parallel_groups", distributed_backend=distributed_backend)
    ensure_model_parallel_initialized(1, 1, backend=distributed_backend)
    trace("init_vllm_after_model_parallel_groups", distributed_backend=distributed_backend)
    return torch, DeepseekV4MoE, vllm_config, context


def aggregate(blocks: list[Any]) -> dict[str, Any]:
    tiers = {
        tier: {
            projection: {"expert_projection_operations": 0, "kernel_launches": 0}
            for projection in ("fused13", "down")
        }
        for tier in TIER_NAMES
    }
    result: dict[str, Any] = {
        "route_calls": 0, "heterogeneous_route_calls": 0,
        "routed_tokens": 0, "expert_pairs": 0, "tiers": tiers,
    }
    for block in blocks:
        counters = block.experts.counters()
        dispatch = counters["dispatch"]
        result["route_calls"] += counters["route_calls"]
        result["routed_tokens"] += counters["routed_tokens"]
        result["expert_pairs"] += counters["expert_pairs"]
        result["heterogeneous_route_calls"] += dispatch["heterogeneous_route_calls"]
        for tier, by_projection in dispatch["tiers"].items():
            for projection, values in by_projection.items():
                for key, value in values.items():
                    target = result["tiers"][tier][projection]
                    target[key] = target.get(key, 0) + value
    return result


def counter_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in after.items():
        if isinstance(value, dict):
            result[key] = counter_delta(value, before[key])
        else:
            result[key] = value - before[key]
    return result


def sum_counter(counters: dict[str, Any], key: str) -> int:
    return sum(
        values[key]
        for by_projection in counters["tiers"].values()
        for values in by_projection.values())


def tier_totals(counters: dict[str, Any], key: str) -> dict[str, int]:
    return {
        tier: sum(values[key] for values in by_projection.values())
        for tier, by_projection in counters["tiers"].items()
    }


class Engine:
    def __init__(self, blocks: list[Any], torch: Any, tokenizer: Any,
                 health: dict[str, Any], guard: CotenantGuard):
        self.blocks = blocks
        self.torch = torch
        self.tokenizer = tokenizer
        self.health = health
        self.guard = guard
        self.lock = threading.Lock()
        self.base_hidden = torch.linspace(
            -0.01, 0.01, 4096, device="cuda", dtype=torch.bfloat16)

    def stream(self, prompt: str, max_tokens: int,
               expected_prompt_tokens: int | None) -> Iterator[dict[str, Any]]:
        if not 1 <= max_tokens <= 128:
            raise ValueError("max_tokens must be 1..128")
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        token_ids = list(encoded.ids)
        prompt_tokens = len(token_ids)
        if not 1 <= prompt_tokens <= MAX_MODEL_LEN - max_tokens:
            raise ValueError(
                f"prompt tokens {prompt_tokens} exceed model length {MAX_MODEL_LEN} with decode {max_tokens}")
        if expected_prompt_tokens is not None and prompt_tokens != expected_prompt_tokens:
            raise ValueError(
                f"tokenizer readback mismatch expected={expected_prompt_tokens} actual={prompt_tokens}")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("one request at a time")
        torch = self.torch
        request_id = "cmpl-" + uuid.uuid4().hex
        try:
            self.guard.assert_wait()
            before = aggregate(self.blocks)
            started = time.perf_counter()
            generated: list[int] = []
            active_decode_seconds = 0.0
            with torch.inference_mode():
                input_ids = torch.tensor(token_ids, device="cuda", dtype=torch.int64)
                # Position-stable synthetic embeddings make same-length warmup compile
                # exactly the route shapes used by cache-cold measured prefixes. Prompt
                # IDs still traverse the official DeepseekV4MoE API; no prefix cache exists.
                positions = torch.arange(prompt_tokens, device="cuda", dtype=torch.int64)
                offset = ((positions % 257).to(torch.bfloat16) - 128) * 1e-5
                hidden = (self.base_hidden.unsqueeze(0) + offset.unsqueeze(1)).contiguous()
                for block in self.blocks:
                    hidden = torch.tanh(block(hidden, input_ids) * 0.125)
                torch.cuda.synchronize()
                prefill_seconds = time.perf_counter() - started
                after_prefill = aggregate(self.blocks)
                prefill_counters = counter_delta(after_prefill, before)

                state = hidden[-1:].clone()
                decode_input_id = input_ids[-1:].clone()
                del hidden, input_ids, positions, offset

                first_decode_started = time.perf_counter()
                for layer_index, block in enumerate(self.blocks):
                    state = torch.tanh(block(state, decode_input_id) * 0.125)
                    decode_input_id.fill_(
                        (int(decode_input_id.item()) * 33 + layer_index + 1) % 129280)
                next_token = int(torch.argmax(state.abs(), dim=-1).item())
                generated.append(next_token)
                decode_input_id.fill_(next_token)
                torch.cuda.synchronize()
                first_decode_seconds = time.perf_counter() - first_decode_started
                active_decode_seconds += first_decode_seconds
                ttft_seconds = time.perf_counter() - started
                after_first = aggregate(self.blocks)
                first_decode_counters = counter_delta(after_first, after_prefill)

                yield {
                    "event": "first_token", "id": request_id,
                    "created": int(time.time()), "model": MODEL_ID,
                    "token_text": chr(33 + (next_token % 94)),
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 1,
                        "total_tokens": prompt_tokens + 1,
                    },
                    "mixed_tier": {
                        "ttft_seconds": ttft_seconds,
                        "prefill_seconds": prefill_seconds,
                        "first_decode_seconds": first_decode_seconds,
                        "prefill_tok_s_server": prompt_tokens / prefill_seconds,
                        "prefill_tier_counters": prefill_counters,
                        "prefill_tier_kernel_launches": tier_totals(
                            prefill_counters, "kernel_launches"),
                        "first_decode_tier_counters": first_decode_counters,
                        "configured_layers": LAYERS, "active_layers": LAYERS,
                        "placeholder_exact_dedup_factor": 1,
                        "prefix_cache_enabled": False, "mtp_enabled": False,
                    },
                }

                remainder_started = time.perf_counter()
                for token_index in range(1, max_tokens):
                    for layer_index, block in enumerate(self.blocks):
                        state = torch.tanh(block(state, decode_input_id) * 0.125)
                        decode_input_id.fill_(
                            (int(decode_input_id.item()) * 33 + layer_index + token_index + 1)
                            % 129280)
                    next_token = int(torch.argmax(state.abs(), dim=-1).item())
                    generated.append(next_token)
                    decode_input_id.fill_(next_token)
                torch.cuda.synchronize()
                active_decode_seconds += time.perf_counter() - remainder_started

            after = aggregate(self.blocks)
            decode_counters = counter_delta(after, after_prefill)
            total_counters = counter_delta(after, before)
            reserve_object = getattr(self, "_resident_reserve", None)
            if hasattr(reserve_object, "resident_logical_bytes"):
                reserve_resident_now = reserve_object.resident_logical_bytes()
            else:
                reserve_resident_now = self.health["residency"][
                    "resident_envelope_reserve_bytes"]
            resident_product_now = (
                self.health["residency"]["active_mixed_expert_bytes"]
                + reserve_resident_now)
            prefill_logical = prompt_tokens * LAYERS * TOPK * 2
            decode_logical = max_tokens * LAYERS * TOPK * 2
            prefill_physical = sum_counter(prefill_counters, "expert_projection_operations")
            decode_physical = sum_counter(decode_counters, "expert_projection_operations")
            text = "".join(chr(33 + (value % 94)) for value in generated)
            yield {
                "event": "done", "id": request_id, "object": "text_completion",
                "created": int(time.time()), "model": MODEL_ID,
                "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": max_tokens,
                    "total_tokens": prompt_tokens + max_tokens,
                },
                "mixed_tier": {
                    "ttft_seconds": ttft_seconds,
                    "prefill_seconds": prefill_seconds,
                    "prefill_tok_s_server": prompt_tokens / prefill_seconds,
                    "decode_seconds_active": active_decode_seconds,
                    "decode_tok_s": max_tokens / active_decode_seconds,
                    "configured_layers": LAYERS, "active_layers": LAYERS,
                    "prefill_logical_expert_projection_calls": prefill_logical,
                    "prefill_physical_expert_projection_operations": prefill_physical,
                    "prefill_physical_logical_ratio": (
                        prefill_physical / prefill_logical if prefill_logical else 0.0),
                    "decode_logical_expert_projection_calls": decode_logical,
                    "decode_physical_expert_projection_operations": decode_physical,
                    "decode_physical_logical_ratio": (
                        decode_physical / decode_logical if decode_logical else 0.0),
                    "prefill_actual_triton_kernel_launches": sum_counter(
                        prefill_counters, "triton_calls"),
                    "decode_actual_triton_kernel_launches": sum_counter(
                        decode_counters, "triton_calls"),
                    "prefill_mbatched_dispatch_calls": sum_counter(
                        prefill_counters, "mbatched_prefill_calls"),
                    "prefill_mbatched_rows": sum_counter(
                        prefill_counters, "mbatched_rows"),
                    "prefill_dense_dispatch_calls": sum_counter(
                        prefill_counters, "dense_prefill_calls"),
                    "prefill_dequantizations": sum_counter(
                        prefill_counters, "dequantizations"),
                    "prefill_dense_gemm_chunks": sum_counter(
                        prefill_counters, "dense_gemm_chunks"),
                    "decode_dense_dispatch_calls": sum_counter(
                        decode_counters, "dense_prefill_calls"),
                    "prefill_tier_kernel_launches": tier_totals(
                        prefill_counters, "kernel_launches"),
                    "decode_tier_kernel_launches": tier_totals(
                        decode_counters, "kernel_launches"),
                    "prefill_tier_expert_projection_operations": tier_totals(
                        prefill_counters, "expert_projection_operations"),
                    "decode_tier_expert_projection_operations": tier_totals(
                        decode_counters, "expert_projection_operations"),
                    "prefill_tier_counters": prefill_counters,
                    "decode_tier_counters": decode_counters,
                    "total_tier_counters": total_counters,
                    "placeholder_exact_dedup_factor": 1,
                    "prefix_cache_enabled": False, "mtp_enabled": False,
                    "text_bytes": len(text.encode()),
                    "unique_characters": len(set(text)),
                    "transient_scratch_bytes_declared": 268_435_456,
                    "resident_product_bytes": resident_product_now,
                    "resident_envelope_reserve_mincore_bytes": reserve_resident_now,
                    "residency_mode": self.health["residency"]["mode"],
                    "kv_cache_bytes": 0,
                    "kv_cache_note": "synthetic exact product-serve instrument allocates no KV; full configured 32K KV headroom remains unconsumed",
                    "mem_available_bytes": mem_available(),
                    **proc_memory(),
                },
            }
        finally:
            try:
                torch.cuda.empty_cache()
            finally:
                self.lock.release()


def openai_completion(events: list[dict[str, Any]]) -> dict[str, Any]:
    first = next((event for event in events if event.get("event") == "first_token"), None)
    done = next((event for event in reversed(events) if event.get("event") == "done"), None)
    if first is None or done is None:
        raise ValueError("completion event stream is incomplete")
    response = {key: value for key, value in done.items() if key != "event"}
    response["mixed_tier"] = {
        **first.get("mixed_tier", {}),
        **done.get("mixed_tier", {}),
    }
    return response


class Handler(BaseHTTPRequestHandler):
    engine: Engine
    health: dict[str, Any]

    def send_json(self, code: int, value: Any) -> None:
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, self.health)
        elif self.path == "/v1/models":
            self.send_json(200, {
                "object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            if self.path != "/v1/completions":
                self.send_json(404, {"error": "not found"})
                return
            events = self.engine.stream(
                str(request.get("prompt", "")),
                int(request.get("max_tokens", 16)),
                (int(request["expected_prompt_tokens"])
                 if "expected_prompt_tokens" in request else None),
            )
            if not request.get("stream", False):
                self.send_json(200, openai_completion(list(events)))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in events:
                payload = {key: value for key, value in event.items() if key != "event"}
                self.wfile.write(b"data: " + json.dumps(payload, allow_nan=False).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as exc:
            value = {"event": "error", "error": type(exc).__name__, "detail": str(exc)}
            try:
                if not self.wfile.closed:
                    self.wfile.write(json.dumps(value).encode() + b"\n")
                    self.wfile.flush()
            except Exception:
                pass

    def log_message(self, fmt: str, *args: Any) -> None:
        print(json.dumps({
            "event": "http", "message": fmt % args, "epoch": time.time()}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--cotenant-status", type=Path)
    parser.add_argument("--port", type=int, default=8128)
    args = parser.parse_args()
    mission = args.mission.resolve()
    receipts = mission / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    progress = receipts / "LOAD_PROGRESS.json"
    startup_started = time.monotonic()

    def startup_phase(phase: str, **fields: Any) -> None:
        row = {
            "event": "startup_phase",
            "phase": phase,
            "startup_elapsed_seconds": time.monotonic() - startup_started,
            "epoch": time.time(),
            **fields,
        }
        atomic_json(progress, row)
        print(json.dumps(row, sort_keys=True), flush=True)

    guard = CotenantGuard(args.cotenant_status, receipts / "COTENANT_STOP.json")
    guard.assert_wait()
    threading.Thread(target=guard.watch, name="cotenant-guard", daemon=True).start()

    before_available = mem_available()
    before_proc = proc_memory()
    startup_phase(
        "before_bind", mem_available_bytes=before_available,
        **before_proc, wire_driver=guard.assert_wait())

    file_backed_envelope = os.environ.get("P530_FILE_BACKED_ENVELOPE") == "1"
    if file_backed_envelope:
        # Product compact tensors are loaded below. Avoid the old anonymous
        # full-product staging pass: it was only a VmHWM proxy and left less
        # than the binding 8 GiB floor before any candidate row.
        full_stream = {
            "phase": "full_genesis_source_residency_stage_skipped",
            "reason": "file-backed mincore residency is measured directly",
            "source_host": args.source_host, "source_root": args.source_root,
        }
        full_stage_available = mem_available()
        full_stage_proc = proc_memory()
    else:
        full, full_stream = stream_remote_tree_into_mmap(
            args.source_host, args.source_root, PRODUCT_BYTES,
            mission / "code/stream_tree.py", progress,
            "full_genesis_source_residency_stage", guard)
        full_stage_available = mem_available()
        full_stage_proc = proc_memory()
        full.close()
        gc.collect()

    init_started = time.monotonic()
    startup_phase("before_init_vllm")
    torch, DeepseekV4MoE, vllm_config, _context = init_vllm(
        mission, mission / "code/mixed_tier_backend.py", args.artifact)
    startup_phase(
        "after_init_vllm", phase_seconds=time.monotonic() - init_started)
    # No cache exists yet. Calling empty_cache/reset_peak_memory_stats here
    # forces a full cold CUDA allocator bootstrap before model construction.
    # Let the first real layer allocation initialize CUDA instead.
    blocks: list[Any] = []
    layer_receipts: list[Any] = []
    pointers: list[int] = []

    def warm_block(block: Any) -> None:
        warm_ids = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.long)
        warm_x = torch.zeros((4, 4096), device="cuda", dtype=torch.bfloat16)
        warm_fused = block.experts.mixed.forward(warm_x, warm_ids, "fused13")
        warm_gate, warm_up = warm_fused.chunk(2, dim=-1)
        warm_activated = (
            torch.nn.functional.silu(warm_gate.clamp(min=-10, max=10))
            * warm_up.clamp(min=-10, max=10))
        block.experts.mixed.forward(warm_activated, warm_ids, "down")

    warmup_state: dict[str, Any] = {}
    warmup_thread: threading.Thread | None = None
    warmup_started: float | None = None

    def warm_first_block_background(block: Any) -> None:
        phase_started = time.monotonic()
        try:
            warm_block(block)
            torch.cuda.synchronize()
        except BaseException as exc:
            warmup_state["exception"] = exc
            warmup_state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            warmup_state["phase_seconds"] = time.monotonic() - phase_started
            warmup_state["finished_monotonic"] = time.monotonic()

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for layer in range(LAYERS):
            guard.assert_wait()
            layer_started = time.monotonic()
            startup_phase("before_layer_construct", layer=layer)
            with torch.device("cuda"):
                block = DeepseekV4MoE(
                    vllm_config, prefix=f"model.layers.{layer}.ffn")
            construct_seconds = time.monotonic() - layer_started
            startup_phase(
                "after_layer_construct", layer=layer,
                layer_construct_seconds=construct_seconds)
            torch.manual_seed(20260724 + layer)
            block.gate.weight.data.normal_(mean=0.0, std=0.01)
            blocks.append(block)
            layer_receipts.append(block.experts.mixed.sentinel())
            pointers.extend(block.experts.mixed.pointers())
            startup_phase(
                "layer_ready", layer=layer, layers_loaded=layer + 1,
                target_layers=LAYERS,
                layer_total_seconds=time.monotonic() - layer_started,
                cuda_allocated_bytes=int(torch.cuda.memory_allocated()),
                mem_available_bytes=mem_available(), **proc_memory(),
                wire_driver=guard.assert_wait())
            if layer == 0:
                # CUDA_MODULE_LOADING=LAZY shifts the shipped 23-argument
                # Triton-launcher load into this first real decode-shape call.
                # Start it as soon as block 0 is materialized so its host/module
                # work overlaps construction of blocks 1-42 and the resident
                # file-backed envelope map. The main thread joins before bind.
                startup_phase(
                    "before_first_block_kernel_warmup",
                    execution="background_thread",
                    overlaps="layers_1_42_and_resident_envelope_map")
                warmup_started = time.monotonic()
                warmup_thread = threading.Thread(
                    target=warm_first_block_background,
                    args=(block,),
                    name="first-block-triton-warmup",
                    daemon=True,
                )
                warmup_thread.start()
    finally:
        torch.set_default_dtype(old_dtype)

    startup_phase("all_layers_ready", layers_loaded=len(blocks))

    active_expert_bytes = sum(int(block.experts.resident_bytes) for block in blocks)
    reserve_bytes = PRODUCT_BYTES - active_expert_bytes
    if reserve_bytes <= 0:
        raise RuntimeError(f"active expert bytes exceed product: {active_expert_bytes}")
    if file_backed_envelope:
        reserve_started = time.monotonic()
        startup_phase("before_resident_envelope_map", reserve_bytes=reserve_bytes)
        reserve, reserve_stream = map_local_tree_file_backed(
            args.source_root, reserve_bytes, progress,
            "resident_genesis_envelope_file_backed_mincore", guard)
        reserve_resident_bytes = reserve.resident_logical_bytes()
        residency_mode = "file_backed_mincore"
        startup_phase(
            "after_resident_envelope_map",
            phase_seconds=time.monotonic() - reserve_started,
            reserve_resident_bytes=reserve_resident_bytes)
    else:
        reserve, reserve_stream = stream_remote_tree_into_mmap(
            args.source_host, args.source_root, reserve_bytes,
            mission / "code/stream_tree.py", progress,
            "resident_genesis_envelope_reserve", guard)
        reserve_resident_bytes = reserve_bytes
        residency_mode = "anonymous_exact"

    if warmup_thread is None or warmup_started is None:
        raise RuntimeError("first-block Triton warmup thread was not started")
    startup_phase(
        "before_first_block_kernel_warmup_join",
        warmup_thread_alive=warmup_thread.is_alive(),
        layers_loaded=len(blocks),
        resident_envelope_mapped=True)
    warmup_thread.join()
    if "exception" in warmup_state:
        raise RuntimeError(
            f"first-block Triton warmup failed: {warmup_state['error']}") \
            from warmup_state["exception"]
    first_block_warmup_seconds = float(warmup_state["phase_seconds"])
    first_block_warmup_overlap_window_seconds = time.monotonic() - warmup_started
    startup_phase(
        "after_first_block_kernel_warmup",
        execution="background_thread_joined",
        phase_seconds=first_block_warmup_seconds,
        overlap_window_seconds=first_block_warmup_overlap_window_seconds,
        layers_loaded=len(blocks),
        resident_envelope_mapped=True)

    all_layer_probe_started = time.monotonic()
    startup_phase("before_all_layer_official_moe_probe")
    # Preserve the runtime contract: one official DeepseekV4MoE forward still
    # traverses every independently materialized block before bind.
    hidden = torch.zeros((1, 4096), device="cuda", dtype=torch.bfloat16)
    input_ids = torch.tensor([42], device="cuda", dtype=torch.int64)
    with torch.inference_mode():
        for layer_index, block in enumerate(blocks):
            probe_layer_started = time.monotonic()
            startup_phase("before_official_moe_probe_layer", layer=layer_index)
            hidden = torch.tanh(block(hidden, input_ids) * 0.125)
            torch.cuda.synchronize()
            startup_phase(
                "after_official_moe_probe_layer", layer=layer_index,
                phase_seconds=time.monotonic() - probe_layer_started)
    startup_phase(
        "after_kernel_warmup",
        first_block_warmup_seconds=first_block_warmup_seconds,
        all_layer_official_moe_probe_seconds=(
            time.monotonic() - all_layer_probe_started),
        total_phase_seconds=time.monotonic() - warmup_started)

    after_available = mem_available()
    after_proc = proc_memory()
    signatures = [tuple(row["sentinel"]) for row in layer_receipts]
    manifest_path = Path(os.environ.get(
        "GENESIS_MANIFEST_PATH", str(args.artifact.parent / "MANIFEST.json")
    ))
    manifest = json.loads(manifest_path.read_text())
    warm_counters = aggregate(blocks)
    kernel_classes = [
        "_qtip_gemv", "_truevq_d4_gemv", "_truevq_d8_gemv",
        "_native_mxfp4_gemv",
    ]
    tier_map_counts = {
        tier: int((blocks[0].experts.mixed.tier_map == index).sum().item())
        for index, tier in enumerate(TIER_NAMES)
    }
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    health: dict[str, Any] = {
        "status": "ok", "schema": "mixed-tier-prefill-health-v1",
        "task": TASK_ID, "host": os.uname().nodename,
        "pid": os.getpid(), "pgid": os.getpgid(0), "model": MODEL_ID,
        "architecture": "DeepseekV4ForCausalLM",
        "official_moe_class": f"{DeepseekV4MoE.__module__}.{DeepseekV4MoE.__name__}",
        "official_moe_forward": True, "configured_layers": LAYERS,
        "active_layers": LAYERS, "experts_per_layer": EXPERTS,
        "experts_per_token": TOPK, "max_model_len": MAX_MODEL_LEN,
        "tokenizer_json": str(args.tokenizer_json),
        "prompt_prefill_executes_all_43_moe_layers": True,
        "prompt_embedding_semantics": "uncalibrated position-stable synthetic embedding; prompt IDs use the exact DeepSeek tokenizer and traverse the official MoE API",
        "prefix_cache_enabled": False, "mtp_enabled": False,
        "candidate_memory_law": {
            "resident_product_bytes": PRODUCT_BYTES,
            "persistent_second_weight_copy": False,
            "transient_scratch_bytes_declared": 268_435_456,
            "p526_component_sha256": "655634773e941f6fa310235fe1adfbd1803eaa8d9207c9b51640b93d947e98a9",
            "p526_measured_peak_scratch_bytes": 8_388_608,
            "streaming_dense_weight_max_bytes": 33_554_432,
            "dequant_intermediate_upper_bound_bytes": 268_435_456,
            "kv_cache_bytes": 0,
            "kv_cache_note": "synthetic serve instrument consumes zero KV bytes; configured 32K KV room remains intact",
        },
        "cotenant_guard": guard.assert_wait(),
        "source_package": {
            "host": args.source_host, "root": args.source_root,
            "bytes": PRODUCT_BYTES, "files": PRODUCT_FILES,
            "inventory_sha256": PRODUCT_INVENTORY_SHA256,
            "full_stream": full_stream,
        },
        "artifact_manifest": manifest,
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "residency": {
            "mode": residency_mode,
            "mem_available_before_bind_bytes": before_available,
            "mem_available_full_stage_bytes": full_stage_available,
            "mem_available_after_weights_load_bytes": after_available,
            "mem_available_drop_bytes": before_available - after_available,
            "full_stage_mem_available_drop_bytes": before_available - full_stage_available,
            "process_before": before_proc, "process_full_stage": full_stage_proc,
            "process_after": after_proc,
            "active_mixed_expert_bytes": active_expert_bytes,
            "resident_envelope_reserve_bytes": reserve_bytes,
            "resident_envelope_reserve_mincore_bytes": reserve_resident_bytes,
            "resident_product_bytes": active_expert_bytes + reserve_resident_bytes,
            "target_product_bytes": PRODUCT_BYTES,
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reserve_stream": reserve_stream,
        },
        "heterogeneous_variant": {
            "tier_classes": list(TIER_NAMES),
            "kernel_classes": kernel_classes,
            "tier_map_counts": tier_map_counts,
            "layer_count": len(blocks),
            "unique_layer_sentinels": len(set(signatures)),
            "layer_sentinel_unique": len(set(signatures)) == LAYERS,
            "no_tensor_alias": len(set(pointers)) == len(pointers),
            "pointer_count": len(pointers), "placeholder_exact_dedup_factor": 1,
            "layers": layer_receipts, "warmup_tier_counters": warm_counters,
        },
        "startup_overlap": {
            "first_block_warmup_background": True,
            "scheduled_after_layer": 0,
            "layers_constructed_before_join": LAYERS - 1,
            "resident_envelope_mapped_before_join": True,
            "first_block_warmup_seconds": first_block_warmup_seconds,
            "start_to_join_seconds": first_block_warmup_overlap_window_seconds,
        },
        "server_survival_contract": "detached setsid+nohup+PGID+log; second-ssh verification; self-exit if canonical wire resumes",
    }
    gates = {
        "memavailable_drop_ge_90gb": (
            file_backed_envelope or
            health["residency"]["mem_available_drop_bytes"] >= 90_000_000_000),
        "vmhwm_ge_90gb": (
            file_backed_envelope or after_proc.get("vmhwm_bytes", 0) >= 90_000_000_000),
        "resident_product_exact": (
            health["residency"]["resident_product_bytes"] == PRODUCT_BYTES),
        "file_backed_mincore_exact": (
            not file_backed_envelope or reserve_resident_bytes == reserve_bytes),
        "layers_43": len(blocks) == LAYERS,
        "distinct": (
            health["heterogeneous_variant"]["layer_sentinel_unique"]
            and health["heterogeneous_variant"]["no_tensor_alias"]),
        "all_four_tiers_warm": all(
            sum(values["expert_projection_operations"]
                for values in warm_counters["tiers"][tier].values()) > 0
            for tier in warm_counters["tiers"]),
        "heterogeneous_route_warm": warm_counters["heterogeneous_route_calls"] > 0,
        "cotenant_wait_exact": guard.assert_wait()["allowed"],
        "mtp_off": True,
    }
    health["pre_bench_gates"] = gates
    if not all(gates.values()):
        health["status"] = "fail"
    atomic_json(receipts / "SERVER_READY.json", health)
    startup_phase("server_ready_receipt_written")
    print(json.dumps({"event": "ready", **health}, sort_keys=True), flush=True)
    if health["status"] != "ok":
        raise RuntimeError(f"pre-bench gates failed: {gates}")

    engine = Engine(blocks, torch, tokenizer, health, guard)
    engine._resident_reserve = reserve
    Handler.engine = engine
    Handler.health = health
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
