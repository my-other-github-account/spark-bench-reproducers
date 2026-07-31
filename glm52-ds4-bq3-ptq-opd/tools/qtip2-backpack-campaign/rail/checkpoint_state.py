"""Fail-closed, weights-only checkpoint validation and overlay utilities."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch

FORMAT = "banana_smasher-basic-repair-v1"
MECHANISM = "physical-vq-codebooks-plus-all-rmsnorms-plus-attention-output-gains"
N_LAYERS = 43
N_CODEBOOK_PARAMS = 2_535_424
N_NORM_PARAMS = 446_080
N_OUTPUT_PARAMS = 43
CODEBOOK_KEY_SURFACE_SHA256 = "4a9b33fc4e86f3f7fb61bb571e58e7f41eb72ea190ecac6b8b3a5349f6f46ff4"
GAIN_CLAMP = 0.25
NATURAL_UPDATES = tuple(range(0, 65, 8))
# UPDATE_000 is the sealed no-repair physical wire and UPDATE_006 is the
# SHA-bound resume seed. Updates 7..30 are the resumed production dose.
SCORABLE_UPDATES = (0, 6, *range(7, 31))
EXPECTED_CHECKPOINT_CONFIG_CANONICAL_SHA256 = "18c7a30738a0111c7eb9daee13b11a47246f2228f1a248a96f2bca491d5cca4b"
EXPECTED_ASSIGNMENT_SHA256 = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
EXPECTED_BASE_HARNESS_SHA256 = "a337be2f538a7d3055e18abfff155f5b766e9079c7bb8db6567b7332554c11ee"
EXPECTED_CONSUMER_SHA256 = "e8aee936aa873607c7209a6a47bea8d41fe9979235d0dac1a7449cf7d2298dae"
EXPECTED_PHYSICAL_CODE76 = 0.05212973475888538
EXPECTED_CODE76_IDS = (
    2, 5, 6, 10, 22, 23, 29, 35, 56, 69, 73, 78, 84, 88, 89, 90,
    102, 108, 117, 124, 128, 142, 163, 168, 187, 197, 221, 228, 238,
    242, 251, 261, 284, 296, 298, 311, 318, 326, 334, 341, 342, 343,
    346, 349, 352, 357, 368, 374, 381, 383, 390, 394, 395, 396, 398,
    402, 403, 409, 411, 412, 421, 423, 428, 448, 449, 451, 463, 467,
    472, 475, 481, 488, 494, 496, 504, 509,
)
CONTAMINATED_IDS = (2, 5, 6, 10)
EXPECTED_CLEAN72_IDS = tuple(win for win in EXPECTED_CODE76_IDS if win not in CONTAMINATED_IDS)
EXPECTED_CLEAN72_SHA256 = "e830b42f9a82ffdd1a45a354478ae74153f44097e74fb14d596863e6bc78292d"
TIER_RE = re.compile(r"d(\d+)_k(\d+)\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def expected_norm_keys() -> set[str]:
    keys = {"model.norm"}
    for layer in range(N_LAYERS):
        prefix = f"model.layers.{layer}"
        keys.update({
            f"{prefix}.input_layernorm",
            f"{prefix}.post_attention_layernorm",
            f"{prefix}.self_attn.q_a_norm",
            f"{prefix}.self_attn.kv_norm",
        })
        if layer >= 2:
            keys.add(f"{prefix}.self_attn.compressor.kv_norm")
        if layer >= 2 and layer % 2 == 0:
            keys.add(f"{prefix}.self_attn.compressor.indexer.kv_norm")
    return keys


def expected_output_keys() -> set[str]:
    return {
        f"model.layers.{layer}.self_attn.o_b_proj.output_log_gain"
        for layer in range(N_LAYERS)
    }


def norm_native_to_checkpoint() -> dict[str, str]:
    result = {"norm.weight": "model.norm"}
    for layer in range(N_LAYERS):
        native = f"layers.{layer}."
        checkpoint = f"model.layers.{layer}."
        result.update({
            native + "attn_norm.weight": checkpoint + "input_layernorm",
            native + "ffn_norm.weight": checkpoint + "post_attention_layernorm",
            native + "attn.q_norm.weight": checkpoint + "self_attn.q_a_norm",
            native + "attn.kv_norm.weight": checkpoint + "self_attn.kv_norm",
        })
        if layer >= 2:
            result[native + "attn.compressor.norm.weight"] = checkpoint + "self_attn.compressor.kv_norm"
        if layer >= 2 and layer % 2 == 0:
            result[native + "attn.indexer.compressor.norm.weight"] = checkpoint + "self_attn.compressor.indexer.kv_norm"
    if set(result.values()) != expected_norm_keys():
        raise AssertionError("internal norm mapping surface mismatch")
    return result


def expected_codebook_keys_from_identity(identity: Mapping[str, Any]) -> dict[str, set[str]]:
    rows = identity.get("layers")
    if not isinstance(rows, list) or [row.get("layer") for row in rows] != list(range(N_LAYERS)):
        raise RuntimeError("checkpoint identity layer surface drift")
    expected: dict[str, set[str]] = {}
    for layer, row in enumerate(rows):
        keys: set[str] = set()
        files = row.get("files")
        if not isinstance(files, list):
            raise RuntimeError(f"checkpoint identity files missing L{layer}")
        for item in files:
            name = item.get("path")
            if not isinstance(name, str) or not name.endswith(".codebook.fp16.bin"):
                continue
            pieces = name.split(".")
            if len(pieces) != 5 or pieces[2:] != ["codebook", "fp16", "bin"]:
                raise RuntimeError(f"malformed codebook identity name L{layer}: {name}")
            tier, projection = pieces[0], pieces[1]
            match = TIER_RE.fullmatch(tier)
            if match is None:
                raise RuntimeError(f"unsupported checkpoint tier L{layer}: {tier}")
            if projection not in {"fused13", "down"}:
                raise RuntimeError(f"unsupported checkpoint projection L{layer}: {projection}")
            keys.add(f"{tier}__{'13' if projection == 'fused13' else '2'}")
        expected[f"L{layer}"] = keys
    return expected


def require_tensor(name: str, tensor: object, *, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"{name} is not a tensor")
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise RuntimeError(f"{name} must be CPU float32, got {tensor.device}/{tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != shape:
        raise RuntimeError(f"{name} shape drift {tuple(tensor.shape)} != {shape}")
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"{name} contains non-finite values")
    return tensor


def validate_state(payload: Mapping[str, Any], sidecar: Mapping[str, Any], update: int) -> dict[str, Any]:
    if update not in SCORABLE_UPDATES:
        raise RuntimeError(f"invalid P487 dose update {update}")
    if set(payload) != {
        "config", "format", "host", "identity", "mechanism", "microprobes",
        "next_update", "optimizer", "saved_unix", "scheduler", "state",
    }:
        raise RuntimeError(f"checkpoint top-level schema drift: {sorted(payload)}")
    exact = {
        "format": FORMAT,
        "mechanism": MECHANISM,
        "host": "compute-node-8",
        "next_update": update,
    }
    drift = {key: (payload.get(key), expected) for key, expected in exact.items() if payload.get(key) != expected}
    if drift:
        raise RuntimeError(f"checkpoint header drift: {drift}")
    if payload.get("identity") != sidecar.get("identity"):
        raise RuntimeError("checkpoint identity differs from immutable sidecar identity")
    config = payload.get("config")
    if not isinstance(config, dict) or canonical_json_sha256(config) != EXPECTED_CHECKPOINT_CONFIG_CANONICAL_SHA256:
        raise RuntimeError("checkpoint config canonical SHA drift")
    required_config = {
        "format": FORMAT,
        "task_id": "PUBLIC_TASK",
        "host": "compute-node-8",
        "assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
        "physical_code76_kld": EXPECTED_PHYSICAL_CODE76,
        "natural_updates": list(NATURAL_UPDATES),
        "steps": 64,
        "probe_every": 8,
        "terminal_selection": "minimum clean72 KLD across predeclared updates 8,16,...,64; ties earliest",
        "known_contaminated_eval_wins": list(CONTAMINATED_IDS),
        "code76_eval_wins": list(EXPECTED_CODE76_IDS),
        "clean72_eval_wins": list(EXPECTED_CLEAN72_IDS),
        "clean72_window_ids_sha256": EXPECTED_CLEAN72_SHA256,
    }
    config_drift = {key: (config.get(key), expected) for key, expected in required_config.items() if config.get(key) != expected}
    if config_drift:
        raise RuntimeError(f"checkpoint config contract drift: {config_drift}")
    identity = payload["identity"]
    if identity.get("base_harness_sha256") != EXPECTED_BASE_HARNESS_SHA256 or identity.get("consumer_sha256") != EXPECTED_CONSUMER_SHA256:
        raise RuntimeError("checkpoint producer identity drift")
    state = payload.get("state")
    if not isinstance(state, dict) or set(state) != {"codebooks", "norms", "outputs"}:
        raise RuntimeError("checkpoint state schema drift")
    codebooks = state["codebooks"]
    norms = state["norms"]
    outputs = state["outputs"]
    if not all(isinstance(value, dict) for value in (codebooks, norms, outputs)):
        raise RuntimeError("checkpoint state maps malformed")
    expected_codebooks = expected_codebook_keys_from_identity(identity)
    if set(codebooks) != set(expected_codebooks):
        raise RuntimeError("checkpoint codebook layer set drift")
    key_surface = {layer_key: sorted(saved) for layer_key, saved in codebooks.items()}
    if canonical_json_sha256(key_surface) != CODEBOOK_KEY_SURFACE_SHA256:
        raise RuntimeError("checkpoint codebook key surface SHA drift")
    codebook_params = 0
    for layer_key, identity_keys in expected_codebooks.items():
        saved = codebooks[layer_key]
        if not set(saved).issubset(identity_keys):
            raise RuntimeError(f"checkpoint codebook key absent from physical identity {layer_key}")
        for name, tensor in saved.items():
            match = TIER_RE.fullmatch(name.split("__", 1)[0])
            if match is None or name.rsplit("__", 1)[1] not in {"13", "2"}:
                raise RuntimeError(f"checkpoint codebook name malformed {layer_key}/{name}")
            d, k = map(int, match.groups())
            require_tensor(f"{layer_key}/{name}", tensor, shape=(k, d))
            codebook_params += tensor.numel()
    if codebook_params != N_CODEBOOK_PARAMS:
        raise RuntimeError(f"checkpoint codebook parameter count drift {codebook_params}")
    if set(norms) != expected_norm_keys():
        raise RuntimeError(f"checkpoint norm key drift missing={sorted(expected_norm_keys()-set(norms))[:8]} extra={sorted(set(norms)-expected_norm_keys())[:8]}")
    norm_params = 0
    for name, tensor in norms.items():
        require_tensor(name, tensor)
        norm_params += tensor.numel()
    if norm_params != N_NORM_PARAMS:
        raise RuntimeError(f"checkpoint norm parameter count drift {norm_params}")
    if set(outputs) != expected_output_keys():
        raise RuntimeError("checkpoint output-gain key drift")
    for name, tensor in outputs.items():
        require_tensor(name, tensor, shape=())
    if len(outputs) != N_OUTPUT_PARAMS:
        raise RuntimeError("checkpoint output-gain count drift")
    saved = float(payload.get("saved_unix", float("nan")))
    if not math.isfinite(saved) or saved <= 0:
        raise RuntimeError("checkpoint saved_unix invalid")
    return {
        "codebooks": codebooks,
        "norms": norms,
        "outputs": outputs,
        "config": config,
        "identity": identity,
        "saved_unix": saved,
    }


def load_checkpoint(path: Path, sidecar_path: Path, update: int) -> tuple[dict[str, Any], dict[str, Any]]:
    sidecar_raw = sidecar_path.read_bytes()
    sidecar = json.loads(sidecar_raw)
    if sidecar.get("schema") != "banana_smasher-basic-checkpoint-v1" or sidecar.get("update") != update:
        raise RuntimeError("checkpoint sidecar schema/update drift")
    expected_sha = sidecar.get("checkpoint_sha256")
    if sha256_file(path) != expected_sha:
        raise RuntimeError("checkpoint payload SHA differs from sidecar")
    # Security/compatibility contract: never unpickle arbitrary checkpoint code.
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload is not a mapping")
    state = validate_state(payload, sidecar, update)
    return state, sidecar


class OverlaySafeOpen:
    """Delegate safetensors reads, replacing only the 235 sealed RMSNorm wires."""

    def __init__(self, handle: object, norms: Mapping[str, torch.Tensor], seen: set[str]):
        self._handle = handle
        self._norms = norms
        self._seen = seen
        self._mapping = norm_native_to_checkpoint()

    def get_tensor(self, name: str) -> torch.Tensor:
        checkpoint_name = self._mapping.get(name)
        if checkpoint_name is None:
            return self._handle.get_tensor(name)
        value = self._norms[checkpoint_name]
        self._seen.add(checkpoint_name)
        return value

    def __enter__(self):
        if hasattr(self._handle, "__enter__"):
            self._handle.__enter__()
        return self

    def __exit__(self, *args):
        if hasattr(self._handle, "__exit__"):
            return self._handle.__exit__(*args)
        return None

    def __getattr__(self, name: str):
        return getattr(self._handle, name)
