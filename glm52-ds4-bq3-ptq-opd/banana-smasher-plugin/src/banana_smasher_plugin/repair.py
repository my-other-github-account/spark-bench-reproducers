from __future__ import annotations

from typing import Any

from safetensors.torch import load_file

from .contract import RuntimeContract


def _resolve(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def apply_dense_norm_repair(module: Any, contract: RuntimeContract) -> tuple[str, ...]:
    state = load_file(str(contract.repair_state), device="cpu")
    applied: list[str] = []
    for key, value in state.items():
        if not key.startswith("norms/"):
            continue
        name = key[len("norms/"):]
        target = _resolve(module, name)
        target.weight.data.copy_(value.to(device=target.weight.device, dtype=target.weight.dtype))
        applied.append(name + ".weight")
    return tuple(sorted(applied))


def load_output_log_gains(contract: RuntimeContract) -> dict[str, float]:
    state = load_file(str(contract.repair_state), device="cpu")
    suffix = ".output_log_gain"
    return {key[len("outputs/"):-len(suffix)]: float(value.item())
            for key, value in state.items()
            if key.startswith("outputs/") and key.endswith(suffix)}
