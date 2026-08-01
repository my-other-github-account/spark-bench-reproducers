from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PackContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeContract:
    pack_root: Path
    layers: tuple[int, ...]
    repair_update: int
    repair_state: Path
    repair_manifest: Path
    tensor_layout_sha256: str


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise PackContractError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackContractError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_runtime_contract(root: str | Path) -> RuntimeContract:
    root = Path(root).resolve()
    cfg = _load(root / "config.json")
    q = cfg.get("quantization_config") or {}
    if q.get("quant_method") != "bs-mixed-tier":
        raise PackContractError("quant_method must be bs-mixed-tier")
    if q.get("format") != "bs-pack" or q.get("format_version") != 1:
        raise PackContractError("unsupported pack format")
    manifest_path = root / q.get("pack_manifest", "")
    manifest = _load(manifest_path)
    if manifest.get("source_format") != "p1016-true-c-native-planes-v1":
        raise PackContractError("source_format must be p1016-true-c-native-planes-v1")
    if manifest.get("quant_method") != "bs-mixed-tier":
        raise PackContractError("manifest quant_method mismatch")
    repair = manifest.get("repair") or {}
    if q.get("repair_format") != "bs-basic-repair-v1" or repair.get("format") != "bs-basic-repair-v1":
        raise PackContractError("repair format mismatch")
    rmanifest = root / repair.get("manifest", "")
    rstate = root / repair.get("state", "")
    if _sha(rmanifest) != repair.get("manifest_sha256"):
        raise PackContractError("repair manifest SHA mismatch")
    if _sha(rstate) != repair.get("state_sha256"):
        raise PackContractError("repair state SHA mismatch")
    nested = _load(rmanifest)
    dense = nested.get("dense_state") or {}
    if dense.get("path") != repair.get("state") or dense.get("sha256") != repair.get("state_sha256"):
        raise PackContractError("nested repair-state binding mismatch")
    layers = tuple(int(x) for x in manifest.get("layers", ()))
    if layers != tuple(range(43)) and layers != (0,):
        raise PackContractError("pack must cover exact runtime layers")
    return RuntimeContract(root, layers, int(repair.get("update")), rstate, rmanifest,
                           str(manifest.get("tensor_layout_sha256")))
