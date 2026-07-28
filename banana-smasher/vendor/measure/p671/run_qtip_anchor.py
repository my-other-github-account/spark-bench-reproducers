#!/usr/bin/env python3
"""P605R: six whole-layer QTIP-2 TRAIN-8 anchor scores on compute-node-6.

Adapts the pinned P539 canonical U030/physical-wire evaluator. Each candidate keeps
all 42 other layers byte-identical and replaces both expert projections for all
256 experts in exactly one target layer from the sealed QTIP aggregate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch

TASK = "PUBLIC_TASK"
ROOT = Path("$HOME/run-bundles/P605R_QTIP2_ANCHORS_C_PUBLIC_TASK_s6")
P500 = Path("$HOME/run-bundles/P500_operator_PUBLIC_TASK_s3")
CLAIM = Path("$HOME/HOST_CLAIM.json")
LAYERS = (27, 30, 34, 35, 38, 42)
WINDOWS = list(range(8))
MANIFEST = ROOT / "inputs/P532_SHARD_B_FINAL_8L_MANIFEST.json"
MANIFEST_SHA = "9f423bf751454a002c6afe7241bc1995cc580182a5964d8be1a21cd9971dbe9c"
STAGE_RECEIPT = ROOT / "receipts/QTIP_STAGE.json"
STAGE_ROOT = ROOT / "scratch/qtip_units"
BASELINE_BANK = ROOT / "inputs/C000_BASELINE_P539.json"
CLASS_BY_WIN = Path("$HOME/run-bundles/BINC_CODECLASS_PUBLIC_TASK/inputs/CLASS_BY_WIN.json")
QTIP_ROOT = Path("$HOME/run-bundles/P532_QTIP2_SHARD_B_PUBLIC_TASK_s6/inputs/qtip-canonical")
TLUT = Path("$HOME/run-bundles/P532_QTIP2_SHARD_B_PUBLIC_TASK_s6/inputs/tlut/PINNED_TLUT.pt")


def sha256(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_claim() -> tuple[bytes, dict[str, Any]]:
    raw = CLAIM.read_bytes()
    claim = json.loads(raw)
    exact = {
        "schema": "p605r-qtip2-anchor-host-claim-v1",
        "status": "CLAIMED",
        "host": "compute-node-6",
        "owner": TASK,
        "task": TASK,
        "task_id": TASK,
        "mission": str(ROOT),
    }
    drift = {key: [claim.get(key), expected] for key, expected in exact.items() if claim.get(key) != expected}
    if drift:
        raise RuntimeError(f"claim drift: {drift}")
    if time.time() > float(claim.get("lease_until_unix", 0)):
        raise RuntimeError("claim lease expired")
    return raw, claim


def configs(assignment: dict[str, Any]) -> dict[str, Any]:
    rows = []
    assignment_sha = canonical_sha(assignment)
    for index, layer in enumerate(LAYERS):
        rows.append({
            "index": index,
            "config_id": f"QTIP2_L{layer:03d}_ALL512",
            "kind": "CHANGED_STATE",
            "target_layer": layer,
            "representative_layers": [layer],
            "source_tier": "U030_DEPLOYED_BASE",
            "target_tier": "qtip2_L16_K2_V2",
            "changed_projection_units": 512,
            "changed_cell_set_sha256": canonical_sha([
                {"layer": layer, "expert": expert, "projection": projection}
                for expert in range(256) for projection in ("fused13", "down")
            ]),
            "assignment_sha256": assignment_sha,
            "assignment_state_sha256": assignment_sha,
            "measurement_scope": "TRAIN8_ONLY_NOT_HELDOUT",
        })
    return {"schema": "p605r-six-qtip-anchor-configs-v1", "status": "PASS", "configs": rows}


def setup_base() -> tuple[Any, dict[str, Any]]:
    base = load_module("p605r_p539_adapted", ROOT / "code/run_configs49_qtip_base.py")
    assignment_obj = json.loads((ROOT / "inputs/NOMINATED_ASSIGNMENT.json").read_text())
    config_obj = configs(assignment_obj["assignment"])
    config_path = ROOT / "inputs/QTIP_ANCHOR_CONFIGS.json"
    atomic_json(config_path, config_obj)
    base.TASK = TASK
    base.ROOT = ROOT
    base.P500 = P500
    base.CLAIM = CLAIM
    base.CONFIGS49 = config_path
    base.CONFIGS49_SHA = sha256(config_path)
    base.ASSIGNMENT = ROOT / "inputs/NOMINATED_ASSIGNMENT.json"
    base.U030 = ROOT / "inputs/UPDATE_030_283aa34e65912a9023c8157e42505b1bd75dff96d5577bacaff879eb7c8e1d9c.pt"
    base.SURFACES = ROOT / "inputs/EMPTY_QTIP_SURFACES.json"
    base.SURFACES_SHA = sha256(base.SURFACES)
    base.TARGET_CACHE = ROOT / "scratch/target_cache"
    base.RUNTIME_SOURCE_RECEIPTS = ROOT / "receipts/runtime_sources"
    base.RESULTS = ROOT / "results"
    base.LOGS = ROOT / "logs/configs49"
    base.RUN = ROOT / "run"
    base.current_claim = current_claim
    base.verify_pins = lambda: (config_obj, assignment_obj, {"layer_records": []})
    return base, config_obj


class QtipResolver:
    """Drop-in ExactTargetResolver for the adapted P539 evaluator."""

    def __init__(self, *, base: Any, assignment: dict[str, Any], state: dict[str, Any], surfaces: dict[str, Any], config: dict[str, Any]):
        self.base = base
        self.assignment = assignment
        self.state = state
        self.config = config
        self.projection_units = 0
        self.overlay_rows: list[dict[str, Any]] = []
        self._kernel = None
        self._expanded = None
        self._manifest_rows: dict[tuple[int, int, str], dict[str, Any]] | None = None

    @staticmethod
    def fwht(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        y = x.contiguous()
        h = 1
        while h < n:
            y = y.reshape(*y.shape[:-1], -1, 2, h)
            a = y[..., 0, :].clone()
            b = y[..., 1, :].clone()
            y = torch.cat((a + b, a - b), dim=-1).reshape(*y.shape[:-3], -1)
            h *= 2
        return y / math.sqrt(n)

    def initialize_decoder(self) -> None:
        if self._kernel is not None:
            return
        kernel_path = QTIP_ROOT / "lib/utils/kernel_decompress.py"
        module = load_module("p605r_qtip_kernel_decompress", kernel_path)
        self._kernel = module.decode_compressed
        tlut = torch.load(TLUT, map_location="cpu", mmap=True, weights_only=True)["tlut"].float().contiguous()
        tlut_sha = hashlib.sha256(tlut.numpy().tobytes()).hexdigest()
        if tlut_sha != "000c7985f6ac0cbece4a9850d3913102f9a6cf6ccb20cacf582d4fa95b569c19":
            raise RuntimeError("TLUT tensor hash drift")
        index = torch.arange(1 << 16, device="cuda")
        quadratic = (index + 1) * index
        sign_flip = 1 - ((quadratic >> 15) & 1) * 2
        lookup = (quadratic >> (16 - 9 - 1)) & 511
        expanded = tlut.to("cuda")[lookup]
        expanded[:, 0] *= sign_flip
        self._expanded = expanded.contiguous()
        manifest = json.loads(MANIFEST.read_text())
        rows = {}
        for row in manifest["unit_rows"]:
            identity = row["identity"]
            key = (int(identity["layer"]), int(identity["expert"]), str(identity["projection"]))
            rows[key] = row
        self._manifest_rows = rows

    def decode(self, payload: dict[str, Any]) -> torch.Tensor:
        assert self._kernel is not None and self._expanded is not None
        shape = tuple(map(int, payload["shape"]))
        raw = self._kernel(
            16, 9, 2, 1, shape[0], shape[1],
            payload["trellis"].to("cuda", non_blocking=False).reshape(-1), self._expanded,
        ) * payload["Wscale"].to("cuda")
        reconstructed = self.fwht(raw.T).T * payload["SV"].float().to("cuda")[:, None]
        reconstructed = self.fwht(reconstructed) * payload["SU"].float().to("cuda")
        return reconstructed

    def apply(self, layer: int, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        target = int(self.config["target_layer"])
        if layer != target:
            return
        self.initialize_decoder()
        assert self._manifest_rows is not None
        started = time.time()
        projection_rows = []
        for projection, destination in (("fused13", gate_up), ("down", down)):
            projection_started = time.time()
            for expert in range(256):
                current_claim()
                row = self._manifest_rows[(layer, expert, projection)]
                artifact = row["artifact"]
                path = STAGE_ROOT / f"L{layer:03d}" / Path(artifact["path"]).name
                if not path.is_file() or path.stat().st_size != int(artifact["bytes"]):
                    raise RuntimeError(f"QTIP staged byte drift: {path}")
                payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
                identity = payload.get("identity", {})
                if identity != {"layer": layer, "expert": expert, "projection": projection}:
                    raise RuntimeError(f"QTIP unit identity drift: {path}")
                if payload.get("geometry") != {"L": 16, "K": 2, "V": 2, "tlut_bits": 9, "decode_mode": "quantlut_sym", "td_x": 16, "td_y": 16}:
                    raise RuntimeError(f"QTIP geometry drift: {path}")
                reconstructed = self.decode(payload)
                if not bool(torch.isfinite(reconstructed).all()):
                    raise RuntimeError(f"nonfinite QTIP decode: {path}")
                destination[expert].copy_(reconstructed.to(torch.bfloat16))
                self.projection_units += 1
                del reconstructed, payload
                if (expert + 1) % 16 == 0:
                    torch.cuda.empty_cache()
                    print(f"[P605R QTIP] L{layer:03d} {projection} {expert + 1}/256", flush=True)
            projection_rows.append({
                "layer": layer, "projection": projection, "units": 256,
                "seconds": time.time() - projection_started,
            })
        self.overlay_rows.append({
            "layer": layer, "units": 512, "seconds": time.time() - started,
            "projections": projection_rows, "manifest_sha256": MANIFEST_SHA,
            "stage_receipt_sha256": sha256(STAGE_RECEIPT),
            "geometry": {"L": 16, "K": 2, "V": 2, "target_bpw": 2.0},
        })


def summarize_delta(candidate: dict[str, Any], layer: int) -> dict[str, Any]:
    baseline = json.loads(BASELINE_BANK.read_text())
    classes = json.loads(CLASS_BY_WIN.read_text())
    cand = {int(row["win"]): float(row["mean"]) for row in candidate["per_window"]}
    bank = {int(row["win"]): float(row["mean"]) for row in baseline["per_window"]}
    per_class = []
    for label in ("agentic", "code", "prose", "reasoning"):
        wins = [win for win in WINDOWS if classes[win] == label]
        deltas = [cand[win] - bank[win] for win in wins]
        mean = statistics.fmean(deltas)
        se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
        per_class.append({
            "schema": "p595-qtip2-anchor-damage-row-v1", "class": label,
            "eval_class": label, "layer": layer, "anchor": f"QTIP2_L{layer:03d}_ALL512",
            "candidate_minus_u030_train8_mean_kld": mean,
            "window_mean_se": se, "window_mean_ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "n_windows": len(wins), "window_ids": wins,
            "candidate_mean_kld": statistics.fmean(cand[win] for win in wins),
            "baseline_mean_kld": statistics.fmean(bank[win] for win in wins),
            "measurement_label": "MEASURED_PAIRED_TRAIN8_NOT_HELDOUT",
        })
    return {
        "schema": "p605r-qtip2-anchor-damage-v1", "status": "PASS_MEASURED_PAIRED_TRAIN8",
        "task_id": TASK, "host": os.uname().nodename, "layer": layer,
        "anchor": f"QTIP2_L{layer:03d}_ALL512", "rows": per_class,
        "candidate_receipt": candidate["runner"]["path"] if False else str(ROOT / f"results/QTIP2_L{layer:03d}_ALL512.json"),
        "candidate_receipt_sha256": sha256(ROOT / f"results/QTIP2_L{layer:03d}_ALL512.json"),
        "baseline_bank": str(BASELINE_BANK), "baseline_bank_sha256": sha256(BASELINE_BANK),
        "qtip_manifest": str(MANIFEST), "qtip_manifest_sha256": MANIFEST_SHA,
        "paired_same_windows": WINDOWS, "heldout512_touched": False,
        "created_unix": time.time(),
    }


def preflight() -> None:
    current_claim()
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("aggregate manifest hash drift")
    stage = json.loads(STAGE_RECEIPT.read_text())
    if stage.get("status") != "PASS" or stage.get("verified_units") != 6 * 512:
        raise RuntimeError("six-layer QTIP stage gate not passed")
    if not BASELINE_BANK.is_file() or json.loads(BASELINE_BANK.read_text()).get("config_id") != "C000_BASELINE":
        raise RuntimeError("baseline TRAIN-8 bank missing/drift")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True, choices=LAYERS)
    args = ap.parse_args()
    layer = args.layer
    preflight()
    base, config_obj = setup_base()
    base.ExactTargetResolver = QtipResolver
    index = LAYERS.index(layer)
    receipt = base.run_one(index, WINDOWS, "CONFIG")
    candidate = json.loads(receipt.read_text())
    damage = summarize_delta(candidate, layer)
    damage_path = ROOT / f"out/QTIP2_ANCHOR_L{layer:03d}/DAMAGE.json"
    atomic_json(damage_path, damage)
    print(json.dumps({
        "status": damage["status"], "layer": layer, "receipt": str(receipt),
        "receipt_sha256": sha256(receipt), "damage": str(damage_path),
        "damage_sha256": sha256(damage_path), "rows": damage["rows"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
