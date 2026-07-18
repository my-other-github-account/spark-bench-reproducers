#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build immutable exact IQ3 d4/d8 serving packs from campaign selections.

The builder flattens a target assignment manifest over a base+delta selected
pack, applies repaired arm4 codebooks to d4_k4096 without touching codes or
scales, and emits ``iq3-vq-wire-v1`` layers consumed by ``moe_w2_cubit``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iq3_vq_wire as wire  # noqa: E402


VQ_TIERS = {
    "vqa",
    "d4_k1024",
    "d4_k2048",
    "d4_k4096",
    "d8_k256",
    "d8_k512",
    "d8_k1024",
    "d8_k2048",
    "d8_k4096",
}


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def md5_file(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def summarize_payload(layers: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize only payload files covered by per-layer SHA256 manifests."""
    files = [entry for layer in layers for entry in layer["files"].values()]
    return {
        "payload_file_count": len(files),
        "payload_total_bytes": sum(int(entry["bytes"]) for entry in files),
    }


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_pt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def assignment_rows(manifest: dict[str, Any], layers: int, experts: int):
    """Normalize current and legacy campaign assignment schemas."""
    result = []
    for layer in range(layers):
        p13 = [None] * experts
        p2 = [None] * experts
        if "assignment" in manifest:
            layer_rows = manifest["assignment"].get(str(layer))
            if not isinstance(layer_rows, dict):
                raise ValueError(f"missing assignment layer {layer}")
            if set(layer_rows) != {str(expert) for expert in range(experts)}:
                raise ValueError(f"incomplete expert domain at layer {layer}")
            for expert in range(experts):
                row = layer_rows[str(expert)]
                if not isinstance(row, dict) or set(row) != {"fused13", "down"}:
                    raise ValueError(f"invalid assignment L{layer} E{expert}: {row!r}")
                p13[expert] = str(row["fused13"])
                p2[expert] = str(row["down"])
        elif "assignments" in manifest:
            for entry in manifest["assignments"][str(layer)]:
                expert_s, projection, tier = entry.split(":", 2)
                if projection not in ("fused13", "down"):
                    raise ValueError(
                        f"unknown projection L{layer} E{expert_s}: {projection}"
                    )
                expert = int(expert_s)
                if not 0 <= expert < experts:
                    raise ValueError(f"expert out of range L{layer}: {expert}")
                target = p13 if projection == "fused13" else p2
                if target[expert] is not None:
                    raise ValueError(
                        f"duplicate assignment L{layer} E{expert} {projection}"
                    )
                target[expert] = tier
        else:
            raise ValueError("manifest has neither 'assignment' nor 'assignments'")
        if any(t is None for t in p13 + p2):
            raise ValueError(f"incomplete assignment layer {layer}")
        result.append((p13, p2))
    return result


class SelectedSource:
    def __init__(
        self,
        delta_root: Path,
        base_root: Path,
        remote_roots: dict[Path, str] | None = None,
    ):
        self.roots = (delta_root, base_root)
        self.remote_roots = {
            root.resolve(): remote.rstrip("/")
            for root, remote in (remote_roots or {}).items()
            if remote
        }
        self.cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.fetched: set[Path] = set()

    @staticmethod
    def _relative_candidates(tier: str, layer: int) -> list[Path]:
        names = (f"layer_{layer:03d}.pt", f"layer_{layer}.pt")
        component = "static" if tier == "static" else tier
        dirs = (Path("selected") / component, Path(component))
        return [directory / name for directory in dirs for name in names]

    @staticmethod
    def _verify_receipt(path: Path, receipt_path: Path) -> dict[str, Any]:
        receipt = json.loads(receipt_path.read_text())
        if int(receipt["bytes"]) != path.stat().st_size:
            raise ValueError(f"{path}: source receipt byte mismatch")
        if receipt.get("md5") != md5_file(path):
            raise ValueError(f"{path}: source receipt MD5 mismatch")
        return receipt

    def _fetch_remote(self, root: Path, tier: str, layer: int) -> None:
        remote = self.remote_roots.get(root.resolve())
        if remote is None:
            return
        relative = self._relative_candidates(tier, layer)[0]
        destination = root / relative
        receipt = destination.with_suffix(".DONE.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--timeout=120",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=10",
                f"{remote}/{relative}",
                f"{remote}/{relative.with_suffix('.DONE.json')}",
                f"{destination.parent}/",
            ],
            check=True,
        )
        self._verify_receipt(destination, receipt)
        self.fetched.update((destination, receipt))

    def candidates(self, root: Path, tier: str, layer: int) -> list[Path]:
        candidates = [
            root / relative for relative in self._relative_candidates(tier, layer)
        ]
        if not any(path.is_file() for path in candidates):
            self._fetch_remote(root, tier, layer)
        return candidates

    def packs(self, tier: str, layer: int) -> list[dict[str, Any]]:
        key = (tier, layer)
        if key not in self.cache:
            found = []
            seen = set()
            for root in self.roots:
                for path in self.candidates(root, tier, layer):
                    if path in seen or not path.exists():
                        continue
                    seen.add(path)
                    pack = load_pt(path)
                    receipt_path = path.with_suffix(".DONE.json")
                    receipt = None
                    if receipt_path.is_file():
                        receipt = self._verify_receipt(path, receipt_path)
                    relative = path.relative_to(root)
                    # Keep generated metadata portable and publication-safe. Remote
                    # hostnames and absolute build paths are execution details, not
                    # artifact identity.
                    origin = f"{root.name}/{relative}"
                    if receipt is not None:
                        origin += (
                            f"#md5={receipt['md5']}#bytes={receipt['bytes']}"
                        )
                    pack["_path"] = origin
                    found.append(pack)
                    break
            if not found:
                raise FileNotFoundError(f"no {tier} layer {layer} in {self.roots}")
            self.cache[key] = found
        return self.cache[key]

    def row(self, tier: str, layer: int, expert: int, projection: str):
        if projection not in ("13", "2"):
            raise ValueError(f"unsupported projection: {projection}")
        static = tier in ("vqa", "fp4")
        pack_tier = "static" if static else tier
        for pack in self.packs(pack_tier, layer):
            ids = (
                pack.get(f"{tier}_ids{projection}")
                if static
                else pack.get(f"expert_ids{projection}")
            )
            if ids is None:
                ids = pack.get("experts", pack.get("expert_ids"))
            if ids is None:
                raise ValueError(
                    f"{pack['_path']}: missing ids for {tier}/{projection}"
                )
            id_list = [int(value) for value in ids.tolist()]
            try:
                row = id_list.index(expert)
            except ValueError:
                continue
            if tier == "vqa":
                return (
                    pack[f"vqa_codes{projection}"][row],
                    pack[f"vqa_sc{projection}"][row],
                    pack[f"cb{projection}"],
                    pack["_path"],
                )
            if tier == "fp4":
                return (
                    pack[f"fp4_wb{projection}"][row],
                    pack[f"fp4_sb{projection}"][row],
                    None,
                    pack["_path"],
                )
            return (
                pack[f"codes{projection}"][row],
                pack[f"sc{projection}"][row],
                pack[f"cb{projection}"],
                pack["_path"],
            )
        raise KeyError(f"missing L{layer} E{expert} {projection} {tier}")

    def clear(self):
        self.cache.clear()
        gc.collect()
        for path in sorted(self.fetched, reverse=True):
            path.unlink(missing_ok=True)
        self.fetched.clear()


def pack_fp4_fragment_major(codes: torch.Tensor) -> torch.Tensor:
    """[N,K] e2m1 nibbles to the existing moe_w4 fragment-major layout."""
    outputs, width = codes.shape
    if outputs % 16 or width % 64:
        raise ValueError(f"FP4 matrix shape must be N%16=K%64=0: {codes.shape}")
    values = codes.view(outputs // 16, 2, 8, width // 64, 2, 2, 4, 4)
    values = values.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    values = values.view(-1, 2).to(torch.int16)
    return (values[:, 0] | (values[:, 1] << 4)).to(torch.uint8).flatten()


def pack_scale_fragment(scales: torch.Tensor) -> torch.Tensor:
    outputs, groups = scales.shape
    if outputs % 16:
        raise ValueError(f"scale outputs must be divisible by 16: {scales.shape}")
    return scales.view(outputs // 16, 16, groups).transpose(1, 2).contiguous().flatten()


def fp4_fragment(codes: torch.Tensor, scales: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert campaign row-major FP4 nibbles to moe_w4 fragment-major bytes."""
    codes = codes.to(torch.uint8).contiguous()
    scales = scales.to(torch.uint8).contiguous()
    outputs, half_k = codes.shape
    q = torch.empty(outputs, half_k * 2, dtype=torch.uint8, device=codes.device)
    q[:, 0::2] = codes & 0x0F
    q[:, 1::2] = codes >> 4

    packed_weights = pack_fp4_fragment_major(q).cpu().numpy()
    packed_scales = pack_scale_fragment(scales).cpu().numpy()
    return packed_weights, packed_scales


def find_arm4_layer(root: Path, layer: int) -> Path:
    """Resolve the sealed VQ3U export name and the legacy helper names."""
    for name in (
        f"vq3u_layer_{layer:03d}.pt",
        f"layer_{layer:03d}.pt",
        f"layer_{layer}.pt",
    ):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"arm4 layer {layer} missing in {root}")


def expand_vq_metadata(packed: dict[str, Any], experts: list[int], total: int):
    if not experts:
        raise ValueError("each projection must contain at least one VQ expert")
    for key in (
        "code_offset",
        "scale_offset",
        "code_row_bytes",
        "dimension",
        "bits",
        "cb_offset",
    ):
        source = packed[key]
        expanded = np.empty(total, dtype=source.dtype)
        expanded[:] = source[0]
        expanded[np.asarray(experts, dtype=np.int64)] = source
        packed[key] = expanded
    return packed


def save_vq(prefix: Path, which: str, packed: dict[str, Any]) -> list[Path]:
    outputs = []
    for key in (
        "codes",
        "scales",
        "codebooks",
        "code_offset",
        "scale_offset",
        "code_row_bytes",
        "dimension",
        "bits",
        "cb_offset",
    ):
        path = Path(f"{prefix}.vq{which}.{key}.npy")
        atomic_save_npy(path, packed[key])
        outputs.append(path)
    return outputs


def build_projection(
    source: SelectedSource,
    assignments: list[str],
    layer: int,
    projection: str,
    width: int,
    arm4: dict[str, Any] | None,
):
    experts = len(assignments)
    vq_experts = [expert for expert, tier in enumerate(assignments) if tier != "fp4"]
    vq_assignments = [assignments[expert] for expert in vq_experts]
    if any(tier not in VQ_TIERS for tier in vq_assignments):
        bad = sorted(set(vq_assignments) - VQ_TIERS)
        raise ValueError(f"unsupported serving tiers: {bad}")
    grouped: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    source_paths = set()
    for expert in vq_experts:
        tier = assignments[expert]
        codes, scales, codebook, path = source.row(tier, layer, expert, projection)
        grouped.setdefault(tier, []).append((codes, scales, codebook))
        source_paths.add(path)
    sources = {}
    source_rows = {}
    overrides = {}
    for tier, rows in grouped.items():
        for _, _, candidate_cb in rows[1:]:
            if not torch.equal(candidate_cb, rows[0][2]):
                raise ValueError(
                    f"L{layer} {projection} {tier}: source codebooks differ"
                )
        sources[tier] = {
            "codes": (
                torch.stack([row[0] for row in rows])
                .cpu().numpy().astype(np.uint16)
            ),
            "scales": torch.stack([row[1] for row in rows]).cpu().numpy(),
            "codebook": rows[0][2].cpu().numpy(),
        }
        source_rows[tier] = np.arange(len(rows), dtype=np.int32)
        if tier == "d4_k4096" and arm4 is not None:
            overrides[tier] = arm4[f"cb{projection}"].cpu().numpy()
    packed = wire.pack_vq_projection(
        vq_assignments,
        source_rows,
        sources,
        width,
        codebook_overrides=overrides,
    )
    expand_vq_metadata(packed, vq_experts, experts)

    fp4_experts = [expert for expert, tier in enumerate(assignments) if tier == "fp4"]
    fp4_planes = []
    fp4_scales = []
    for expert in fp4_experts:
        codes, scales, _, path = source.row("fp4", layer, expert, projection)
        planes, scale_bytes = fp4_fragment(codes, scales)
        fp4_planes.append(planes.reshape(-1))
        fp4_scales.append(scale_bytes.reshape(-1))
        source_paths.add(path)
    kinds = [2 if tier == "fp4" else 0 for tier in assignments]
    slots = [0] * experts
    for slot, expert in enumerate(fp4_experts):
        slots[expert] = slot
    return packed, kinds, slots, fp4_planes, fp4_scales, sorted(source_paths)


def build_layer(
    layer: int,
    assignments: tuple[list[str], list[str]],
    source: SelectedSource,
    arm4_root: Path | None,
    output: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    prefix = output / f"layer_{layer:03d}"
    if Path(f"{prefix}.meta.json").exists():
        return json.loads(Path(f"{prefix}.meta.json").read_text())
    arm4 = None
    arm4_path = None
    if arm4_root:
        arm4_path = find_arm4_layer(arm4_root, layer)
        arm4 = load_pt(arm4_path)

    dimensions = {"E": len(assignments[0]), "N13": 4096, "K13": 4096,
                  "N2": 4096, "K2": 2048}
    files = []
    layer_meta: dict[str, Any] = {
        "format": "iq3-vq-wire-v1",
        **dimensions,
        "layer": layer,
        "source_manifest_sha256": manifest_sha256,
        "created_unix": int(time.time()),
    }
    if arm4_path is not None:
        layer_meta["arm4_source"] = {
            "file": arm4_path.name,
            "sha256": sha256_file(arm4_path),
        }
    source_paths = set()
    for which, tiers, width in (
        ("13", assignments[0], dimensions["K13"]),
        ("2", assignments[1], dimensions["K2"]),
    ):
        packed, kinds, slots, fp4_planes, fp4_scales, used = build_projection(
            source, tiers, layer, which, width, arm4
        )
        files.extend(save_vq(prefix, which, packed))
        layer_meta[f"kind{which}"] = kinds
        layer_meta[f"slot{which}"] = slots
        source_paths.update(used)
        if fp4_planes:
            plane_path = Path(f"{prefix}.fp4.planes{which}.npy")
            scale_path = Path(f"{prefix}.fp4.sc{which}.npy")
            atomic_save_npy(plane_path, np.stack(fp4_planes))
            atomic_save_npy(scale_path, np.stack(fp4_scales))
            files.extend((plane_path, scale_path))
    layer_meta["source_paths"] = sorted(source_paths)
    layer_meta["files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    }
    meta_path = Path(f"{prefix}.meta.json")
    tmp = meta_path.with_suffix(meta_path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(layer_meta, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, meta_path)
    source.clear()
    return layer_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--delta-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument(
        "--delta-remote",
        help="Optional rsync source host:/absolute/root for layer-streamed delta rows",
    )
    parser.add_argument(
        "--base-remote",
        help="Optional rsync source host:/absolute/root for layer-streamed base rows",
    )
    parser.add_argument("--arm4-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=43)
    parser.add_argument("--experts", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_raw = args.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    source_manifest_path = args.output / "SOURCE_MANIFEST.json"
    source_manifest_tmp = source_manifest_path.with_suffix(
        source_manifest_path.suffix + f".tmp.{os.getpid()}"
    )
    source_manifest_tmp.write_bytes(manifest_raw)
    os.replace(source_manifest_tmp, source_manifest_path)
    assignments = assignment_rows(manifest, args.layers, args.experts)
    source = SelectedSource(
        args.delta_root,
        args.base_root,
        remote_roots={
            args.delta_root: args.delta_remote,
            args.base_root: args.base_remote,
        },
    )
    layers = []
    layer_metas = []
    for layer in range(args.layers):
        started = time.time()
        meta = build_layer(
            layer,
            assignments[layer],
            source,
            args.arm4_root,
            args.output,
            manifest_sha256,
        )
        layer_metas.append(meta)
        layers.append({
            "layer": layer,
            "meta_sha256": sha256_file(args.output / f"layer_{layer:03d}.meta.json"),
            "seconds": round(time.time() - started, 3),
            "files": len(meta["files"]),
        })
        print(json.dumps(layers[-1]), flush=True)

    receipt_names = (
        "SELECTED_PACKS.COMPLETE",
        "DELTA_PACK_MANIFEST.json",
        "PACK_MANIFEST.json",
        "EXPORT_META.json",
        "ARM4_EXPORT_META.json",
        "EXPORT_REPORT.json",
    )
    source_receipts = {}
    for root in (args.delta_root, args.base_root, args.arm4_root):
        if root is None:
            continue
        for name in receipt_names:
            path = root / name
            if path.exists():
                source_receipts[f"{root.name}/{name}"] = sha256_file(path)

    pack_manifest = {
        "format": "iq3-vq-wire-pack-v1",
        "source_manifest": args.manifest.name,
        "source_manifest_copy": source_manifest_path.name,
        "source_manifest_sha256": manifest_sha256,
        "delta_root": args.delta_root.name,
        "base_root": args.base_root.name,
        "arm4_root": args.arm4_root.name if args.arm4_root else None,
        "source_receipts": source_receipts,
        "layers": layers,
        "created_unix": int(time.time()),
    }
    pack_manifest.update(summarize_payload(layer_metas))
    manifest_path = args.output / "PACK_MANIFEST.json"
    manifest_path.write_text(json.dumps(pack_manifest, sort_keys=True, indent=2) + "\n")
    (args.output / "PACK_MANIFEST.json.sha256").write_text(
        f"{sha256_file(manifest_path)}  PACK_MANIFEST.json\n"
    )
    complete_path = args.output / "PACK_COMPLETE"
    complete_tmp = complete_path.with_suffix(f".tmp.{os.getpid()}")
    complete_tmp.write_text(json.dumps({
        "format": "iq3-vq-wire-pack-complete-v1",
        "pack_manifest_sha256": sha256_file(manifest_path),
    }, sort_keys=True) + "\n")
    os.replace(complete_tmp, complete_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
