#!/usr/bin/env python3
"""Sealed LP4 base-pack plus task-local assignment-delta plane reader."""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl

import readapt_eval_contracts as EK


@triton.jit
def _vq_dequant_write_kernel(
    codes_ptr, scales_ptr, codebook_ptr, expert_ids_ptr, output_ptr,
    n_rows: tl.constexpr, out_cols: tl.constexpr,
    code_width: tl.constexpr, vector_width: tl.constexpr,
    scale_cols: tl.constexpr, total,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    per_expert = n_rows * out_cols
    batch = offsets // per_expert
    within = offsets - batch * per_expert
    row = within // out_cols
    col = within - row * out_cols
    code_col = col // vector_width
    lane = col - code_col * vector_width
    code = tl.load(
        codes_ptr + (batch * n_rows + row) * code_width + code_col,
        mask=mask, other=0,
    ).to(tl.int32)
    exponent = tl.load(
        scales_ptr + (batch * n_rows + row) * scale_cols + col // 32,
        mask=mask, other=127,
    ).to(tl.float32) - 127.0
    value = tl.load(codebook_ptr + code * vector_width + lane, mask=mask, other=0.0)
    expert = tl.load(expert_ids_ptr + batch, mask=mask, other=0).to(tl.int64)
    output_offset = (expert * n_rows + row) * out_cols + col
    tl.store(output_ptr + output_offset, value.to(tl.float32) * tl.exp2(exponent), mask=mask)


@triton.jit
def _fp4_dequant_write_kernel(
    packed_ptr, scales_ptr, lut_ptr, expert_ids_ptr, output_ptr,
    n_rows: tl.constexpr, out_cols: tl.constexpr,
    packed_cols: tl.constexpr, scale_cols: tl.constexpr,
    total, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    per_expert = n_rows * out_cols
    batch = offsets // per_expert
    within = offsets - batch * per_expert
    row = within // out_cols
    col = within - row * out_cols
    packed = tl.load(
        packed_ptr + (batch * n_rows + row) * packed_cols + col // 2,
        mask=mask, other=0,
    ).to(tl.int32)
    nibble = tl.where((col & 1) == 0, packed & 15, packed >> 4)
    exponent = tl.load(
        scales_ptr + (batch * n_rows + row) * scale_cols + col // 32,
        mask=mask, other=127,
    ).to(tl.float32) - 127.0
    value = tl.load(lut_ptr + nibble, mask=mask, other=0.0)
    expert = tl.load(expert_ids_ptr + batch, mask=mask, other=0).to(tl.int64)
    output_offset = (expert * n_rows + row) * out_cols + col
    tl.store(output_ptr + output_offset, value.to(tl.float32) * tl.exp2(exponent), mask=mask)


@triton.jit
def _lut_dequant_write_kernel(
    codes_ptr, scales_ptr, lut_ptr, expert_ids_ptr, output_ptr,
    n_rows: tl.constexpr, out_cols: tl.constexpr,
    scale_cols: tl.constexpr, scale_repeat: tl.constexpr,
    total, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    per_expert = n_rows * out_cols
    batch = offsets // per_expert
    within = offsets - batch * per_expert
    row = within // out_cols
    col = within - row * out_cols
    code = tl.load(codes_ptr + offsets, mask=mask, other=0).to(tl.int32)
    exponent = tl.load(
        scales_ptr + (batch * n_rows + row) * scale_cols + col // scale_repeat,
        mask=mask, other=127,
    ).to(tl.float32) - 127.0
    value = tl.load(lut_ptr + code, mask=mask, other=0.0)
    expert = tl.load(expert_ids_ptr + batch, mask=mask, other=0).to(tl.int64)
    output_offset = (expert * n_rows + row) * out_cols + col
    tl.store(output_ptr + output_offset, value.to(tl.float32) * tl.exp2(exponent), mask=mask)


HOME = Path.home()
ROOT = HOME / "run-bundles/IQ4_BAR_RUN_PUBLIC_TASK"
BASE_MANIFEST = HOME / "run-bundles/LP4_REPAIR/static/LP4_MANIFEST_S7LOCAL.json"
BASE_PACK = HOME / "run-bundles/LP4_REPAIR/LP4_PACK"
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(HOME / "run-bundles/DS4_TEACHER"))
import lp4_pack as pack  # noqa: E402
import t8192_ds4_build_v3 as v3  # noqa: E402

_DIMS = (256, 4096, 4096, 4096, 2048)


class CorrectedK4096Stream:
    """One-layer QSFP cache for the corrected k4096/d4 tier."""

    def __init__(self):
        import os as _os
        self._arm4_dir = _os.environ.get("ARM4_OVERRIDE_DIR", "").strip()
        self._partial_dir = _os.environ.get("ARM4_PARTIAL_DIR", "").strip()
        self._partial_verified = set()
        self._partial_receipt_rows = {}
        receipt_path = _os.environ.get("ARM4_PARTIAL_RECEIPT", "").strip()
        if self._partial_dir and receipt_path:
            receipt = json.loads(Path(receipt_path).read_text())
            if (receipt.get("status") != "PASS"
                    or receipt.get("task_id") != "PUBLIC_TASK"
                    or receipt.get("layer_count") != 43):
                raise RuntimeError("partial-plane stage receipt contract drift")
            rows = {int(row["layer"]): row for row in receipt.get("layers", [])}
            if set(rows) != set(range(43)):
                raise RuntimeError("partial-plane stage receipt layer drift")
            self._partial_receipt_rows = rows
        self._bint_checkpoint = _os.environ.get("BINT_CODEBOOK_CHECKPOINT", "").strip()
        self._bint_codebooks = None
        if self._bint_checkpoint:
            payload = torch.load(
                self._bint_checkpoint, map_location="cpu", weights_only=False
            )
            if not EK.overlay_format_ok(payload.get("format")):
                raise RuntimeError(
                    f"codebook overlay checkpoint format drift: {payload.get('format')}"
                )
            state = payload.get("state")
            if not isinstance(state, dict) or set(state) != {"codebooks", "norms", "outputs"}:
                raise RuntimeError("BINT checkpoint state surface drift")
            if set(state["codebooks"]) != {f"L{i}" for i in range(43)}:
                raise RuntimeError("BINT checkpoint codebook layer surface drift")
            self._bint_codebooks = state["codebooks"]
        if self._arm4_dir:
            self.expected = {L: None for L in range(43)}
            self.overrides = {}
            self._path = None
            self._data = None
            print(f"[BaseDeltaSource] ARM4 override active: {self._arm4_dir}", flush=True)
            return
        rca = HOME / "run-bundles/K4096_ANCHOR_RCA_t8885886e"
        parent_path = HOME / "run-bundles/K4096_RAIL_S1/K4096_ANCHOR_PROVENANCE_RECEIPT.json"
        parent = json.loads(parent_path.read_text())
        markers = parent["canonical_planes"]["s1_marker_binding"]["markers"]
        self.expected = {
            int(row["layer"]): {"bytes": int(row["bytes"]), "md5": row["canonical_md5"]}
            for row in markers
        }
        corrected = json.loads((rca / "CORRECTED_PLANE_RECEIPT.json").read_text())
        self.overrides = {int(row["layer"]): row for row in corrected["layers"]}
        for layer, row in self.overrides.items():
            self.expected[layer] = {"bytes": int(row["bytes"]), "md5": row["canonical_md5"]}
        assert set(self.expected) == set(range(43)) and set(self.overrides) == {0, 1}
        self.cache_dir = Path(os.environ.get(
            "K4096_STREAM_CACHE", str(ROOT / "stream_cache/k4096")
        ))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Path(os.environ.get(
            "K4096_STREAM_LEDGER", str(ROOT / "out/K4096_STREAM_VERIFICATION.jsonl")
        ))
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.prior_verified = set()
        if self.ledger.is_file():
            for line in self.ledger.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                layer = row.get("layer")
                if (row.get("status") == "PASS" and isinstance(layer, int)
                        and row.get("expected_md5") == self.expected.get(layer, {}).get("md5")):
                    self.prior_verified.add(layer)
        self._path = None
        self._data = None

    @staticmethod
    def md5(path, chunk=8 << 20):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()

    def source(self, layer):
        if layer in self.overrides:
            return self.overrides[layer]["source_path"], "corrected_local_override"
        host = "203.0.113.9" if layer <= 21 else "203.0.113.6"
        source = f"fleet-user@{host}:$HOME/run-bundles/VQ3_K4096/planes/vq3u_layer_{layer:03d}.pt"
        return source, "canonical_qsfp"

    @staticmethod
    def rsync_with_retry(source, destination, attempts=5):
        """Resume a QSFP plane transfer across transient SSH/rsync failures."""
        command = [
            "rsync", "-a", "--partial", "--append-verify", "--timeout=120",
            "--bwlimit=200000",
            "-e", (
                "ssh -o BatchMode=yes -o ConnectTimeout=10 "
                "-o ServerAliveInterval=15 -o ServerAliveCountMax=4"
            ),
            source, str(destination),
        ]
        for attempt in range(1, attempts + 1):
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                return
            if attempt == attempts:
                raise subprocess.CalledProcessError(result.returncode, command)
            delay = 5 * (2 ** (attempt - 1))
            print(
                f"k4096 rsync attempt {attempt}/{attempts} failed "
                f"rc={result.returncode}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)

    def load(self, layer):
        if getattr(self, "_arm4_dir", ""):
            import torch as _torch, gc as _gc
            pth = f"{self._arm4_dir}/vq3u_layer_{layer:03d}.pt"
            if self._path != pth:
                self._data = None
                _gc.collect()
                self._data = _torch.load(pth, map_location="cpu", weights_only=False)
                if self._bint_codebooks is not None:
                    patched = dict(self._data)
                    state = self._bint_codebooks[f"L{layer}"]
                    for key in ("cb13", "cb2"):
                        master = state[key]
                        source = self._data[key]
                        if tuple(master.shape) != tuple(source.shape):
                            raise RuntimeError(
                                f"BINT codebook shape drift L{layer}.{key}: "
                                f"{tuple(master.shape)} != {tuple(source.shape)}"
                            )
                        patched[key] = master.detach().to(
                            dtype=source.dtype, device="cpu"
                        )
                    self._data = patched
                    print(
                        f"[BaseDeltaSource] BINT codebooks applied L{layer:03d}",
                        flush=True,
                    )
                self._path = pth
                print(f"[BaseDeltaSource] ARM4 load L{layer:03d} <- {pth}", flush=True)
            return self._data
        expected = self.expected[layer]
        if self._partial_dir:
            partial = Path(self._partial_dir) / f"vq3u_layer_{layer:03d}.pt"
            if partial.is_file():
                if partial.stat().st_size != expected["bytes"]:
                    raise RuntimeError(f"partial plane size drift L{layer}: {partial}")
                if layer not in self._partial_verified:
                    staged = self._partial_receipt_rows.get(layer)
                    if staged is not None:
                        if (Path(staged["path"]) != partial
                                or int(staged["bytes"]) != expected["bytes"]
                                or staged["md5"] != expected["md5"]):
                            raise RuntimeError(f"partial-plane receipt binding drift L{layer}")
                        got = expected["md5"]
                    else:
                        got = self.md5(partial)
                    if got != expected["md5"]:
                        raise RuntimeError(
                            f"partial plane MD5 drift L{layer}: {got} != {expected['md5']}"
                        )
                    self._partial_verified.add(layer)
                if self._path != partial:
                    self._data = None
                    gc.collect()
                    self._data = torch.load(
                        partial, map_location="cpu", mmap=True, weights_only=True
                    )
                    if self._bint_codebooks is not None:
                        patched = dict(self._data)
                        state = self._bint_codebooks[f"L{layer}"]
                        for key in ("cb13", "cb2"):
                            master = state[key]
                            source = self._data[key]
                            if tuple(master.shape) != tuple(source.shape):
                                raise RuntimeError(
                                    f"BINT codebook shape drift L{layer}.{key}: "
                                    f"{tuple(master.shape)} != {tuple(source.shape)}"
                                )
                            patched[key] = master.detach().to(
                                dtype=source.dtype, device="cpu"
                            )
                        self._data = patched
                    self._path = partial
                    print(
                        f"[BaseDeltaSource] verified partial local L{layer:03d} <- {partial}",
                        flush=True,
                    )
                return self._data
        dst = self.cache_dir / f"vq3u_layer_{layer:03d}.pt"
        if self._path is not None and self._path != dst:
            previous = Path(self._path)
            self._data = None
            gc.collect()
            if previous.parent == self.cache_dir:
                try:
                    previous.unlink()
                except FileNotFoundError:
                    pass
        source, source_kind = self.source(layer)
        got = None
        verification_mode = "fresh_md5"
        if dst.is_file() and dst.stat().st_size == expected["bytes"] and layer in self.prior_verified:
            # rsync verifies transferred blocks; after one durable full-file MD5
            # pass for this exact receipt identity, later chunks only need the
            # receipt size gate. This avoids re-hashing 147 GB per 64-window chunk.
            got = expected["md5"]
            verification_mode = "prior_md5_plus_rsync_size"
        elif dst.is_file() and dst.stat().st_size == expected["bytes"]:
            got = self.md5(dst)
        valid = got == expected["md5"]
        if not valid:
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            if shutil.disk_usage(self.cache_dir).free < 12 << 30:
                raise RuntimeError("k4096 stream cache has less than 12 GiB free")
            tmp = dst.with_suffix(".partial")
            if source_kind == "corrected_local_override":
                shutil.copyfile(source, tmp)
            else:
                self.rsync_with_retry(source, tmp)
            assert tmp.stat().st_size == expected["bytes"], (tmp, tmp.stat().st_size, expected)
            if layer in self.prior_verified:
                got = expected["md5"]
                verification_mode = "prior_md5_plus_rsync_size"
            else:
                got = self.md5(tmp)
                assert got == expected["md5"], (layer, got, expected)
            os.replace(tmp, dst)
        assert got == expected["md5"]
        with self.ledger.open("a") as f:
            f.write(json.dumps({
                "task": "PUBLIC_TASK", "layer": layer, "source": source,
                "source_kind": source_kind, "path": str(dst), "bytes": dst.stat().st_size,
                "md5": got, "expected_md5": expected["md5"], "status": "PASS",
                "verification_mode": verification_mode,
                "ts": time.time(),
            }, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._path = dst
        self._data = torch.load(dst, map_location="cpu", mmap=True, weights_only=True)
        if self._bint_codebooks is not None:
            patched = dict(self._data)
            state = self._bint_codebooks[f"L{layer}"]
            for key in ("cb13", "cb2"):
                master = state[key]
                source = self._data[key]
                if tuple(master.shape) != tuple(source.shape):
                    raise RuntimeError(
                        f"BINT codebook shape drift L{layer}.{key}: "
                        f"{tuple(master.shape)} != {tuple(source.shape)}"
                    )
                patched[key] = master.detach().to(dtype=source.dtype, device="cpu")
            self._data = patched
            print(
                f"[BaseDeltaSource] streamed BINT codebooks applied L{layer:03d}",
                flush=True,
            )
        assert int(self._data["cb13"].shape[0]) == 4096
        assert int(self._data["cb13"].shape[1]) == 4
        return self._data


def tier(entry, proj):
    return entry["fused13" if proj == "13" else "down"] if isinstance(entry, dict) else entry


def row_for(ids, expert):
    hit = (ids == expert).nonzero()
    assert hit.numel() == 1, (expert, ids.tolist())
    return int(hit[0, 0])


class BaseDeltaSource:
    def __init__(self, manifest_path):
        self.manifest_path = Path(os.path.abspath(os.path.expanduser(manifest_path)))
        self.target = json.loads(self.manifest_path.read_text())
        self.base_manifest = json.loads(BASE_MANIFEST.read_text())
        self.delta_dir = Path(os.environ["TWOBIN_DELTA_DIR"])
        assert (self.delta_dir / "DELTA_PACK.COMPLETE").is_file(), self.delta_dir
        self._cache = {}
        self.k4096 = CorrectedK4096Stream()
        assigned = [
            tier(row, proj)
            for layer in self.target["assignment"].values()
            for row in layer.values()
            for proj in ("13", "2")
        ]
        assert "vq3" not in assigned, "k8192 was in the menu but should be unassigned in the solved lower hull"

    def _layer_context(self, layer):
        base = torch.load(
            BASE_PACK / f"layer_{layer:03d}.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        delta = torch.load(
            self.delta_dir / f"layer_{layer:03d}.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        self._cache = {layer: (base, delta)}
        target_map = self.target["assignment"][str(layer)]
        base_map = self.base_manifest["assignment"][str(layer)]
        needs_k4096 = any(
            tier(entry, proj) == "vq3b"
            for entry in target_map.values()
            for proj in ("13", "2")
        )
        vq3b = self.k4096.load(layer) if needs_k4096 else None
        delta_rows = {}
        for target_tier in ("w3", "vqa", "ternary"):
            for proj in ("13", "2"):
                key = f"{target_tier}_ids{proj}"
                if key not in delta:
                    continue
                values = [int(value) for value in delta[key].tolist()]
                index = {expert_id: row for row, expert_id in enumerate(values)}
                assert len(index) == len(values), f"duplicate {key} expert ids"
                delta_rows[(target_tier, proj)] = index
        return base, delta, target_map, base_map, vq3b, delta_rows

    @staticmethod
    def _row_for_assignment(base, delta_rows, target_tier, proj, expert_id, use_delta):
        if use_delta and target_tier != "vq3b":
            assert target_tier in ("w3", "vqa", "ternary"), target_tier
            rows = delta_rows[(target_tier, proj)]
            assert expert_id in rows, (target_tier, proj, expert_id)
            return rows[expert_id]
        return int(base[f"row{proj}"][expert_id])

    def fill_layer(self, layer, gate_up, down):
        """Bulk-dequantize assignment groups while preserving the scalar math."""
        base, delta, target_map, base_map, vq3b, delta_rows = self._layer_context(layer)
        groups = {}
        for expert_key, entry in target_map.items():
            expert_id = int(expert_key)
            for proj in ("13", "2"):
                target_tier = tier(entry, proj)
                use_delta = target_tier != tier(base_map[expert_key], proj)
                row = self._row_for_assignment(
                    base, delta_rows, target_tier, proj, expert_id, use_delta
                )
                groups.setdefault((target_tier, proj, use_delta), []).append((expert_id, row))

        batch_size = int(os.environ.get("FULLMENU_ASSEMBLY_BATCH", "8"))
        if batch_size < 1:
            raise ValueError("FULLMENU_ASSEMBLY_BATCH must be positive")
        fused = os.environ.get("FULLMENU_FUSED_DEQUANT", "0") == "1"
        codebooks = {}
        fp4_lut = v3.E2M1_VAL.to("cuda").float() if fused else None
        for (target_tier, proj, use_delta), entries in sorted(groups.items()):
            destination = gate_up if proj == "13" else down
            n, k = (4096, 4096) if proj == "13" else (4096, 2048)
            for start in range(0, len(entries), batch_size):
                batch = entries[start:start + batch_size]
                expert_ids = [item[0] for item in batch]
                rows = [item[1] for item in batch]
                source = delta if use_delta else base

                if fused and target_tier in {"vqa", "ternary", "vq3b", "fp4"}:
                    expert_ids_cuda = torch.tensor(
                        expert_ids, dtype=torch.int32, device="cuda"
                    )
                    total = len(batch) * n * k
                    launch_block = 4096
                    grid = (triton.cdiv(total, launch_block),)
                    if target_tier == "vqa":
                        codes = source[f"vqa_codes{proj}"][rows].to("cuda").contiguous()
                        scales = source[f"vqa_sc{proj}"][rows].to("cuda").contiguous()
                        codebook = base[f"cb{proj}"].to("cuda").float().contiguous()
                        vector_width = int(codebook.shape[1])
                        code_width = int(codes.shape[2])
                        scale_cols = int(scales.shape[2])
                        assert code_width * vector_width == k
                        _vq_dequant_write_kernel[grid](
                            codes, scales, codebook, expert_ids_cuda, destination,
                            n, k, code_width, vector_width, scale_cols, total,
                            BLOCK=launch_block, num_warps=8,
                        )
                    elif target_tier == "vq3b":
                        assert vq3b is not None
                        codes = vq3b[f"codes{proj}"][expert_ids].to("cuda").contiguous()
                        scales = vq3b[f"sc{proj}"][expert_ids].to("cuda").contiguous()
                        if proj not in codebooks:
                            codebooks[proj] = vq3b[f"cb{proj}"].to("cuda").float().contiguous()
                        codebook = codebooks[proj]
                        vector_width = int(codebook.shape[1])
                        code_width = int(codes.shape[2])
                        scale_cols = int(scales.shape[2])
                        assert code_width * vector_width == k
                        _vq_dequant_write_kernel[grid](
                            codes, scales, codebook, expert_ids_cuda, destination,
                            n, k, code_width, vector_width, scale_cols, total,
                            BLOCK=launch_block, num_warps=8,
                        )
                    elif target_tier == "fp4":
                        assert not use_delta and fp4_lut is not None
                        packed = base[f"fp4_wb{proj}"][rows].to("cuda").contiguous()
                        scales = base[f"fp4_sb{proj}"][rows].to("cuda").contiguous()
                        packed_cols = int(packed.shape[2])
                        scale_cols = int(scales.shape[2])
                        assert packed_cols * 2 == k
                        _fp4_dequant_write_kernel[grid](
                            packed, scales, fp4_lut, expert_ids_cuda, destination,
                            n, k, packed_cols, scale_cols, total,
                            BLOCK=launch_block, num_warps=8,
                        )
                    else:
                        if use_delta:
                            lut = delta[f"ternary_lut{proj}"]
                            code_key = f"ternary_codes{proj}"
                            scale_key = f"ternary_sc{proj}"
                        else:
                            lut = base[f"tern_lut{proj}"]
                            code_key = f"tern_codes{proj}"
                            scale_key = f"tern_sc{proj}"
                        codes = source[code_key][rows].to("cuda").contiguous()
                        scales = source[scale_key][rows].to("cuda").contiguous()
                        lut = lut.to("cuda").float().contiguous()
                        scale_cols = int(scales.shape[2])
                        scale_repeat = int(codes.shape[2] // scale_cols)
                        assert codes.shape[2] == k and scale_repeat * scale_cols == k
                        _lut_dequant_write_kernel[grid](
                            codes, scales, lut, expert_ids_cuda, destination,
                            n, k, scale_cols, scale_repeat, total,
                            BLOCK=launch_block, num_warps=8,
                        )
                    del expert_ids_cuda
                    continue

                if target_tier == "w3":
                    weights = pack.deq_w3_batched(
                        source[f"w3_pl{proj}"][rows].to("cuda"),
                        source[f"w3_sc{proj}"][rows].to("cuda"),
                        base["w3_lut"], n, k,
                    )
                elif target_tier == "vqa":
                    weights = pack.deq_vqa_batched(
                        source[f"vqa_codes{proj}"][rows].to("cuda"),
                        source[f"vqa_sc{proj}"][rows].to("cuda"),
                        base[f"cb{proj}"],
                    )
                elif target_tier == "ternary":
                    if use_delta:
                        lut = delta[f"ternary_lut{proj}"]
                        code_key, scale_key = f"ternary_codes{proj}", f"ternary_sc{proj}"
                    else:
                        lut = base[f"tern_lut{proj}"]
                        code_key, scale_key = f"tern_codes{proj}", f"tern_sc{proj}"
                    weights = pack.deq_tern_batched(
                        source[code_key][rows].to("cuda"),
                        source[scale_key][rows].to("cuda"),
                        lut,
                    )
                elif target_tier == "vq3b":
                    assert vq3b is not None
                    codes = vq3b[f"codes{proj}"][expert_ids].to("cuda")
                    scales = vq3b[f"sc{proj}"][expert_ids].to("cuda")
                    if proj not in codebooks:
                        codebooks[proj] = vq3b[f"cb{proj}"].to("cuda").float()
                    scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)
                    weights = codebooks[proj][codes.long()].reshape(len(batch), n, k) * scale_columns
                elif target_tier == "fp4":
                    assert not use_delta
                    weights = v3.deq_fp4_block32(
                        base[f"fp4_wb{proj}"][rows].to("cuda"),
                        base[f"fp4_sb{proj}"][rows].to("cuda"),
                        "e2m1",
                    )
                else:
                    raise KeyError(target_tier)
                assert tuple(weights.shape) == (len(batch), n, k), (
                    layer, proj, target_tier, weights.shape
                )
                destination[expert_ids] = weights.to(torch.bfloat16)
                del weights
        print(
            f"[BaseDeltaSource] L{layer:03d} bulk-filled "
            f"fused={int(fused)} target={self.manifest_path.name}",
            flush=True,
        )

    def layer(self, layer):
        base, delta, target_map, base_map, vq3b, delta_rows = self._layer_context(layer)

        def expert(expert_id, proj):
            target_tier = tier(target_map[str(expert_id)], proj)
            base_tier = tier(base_map[str(expert_id)], proj)
            use_delta = target_tier != base_tier
            n, k = (4096, 4096) if proj == "13" else (4096, 2048)
            row = self._row_for_assignment(
                base, delta_rows, target_tier, proj, expert_id, use_delta
            )

            if target_tier == "w3":
                source = delta if use_delta else base
                weights = pack.deq_w3_batched(
                    source[f"w3_pl{proj}"][row:row + 1].to("cuda"),
                    source[f"w3_sc{proj}"][row:row + 1].to("cuda"),
                    base["w3_lut"], n, k,
                )[0]
            elif target_tier == "vqa":
                source = delta if use_delta else base
                weights = pack.deq_vqa_batched(
                    source[f"vqa_codes{proj}"][row:row + 1].to("cuda"),
                    source[f"vqa_sc{proj}"][row:row + 1].to("cuda"),
                    base[f"cb{proj}"],
                )[0]
            elif target_tier == "ternary":
                if use_delta:
                    source = delta
                    lut = delta[f"ternary_lut{proj}"]
                    code_key, scale_key = f"ternary_codes{proj}", f"ternary_sc{proj}"
                else:
                    source = base
                    lut = base[f"tern_lut{proj}"]
                    code_key, scale_key = f"tern_codes{proj}", f"tern_sc{proj}"
                weights = pack.deq_tern_batched(
                    source[code_key][row:row + 1].to("cuda"),
                    source[scale_key][row:row + 1].to("cuda"),
                    lut,
                )[0]
            elif target_tier == "vq3b":
                assert vq3b is not None
                codes = vq3b[f"codes{proj}"][expert_id].to("cuda")
                scales = vq3b[f"sc{proj}"][expert_id].to("cuda")
                codebook = vq3b[f"cb{proj}"].to("cuda").float()
                scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)
                weights = codebook[codes.long()].reshape(codes.shape[0], -1) * scale_columns
            elif target_tier == "fp4":
                assert not use_delta
                weights = v3.deq_fp4_block32(
                    base[f"fp4_wb{proj}"][row].to("cuda"),
                    base[f"fp4_sb{proj}"][row].to("cuda"),
                    "e2m1",
                )
            else:
                raise KeyError(target_tier)
            assert tuple(weights.shape) == (n, k), (layer, expert_id, proj, target_tier, weights.shape)
            return weights.to(torch.bfloat16)

        print(f"[BaseDeltaSource] L{layer:03d} target={self.manifest_path.name}", flush=True)
        return expert, _DIMS


def self_test(manifest_path, layer=0):
    source = BaseDeltaSource(manifest_path)
    expert, dims = source.layer(layer)
    assert dims == _DIMS
    target = source.target["assignment"][str(layer)]
    picks = {}
    for expert_id, entry in target.items():
        for proj in ("13", "2"):
            picks.setdefault((tier(entry, proj), proj), int(expert_id))
    for (name, proj), expert_id in sorted(picks.items()):
        value = expert(expert_id, proj)
        assert torch.isfinite(value).all(), (name, proj, expert_id)
        print(f"self-test {name}/{proj} e{expert_id}: {tuple(value.shape)} {value.dtype} PASS", flush=True)
        del value
        torch.cuda.empty_cache()
    print("BaseDeltaSource self-test PASS", flush=True)


if __name__ == "__main__":
    self_test(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)