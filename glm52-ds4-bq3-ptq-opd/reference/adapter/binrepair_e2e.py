#!/usr/bin/env python3
"""Public BINREPAIR compatibility layer for the sealed BQ3 Combo-V4 wire.

The 101.95 GB step-32 wire is supplied externally. This module preserves
BINREPAIR's model, corpus, activation-cache, loss, checkpoint, and forward API,
but dequants routed experts directly from that compact exact wire instead of
requiring the 147 GB all-k4096 source bank plus delta pack.

Only the d4/k4096 codebooks are trainable.  Packed codes, block-32 scales,
other VQ tiers, FP4 rows, routing, and all frozen model weights remain exact.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Import the sealed base API from a distinct, source-sealed path. Importing by
# module name here would resolve this compatibility wrapper again and recurse.
BASE_HARNESS = Path(os.path.expanduser(os.environ["BR_BASE_HARNESS"]))
_spec = importlib.util.spec_from_file_location("bint_binrepair_base", BASE_HARNESS)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load sealed BINREPAIR base: {BASE_HARNESS}")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
for _name, _value in vars(_base).items():
    if not _name.startswith("_"):
        globals()[_name] = _value

WIRE_DIR = Path(os.path.expanduser(os.environ["BR_WIRE_DIR"]))
PACK_MANIFEST = WIRE_DIR / "PACK_MANIFEST.json"
WIRE_RECEIPT = WIRE_DIR / "V4_STEP32_WIRE_RECEIPT.json"
_EXPECTED_STEP32 = "fae41d519193269aec4b2221c97a1dc00e0b00d3d66074d917a78489fac2149c"

if not (WIRE_DIR / "PACK_COMPLETE").exists():
    raise RuntimeError(f"sealed wire missing PACK_COMPLETE: {WIRE_DIR}")
_pack = json.loads(PACK_MANIFEST.read_text())
if _pack.get("checkpoint_sha256") != _EXPECTED_STEP32:
    raise RuntimeError("wire checkpoint identity drift")


def _npy(layer: int, projection: str, name: str):
    return np.load(
        WIRE_DIR / f"layer_{layer:03d}.vq{projection}.{name}.npy",
        mmap_mode="r",
    )


def _torch_copy(array, *, device=DEV):
    # Explicit copy avoids undefined writes through read-only NumPy memmaps.
    return torch.from_numpy(np.array(array, copy=True)).to(device)


def _unpack_indices(raw: torch.Tensor, bits: int, groups: int) -> torch.Tensor:
    """Unpack little-endian variable-bit indices from [N,row_bytes] uint8."""
    pos = torch.arange(groups, device=raw.device, dtype=torch.int64) * bits
    byte = torch.div(pos, 8, rounding_mode="floor")
    shift = pos - byte * 8
    padded = F.pad(raw, (0, 2))
    words = (
        padded[:, byte].to(torch.int32)
        | (padded[:, byte + 1].to(torch.int32) << 8)
        | (padded[:, byte + 2].to(torch.int32) << 16)
    )
    return ((words >> shift) & ((1 << bits) - 1)).long()


class WireExperts(nn.Module):
    """Exact compact-wire routed experts with k4096 codebook STE masters."""

    def __init__(self, layer: int, pilot: bool):
        super().__init__()
        self.L = layer
        self.pilot = pilot
        self.limit = 10.0
        self.act = F.silu
        self.meta = json.loads((WIRE_DIR / f"layer_{layer:03d}.meta.json").read_text())
        self.proj = {}
        self.tiers = {"13": [], "2": []}
        for which in ("13", "2"):
            state = {
                name: _npy(layer, which, name)
                for name in (
                    "codes", "scales", "codebooks", "code_offset",
                    "scale_offset", "code_row_bytes", "dimension", "bits",
                    "cb_offset",
                )
            }
            self.proj[which] = state
            self.tiers[which] = [
                "vq3b" if int(b) == 12 and int(d) == 4 else "frozen"
                for b, d in zip(state["bits"], state["dimension"])
            ]
        # Every layer keeps an fp32 master for the shared d4/k4096 codebook.
        # Offset 12288 = (2^8 + 2^10 + 2^11) * d4 elements.
        self.cb13 = nn.Parameter(
            _torch_copy(self.proj["13"]["codebooks"][12288:28672]).float().reshape(4096, 4)
        )
        self.cb2 = nn.Parameter(
            _torch_copy(self.proj["2"]["codebooks"][12288:28672]).float().reshape(4096, 4)
        )

    def _vq_parts(self, expert: int, which: str):
        state = self.proj[which]
        nout, width = (N13, K13) if which == "13" else (N2, K2)
        bits = int(state["bits"][expert])
        dim = int(state["dimension"][expert])
        row_bytes = int(state["code_row_bytes"][expert])
        co = int(state["code_offset"][expert])
        so = int(state["scale_offset"][expert])
        raw = _torch_copy(state["codes"][co:co + nout * row_bytes]).reshape(nout, row_bytes)
        scales = _torch_copy(state["scales"][so:so + nout * (width // 32)]).reshape(
            nout, width // 32
        )
        indices = _unpack_indices(raw, bits, width // dim)
        return bits, dim, indices, scales

    def _vq_weight(self, expert: int, which: str):
        bits, dim, indices, scales = self._vq_parts(expert, which)
        if bits == 12 and dim == 4:
            cb = self.cb13 if which == "13" else self.cb2
            return Vq3bDeqFn.apply(cb, indices, scales)
        state = self.proj[which]
        off = int(state["cb_offset"][expert])
        cb = _torch_copy(state["codebooks"][off:off + (1 << bits) * dim]).float().reshape(
            1 << bits, dim
        )
        values = cb[indices].reshape(indices.shape[0], -1)
        scale = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)
        return (values * scale).to(torch.bfloat16)

    def _fp4_weight(self, expert: int, which: str):
        slot = int(self.meta[f"slot{which}"][expert])
        if slot < 0:
            raise RuntimeError(f"invalid FP4 slot L{self.L} e{expert} {which}")
        planes = np.load(
            WIRE_DIR / f"layer_{self.L:03d}.fp4.planes{which}.npy", mmap_mode="r"
        )
        scales = np.load(
            WIRE_DIR / f"layer_{self.L:03d}.fp4.sc{which}.npy", mmap_mode="r"
        )
        weight = v3.deq_fp4_block32(
            _torch_copy(planes[slot]), _torch_copy(scales[slot]), "e2m1"
        )
        rows, cols = (N13, K13) if which == "13" else (N2, K2)
        return weight.reshape(rows, cols)

    def _weight(self, expert: int, which: str):
        kind = int(self.meta[f"kind{which}"][expert])
        if kind == 0:
            return self._vq_weight(expert, which)
        if kind == 2:
            return self._fp4_weight(expert, which)
        raise RuntimeError(f"unsupported wire kind L{self.L} e{expert} {which}: {kind}")

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e_ in hit:
            expert = int(e_[0])
            top_k_pos, token_idx = torch.where(mask[expert])
            xt = hidden_states[token_idx]
            w13 = self._weight(expert, "13")
            cur = F.linear(xt, w13)
            gate, up = cur.chunk(2, dim=-1)
            cur = self.act(gate.clamp(max=self.limit)) * up.clamp(
                min=-self.limit, max=self.limit
            )
            w2 = self._weight(expert, "2")
            cur = F.linear(cur, w2) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
            del xt, w13, w2, cur
        return final


# Name expected by the sealed Tailfix trainer.
K4096Experts = WireExperts


def codes_hash():
    """Compact-wire identity used by all resumable continuation checkpoints."""
    receipt = json.loads(WIRE_RECEIPT.read_text())
    return {
        "wire_pack_manifest_sha256": hashlib.sha256(PACK_MANIFEST.read_bytes()).hexdigest(),
        "wire_receipt_sha256": hashlib.sha256(WIRE_RECEIPT.read_bytes()).hexdigest(),
        "wire_checkpoint_sha256": _pack["checkpoint_sha256"],
        "wire_replay_sha256": receipt.get("replay_sha256"),
    }


def gradcheck(student):
    """Cheap exact-path gradient receipt on one live k4096 wire row."""
    layer = TRAINABLE[0]
    te = student.experts[layer]
    expert = next(e for e, tier in enumerate(te.tiers["13"]) if tier == "vq3b")
    _bits, _dim, indices, scales = te._vq_parts(expert, "13")
    cb = nn.Parameter(te.cb13.detach().clone())
    weight = Vq3bDeqFn.apply(cb, indices[:8], scales[:8])
    torch.manual_seed(0)
    probe = torch.randn_like(weight, dtype=torch.float32)
    (weight.float() * probe).sum().backward()
    finite = cb.grad is not None and bool(torch.isfinite(cb.grad).all())
    nonzero = cb.grad is not None and int(torch.count_nonzero(cb.grad)) > 0
    emit(event="wire_gradcheck", layer=layer, expert=expert, finite=finite, nonzero=nonzero)
    if not finite or not nonzero:
        raise RuntimeError("wire codebook gradient check failed")
