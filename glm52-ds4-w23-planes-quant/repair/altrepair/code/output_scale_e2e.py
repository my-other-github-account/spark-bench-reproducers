#!/usr/bin/env python3
"""Orthogonal BINREPAIR track: per-expert output-gain repair.

This wraps the proven ``binrepair_e2e.py`` artifact loader/loss/probe harness but
keeps every quantized weight and codebook frozen.  Each selected MoE layer gets
one learned log-gain for every expert's fused13 projection and one for its down
projection (256 * 2 scalars/layer).  With all 43 layers selected this is 22,016
trainable scalars.

The gains initialize to exactly 1.0, so step-0 logits and KLD must reproduce the
same ledger baseline as the frozen BINREPAIR student.  Checkpoints contain only
the gain side table and optimizer state; no quantized bytes are rewritten.

Required environment is the same as binrepair_e2e.py.  Additional variables:
  ALT_BINREPAIR_BASE  path to the proven base harness
  ALT_GAIN_CLAMP      absolute clamp on log-gain (default 0.25)
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


BASE_PATH = Path(os.path.expanduser(os.environ.get(
    "ALT_BINREPAIR_BASE", str(Path(__file__).with_name("binrepair_e2e.py"))
)))
GAIN_CLAMP = float(os.environ.get("ALT_GAIN_CLAMP", "0.25"))
BASELINE_SEED = os.environ.get("ALT_BASELINE_CKPT", "")
if not BASE_PATH.is_file():
    raise FileNotFoundError(f"ALT_BINREPAIR_BASE not found: {BASE_PATH}")
if not (0.0 < GAIN_CLAMP <= math.log(2.0)):
    raise ValueError(f"ALT_GAIN_CLAMP must be in (0, log(2)], got {GAIN_CLAMP}")

_spec = importlib.util.spec_from_file_location("binrepair_base", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load base harness: {BASE_PATH}")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

MECHANISM = "per-expert-output-log-gain"
FORMAT = "altrepair-output-scale-v1"


def apply_log_gain(x: torch.Tensor, log_gain: torch.Tensor) -> torch.Tensor:
    """Apply a positive bounded gain while preserving exact identity at zero."""
    gain = torch.exp(log_gain.clamp(-GAIN_CLAMP, GAIN_CLAMP)).to(x.dtype)
    return x * gain


class OutputScaleExperts(B.K4096Experts):
    """Frozen artifact experts with two trainable output gains per expert."""

    def __init__(self, layer: int, pilot: bool):
        # False is intentional: vq3b codebooks remain frozen and dense-dequant
        # through the exact sealed reader path.
        super().__init__(layer, False)
        self.pilot = pilot
        if pilot:
            # Keep the cb13/cb2 attribute names because the base harness builds
            # its optimizer from those two attributes.  In this wrapper they
            # are explicitly log-gain vectors, not codebooks.
            self.cb13 = nn.Parameter(torch.zeros(
                B.E, dtype=torch.float32, device=B.DEV))
            self.cb2 = nn.Parameter(torch.zeros(
                B.E, dtype=torch.float32, device=B.DEV))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=B.E).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

        # All artifact weights, including vq3b, stay on the frozen dense path.
        gu, dn = self._dense(skip_vq3b=False)
        for e_ in hit:
            e = int(e_[0])
            top_k_pos, token_idx = torch.where(mask[e])
            xt = hidden_states[token_idx]

            cur = F.linear(xt, gu[e])
            if self.pilot:
                cur = apply_log_gain(cur, self.cb13[e])
            gate, up = cur.chunk(2, dim=-1)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            cur = self.act(gate) * up

            cur = F.linear(cur, dn[e])
            if self.pilot:
                cur = apply_log_gain(cur, self.cb2[e])
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
            del cur, xt

        self._evict_planes()
        return final


def instant_artifact_identity():
    """Seconds-only identity gate; ledger parity is the real integrity check."""
    plane_sizes = {}
    delta_sizes = {}
    for layer in B.TRAINABLE:
        plane = B.VQ3B_DIR / f"vq3u_layer_{layer:03d}.pt"
        delta = B.DELTA_DIR / f"layer_{layer:03d}.pt"
        plane_sizes[str(layer)] = plane.stat().st_size
        delta_sizes[str(layer)] = delta.stat().st_size

    # One header/meta load follows the campaign's instant-check policy.  Do not
    # content-hash the 43 large planes absent a concrete integrity suspicion.
    sample_layer = B.TRAINABLE[0]
    sample = torch.load(
        B.VQ3B_DIR / f"vq3u_layer_{sample_layer:03d}.pt",
        map_location="cpu", mmap=True, weights_only=True)
    sample_header = {
        key: {"shape": list(sample[key].shape), "dtype": str(sample[key].dtype)}
        for key in ("cb13", "cb2", "codes13", "codes2", "sc13", "sc2")
    }
    return {
        "policy": "exists-size-plus-one-header; ledger-parity-is-integrity-gate",
        "plane_sizes": plane_sizes,
        "delta_sizes": delta_sizes,
        "sample_layer": sample_layer,
        "sample_header": sample_header,
    }


def gradcheck(student):
    """Finite-difference check of the exact gain helper and parameter wiring."""
    te = student.experts[B.TRAINABLE[0]]
    expert = 0
    p = te.cb13[expert]
    x32 = torch.tensor([0.25, -0.5, 1.5, 2.0], device=B.DEV)

    # Identity at initialization must also be bit-exact in the BF16 path.
    xbf = x32.to(torch.bfloat16)
    if not torch.equal(apply_log_gain(xbf, p), xbf):
        raise AssertionError("zero log-gain is not BF16 bit-exact identity")

    te.zero_grad(set_to_none=True)
    loss = apply_log_gain(x32, p).sum()
    loss.backward()
    autograd = float(te.cb13.grad[expert])
    original = float(p.detach())
    h = 1e-3
    with torch.no_grad():
        p.copy_(torch.tensor(original + h, device=B.DEV))
        plus = float(apply_log_gain(x32, p).sum())
        p.copy_(torch.tensor(original - h, device=B.DEV))
        minus = float(apply_log_gain(x32, p).sum())
        p.copy_(torch.tensor(original, device=B.DEV))
    finite_difference = (plus - minus) / (2.0 * h)
    rel = abs(finite_difference - autograd) / max(
        abs(finite_difference), abs(autograd), 1e-8)
    B.emit(event="output_gain_gradcheck", layer=B.TRAINABLE[0],
           expert=expert, autograd=autograd, fd=finite_difference,
           rel=round(rel, 7), bf16_identity=True)
    if rel >= 2e-3:
        raise AssertionError(f"output gain gradcheck FAIL rel={rel}")
    te.zero_grad(set_to_none=True)


def state_named(student):
    return {
        f"L{layer}": {
            "log_gain13": student.experts[layer].cb13.detach().cpu(),
            "log_gain2": student.experts[layer].cb2.detach().cpu(),
        }
        for layer in B.TRAINABLE
    }


def save_ckpt(path, student, opt, next_step, base, identity, best_mean):
    payload = {
        "format": FORMAT,
        "mechanism": MECHANISM,
        "manifest_md5": B.AMD5,
        "artifact_identity": identity,
        "trainable_layers": B.TRAINABLE,
        "gain_clamp": GAIN_CLAMP,
        "lr": B.LR,
        "steps_target": B.STEPS,
        "train_wins": B.TRAIN_WINS,
        "probe_wins": B.PROBE_WINS,
        "cache_id": B.CACHE_ID,
        "next_step": next_step,
        "baseline": base,
        "best_probe_mean": best_mean,
        "state": state_named(student),
        "optimizer": opt.state_dict(),
        "saved_ts": time.time(),
        "host": os.uname().nodename,
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def try_resume(student, opt, identity):
    if not B.LATEST.exists():
        if not BASELINE_SEED:
            return 0, None, None
        seed_path = Path(os.path.expanduser(BASELINE_SEED))
        if not seed_path.is_file():
            return 0, None, None
        seed = torch.load(seed_path, map_location="cpu", weights_only=False)
        expected = {
            "manifest_md5": B.AMD5,
            "train_wins": B.TRAIN_WINS,
            "probe_wins": B.PROBE_WINS,
        }
        bad = {key: (seed.get(key), value) for key, value in expected.items()
               if seed.get(key) != value}
        baseline = seed.get("baseline")
        required = set(B.TRAIN_WINS) | set(B.PROBE_WINS)
        present = {int(key) for key in baseline} if isinstance(baseline, dict) else set()
        if bad or not required.issubset(present):
            raise RuntimeError(
                f"baseline seed mismatch: fields={list(bad)}, "
                f"missing_wins={sorted(required-present)}")
        B.emit(event="baseline_seeded", source=str(seed_path),
               count=len(baseline), verification="own step0 8-probe panel follows")
        return 0, baseline, None

    ckpt = torch.load(B.LATEST, map_location="cpu", weights_only=False)
    expected = {
        "format": FORMAT,
        "mechanism": MECHANISM,
        "manifest_md5": B.AMD5,
        "artifact_identity": identity,
        "trainable_layers": B.TRAINABLE,
        "gain_clamp": GAIN_CLAMP,
        "lr": B.LR,
        "train_wins": B.TRAIN_WINS,
        "probe_wins": B.PROBE_WINS,
        "cache_id": B.CACHE_ID,
    }
    bad = {key: (ckpt.get(key), value) for key, value in expected.items()
           if ckpt.get(key) != value}
    if bad:
        raise RuntimeError(f"output-scale resume identity mismatch: {list(bad)}")
    for layer in B.TRAINABLE:
        state = ckpt["state"][f"L{layer}"]
        student.experts[layer].cb13.data.copy_(state["log_gain13"].to(B.DEV))
        student.experts[layer].cb2.data.copy_(state["log_gain2"].to(B.DEV))
    opt.load_state_dict(ckpt["optimizer"])
    B.emit(event="output_gain_resumed", next_step=ckpt["next_step"])
    return (ckpt["next_step"], ckpt.get("baseline"),
            ckpt.get("best_probe_mean"))


def enrich_final():
    if not B.FINAL.exists() or not B.BEST.exists():
        return
    result = json.loads(B.FINAL.read_text())
    ckpt = torch.load(B.BEST, map_location="cpu", weights_only=False)
    stats = {}
    for projection, key in (("fused13", "log_gain13"), ("down", "log_gain2")):
        logs = torch.cat([
            ckpt["state"][f"L{layer}"][key].float()
            for layer in B.TRAINABLE
        ])
        gains = logs.clamp(-GAIN_CLAMP, GAIN_CLAMP).exp()
        stats[projection] = {
            "min": float(gains.min()),
            "max": float(gains.max()),
            "mean": float(gains.mean()),
            "std": float(gains.std()),
        }
    result.update({
        "format": FORMAT,
        "mechanism": MECHANISM,
        "gain_clamp": GAIN_CLAMP,
        "n_trainable_params": len(B.TRAINABLE) * B.E * 2,
        "best_gain_stats": stats,
    })
    B.atomic_json(B.FINAL, result)
    B.status(**result)
    B.emit(event="output_gain_finalized", mechanism=MECHANISM,
           n_trainable_params=result["n_trainable_params"],
           best_gain_stats=stats)


def main():
    setattr(B, "K4096Experts", OutputScaleExperts)
    setattr(B, "codes_hash", instant_artifact_identity)
    setattr(B, "gradcheck", gradcheck)
    setattr(B, "state_named", state_named)
    setattr(B, "save_ckpt", save_ckpt)
    setattr(B, "try_resume", try_resume)
    B.main()
    enrich_final()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        try:
            B.status(state="failed", mechanism=MECHANISM,
                     error=f"{type(exc).__name__}: {exc}")
            B.emit(event="failed", mechanism=MECHANISM,
                   error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
