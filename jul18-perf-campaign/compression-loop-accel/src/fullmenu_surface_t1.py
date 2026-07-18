#!/usr/bin/env python3
"""Trainable exact full-menu expert surface for R5 prerepair+re-repair.

The source semantics exactly match R4V0_GATE/fullmenu_delta_source.py:
unchanged assignment rows come from pack_3tier; changed rows come from
sparse delta_3tier/selected. Codebooks are layer-shared and verified equal
across the two surfaces before becoming one fp32 master parameter.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

VQ_TIERS = {
    "d4_k1024", "d4_k2048", "d4_k4096",
    "d8_k256", "d8_k512", "d8_k1024", "d8_k2048", "d8_k4096",
}
STATIC_TIERS = {"vqa", "fp4"}


def tier(entry: object, projection: str) -> str:
    if isinstance(entry, dict):
        return str(entry["fused13" if projection == "13" else "down"])
    return str(entry)


def component(tier_name: str) -> str:
    return "static" if tier_name in STATIC_TIERS else tier_name


def parameter_key(tier_name: str, projection: str) -> str:
    return f"{tier_name}__{projection}"


# ---------------------------------------------------------------------------
# T1 ACCEL (<source-task>): GPU-side cache for FROZEN (codes, scales) row gathers.
#
# In the r5-fullmenu-combo mechanism only codebooks + rmsnorms + attn output
# gains train; codes/scales are frozen for the whole run. The sealed path
# re-gathers the same mmap rows + re-uploads H2D on EVERY forward (and again
# on the checkpoint recompute inside backward, x grad-accum), because
# _evict_payloads() madvise(DONTNEED)s the payload mmaps after each forward.
# This cache keeps the exact device copies the sealed path would have created,
# keyed by (layer, projection, source, tier, rows-chunk). On a hit the mmap
# gather + H2D is skipped and the SAME tensors (bitwise: they are the same
# device copies) feed the autograd Function, so codebook grads are untouched.
#
# Enable with R5_CODES_CACHE=1; VRAM bounded by R5_CODES_CACHE_GIB (default 8),
# evicting whole least-recently-used layers. Default OFF => sealed behavior.
# ---------------------------------------------------------------------------
_CODES_CACHE_ENABLED = os.environ.get("R5_CODES_CACHE", "0") == "1"
_CODES_CACHE_CAP_BYTES = int(
    float(os.environ.get("R5_CODES_CACHE_GIB", "8")) * (1 << 30)
)
_codes_cache: "OrderedDict[int, dict[Any, tuple[torch.Tensor, torch.Tensor]]]" = (
    OrderedDict()
)
_codes_cache_bytes = 0


def _codes_cache_pair(layer, key, payload, name_a, name_b, rows):
    """Return (payload[name_a][rows].cuda(), payload[name_b][rows].cuda()),
    served from the frozen-rows GPU cache when enabled."""
    global _codes_cache_bytes
    if not _CODES_CACHE_ENABLED:
        return payload[name_a][rows].to("cuda"), payload[name_b][rows].to("cuda")
    layer_cache = _codes_cache.get(layer)
    if layer_cache is None:
        layer_cache = {}
        _codes_cache[layer] = layer_cache
    _codes_cache.move_to_end(layer)
    hit = layer_cache.get(key)
    if hit is not None:
        return hit
    pair = (payload[name_a][rows].to("cuda"), payload[name_b][rows].to("cuda"))
    layer_cache[key] = pair
    _codes_cache_bytes += sum(t.untyped_storage().nbytes() for t in pair)
    while _codes_cache_bytes > _CODES_CACHE_CAP_BYTES and len(_codes_cache) > 1:
        _evicted_layer, evicted = _codes_cache.popitem(last=False)
        _codes_cache_bytes -= sum(
            t.untyped_storage().nbytes() for p in evicted.values() for t in p
        )
    return pair


class GenericVQDeqFn(torch.autograd.Function):
    """Exact fp16-wire VQ forward with STE gradient to a shared codebook."""

    @staticmethod
    def forward(ctx, codebook32, codes, scales):
        wire = codebook32.detach().to(torch.float16).float()
        scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
        out = (wire[codes.long()].reshape(codes.shape[0], -1) * scale_columns).to(
            torch.bfloat16
        )
        ctx.save_for_backward(codes, scales)
        ctx.k = int(codebook32.shape[0])
        ctx.d = int(codebook32.shape[1])
        return out

    @staticmethod
    def backward(ctx, grad):
        codes, scales = ctx.saved_tensors
        scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
        grouped = (grad.float() * scale_columns).reshape(
            codes.shape[0], codes.shape[1], ctx.d
        )
        grad_codebook = torch.zeros(
            ctx.k, ctx.d, device=grad.device, dtype=torch.float32
        )
        grad_codebook.index_add_(
            0, codes.reshape(-1).long(), grouped.reshape(-1, ctx.d)
        )
        return grad_codebook, None, None


class BatchedGenericVQDeqFn(torch.autograd.Function):
    """Batched exact fp16-wire VQ dequant with one shared codebook."""

    @staticmethod
    def forward(ctx, codebook32, codes, scales):
        wire = codebook32.detach().to(torch.float16).float()
        scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
        out = (wire[codes.long()].reshape(codes.shape[0], codes.shape[1], -1) * scale_columns).to(
            torch.bfloat16
        )
        ctx.save_for_backward(codes, scales)
        ctx.k = int(codebook32.shape[0])
        ctx.d = int(codebook32.shape[1])
        return out

    @staticmethod
    def backward(ctx, grad):
        codes, scales = ctx.saved_tensors
        scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
        grouped = (grad.float() * scale_columns).reshape(
            codes.shape[0], codes.shape[1], codes.shape[2], ctx.d
        )
        grad_codebook = torch.zeros(
            ctx.k, ctx.d, device=grad.device, dtype=torch.float32
        )
        grad_codebook.index_add_(
            0, codes.reshape(-1).long(), grouped.reshape(-1, ctx.d)
        )
        return grad_codebook, None, None


class FullMenuExperts(nn.Module):
    def __init__(self, layer: int, pilot: bool):
        super().__init__()
        if not pilot:
            raise AssertionError(f"R5 requires all 43 layers trainable, got L{layer}")
        self.L = layer
        self.pilot = pilot
        self.limit = 10.0
        self.act = F.silu
        self.root = Path(os.environ["R5_INPUT_ROOT"]).expanduser().resolve()
        self.target_manifest = json.loads(
            (self.root / "TRUE_CORRECTED_MANIFEST.json").read_text()
        )
        self.base_manifest = json.loads(
            (self.root / "OLD_BIN_MANIFEST.json").read_text()
        )
        self.target_map = self.target_manifest["assignment"][str(layer)]
        self.base_map = self.base_manifest["assignment"][str(layer)]
        if set(self.target_map) != set(self.base_map):
            raise AssertionError(f"expert domain mismatch L{layer}")

        base_needed: set[str] = set()
        delta_needed: set[str] = set()
        self.routes: dict[tuple[int, str], tuple[str, str, int]] = {}
        for expert_id in range(256):
            target_entry = self.target_map[str(expert_id)]
            base_entry = self.base_map[str(expert_id)]
            for projection in ("13", "2"):
                target_tier = tier(target_entry, projection)
                base_tier = tier(base_entry, projection)
                source = "delta" if target_tier != base_tier else "base"
                (delta_needed if source == "delta" else base_needed).add(
                    component(target_tier)
                )
                self.routes[(expert_id, projection)] = (source, target_tier, -1)

        self.payloads: dict[str, dict[str, dict[str, Any]]] = {"base": {}, "delta": {}}
        for source, names in (("base", base_needed), ("delta", delta_needed)):
            prefix = (
                self.root / "pack_3tier"
                if source == "base"
                else self.root / "delta_3tier/selected"
            )
            for name in sorted(names):
                path = prefix / name / f"layer_{layer:03d}.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)
                self.payloads[source][name] = torch.load(
                    path, map_location="cpu", mmap=True, weights_only=True
                )

        # Resolve every row once, failing closed on absent/duplicate expert IDs.
        for (expert_id, projection), (source, target_tier, _row) in list(
            self.routes.items()
        ):
            payload = self.payloads[source][component(target_tier)]
            prefix = target_tier if target_tier in STATIC_TIERS else "expert"
            ids = payload[f"{prefix}_ids{projection}"]
            hits = (ids == expert_id).nonzero()
            if hits.numel() != 1:
                raise AssertionError(
                    f"missing/duplicate row L{layer} e{expert_id} p{projection} "
                    f"tier={target_tier} source={source}"
                )
            self.routes[(expert_id, projection)] = (
                source,
                target_tier,
                int(hits[0, 0]),
            )

        self.groups: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
        for (expert_id, projection), (source, target_tier, row) in self.routes.items():
            self.groups.setdefault((projection, source, target_tier), []).append(
                (expert_id, row)
            )

        # One trainable master per live layer/tier/projection. Base and delta
        # carry redundant copies; they must be bit-identical (verified on s8
        # globally before staging and repeated here for each live pair).
        self.codebooks = nn.ParameterDict()
        live_tiers = {
            target_tier for _source, target_tier, _row in self.routes.values()
            if target_tier != "fp4"
        }
        for target_tier in sorted(live_tiers):
            name = component(target_tier)
            for projection in ("13", "2"):
                candidates = []
                for source in ("base", "delta"):
                    payload = self.payloads[source].get(name)
                    if payload is not None and f"cb{projection}" in payload:
                        candidates.append(payload[f"cb{projection}"])
                if not candidates:
                    raise AssertionError(
                        f"missing codebook L{layer} {target_tier} p{projection}"
                    )
                if any(not torch.equal(candidates[0], other) for other in candidates[1:]):
                    raise AssertionError(
                        f"base/delta codebook mismatch L{layer} {target_tier} p{projection}"
                    )
                self.codebooks[parameter_key(target_tier, projection)] = nn.Parameter(
                    candidates[0].float().to("cuda")
                )

    def named_codebooks(self):
        return self.codebooks.items()

    def _weight(self, expert_id: int, projection: str):
        source, target_tier, row = self.routes[(expert_id, projection)]
        payload = self.payloads[source][component(target_tier)]
        if target_tier in VQ_TIERS:
            codebook = self.codebooks[parameter_key(target_tier, projection)]
            return GenericVQDeqFn.apply(
                codebook,
                payload[f"codes{projection}"][row].to("cuda"),
                payload[f"sc{projection}"][row].to("cuda"),
            )
        if target_tier == "vqa":
            codebook = self.codebooks[parameter_key(target_tier, projection)]
            return GenericVQDeqFn.apply(
                codebook,
                payload[f"vqa_codes{projection}"][row].to("cuda"),
                payload[f"vqa_sc{projection}"][row].to("cuda"),
            )
        if target_tier == "fp4":
            # Import through the sealed base harness to preserve exact E2M1
            # nibble/scaling convention.
            import t8192_ds4_build_v3 as v3

            return v3.deq_fp4_block32(
                payload[f"fp4_wb{projection}"][row].to("cuda"),
                payload[f"fp4_sb{projection}"][row].to("cuda"),
                "e2m1",
            )
        raise KeyError(target_tier)

    def _weights_for(self, projection: str, hit_ids: list[int]):
        weights = {}
        hit_set = set(hit_ids)
        chunk_size = int(os.environ.get("R5_DEQ_CHUNK", "4"))
        for (group_projection, source, target_tier), pairs in self.groups.items():
            if group_projection != projection:
                continue
            selected = [(expert, row) for expert, row in pairs if expert in hit_set]
            if not selected:
                continue
            payload = self.payloads[source][component(target_tier)]
            for start in range(0, len(selected), chunk_size):
                chunk = selected[start : start + chunk_size]
                experts = [expert for expert, _row in chunk]
                rows = torch.tensor([row for _expert, row in chunk], dtype=torch.long)
                cache_key = (
                    projection,
                    source,
                    target_tier,
                    tuple(row for _expert, row in chunk),
                )
                if target_tier in VQ_TIERS:
                    codebook = self.codebooks[parameter_key(target_tier, projection)]
                    codes_gpu, sc_gpu = _codes_cache_pair(
                        self.L, cache_key, payload,
                        f"codes{projection}", f"sc{projection}", rows,
                    )
                    batch = BatchedGenericVQDeqFn.apply(codebook, codes_gpu, sc_gpu)
                elif target_tier == "vqa":
                    codebook = self.codebooks[parameter_key(target_tier, projection)]
                    codes_gpu, sc_gpu = _codes_cache_pair(
                        self.L, cache_key, payload,
                        f"vqa_codes{projection}", f"vqa_sc{projection}", rows,
                    )
                    batch = BatchedGenericVQDeqFn.apply(codebook, codes_gpu, sc_gpu)
                elif target_tier == "fp4":
                    import t8192_ds4_build_v3 as v3

                    wb_gpu, sb_gpu = _codes_cache_pair(
                        self.L, cache_key, payload,
                        f"fp4_wb{projection}", f"fp4_sb{projection}", rows,
                    )
                    batch = v3.deq_fp4_block32(wb_gpu, sb_gpu, "e2m1")
                else:
                    raise KeyError(target_tier)
                for index, expert in enumerate(experts):
                    weights[expert] = batch[index]
        if set(weights) != hit_set:
            raise AssertionError(
                f"weight materialization mismatch L{self.L} p{projection}: "
                f"missing={sorted(hit_set-set(weights))}"
            )
        return weights

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=256).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        hit_ids = [int(expert_[0]) for expert_ in hit]
        weights13 = self._weights_for("13", hit_ids)
        intermediates = {}
        positions = {}
        for expert_id in hit_ids:
            top_k_pos, token_idx = torch.where(mask[expert_id])
            xt = hidden_states[token_idx]
            current = F.linear(xt, weights13[expert_id])
            gate, up = current.chunk(2, dim=-1)
            intermediates[expert_id] = F.silu(gate.clamp(max=self.limit)) * up.clamp(
                min=-self.limit, max=self.limit
            )
            positions[expert_id] = (top_k_pos, token_idx)
            del current, xt
        del weights13
        weights2 = self._weights_for("2", hit_ids)
        for expert_id in hit_ids:
            top_k_pos, token_idx = positions[expert_id]
            current = F.linear(intermediates[expert_id], weights2[expert_id]) * top_k_weights[
                token_idx, top_k_pos, None
            ]
            final.index_add_(0, token_idx, current.to(final.dtype))
            del current
        del weights2, intermediates, positions
        self._evict_payloads()
        return final

    def _evict_payloads(self):
        # The sealed base trainer's madvise helper prevents mmap page-cache
        # residency from starving unified-memory cudaMalloc.
        import lp4_train as trainer

        for by_component in self.payloads.values():
            for payload in by_component.values():
                for value in payload.values():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.device.type == "cpu"
                        and value.untyped_storage().nbytes() > (16 << 20)
                    ):
                        trainer.evict_tensor(value)


def surface_state(student) -> dict[str, dict[str, torch.Tensor]]:
    return {
        f"L{layer}": {
            name: parameter.detach().cpu()
            for name, parameter in student.experts[layer].named_codebooks()
        }
        for layer in range(43)
    }


def load_surface_state(student, state) -> None:
    for layer in range(43):
        live = dict(student.experts[layer].named_codebooks())
        saved = state[f"L{layer}"]
        if set(live) != set(saved):
            raise RuntimeError(f"codebook state keys mismatch L{layer}")
        for name, parameter in live.items():
            parameter.data.copy_(saved[name].to("cuda"))


def surface_parameters(student) -> list[nn.Parameter]:
    return [
        parameter
        for layer in range(43)
        for _name, parameter in student.experts[layer].named_codebooks()
    ]


def gradcheck(student, base_module) -> None:
    """One exact fp16-wire finite-difference check on the live R5 surface."""
    experts = student.experts[0]
    expert_id = next(
        expert
        for expert in range(256)
        if experts.routes[(expert, "13")][1] != "fp4"
    )
    source, target_tier, row = experts.routes[(expert_id, "13")]
    payload = experts.payloads[source][component(target_tier)]
    if target_tier == "vqa":
        codes = payload["vqa_codes13"][row].to("cuda")
        scales = payload["vqa_sc13"][row].to("cuda")
    else:
        codes = payload["codes13"][row].to("cuda")
        scales = payload["sc13"][row].to("cuda")
    master = experts.codebooks[parameter_key(target_tier, "13")]
    code_index = int(codes[0, 0])
    column = 0
    # Isolate exactly one master entry so finite differences do not subtract
    # two enormous, cancellation-heavy full-matrix random dot products.
    gradient_seed = torch.zeros(4096, 4096, device="cuda")
    matches = (codes == code_index).nonzero()
    gradient_seed[matches[:, 0], matches[:, 1] * master.shape[1] + column] = 1.0
    scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)

    def reference(value):
        wire = value.detach().to(torch.float16).float()
        weight = wire[codes.long()].reshape(codes.shape[0], -1) * scale_columns
        return (weight * gradient_seed).sum().item()

    loss = (GenericVQDeqFn.apply(master, codes, scales).float() * gradient_seed).sum()
    loss.backward()
    autograd_value = master.grad[code_index, column].item()
    original = master.data[code_index, column].item()
    step = max(abs(original) / 16.0, 1e-2)
    with torch.no_grad():
        master.data[code_index, column] = original + step
        plus = reference(master.data)
        plus_wire = float(torch.tensor(original + step).to(torch.float16))
        master.data[code_index, column] = original - step
        minus = reference(master.data)
        minus_wire = float(torch.tensor(original - step).to(torch.float16))
        master.data[code_index, column] = original
    master.grad = None
    finite_difference = (plus - minus) / (plus_wire - minus_wire)
    relative = abs(finite_difference - autograd_value) / max(
        abs(finite_difference), abs(autograd_value), 1e-8
    )
    base_module.emit(
        event="r5_fullmenu_gradcheck",
        layer=0,
        expert=expert_id,
        tier=target_tier,
        autograd=autograd_value,
        fd=finite_difference,
        rel=relative,
    )
    if relative >= 0.02:
        raise AssertionError(f"R5 fullmenu gradcheck failed rel={relative}")


def quick_identity(input_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(input_root).expanduser().resolve()
    files = [
        root / "TRUE_CORRECTED_MANIFEST.json",
        root / "OLD_BIN_MANIFEST.json",
        root / "SCORE_3TIER.json",
        root / "delta_3tier/DELTA_PACK.COMPLETE",
        root / "delta_3tier/DELTA_PACK_MANIFEST.json",
    ]
    result = {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    stage = root.parent / "receipts/STAGE_COMPLETE.json"
    if stage.is_file():
        result["stage_receipt_sha256"] = hashlib.sha256(stage.read_bytes()).hexdigest()
    return result
