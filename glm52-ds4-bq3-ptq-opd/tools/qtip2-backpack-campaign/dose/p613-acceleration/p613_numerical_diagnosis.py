#!/usr/bin/env python3
"""Localize P613 batch-shape numerical drift at real-model layer boundaries."""
from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path("$HOME/run-bundles/P613_ACTCACHE_ACCEL_PUBLIC_TASK_s5w")
CODE = ROOT / "code"
WINS = list(range(20, 28))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_only(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    return None


def matching_row(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor | None:
    if tuple(candidate.shape) == tuple(reference.shape):
        return candidate
    if candidate.ndim != reference.ndim or candidate.shape[1:] != reference.shape[1:]:
        return None
    if candidate.shape[0] < reference.shape[0]:
        return None
    # The model flattens B*T*HC before routed experts. Window 0 remains the
    # first contiguous reference.shape[0] rows in that representation.
    return candidate[: reference.shape[0]]


def compare(reference: torch.Tensor, candidate: torch.Tensor):
    selected = matching_row(reference, candidate)
    if selected is None:
        return {
            "comparable": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    delta = (reference.float() - selected.float()).abs()
    return {
        "comparable": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "exact_equal": bool(torch.equal(reference, selected)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "nonzero": int(torch.count_nonzero(delta)),
    }


def main():
    benchmark = load_module("p613_benchmark_support", CODE / "p613_profile_benchmark.py")
    benchmark.ensure_inputs()
    benchmark.set_env()
    os.environ["BR_OUTDIR"] = str(ROOT / "bench" / "diagnosis")
    sys.path.insert(0, str(CODE))
    base = load_module(
        "p613_diagnosis_base", CODE / "base_binrepair_e2e_accel.py"
    )
    from genesis_physical_surface import GenesisPhysicalExperts
    base.T.TrainableExperts = GenesisPhysicalExperts
    base.T.PILOT = tuple(range(43))

    assembled_started = time.perf_counter()
    student = base.T.Student()
    torch.cuda.synchronize()
    assembled_seconds = time.perf_counter() - assembled_started
    corpus = base.T.load_corpus()
    model, config = student.model, student.config
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_sliding_window_causal_mask

    def context(wins):
        ids = torch.stack([base.T.window_ids(corpus, win)[0] for win in wins]).to(base.DEV)
        pos = torch.arange(ids.shape[1], device=base.DEV).unsqueeze(0)
        embeds = model.model.embed_tokens(ids)
        pe = {
            "main": model.model.rotary_emb(
                embeds[:1], position_ids=pos, layer_type="main"
            ),
            "compress": model.model.rotary_emb(
                embeds[:1], position_ids=pos, layer_type="compress"
            ),
        }
        mask = create_sliding_window_causal_mask(
            config=config,
            inputs_embeds=embeds,
            attention_mask=None,
            past_key_values=DynamicCache(config=config),
            position_ids=pos,
        )
        hidden = embeds.unsqueeze(2).expand(
            -1, -1, config.hc_mult, -1
        ).contiguous()
        return ids, pos, pe, mask, hidden

    def run(wins, capture_children=False, stable=False):
        ids, pos, pe, mask, hidden = context(wins)
        captures = {}
        handles = []
        if capture_children:
            for name, child in model.model.layers[0].named_children():
                def hook(_module, _args, output, module_name=name):
                    tensor = tensor_only(output)
                    if tensor is not None:
                        captures[module_name] = tensor.detach().cpu().clone()
                handles.append(child.register_forward_hook(hook))
        layer_outputs = []
        started = time.perf_counter()
        with torch.no_grad():
            for layer_index in range(4):
                if stable:
                    hidden = base.batch_stable_layer(
                        model.model.layers[layer_index], hidden,
                        position_embeddings=pe, position_ids=pos,
                        attention_mask=mask, input_ids=ids,
                        past_key_values=DynamicCache(config=config),
                    )
                else:
                    hidden = model.model.layers[layer_index](
                        hidden,
                        position_embeddings=pe,
                        position_ids=pos,
                        attention_mask=mask,
                        input_ids=ids,
                        past_key_values=DynamicCache(config=config),
                    )
                torch.cuda.synchronize()
                layer_outputs.append(hidden.detach().cpu().clone())
        seconds = time.perf_counter() - started
        for handle in handles:
            handle.remove()
        del ids, pos, pe, mask, hidden
        torch.cuda.empty_cache()
        gc.collect()
        return layer_outputs, captures, seconds

    serial_layers, serial_children, serial_seconds = run([WINS[0]], True, True)
    batch_layers, batch_children, batch_seconds = run(WINS, True, True)
    repeat_layers, _repeat_children, repeat_seconds = run(WINS, False, True)

    child_comparisons = {}
    for name in sorted(set(serial_children) & set(batch_children)):
        child_comparisons[name] = compare(serial_children[name], batch_children[name])

    report = {
        "schema": "p613-batch-shape-numerical-fix-v1",
        "task_id": "PUBLIC_TASK",
        "host": os.uname().nodename,
        "assembled_seconds": assembled_seconds,
        "serial_win": WINS[0],
        "batch_wins": WINS,
        "serial_4layer_seconds": serial_seconds,
        "batch8_4layer_seconds": batch_seconds,
        "batch8_repeat_4layer_seconds": repeat_seconds,
        "layer_serial_vs_batch8": [
            {"layer": index, **compare(serial_layers[index], batch_layers[index])}
            for index in range(4)
        ],
        "layer_batch8_repeatability": [
            {"layer": index, **compare(batch_layers[index], repeat_layers[index])}
            for index in range(4)
        ],
        "layer0_child_serial_vs_batch8": child_comparisons,
    }
    out = ROOT / "DIAGNOSIS_FIX.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
