#!/usr/bin/env python3
"""Combo-V4 step-32 tail repair with exact-lineage matched controls.

The exact BQ3 k4096-menu artifact is assembled by the sealed BINREPAIR base
harness. The joint surface is the union of three above-floor mechanisms:

* all 43 layers' vq3b codebooks (fp32 masters -> fp16 wire STE),
* all RMSNorm gammas (fp32 masters -> BF16 wire),
* one bounded output log-gain after every self_attn.o_b_proj.

The prior all-RMSNorm arm already contains every attention RMSNorm used by the
attention arm. Those 148 tensors are exposed exactly once. Fresh runs seed the
stronger all-RMSNorm checkpoint for the shared tensors and import only the 43
unique output gains from the attention checkpoint. Codebooks seed from arm4.

All arms start from the exact frozen Combo-V4 step-32 checkpoint.  The pure
plan contract fixes a fresh/disjoint 48-window train pool, a shared 16-window
held-out pool, equal total exposure, and one of three objectives:

* control: scorer-exact mean KLD, balanced class exposure;
* tail: detached token/window-tail weighting, same schedule as control;
* tail_class40: same tail objective with >=40% code and >=40% reasoning
  target gradient exposure.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

import combo_core as C
import combo_v2_core as V2
import tailfix_core as TF


FORMAT = "combo-v4-tailfix-v1"
MECHANISM = "combo-v4-step32-tailfix-codebooks-norms-output-gains"
GAIN_CLAMP = float(os.environ.get("COMBO_GAIN_CLAMP", "0.25"))
OUTPUT_LR = float(os.environ.get("COMBO_OUTPUT_LR", "1e-2"))
MIN_LR_RATIO = float(os.environ.get("COMBO_MIN_LR_RATIO", "0.1"))
SEED = int(os.environ.get("COMBO_SEED", "1701"))
ARM = os.environ.get("TAILFIX_ARM", "control")
POOL_ROLE = f"TAILFIX_{ARM}"
PURITY_WINS: list[int] = []


class WireBf16(nn.Module):
    def forward(self, master: torch.Tensor) -> torch.Tensor:
        return master.to(torch.bfloat16)


def _output_gain_hook(module, _inputs, output):
    log_gain = module._combo_output_log_gain
    gain = torch.exp(log_gain.clamp(-GAIN_CLAMP, GAIN_CLAMP)).to(output.dtype)
    return output * gain


def attach_output_gain(module: nn.Module) -> nn.Parameter:
    wire = module._parameters.get("weight")
    if wire is None:
        raise RuntimeError("o_b_proj has no weight parameter")
    if hasattr(module, "_combo_output_log_gain"):
        raise RuntimeError("combo output gain already attached")
    gain = nn.Parameter(torch.zeros((), dtype=torch.float32, device=wire.device))
    module.register_parameter("_combo_output_log_gain", gain)
    module.register_forward_hook(_output_gain_hook)
    return gain


def self_test() -> None:
    torch.manual_seed(17)
    linear = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    linear.weight.requires_grad_(False)
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    before = linear(x).detach().clone()
    gain = attach_output_gain(linear)
    after = linear(x)
    if not torch.equal(before, after):
        raise AssertionError("zero output gain changed BF16 output")
    after.float().sum().backward()
    output_gradient_finite = gain.grad is not None and bool(torch.isfinite(gain.grad).all())

    dummy = nn.Module()
    dummy.weight = nn.Parameter(
        torch.tensor([1.0, 1.5, -0.75], dtype=torch.bfloat16), requires_grad=False
    )
    norm_before = dummy.weight.detach().clone()
    parametrize.register_parametrization(dummy, "weight", WireBf16(), unsafe=True)
    master = dummy.parametrizations.weight.original
    master.data = master.data.float()
    master.requires_grad_(True)
    if not torch.equal(norm_before, dummy.weight.detach()):
        raise AssertionError("BF16 norm master changed initialization")
    (dummy.weight.float() * torch.tensor([0.5, -1.0, 2.0])).sum().backward()
    norm_gradient_finite = master.grad is not None and bool(torch.isfinite(master.grad).all())
    tail_weights = TF.normalized_tail_weights([1.0, 2.0, 3.0, 100.0], 0.75, 4.0)
    if not output_gradient_finite or not norm_gradient_finite or tail_weights[-1] <= tail_weights[0]:
        raise AssertionError("tailfix self-test failed")
    print(json.dumps({
        "self_test": "PASS",
        "bf16_identity": True,
        "norm_gradient_finite": norm_gradient_finite,
        "output_gradient_finite": output_gradient_finite,
        "tail_weighting": "PASS",
        "semantic_overlap_policy": "frozen-step32-state",
    }, sort_keys=True))


if "--self-test" in sys.argv:
    self_test()
    raise SystemExit(0)

if ARM not in TF.ARMS:
    raise ValueError(f"TAILFIX_ARM must be one of {TF.ARMS}, got {ARM}")
if not (0.0 < GAIN_CLAMP <= math.log(2.0)):
    raise ValueError(f"invalid COMBO_GAIN_CLAMP={GAIN_CLAMP}")
if OUTPUT_LR <= 0.0:
    raise ValueError(f"invalid COMBO_OUTPUT_LR={OUTPUT_LR}")
if not 0.0 <= MIN_LR_RATIO <= 1.0:
    raise ValueError(f"invalid COMBO_MIN_LR_RATIO={MIN_LR_RATIO}")

BASE_PATH = Path(os.path.expanduser(os.environ["COMBO_BINREPAIR_BASE"]))
START_PATH = Path(os.path.expanduser(os.environ["TAILFIX_START_CKPT"]))
PLAN_PATH = Path(os.path.expanduser(os.environ["TAILFIX_PLAN"]))
for required in (BASE_PATH, START_PATH, PLAN_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)
PLAN = json.loads(PLAN_PATH.read_text())
TF.validate_plan(PLAN)
if PLAN["arm"] != ARM:
    raise ValueError(f"plan arm={PLAN['arm']} does not match TAILFIX_ARM={ARM}")
TRAINING_SCHEDULE = [list(map(int, item["wins"])) for item in PLAN["schedule"]]
TRAINING_CLASSES = [str(item["class"]) for item in PLAN["schedule"]]
TAIL_TOKEN_QUANTILE = PLAN["tail_token_quantile"]
TAIL_TOKEN_BOOST = PLAN["tail_token_boost"]
TAIL_WINDOW_QUANTILE = PLAN["tail_window_quantile"]
TAIL_WINDOW_BOOST = PLAN["tail_window_boost"]
TRAJECTORY_PATH = (
    Path(os.path.expanduser(os.environ["TAILFIX_TRAJECTORY_FILE"]))
    if ARM in {"trajectory_micro", "trajectory_full"} else None
)
TRAJECTORY_WEIGHT = float(PLAN.get("trajectory_weight") or 0.0)
BASELINE_SOURCE = os.environ.get("TAILFIX_BASELINE_SOURCE")
SKIP_FULL_PROBE = os.environ.get("TAILFIX_SKIP_FULL_PROBE", "0") == "1"
BINDING_GATE_STOP = os.environ.get("TAILFIX_BINDING_GATE_STOP", "0") == "1"

_spec = importlib.util.spec_from_file_location("combo_binrepair_base", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load base harness: {BASE_PATH}")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

TAIL_BEST = Path(str(B.PREFIX) + ".tailbest.pt")
TAIL_BASELINE = Path(str(B.PREFIX) + ".heldout_tail_baseline.json")
TAIL_SELECTION = Path(str(B.PREFIX) + ".heldout_tail_selection.json")
PROBE_CLASS_BY_WIN = {
    int(win): cls
    for cls, wins in PLAN["probe_by_class"].items()
    for win in wins
}
if set(PROBE_CLASS_BY_WIN) != set(map(int, PLAN["probe_wins"])):
    raise ValueError("probe_by_class does not exactly cover probe_wins")

EXPECTED_LAYERS = list(range(43))
EXPECTED_NORM_PARAMS = 446080
EXPECTED_ATTN_NORM_PARAMS = 89728
EXPECTED_OUTPUT_GAINS = 43
EXPECTED_CODEBOOK_PARAMS = 1409024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def expose_combo_parameters(student):
    norms = []
    outputs = []
    modules = list(student.model.named_modules())
    for module_name, module in modules:
        leaf = module_name.rsplit(".", 1)[-1].lower()
        if "norm" not in leaf:
            continue
        wire = module._parameters.get("weight")
        if wire is None or wire.ndim != 1:
            continue
        before = wire.detach().clone()
        parametrize.register_parametrization(
            module, "weight", WireBf16(), unsafe=True
        )
        master = module.parametrizations.weight.original
        master.data = master.data.float()
        master.requires_grad_(True)
        if module.weight.dtype != torch.bfloat16 or not torch.equal(
            module.weight.detach(), before
        ):
            raise AssertionError(f"norm parametrization changed init: {module_name}")
        norms.append((module_name, module, master))

    for module_name, module in modules:
        if module_name.endswith(".self_attn.o_b_proj"):
            outputs.append(
                (
                    module_name + ".output_log_gain",
                    module,
                    attach_output_gain(module),
                )
            )

    norm_count = sum(parameter.numel() for _n, _m, parameter in norms)
    if norm_count != EXPECTED_NORM_PARAMS or len(outputs) != EXPECTED_OUTPUT_GAINS:
        raise RuntimeError(
            "sealed combo surface mismatch: "
            f"norm={norm_count} outputs={len(outputs)}"
        )
    return norms, outputs


def codebook_params(student):
    params = [
        parameter
        for layer in EXPECTED_LAYERS
        for parameter in (student.experts[layer].cb13, student.experts[layer].cb2)
    ]
    count = sum(parameter.numel() for parameter in params)
    if count != EXPECTED_CODEBOOK_PARAMS:
        raise RuntimeError(f"sealed codebook surface mismatch: {count}")
    return params


def baseline_dict(checkpoint):
    raw = checkpoint.get("baseline")
    if not isinstance(raw, dict):
        raise RuntimeError("seed checkpoint lacks baseline dictionary")
    return {int(key): float(value) for key, value in raw.items()}


def validate_and_load_seeds(student, norms, outputs):
    checkpoint_sha256 = sha256_file(START_PATH)
    checkpoint = torch.load(START_PATH, map_location="cpu", weights_only=False)
    TF.validate_start_header(checkpoint, checkpoint_sha256)
    load_named(student, norms, outputs, checkpoint["state"])
    baseline = {
        int(key): float(value)
        for key, value in (checkpoint.get("baseline") or {}).items()
    }
    return baseline, {
        "semantic_overlap_tensors": 148,
        "semantic_overlap_params": EXPECTED_ATTN_NORM_PARAMS,
        "overlap_policy": "exact repaired Combo-V4 step32 state; optimizer and scheduler initialized fresh",
        "seed_sha256": {"combo_v4_step32": checkpoint_sha256},
        "seed_paths": {"combo_v4_step32": str(START_PATH)},
        "warm_start": {
            "path": str(START_PATH),
            "sha256": checkpoint_sha256,
            "source_pool_role": checkpoint["pool_role"],
            "source_next_step": checkpoint["next_step"],
            "optimizer_reused": False,
            "scheduler_reused": False,
        },
    }


def gradcheck_aux(norms, outputs):
    name, module, master = norms[0]
    module.zero_grad(set_to_none=True)
    values = torch.tensor([0.25, -0.5, 1.5, 2.0], device=B.DEV)
    p = master.reshape(-1)[0]
    loss = (values.to(torch.bfloat16) * p.to(torch.bfloat16)).float().sum()
    loss.backward()
    if master.grad is None or not torch.isfinite(master.grad).all():
        raise AssertionError(f"norm gradient missing/non-finite: {name}")
    norm_grad = float(master.grad.reshape(-1)[0])
    module.zero_grad(set_to_none=True)

    name, module, gain = outputs[0]
    module.zero_grad(set_to_none=True)
    x = torch.tensor([0.25, -0.5, 1.5, 2.0], device=B.DEV, dtype=torch.bfloat16)
    y = x * torch.exp(gain.clamp(-GAIN_CLAMP, GAIN_CLAMP)).to(x.dtype)
    y.float().sum().backward()
    if gain.grad is None or not torch.isfinite(gain.grad).all():
        raise AssertionError(f"output gradient missing/non-finite: {name}")
    output_grad = float(gain.grad)
    module.zero_grad(set_to_none=True)
    B.emit(
        event="combo_aux_gradcheck",
        norm_parameter=norms[0][0],
        norm_grad=norm_grad,
        output_parameter=outputs[0][0],
        output_grad=output_grad,
    )


def state_named(student, norms, outputs):
    return {
        "codebooks": B.state_named(student),
        "norms": {
            name: parameter.detach().cpu()
            for name, _module, parameter in norms
        },
        "outputs": {
            name: parameter.detach().cpu()
            for name, _module, parameter in outputs
        },
    }


def load_named(student, norms, outputs, state):
    for layer in EXPECTED_LAYERS:
        layer_state = state["codebooks"][f"L{layer}"]
        student.experts[layer].cb13.data.copy_(layer_state["cb13"].to(B.DEV))
        student.experts[layer].cb2.data.copy_(layer_state["cb2"].to(B.DEV))
    norm_map = {name: parameter for name, _module, parameter in norms}
    output_map = {name: parameter for name, _module, parameter in outputs}
    if set(norm_map) != set(state["norms"]) or set(output_map) != set(state["outputs"]):
        raise RuntimeError("combo resume parameter names mismatch")
    for name, parameter in norm_map.items():
        parameter.data.copy_(state["norms"][name].to(B.DEV))
    for name, parameter in output_map.items():
        parameter.data.copy_(state["outputs"][name].to(B.DEV))


def load_v2_warm_start(student, norms, outputs, seed_info):
    """Compatibility shim: exact step-32 state was already loaded above."""
    return seed_info


def identity(seed_info):
    return {
        "manifest_md5": B.AMD5,
        "source_codes_hash": B.codes_hash(),
        "tailfix_arm": ARM,
        "tailfix_plan_sha256": PLAN["plan_sha256"],
        "objective": PLAN["objective"],
        "start_checkpoint_sha256": TF.START_SHA256,
        **seed_info,
    }


def save_checkpoint(
    path,
    student,
    norms,
    outputs,
    optimizer,
    scheduler,
    next_step,
    baseline,
    artifact_identity,
    best_mean,
    step0_mean,
):
    payload = {
        "format": FORMAT,
        "mechanism": MECHANISM,
        "manifest_md5": B.AMD5,
        "artifact_identity": artifact_identity,
        "pool_role": POOL_ROLE,
        "train_wins": B.TRAIN_WINS,
        "probe_wins": B.PROBE_WINS,
        "codebook_lr": B.LR,
        "norm_lr": float(os.environ.get("COMBO_NORM_LR", "1e-4")),
        "output_lr": OUTPUT_LR,
        "min_lr_ratio": MIN_LR_RATIO,
        "steps_target": B.STEPS,
        "batch": B.BATCH,
        "next_step": next_step,
        "baseline": baseline,
        "step0_combo_mean": step0_mean,
        "best_probe_mean": best_mean,
        "state": state_named(student, norms, outputs),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "saved_ts": time.time(),
        "host": os.uname().nodename,
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def try_resume(student, norms, outputs, optimizer, scheduler, artifact_identity):
    if not B.LATEST.exists():
        return 0, None, None, None
    checkpoint = torch.load(B.LATEST, map_location="cpu", weights_only=False)
    expected = {
        "format": FORMAT,
        "mechanism": MECHANISM,
        "manifest_md5": B.AMD5,
        "artifact_identity": artifact_identity,
        "pool_role": POOL_ROLE,
        "train_wins": B.TRAIN_WINS,
        "probe_wins": B.PROBE_WINS,
        "codebook_lr": B.LR,
        "norm_lr": float(os.environ.get("COMBO_NORM_LR", "1e-4")),
        "output_lr": OUTPUT_LR,
        "min_lr_ratio": MIN_LR_RATIO,
        "batch": B.BATCH,
    }
    bad = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if bad:
        raise RuntimeError(f"combo resume identity mismatch: {list(bad)}")
    load_named(student, norms, outputs, checkpoint["state"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    B.emit(event="combo_resumed", next_step=checkpoint["next_step"])
    return (
        checkpoint["next_step"],
        checkpoint.get("baseline"),
        checkpoint.get("best_probe_mean"),
        checkpoint.get("step0_combo_mean"),
    )


def _normalized_detached_weights(values, quantile, boost):
    detached = values.detach().float()
    threshold = torch.quantile(detached, float(quantile))
    raw = torch.where(detached >= threshold, float(boost), 1.0)
    return raw / raw.mean(), float(threshold)


def _token_kld(student, hidden, real_len, win):
    idx, lp_n, p_n = B.T.teacher_rows(win)
    logits = student.model.lm_head(hidden[:real_len].to(torch.bfloat16))
    q = logits.gather(1, idx[:real_len]).float()
    lq_n = q - q.logsumexp(-1, keepdim=True)
    values = (p_n[:real_len] * (lp_n[:real_len] - lq_n)).sum(-1)
    del idx, lp_n, p_n, logits
    return values


def _tail_stats(values):
    values = values.detach().float().cpu().contiguous()
    return {
        "n_positions": int(values.numel()),
        "mean": float(values.mean()),
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "max": float(values.max()),
        "pct_positions_gt_0_5": float((values > 0.5).float().mean() * 100.0),
    }


def heldout_token_values(student, corpus, acache, win):
    """Scorer-exact held-out per-position KL without a second forward pass."""
    with torch.no_grad():
        ids, real_len = B.T.window_ids(corpus, win)
        batch_ids = ids.unsqueeze(0).to(B.DEV)
        hidden = acache.get(corpus, win)
        hidden = B.fast_forward(student, hidden, batch_ids, False)
        values = _token_kld(student, hidden[0], real_len, win).detach().float().cpu()
        del hidden, batch_ids
    return values


def summarize_heldout_tail(token_values_by_win):
    by_class = {cls: [] for cls in TF.CLASSES}
    per_window = {}
    for win, values in token_values_by_win.items():
        cls = PROBE_CLASS_BY_WIN[int(win)]
        by_class[cls].append(values)
        per_window[str(win)] = {"class": cls, **_tail_stats(values)}
    if any(not parts for parts in by_class.values()):
        raise RuntimeError("held-out tail summary is missing a represented class")
    all_values = torch.cat(list(token_values_by_win.values()))
    return {
        "instrument": "held-out scorer-exact per-position KL on teacher top-8192 support",
        "n_windows": len(token_values_by_win),
        "global": _tail_stats(all_values),
        "by_class": {
            cls: _tail_stats(torch.cat(parts))
            for cls, parts in by_class.items()
        },
        "per_window": per_window,
    }


def tailfix_batch_loss(student, corpus, acache, wins, requires_grad):
    """Matched mean-KLD control or detached tail-weighted objective."""
    real_lengths = [B.T.window_ids(corpus, win)[1] for win in wins]
    ids = torch.stack([B.T.window_ids(corpus, win)[0] for win in wins]).to(B.DEV)
    hidden = torch.cat([acache.get(corpus, win) for win in wins], 0)
    hidden = B.fast_forward(student, hidden, ids, requires_grad)
    token_values = [
        _token_kld(student, hidden[j], real_lengths[j], win)
        for j, win in enumerate(wins)
    ]
    window_means = torch.stack([values.mean() for values in token_values])
    if ARM in {"control", "trajectory_micro", "trajectory_full"}:
        return window_means.mean(), {
            "objective": "mean_kld",
            "window_kld_means": [float(value.detach()) for value in window_means],
        }

    tail_losses = []
    token_thresholds = []
    token_tail_mass = []
    for values in token_values:
        weights, threshold = _normalized_detached_weights(
            values, TAIL_TOKEN_QUANTILE, TAIL_TOKEN_BOOST
        )
        tail_losses.append((values * weights).mean())
        token_thresholds.append(threshold)
        token_tail_mass.append(float(weights[values.detach().float() >= threshold].sum() / weights.sum()))
    tail_losses_tensor = torch.stack(tail_losses)
    window_weights, window_threshold = _normalized_detached_weights(
        window_means, TAIL_WINDOW_QUANTILE, TAIL_WINDOW_BOOST
    )
    loss = (tail_losses_tensor * window_weights).mean()
    return loss, {
        "objective": "tail_weighted_kld",
        "window_kld_means": [float(value.detach()) for value in window_means],
        "token_tail_thresholds": token_thresholds,
        "token_tail_mass": token_tail_mass,
        "window_tail_threshold": window_threshold,
        "window_weights": [float(value) for value in window_weights],
    }


def load_trajectory_rows(path: Path) -> list[dict[str, object]]:
    """Load sealed FP-visible trajectories and enforce the plan identity."""
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_source_sha = PLAN.get("trajectory_source_sha256")
    if expected_source_sha is not None and sha256_file(path) != expected_source_sha:
        raise RuntimeError("trajectory source digest mismatch")
    by_task = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = str(row["task_id"])
        if task_id in by_task:
            raise RuntimeError(f"duplicate trajectory task: {task_id}")
        token_ids = list(map(int, row["token_ids"]))
        score_start = int(row["score_start"])
        if not 1 <= score_start < len(token_ids):
            raise RuntimeError(f"invalid trajectory score span: {task_id}")
        if int(row["n_score_tokens"]) != len(token_ids) - score_start:
            raise RuntimeError(f"trajectory score length mismatch: {task_id}")
        by_task[task_id] = {
            "task_id": task_id,
            "token_ids": token_ids,
            "score_start": score_start,
            "n_score_tokens": int(row["n_score_tokens"]),
        }
    holdout = set(map(str, PLAN.get("trajectory_holdout_task_ids", [])))
    contamination = sorted(holdout & set(by_task))
    if contamination:
        raise RuntimeError(f"trajectory source contains held-out tasks: {contamination}")
    selected = []
    for task_id in map(str, PLAN["trajectory_task_ids"]):
        if task_id in holdout:
            raise RuntimeError(f"trajectory schedule contains held-out task: {task_id}")
        if task_id not in by_task:
            raise RuntimeError(f"missing trajectory task: {task_id}")
        selected.append(by_task[task_id])
    expected = int(PLAN["steps"]) * int(PLAN["trajectory_batch"])
    if len(selected) != expected:
        raise RuntimeError("trajectory schedule size drift")
    return selected


def trajectory_batch_nll(student, rows, requires_grad):
    """Full-vocabulary hard NLL on frozen FP-visible continuation tokens.

    Right padding is causally after every scored token and cannot alter a
    scored logit. It lets four trajectories share one full-model forward.
    """
    max_len = max(len(row["token_ids"]) for row in rows)
    ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=B.DEV)
    for j, row in enumerate(rows):
        values = torch.tensor(row["token_ids"], dtype=torch.long, device=B.DEV)
        ids[j, : values.numel()] = values
    embeds = student.model.model.embed_tokens(ids)
    hidden = embeds.unsqueeze(2).expand(
        -1, -1, student.config.hc_mult, -1
    ).contiguous()
    del embeds
    if requires_grad:
        hidden = B.fast_forward(student, hidden, ids, True)
    else:
        with torch.no_grad():
            hidden = B.fast_forward(student, hidden, ids, False)
    losses = []
    per_task = []
    for j, row in enumerate(rows):
        real_len = len(row["token_ids"])
        score_start = int(row["score_start"])
        logits = student.model.lm_head(
            hidden[j, score_start - 1 : real_len - 1].to(torch.bfloat16)
        ).float()
        targets = ids[j, score_start:real_len]
        loss = F.cross_entropy(logits, targets, reduction="mean")
        losses.append(loss)
        per_task.append({
            "task_id": str(row["task_id"]),
            "n_score_tokens": int(row["n_score_tokens"]),
            "nll": float(loss.detach()),
        })
        del logits
    return torch.stack(losses).mean(), per_task


def reused_baseline(prefix: str):
    """Reuse the control's exact frozen-step32 held-out pass, never its state."""
    source = Path(os.path.expanduser(prefix))
    checkpoint_path = Path(str(source) + ".latest.pt")
    tail_path = Path(str(source) + ".heldout_tail_baseline.json")
    if not checkpoint_path.is_file() or not tail_path.is_file():
        raise FileNotFoundError(f"control baseline incomplete: {source}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_sha = checkpoint.get("artifact_identity", {}).get("start_checkpoint_sha256")
    if source_sha != TF.BASELINE_START_SHA256:
        raise RuntimeError(f"control baseline lineage mismatch: {source_sha}")
    baseline = baseline_dict(checkpoint)
    tail = json.loads(tail_path.read_text())
    if set(baseline) != set(map(int, PLAN["probe_wins"])):
        raise RuntimeError("control baseline probe set mismatch")
    if set(tail.get("by_class", {})) != set(TF.CLASSES):
        raise RuntimeError("control baseline class set mismatch")
    return baseline, tail, {
        "prefix": str(source),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "tail": str(tail_path),
        "tail_sha256": sha256_file(tail_path),
    }


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    TF.validate_plan(PLAN)
    selected_pool = list(map(int, PLAN["train_wins"]))
    selected_probes = list(map(int, PLAN["probe_wins"]))
    if selected_pool != B.TRAIN_WINS:
        raise RuntimeError("tailfix plan train pool does not match BR_TRAIN")
    if selected_probes != B.PROBE_WINS:
        raise RuntimeError("tailfix plan held-out pool does not match BR_PROBE")
    if B.STEPS != len(TRAINING_SCHEDULE) or B.BATCH != int(PLAN["batch"]):
        raise RuntimeError("tailfix matched schedule/budget drift")
    v2_layout = {
        "disjoint": True,
        "train_window_count": len(selected_pool),
        "heldout_window_count": len(selected_probes),
        "total_window_exposures": PLAN["total_window_exposures"],
        "plan_sha256": PLAN["plan_sha256"],
        "arm": ARM,
        "objective": PLAN["objective"],
        "target_gradient_mass_by_class": PLAN["target_gradient_mass_by_class"],
    }

    B.OUTDIR.mkdir(parents=True, exist_ok=True)
    B.status(
        state="starting",
        mechanism=MECHANISM,
        pool_role=POOL_ROLE,
        arm=ARM,
        objective=PLAN["objective"],
        plan_sha256=PLAN["plan_sha256"],
        target_gradient_mass_by_class=PLAN["target_gradient_mass_by_class"],
        manifest_md5=B.AMD5,
        codebook_lr=B.LR,
        norm_lr=float(os.environ.get("COMBO_NORM_LR", "1e-4")),
        output_lr=OUTPUT_LR,
        min_lr_ratio=MIN_LR_RATIO,
        steps=B.STEPS,
        batch=B.BATCH,
        train_wins=B.TRAIN_WINS,
        probe_wins=B.PROBE_WINS,
        purity_wins=PURITY_WINS,
        v2_layout=v2_layout,
    )
    B.emit(
        event="start",
        mechanism=MECHANISM,
        pool_role=POOL_ROLE,
        arm=ARM,
        objective=PLAN["objective"],
        plan_sha256=PLAN["plan_sha256"],
        target_gradient_mass_by_class=PLAN["target_gradient_mass_by_class"],
        seed=SEED,
        train_wins=B.TRAIN_WINS,
        probe_wins=B.PROBE_WINS,
        purity_wins=PURITY_WINS,
        v2_layout=v2_layout,
    )

    B.T.TrainableExperts = B.K4096Experts
    B.T.PILOT = tuple(EXPECTED_LAYERS)
    student = B.T.Student()
    norms, outputs = expose_combo_parameters(student)
    codebooks = codebook_params(student)
    baseline, seed_info = validate_and_load_seeds(student, norms, outputs)
    seed_info = load_v2_warm_start(student, norms, outputs, seed_info)
    artifact_identity = identity(seed_info)
    norm_params = [parameter for _name, _module, parameter in norms]
    output_params = [parameter for _name, _module, parameter in outputs]
    norm_lr = float(os.environ.get("COMBO_NORM_LR", "1e-4"))
    optimizer = torch.optim.Adam(
        [
            {"params": codebooks, "lr": B.LR, "group_name": "codebooks"},
            {"params": norm_params, "lr": norm_lr, "group_name": "norms"},
            {"params": output_params, "lr": OUTPUT_LR, "group_name": "outputs"},
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda step: C.cosine_multiplier(step, B.STEPS, MIN_LR_RATIO),
            lambda step: C.cosine_multiplier(step, B.STEPS, MIN_LR_RATIO),
            lambda step: C.cosine_multiplier(step, B.STEPS, MIN_LR_RATIO),
        ],
    )
    all_params = codebooks + norm_params + output_params
    B.emit(
        event="assembled",
        n_codebook_params=sum(parameter.numel() for parameter in codebooks),
        n_norm_params=sum(parameter.numel() for parameter in norm_params),
        n_output_params=sum(parameter.numel() for parameter in output_params),
        n_trainable_params=sum(parameter.numel() for parameter in all_params),
        semantic_overlap_tensors=seed_info["semantic_overlap_tensors"],
        semantic_overlap_params=seed_info["semantic_overlap_params"],
        overlap_policy=seed_info["overlap_policy"],
        seed_sha256=seed_info["seed_sha256"],
        warm_start=seed_info.get("warm_start"),
    )

    corpus = B.T.load_corpus()
    trajectory_rows = (
        load_trajectory_rows(TRAJECTORY_PATH)
        if ARM in {"trajectory_micro", "trajectory_full"} and TRAJECTORY_PATH is not None
        else []
    )
    if ARM in {"trajectory_micro", "trajectory_full"}:
        B.emit(
            event="trajectory_corpus",
            path=str(TRAJECTORY_PATH),
            sha256=sha256_file(TRAJECTORY_PATH),
            tasks=[row["task_id"] for row in trajectory_rows],
            weight=TRAJECTORY_WEIGHT,
            caveat=PLAN["trajectory_caveat"],
        )
    if B.GRADCHECK:
        B.gradcheck(student)
        gradcheck_aux(norms, outputs)
    acache = B.ActCache(student)
    start_step, resumed_baseline, best_mean, step0_mean = try_resume(
        student, norms, outputs, optimizer, scheduler, artifact_identity
    )
    baseline_tail = None
    if resumed_baseline is not None:
        baseline = {int(key): float(value) for key, value in resumed_baseline.items()}
        if not TAIL_BASELINE.is_file():
            raise RuntimeError("resumed tailfix checkpoint is missing held-out tail baseline")
        baseline_tail = json.loads(TAIL_BASELINE.read_text())
    elif start_step == 0 and BASELINE_SOURCE:
        baseline, baseline_tail, baseline_reuse = reused_baseline(BASELINE_SOURCE)
        B.atomic_json(TAIL_BASELINE, baseline_tail)
        B.emit(event="heldout_baseline_reused", source=baseline_reuse)
    elif start_step == 0:
        baseline = {}
        baseline_token_values = {}
        for win in B.PROBE_WINS:
            t0 = time.time()
            token_values = heldout_token_values(student, corpus, acache, win)
            value = float(token_values.mean())
            if not math.isfinite(value) or not 0.0 <= value < 5.0:
                raise RuntimeError(f"non-physical held-out baseline win={win}: {value}")
            baseline[win] = value
            baseline_token_values[win] = token_values
            win_tail = _tail_stats(token_values)
            B.emit(
                event="heldout_baseline",
                win=win,
                source_class=PROBE_CLASS_BY_WIN[win],
                kld=value,
                p95=win_tail["p95"],
                p99=win_tail["p99"],
                secs=round(time.time() - t0, 1),
            )
            torch.cuda.empty_cache()
        baseline_tail = summarize_heldout_tail(baseline_token_values)
        B.atomic_json(TAIL_BASELINE, baseline_tail)
        B.emit(event="heldout_tail_baseline", stats=baseline_tail)
    else:
        raise RuntimeError("resumed tailfix checkpoint is missing held-out baseline")
    if set(baseline) != set(B.PROBE_WINS):
        raise RuntimeError("tailfix held-out baseline set drift")
    if baseline_tail is None:
        raise RuntimeError("tailfix held-out tail baseline missing")

    def probe(step):
        values = {}
        token_values_by_win = {}
        for win in B.PROBE_WINS:
            t0 = time.time()
            token_values = heldout_token_values(student, corpus, acache, win)
            value = float(token_values.mean())
            values[win] = value
            token_values_by_win[win] = token_values
            win_tail = _tail_stats(token_values)
            B.emit(
                event="probe",
                step=step,
                win=win,
                source_class=PROBE_CLASS_BY_WIN[win],
                kld=value,
                p95=win_tail["p95"],
                p99=win_tail["p99"],
                baseline=baseline[win],
                delta_pct=round((baseline[win] - value) / baseline[win] * 100, 4),
                secs=round(time.time() - t0, 1),
            )
            torch.cuda.empty_cache()
        tail_summary = summarize_heldout_tail(token_values_by_win)
        mean = sum(values.values()) / len(values)
        base_mean = sum(baseline[win] for win in B.PROBE_WINS) / len(B.PROBE_WINS)
        delta = (base_mean - mean) / base_mean * 100
        B.emit(
            event="probe_mean",
            step=step,
            mean=mean,
            baseline_mean=base_mean,
            delta_pct=delta,
            floor_label=C.floor_label(delta),
            hypothesis_band=C.hypothesis_band(delta),
        )
        B.emit(event="heldout_tail_probe", step=step, stats=tail_summary)
        B.status(
            state="running",
            last_probe_step=step,
            last_probe_mean=mean,
            baseline_probe_mean=base_mean,
            last_probe_delta_pct=delta,
            last_probe_tail=tail_summary,
            floor_label=C.floor_label(delta),
            hypothesis_band=C.hypothesis_band(delta),
        )
        return mean, tail_summary

    def purity_probe(stage):
        values = {}
        for win in PURITY_WINS:
            t0 = time.time()
            value = B.kld_window(student, corpus, acache, win)
            values[win] = value
            B.emit(
                event="purity_probe",
                stage=stage,
                win=win,
                kld=value,
                secs=round(time.time() - t0, 1),
            )
            torch.cuda.empty_cache()
        B.emit(
            event="purity_probe_mean",
            stage=stage,
            mean=sum(values.values()) / len(values),
            wins=PURITY_WINS,
        )
        return values

    purity_warm = {}
    purity_path = Path(str(B.PREFIX) + ".purity_warm.json")
    if POOL_ROLE == "V2":
        if start_step == 0:
            purity_warm = purity_probe("warm_start")
            B.atomic_json(purity_path, purity_warm)
        else:
            if not purity_path.is_file():
                raise RuntimeError("resumed COMBO-V2 missing warm-start purity baseline")
            purity_warm = {
                int(key): float(value)
                for key, value in json.loads(purity_path.read_text()).items()
            }

    if start_step == 0:
        step0_mean = sum(baseline.values()) / len(baseline)
        best_mean = step0_mean
        B.emit(
            event="probe_mean",
            step=0,
            mean=step0_mean,
            baseline_mean=step0_mean,
            delta_pct=0.0,
            floor_label=C.floor_label(0.0),
            hypothesis_band=C.hypothesis_band(0.0),
            source="single exact frozen-step32 baseline pass",
        )
        save_checkpoint(
            B.BEST,
            student,
            norms,
            outputs,
            optimizer,
            scheduler,
            0,
            baseline,
            artifact_identity,
            best_mean,
            step0_mean,
        )
        save_checkpoint(
            B.LATEST,
            student,
            norms,
            outputs,
            optimizer,
            scheduler,
            0,
            baseline,
            artifact_identity,
            best_mean,
            step0_mean,
        )
        if ARM != "control":
            save_checkpoint(
                TAIL_BEST,
                student,
                norms,
                outputs,
                optimizer,
                scheduler,
                0,
                baseline,
                artifact_identity,
                best_mean,
                step0_mean,
            )
        tail_selection = {
            "arm": ARM,
            "selection_rule": (
                "heldout_mean_kld" if ARM == "control"
                else "code_p99_improves_with_all_represented_class_means_within_1pct"
            ),
            "selected": False,
            "selected_step": 0,
            "selected_checkpoint": str(B.BEST if ARM == "control" else TAIL_BEST),
            "selected_probe_mean": step0_mean,
            "selected_tail_stats": baseline_tail,
            "decision": {"reason": "warm_start"},
        }
        B.atomic_json(TAIL_SELECTION, tail_selection)
    else:
        if not TAIL_SELECTION.is_file():
            raise RuntimeError("resumed tailfix checkpoint is missing tail selection receipt")
        tail_selection = json.loads(TAIL_SELECTION.read_text())
    if best_mean is None or step0_mean is None:
        raise RuntimeError("combo best/step0 mean missing")

    stall = 0
    started = time.time()
    for step in range(start_step, B.STEPS):
        if (time.time() - started) / 3600 > B.MAX_HOURS:
            B.emit(event="wall_guard", step=step)
            break
        wins = TRAINING_SCHEDULE[step]
        batch_class = TRAINING_CLASSES[step]
        t0 = time.time()
        standard_loss, objective_stats = tailfix_batch_loss(
            student, corpus, acache, wins, True
        )
        kld_pre = float(standard_loss.detach())
        optimizer.zero_grad(set_to_none=True)
        if ARM in {"trajectory_micro", "trajectory_full"}:
            # Backprop the standard term before constructing the trajectory
            # graph. This preserves the exact summed gradient while avoiding
            # two full batch graphs resident at once on unified memory.
            standard_loss.backward()
            del standard_loss
            torch.cuda.empty_cache()
            trajectory_batch = int(PLAN["trajectory_batch"])
            lo = step * trajectory_batch
            rows = trajectory_rows[lo : lo + trajectory_batch]
            trajectory_loss, trajectory_stats = trajectory_batch_nll(student, rows, True)
            weighted_trajectory_loss = TRAJECTORY_WEIGHT * trajectory_loss
            weighted_trajectory_loss.backward()
            objective_stats = {
                **objective_stats,
                "objective": PLAN["objective"],
                "standard_mean_kld": kld_pre,
                "trajectory_weight": TRAJECTORY_WEIGHT,
                "trajectory_hard_nll": float(trajectory_loss.detach()),
                "combined_pre_update": kld_pre + float(weighted_trajectory_loss.detach()),
                "trajectory_tasks": trajectory_stats,
                "trajectory_caveat": PLAN["trajectory_caveat"],
            }
            loss = weighted_trajectory_loss
            del trajectory_loss
        else:
            loss = standard_loss
            loss.backward()
        group_grad_norms = {}
        for name, params in (
            ("codebooks", codebooks),
            ("norms", norm_params),
            ("outputs", output_params),
        ):
            group_grad_norms[name] = float(
                sum(
                    parameter.grad.norm() ** 2
                    for parameter in params
                    if parameter.grad is not None
                )
                ** 0.5
            )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        del loss
        torch.cuda.empty_cache()
        next_step = step + 1
        save_checkpoint(
            B.LATEST,
            student,
            norms,
            outputs,
            optimizer,
            scheduler,
            next_step,
            baseline,
            artifact_identity,
            best_mean,
            step0_mean,
        )
        if next_step % 4 == 0:
            save_checkpoint(
                Path(str(B.PREFIX) + f".step{next_step:04d}.pt"),
                student,
                norms,
                outputs,
                optimizer,
                scheduler,
                next_step,
                baseline,
                artifact_identity,
                best_mean,
                step0_mean,
            )
        B.emit(
            event="step",
            step=next_step,
            train_wins=wins,
            batch_class=batch_class,
            target_gradient_mass_by_class=PLAN["target_gradient_mass_by_class"],
            kld_pre_update=kld_pre,
            objective_stats=objective_stats,
            group_grad_norms=group_grad_norms,
            lrs={
                group["group_name"]: group["lr"]
                for group in optimizer.param_groups
            },
            secs=round(time.time() - t0, 1),
            mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1),
        )
        B.status(
            state="running",
            next_step=next_step,
            last_batch_class=batch_class,
            last_kld_pre_update=kld_pre,
            last_objective_stats=objective_stats,
            last_group_grad_norms=group_grad_norms,
        )
        if (not SKIP_FULL_PROBE) and (
            next_step % B.PROBE_EVERY == 0 or next_step == B.STEPS
        ):
            mean, tail_summary = probe(next_step)
            mean_improved = mean < best_mean - 1e-6
            if mean_improved:
                best_mean = mean
                stall = 0
                save_checkpoint(
                    B.BEST,
                    student,
                    norms,
                    outputs,
                    optimizer,
                    scheduler,
                    next_step,
                    baseline,
                    artifact_identity,
                    best_mean,
                    step0_mean,
                )
                B.emit(event="best", step=next_step, mean=mean)
            else:
                stall += 1
                B.emit(event="stall", step=next_step, count=stall)

            if ARM == "control":
                tail_selection = {
                    "arm": ARM,
                    "selection_rule": "heldout_mean_kld",
                    "selected": mean_improved,
                    "selected_step": next_step if mean_improved else 0,
                    "selected_checkpoint": str(B.BEST),
                    "selected_probe_mean": best_mean,
                    "selected_tail_stats": tail_summary if mean_improved else baseline_tail,
                    "candidate_step": next_step,
                    "candidate_probe_mean": mean,
                    "candidate_tail_stats": tail_summary,
                    "decision": {
                        "reason": "heldout_mean_improved" if mean_improved else "heldout_mean_not_improved"
                    },
                }
            else:
                decision = TF.tail_candidate_decision(
                    baseline_tail["by_class"], tail_summary["by_class"]
                )
                if decision["selected"]:
                    save_checkpoint(
                        TAIL_BEST,
                        student,
                        norms,
                        outputs,
                        optimizer,
                        scheduler,
                        next_step,
                        baseline,
                        artifact_identity,
                        best_mean,
                        step0_mean,
                    )
                tail_selection = {
                    "arm": ARM,
                    "selection_rule": "code_p99_improves_with_all_represented_class_means_within_1pct",
                    "selected": decision["selected"],
                    "selected_step": next_step if decision["selected"] else 0,
                    "selected_checkpoint": str(TAIL_BEST),
                    "selected_probe_mean": mean if decision["selected"] else step0_mean,
                    "selected_tail_stats": tail_summary if decision["selected"] else baseline_tail,
                    "candidate_step": next_step,
                    "candidate_probe_mean": mean,
                    "candidate_tail_stats": tail_summary,
                    "decision": decision,
                }
            B.atomic_json(TAIL_SELECTION, tail_selection)
            B.emit(event="tail_selection", step=next_step, **tail_selection)
            if (
                BINDING_GATE_STOP
                and ARM == "trajectory_full"
                and decision.get("mean_guard_violations")
            ):
                gate_path = Path(str(B.PREFIX) + ".binding_gate_stop.json")
                candidate_checkpoint = Path(
                    str(B.PREFIX) + f".step{next_step:04d}.pt"
                )
                gate = {
                    "format": "tailfix-trajectory-binding-gate-v2",
                    "bin": PLAN.get("bin", "PTQ-OPD"),
                    "state": "stopped_binding_all_class_regression",
                    "arm": ARM,
                    "step": next_step,
                    "reason": "one or more represented held-out class means regressed by more than 1pct",
                    "plan_sha256": PLAN["plan_sha256"],
                    "trajectory_holdout_seed": PLAN.get("trajectory_holdout_seed"),
                    "trajectory_holdout_task_ids": PLAN.get("trajectory_holdout_task_ids"),
                    "trajectory_holdout_sha256": PLAN.get("trajectory_holdout_sha256"),
                    "trajectory_source_sha256": PLAN.get("trajectory_source_sha256"),
                    "decision": decision,
                    "candidate_probe_mean": mean,
                    "candidate_tail_stats": tail_summary,
                    "baseline_tail_stats": baseline_tail,
                    "candidate_checkpoint": str(candidate_checkpoint),
                    "selected_checkpoint": str(START_PATH),
                    "ts": time.time(),
                }
                B.atomic_json(gate_path, gate)
                B.emit(event="binding_all_class_stop", receipt=str(gate_path), **gate)
                B.status(
                    state="stopped_binding_all_class_regression",
                    next_step=next_step,
                    binding_gate_receipt=str(gate_path),
                    selected_checkpoint=str(START_PATH),
                    rejected_checkpoint=str(candidate_checkpoint),
                    decision=decision,
                )
                return
            if stall >= B.EARLY_STOP:
                B.emit(event="early_stop", step=next_step)
                break

    if ARM in {"trajectory_micro", "trajectory_full"} and SKIP_FULL_PROBE:
        latest = torch.load(B.LATEST, map_location="cpu", weights_only=False)
        if int(latest.get("next_step", -1)) != B.STEPS:
            raise RuntimeError("trajectory micro-dose did not reach its atomic final step")
        tail_selection = {
            "arm": ARM,
            "selection_rule": "pending frozen fast-signal panel",
            "selected": True,
            "selected_step": B.STEPS,
            "selected_checkpoint": str(B.LATEST),
            "selected_probe_mean": step0_mean,
            "selected_tail_stats": baseline_tail,
            "decision": {
                "reason": "microdose_complete_pending_32prefix_and_32window_signal",
                "full_probe_skipped_for_deadline": True,
            },
        }
        B.atomic_json(TAIL_SELECTION, tail_selection)
        B.emit(event="trajectory_micro_complete", **tail_selection)

    purity = None
    if POOL_ROLE == "V2":
        best_checkpoint = torch.load(B.BEST, map_location="cpu", weights_only=False)
        load_named(student, norms, outputs, best_checkpoint["state"])
        purity_final = purity_probe("sealed_best")
        purity = V2.purity_metrics(warm_start=purity_warm, final=purity_final)

    base_mean = sum(baseline[win] for win in B.PROBE_WINS) / len(B.PROBE_WINS)
    step0_delta = (base_mean - step0_mean) / base_mean * 100
    mean_best_delta = (base_mean - best_mean) / base_mean * 100
    selected_mean = float(tail_selection["selected_probe_mean"])
    selected_checkpoint = str(tail_selection["selected_checkpoint"])
    selected_delta = (base_mean - selected_mean) / base_mean * 100
    result = {
        "state": "completed",
        "format": FORMAT,
        "mechanism": MECHANISM,
        "manifest_md5": B.AMD5,
        "pool_role": POOL_ROLE,
        "arm": ARM,
        "objective": PLAN["objective"],
        "trajectory_weight": PLAN.get("trajectory_weight"),
        "trajectory_task_ids": PLAN.get("trajectory_task_ids"),
        "trajectory_holdout_seed": PLAN.get("trajectory_holdout_seed"),
        "trajectory_holdout_task_ids": PLAN.get("trajectory_holdout_task_ids"),
        "trajectory_holdout_sha256": PLAN.get("trajectory_holdout_sha256"),
        "trajectory_source_sha256": PLAN.get("trajectory_source_sha256"),
        "trajectory_caveat": PLAN.get("trajectory_caveat"),
        "fast_signal_pending": ARM in {"trajectory_micro", "trajectory_full"} and SKIP_FULL_PROBE,
        "plan_sha256": PLAN["plan_sha256"],
        "start_checkpoint_sha256": TF.START_SHA256,
        "target_gradient_mass_by_class": PLAN["target_gradient_mass_by_class"],
        "exposures_by_class": PLAN["exposures_by_class"],
        "schedule": PLAN["schedule"],
        "train_wins": B.TRAIN_WINS,
        "probe_wins": B.PROBE_WINS,
        "purity_wins": PURITY_WINS,
        "held_out": True,
        "n_probe": len(B.PROBE_WINS),
        "n_codebook_params": sum(parameter.numel() for parameter in codebooks),
        "n_norm_params": sum(parameter.numel() for parameter in norm_params),
        "n_output_params": sum(parameter.numel() for parameter in output_params),
        "n_trainable_params": sum(parameter.numel() for parameter in all_params),
        "semantic_overlap_tensors": seed_info["semantic_overlap_tensors"],
        "semantic_overlap_params": seed_info["semantic_overlap_params"],
        "overlap_policy": seed_info["overlap_policy"],
        "codebook_lr": B.LR,
        "norm_lr": norm_lr,
        "output_lr": OUTPUT_LR,
        "min_lr_ratio": MIN_LR_RATIO,
        "baseline_probe_mean": base_mean,
        "step0_combo_mean": step0_mean,
        "step0_combo_delta_pct": step0_delta,
        "best_probe_mean": selected_mean,
        "best_delta_pct": selected_delta,
        "mean_best_probe_mean": best_mean,
        "mean_best_delta_pct": mean_best_delta,
        "floor_label": C.floor_label(selected_delta),
        "hypothesis_band": C.hypothesis_band(selected_delta),
        "effect_floor_pct": 2.6,
        "best_checkpoint": selected_checkpoint,
        "mean_best_checkpoint": str(B.BEST),
        "heldout_tail_baseline": baseline_tail,
        "heldout_tail_selection": tail_selection,
        "latest": str(B.LATEST),
        "seed_sha256": seed_info["seed_sha256"],
        "warm_start": seed_info.get("warm_start"),
        "v2_layout": v2_layout,
        "purity": purity,
        "host": os.uname().nodename,
        "ts": time.time(),
    }
    B.atomic_json(B.FINAL, result)
    B.emit(event="completed", **result)
    B.status(**result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        try:
            B.status(state="failed", mechanism=MECHANISM, error=f"{type(exc).__name__}: {exc}")
            B.emit(event="failed", mechanism=MECHANISM, error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
