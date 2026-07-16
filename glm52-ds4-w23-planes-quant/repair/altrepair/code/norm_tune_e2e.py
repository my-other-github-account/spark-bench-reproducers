#!/usr/bin/env python3
"""Orthogonal BINREPAIR mechanism #2: RMSNorm-gamma tuning.

Uses the proven frozen k4096-menu artifact and scorer-exact KL harness.  All
quantized experts, codebooks, attention weights, router weights, embeddings,
and heads remain frozen.  RMSNorm weights are exposed through a float32 master
with a BF16 forward parametrization, preserving the exact step-0 training graph
while avoiding low-precision optimizer-state updates.

Additional environment:
  ALT_BINREPAIR_BASE  path to binrepair_e2e.py
  NT_SCOPE            all (default) or block (input/post-attention + final)
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize


BASE_PATH = Path(os.path.expanduser(os.environ.get(
    "ALT_BINREPAIR_BASE", str(Path(__file__).with_name("binrepair_e2e.py"))
)))
SCOPE = os.environ.get("NT_SCOPE", "all")
if SCOPE not in {"all", "block"}:
    raise ValueError(f"NT_SCOPE must be all or block, got {SCOPE}")
if not BASE_PATH.is_file():
    raise FileNotFoundError(f"ALT_BINREPAIR_BASE not found: {BASE_PATH}")
_spec = importlib.util.spec_from_file_location("binrepair_base_norm", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load base harness: {BASE_PATH}")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

FORMAT = "altrepair-rmsnorm-v1"
MECHANISM = "rmsnorm-gamma"
BASELINE_SEED = os.environ.get("ALT_BASELINE_CKPT", "")


class FrozenArtifactExperts(B.K4096Experts):
    """Exact artifact reader with no trainable expert/codebook parameters."""

    def __init__(self, layer: int, pilot: bool):
        super().__init__(layer, False)
        self.pilot = False


class WireBf16(nn.Module):
    """Float32 master -> exact BF16 weight used by the frozen model graph."""

    def forward(self, master: torch.Tensor) -> torch.Tensor:
        return master.to(torch.bfloat16)


def _is_selected(module_name: str) -> bool:
    leaf = module_name.rsplit(".", 1)[-1].lower()
    if SCOPE == "all":
        return "norm" in leaf
    return (leaf in {"input_layernorm", "post_attention_layernorm"}
            or module_name == "model.norm")


def expose_norm_masters(student):
    selected = []
    # Snapshot because registration mutates module parameter traversal.
    for module_name, module in list(student.model.named_modules()):
        if not _is_selected(module_name):
            continue
        wire = module._parameters.get("weight")
        if wire is None or wire.ndim != 1:
            continue
        before = wire.detach().clone()
        # Register while the original remains BF16, then promote only the
        # hidden original parameter. PyTorch rejects a dtype-changing
        # right_inverse even with unsafe=True, but supports this explicit
        # promotion while the exposed forward remains BF16.
        parametrize.register_parametrization(
            module, "weight", WireBf16(), unsafe=True)
        master = module.parametrizations.weight.original
        master.data = master.data.float()
        master.requires_grad_(True)
        after = module.weight.detach()
        if after.dtype != torch.bfloat16 or not torch.equal(after, before):
            raise AssertionError(f"norm parametrization changed init: {module_name}")
        selected.append((module_name, module, master))
    if not selected:
        raise RuntimeError(f"no RMSNorm weights selected for scope={SCOPE}")
    return selected


def instant_artifact_identity():
    plane_sizes = {}
    delta_sizes = {}
    for layer in range(B.T.config.num_hidden_layers if hasattr(B.T, "config") else 43):
        plane = B.VQ3B_DIR / f"vq3u_layer_{layer:03d}.pt"
        delta = B.DELTA_DIR / f"layer_{layer:03d}.pt"
        plane_sizes[str(layer)] = plane.stat().st_size
        delta_sizes[str(layer)] = delta.stat().st_size
    sample = torch.load(B.VQ3B_DIR / "vq3u_layer_000.pt",
                        map_location="cpu", mmap=True, weights_only=True)
    return {
        "policy": "exists-size-plus-one-header; ledger-parity-is-integrity-gate",
        "plane_sizes": plane_sizes,
        "delta_sizes": delta_sizes,
        "sample_header": {
            key: {"shape": list(sample[key].shape),
                  "dtype": str(sample[key].dtype)}
            for key in ("cb13", "cb2", "codes13", "codes2", "sc13", "sc2")
        },
    }


def gradcheck(selected):
    name, module, master = selected[0]
    x = torch.tensor([0.25, -0.5, 1.5, 2.0], device=B.DEV)
    p = master.reshape(-1)[0]
    module.zero_grad(set_to_none=True)
    # Test the same FP32-master -> BF16 cast path directly.  A large enough
    # perturbation crosses BF16 bins, with the realized BF16 step in the FD
    # denominator just like the codebook STE check.
    y = (x.to(torch.bfloat16) * p.to(torch.bfloat16)).float().sum()
    y.backward()
    autograd = float(master.grad.reshape(-1)[0])
    original = float(p.detach())
    h = max(abs(original) / 64.0, 1e-2)
    with torch.no_grad():
        p.copy_(torch.tensor(original + h, device=B.DEV))
        plus_wire = float(p.to(torch.bfloat16))
        plus = float((x * plus_wire).sum())
        p.copy_(torch.tensor(original - h, device=B.DEV))
        minus_wire = float(p.to(torch.bfloat16))
        minus = float((x * minus_wire).sum())
        p.copy_(torch.tensor(original, device=B.DEV))
    finite_difference = (plus - minus) / (plus_wire - minus_wire)
    rel = abs(finite_difference - autograd) / max(
        abs(finite_difference), abs(autograd), 1e-8)
    B.emit(event="norm_gradcheck", parameter=name, autograd=autograd,
           fd=finite_difference, rel=round(rel, 7), init_identity=True)
    if rel >= 2e-3:
        raise AssertionError(f"norm gradcheck FAIL rel={rel}")
    module.zero_grad(set_to_none=True)


def state_named(selected):
    return {name: master.detach().cpu() for name, _module, master in selected}


def save_ckpt(path, selected, optimizer, next_step, baseline, identity,
              best_mean):
    payload = {
        "format": FORMAT,
        "mechanism": MECHANISM,
        "scope": SCOPE,
        "manifest_md5": B.AMD5,
        "artifact_identity": identity,
        "lr": B.LR,
        "steps_target": B.STEPS,
        "train_wins": B.TRAIN_WINS,
        "probe_wins": B.PROBE_WINS,
        "next_step": next_step,
        "baseline": baseline,
        "best_probe_mean": best_mean,
        "state": state_named(selected),
        "optimizer": optimizer.state_dict(),
        "saved_ts": time.time(),
        "host": os.uname().nodename,
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def try_resume(selected, optimizer, identity):
    if not B.LATEST.exists():
        if not BASELINE_SEED:
            return 0, None, None
        seed_path = Path(os.path.expanduser(BASELINE_SEED))
        if not seed_path.is_file():
            raise FileNotFoundError(f"baseline seed not found: {seed_path}")
        seed = torch.load(seed_path, map_location="cpu", weights_only=False)
        baseline = seed.get("baseline")
        required = set(B.TRAIN_WINS) | set(B.PROBE_WINS)
        present = ({int(key) for key in baseline}
                   if isinstance(baseline, dict) else set())
        seed_train = {int(win) for win in seed.get("train_wins", [])}
        bad = []
        if seed.get("manifest_md5") != B.AMD5:
            bad.append("manifest_md5")
        if seed_train != set(B.TRAIN_WINS):
            bad.append("train_window_set")
        if seed.get("probe_wins") != B.PROBE_WINS:
            bad.append("probe_wins")
        if not required.issubset(present):
            bad.append(f"missing_windows={sorted(required-present)}")
        if bad:
            raise RuntimeError(f"baseline seed mismatch: {bad}")
        B.emit(event="baseline_seeded", source=str(seed_path),
               count=len(baseline),
               verification="own step0 8-probe panel follows")
        return 0, baseline, None
    ckpt = torch.load(B.LATEST, map_location="cpu", weights_only=False)
    expected = {
        "format": FORMAT,
        "mechanism": MECHANISM,
        "scope": SCOPE,
        "manifest_md5": B.AMD5,
        "artifact_identity": identity,
        "lr": B.LR,
        "train_wins": B.TRAIN_WINS,
        "probe_wins": B.PROBE_WINS,
    }
    bad = {key: (ckpt.get(key), value) for key, value in expected.items()
           if ckpt.get(key) != value}
    if bad:
        raise RuntimeError(f"norm resume identity mismatch: {list(bad)}")
    for name, _module, master in selected:
        master.data.copy_(ckpt["state"][name].to(B.DEV))
    optimizer.load_state_dict(ckpt["optimizer"])
    B.emit(event="norm_resumed", next_step=ckpt["next_step"])
    return (ckpt["next_step"], ckpt.get("baseline"),
            ckpt.get("best_probe_mean"))


def main():
    B.OUTDIR.mkdir(parents=True, exist_ok=True)
    B.status(state="starting", mechanism=MECHANISM, scope=SCOPE,
             tag=B.TAG, manifest_md5=B.AMD5, lr=B.LR, steps=B.STEPS,
             batch=B.BATCH, train_wins=B.TRAIN_WINS,
             probe_wins=B.PROBE_WINS)
    B.emit(event="start", mechanism=MECHANISM, scope=SCOPE, tag=B.TAG,
           manifest_md5=B.AMD5, lr=B.LR, steps=B.STEPS, batch=B.BATCH,
           train_wins=B.TRAIN_WINS, probe_wins=B.PROBE_WINS)

    identity = instant_artifact_identity()
    B.emit(event="artifact_identity", identity=identity)
    setattr(B, "K4096Experts", FrozenArtifactExperts)
    B.T.TrainableExperts = FrozenArtifactExperts
    B.T.PILOT = tuple(range(43))
    student = B.T.Student()
    selected = expose_norm_masters(student)
    params = [master for _name, _module, master in selected]
    B.emit(event="assembled", mechanism=MECHANISM, scope=SCOPE,
           n_tensors=len(params), n_trainable_params=sum(p.numel() for p in params),
           names=[name for name, _module, _master in selected])
    corpus = B.T.load_corpus()
    gradcheck(selected)
    acache = B.ActCache(student)
    all_wins = B.TRAIN_WINS + [
        win for win in B.PROBE_WINS if win not in B.TRAIN_WINS]
    optimizer = torch.optim.Adam(params, lr=B.LR)
    start_step, baseline, best_mean = try_resume(selected, optimizer, identity)

    ref = {}
    if B.REF_KLD_PATH and os.path.exists(os.path.expanduser(B.REF_KLD_PATH)):
        ref = {int(key): float(value) for key, value in json.loads(
            Path(os.path.expanduser(B.REF_KLD_PATH)).read_text()).items()}
    if baseline is None:
        baseline = {}
        for win in all_wins:
            t0 = time.time()
            value = B.kld_window(student, corpus, acache, win)
            if not (value == value and value < 5.0):
                raise AssertionError(f"non-physical baseline {win}: {value}")
            baseline[win] = value
            ledger = ref.get(win)
            B.emit(event="baseline", win=win, kld=value, ledger_ref=ledger,
                   ledger_delta=None if ledger is None else round(value-ledger, 6),
                   secs=round(time.time()-t0, 1))
        save_ckpt(B.LATEST, selected, optimizer, 0, baseline, identity, None)
    baseline = {int(key): float(value) for key, value in baseline.items()}

    def probe(step):
        values = {}
        for win in B.PROBE_WINS:
            t0 = time.time()
            value = B.kld_window(student, corpus, acache, win)
            values[win] = value
            B.emit(event="probe", step=step, win=win, kld=value,
                   baseline=baseline[win],
                   delta_pct=round((baseline[win]-value)/baseline[win]*100, 4),
                   secs=round(time.time()-t0, 1))
            torch.cuda.empty_cache()
        mean = sum(values.values()) / len(values)
        base_mean = sum(baseline[win] for win in B.PROBE_WINS) / len(B.PROBE_WINS)
        B.emit(event="probe_mean", step=step, mean=round(mean, 6),
               baseline_mean=round(base_mean, 6),
               delta_pct=round((base_mean-mean)/base_mean*100, 4))
        B.status(state="running", last_probe_step=step,
                 last_probe_mean=mean, baseline_probe_mean=base_mean)
        return mean

    if start_step == 0:
        best_mean = probe(0)
        save_ckpt(B.BEST, selected, optimizer, 0, baseline, identity, best_mean)
    if best_mean is None:
        best_mean = sum(baseline[win] for win in B.PROBE_WINS)/len(B.PROBE_WINS)

    # Checkpoints are written immediately after an optimizer step, before a
    # due probe panel.  If the wrapper/SSH session dies in that narrow window,
    # next_step can already equal STEPS and the training loop below is empty.
    # Recover the binding panel from the checkpointed post-update state rather
    # than silently finalizing with the previous panel's best.
    if (start_step > 0 and
            (start_step % B.PROBE_EVERY == 0 or start_step == B.STEPS)):
        existing = {}
        if B.JLOG.exists():
            for line in B.JLOG.read_text().splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if (event.get("event") == "probe"
                        and int(event.get("step", -1)) == start_step):
                    existing[int(event["win"])] = event
        if set(existing) == set(B.PROBE_WINS):
            recovered_mean = sum(
                float(existing[win]["kld"]) for win in B.PROBE_WINS
            ) / len(B.PROBE_WINS)
            B.emit(event="resume_probe_found", step=start_step,
                   mean=round(recovered_mean, 6))
        else:
            B.emit(event="resume_probe_recovery", step=start_step,
                   found_wins=sorted(existing), expected_wins=B.PROBE_WINS)
            recovered_mean = probe(start_step)
        if recovered_mean < best_mean-1e-6:
            best_mean = recovered_mean
            save_ckpt(B.BEST, selected, optimizer, start_step, baseline,
                      identity, best_mean)
            B.emit(event="best_recovered", step=start_step,
                   mean=round(best_mean, 6))

    t_run = time.time()
    groups = max(1, (len(B.TRAIN_WINS)+B.BATCH-1)//B.BATCH)
    stall = 0
    for step in range(start_step, B.STEPS):
        if (time.time()-t_run)/3600 > B.MAX_HOURS:
            B.emit(event="wall_guard", step=step)
            break
        group = step % groups
        wins = B.TRAIN_WINS[group*B.BATCH:(group+1)*B.BATCH]
        t0 = time.time()
        loss = B.batch_loss(student, corpus, acache, wins, True)
        kld_pre = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(sum((p.grad.norm()**2 for p in params
                              if p.grad is not None))**0.5)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del loss
        torch.cuda.empty_cache()
        next_step = step+1
        save_ckpt(B.LATEST, selected, optimizer, next_step, baseline,
                  identity, best_mean)
        B.emit(event="step", step=next_step, train_wins=wins,
               kld_pre_update=kld_pre, grad_norm=grad_norm,
               secs=round(time.time()-t0, 1),
               mem_gb=round(torch.cuda.max_memory_allocated()/1e9, 1))
        B.status(state="running", next_step=next_step,
                 last_kld_pre_update=kld_pre, last_grad_norm=grad_norm)
        if next_step % B.PROBE_EVERY == 0 or next_step == B.STEPS:
            mean = probe(next_step)
            if mean < best_mean-1e-6:
                best_mean = mean
                stall = 0
                save_ckpt(B.BEST, selected, optimizer, next_step, baseline,
                          identity, best_mean)
                B.emit(event="best", step=next_step, mean=round(mean, 6))
            else:
                stall += 1
                B.emit(event="stall", step=next_step, count=stall)
                if stall >= B.EARLY_STOP:
                    B.emit(event="early_stop", step=next_step)
                    break

    base_mean = sum(baseline[win] for win in B.PROBE_WINS)/len(B.PROBE_WINS)
    result = {
        "state": "completed", "format": FORMAT, "mechanism": MECHANISM,
        "scope": SCOPE, "tag": B.TAG, "manifest_md5": B.AMD5,
        "n_tensors": len(params),
        "n_trainable_params": sum(p.numel() for p in params),
        "lr": B.LR, "baseline_probe_mean": base_mean,
        "best_probe_mean": best_mean,
        "best_delta_pct": (base_mean-best_mean)/base_mean*100,
        "best_checkpoint": str(B.BEST), "latest": str(B.LATEST),
        "host": os.uname().nodename, "ts": time.time(),
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
            B.status(state="failed", mechanism=MECHANISM,
                     error=f"{type(exc).__name__}: {exc}")
            B.emit(event="failed", mechanism=MECHANISM,
                   error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
