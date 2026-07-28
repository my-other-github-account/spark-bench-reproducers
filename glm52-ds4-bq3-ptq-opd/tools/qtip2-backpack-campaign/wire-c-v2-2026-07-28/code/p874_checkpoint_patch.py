# P874 checkpoint patch — minimal, surgical, applied to eval_package/t8192_ds4_build_v3.py
# Design (P872 spec): after each layer's hidden states are computed, persist
# {layer, hidden tensors (cpu), wins, rng-free deterministic context, binding SHAs}
# atomically to CKPT_DIR; rolling retention 2. On start with P874_RESUME=1, load
# newest valid checkpoint, verify binding SHAs match, seed hidden and skip layers <= ckpt layer.
# Env: P874_CHECKPOINT=1 enables writes, P874_CKPT_DIR sets dir,
#      P874_BINDING_SHAS=<json path> optional binding file (inventory/adapter/assignment shas)
# All hooks are log-and-continue on write failure (a checkpoint failure must never kill the walk).
import hashlib
import json
import os
import time
from pathlib import Path

import torch

CKPT_SCHEMA = "p874-anchor-walk-ckpt-v1"


def _dir() -> Path | None:
    d = os.environ.get("P874_CKPT_DIR")
    if not d:
        return None
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _binding() -> dict:
    p = os.environ.get("P874_BINDING_SHAS")
    if p and Path(p).exists():
        try:
            return json.loads(Path(p).read_text())
        except Exception:
            return {"binding_error": "unreadable"}
    return {}


def enabled() -> bool:
    return os.environ.get("P874_CHECKPOINT", "0") == "1" and _dir() is not None


def resume_requested() -> bool:
    return os.environ.get("P874_RESUME", "0") == "1" and _dir() is not None


def save(layer: int, wins, hidden, extra: dict | None = None) -> None:
    """Persist post-layer hidden states. Log-and-continue on any failure."""
    if not enabled():
        return
    try:
        d = _dir()
        started = time.time()
        payload = {
            "schema": CKPT_SCHEMA,
            "layer": int(layer),
            "wins": list(wins),
            "binding": _binding(),
            "hidden": [h.detach().to("cpu", copy=True) for h in hidden],
            "extra": extra or {},
            "created_unix": time.time(),
        }
        tmp = d / f".tmp_anchor_L{layer:03d}.pt"
        final = d / f"anchor_L{layer:03d}.pt"
        torch.save(payload, tmp)
        os.replace(tmp, final)
        # rolling retention: keep newest 2
        kept = sorted(d.glob("anchor_L*.pt"), key=lambda p: p.stat().st_mtime)
        for old in kept[:-2]:
            try:
                old.unlink()
            except OSError:
                pass
        print(f"[P874] ckpt L{layer:03d} saved {time.time()-started:.2f}s", flush=True)
        if int(layer) == 3 and os.environ.get("P880_FIRST_LAYER_GATE", "0") == "1":
            root = Path(os.environ["P770_ROOT"])
            gate = root / "run/FIRST_SCORED_LAYER_GATE.RAW.json"
            release = root / "run/FIRST_SCORED_LAYER_GATE.RELEASE.json"
            h = hashlib.sha256()
            with final.open("rb") as fh:
                for block in iter(lambda: fh.read(8 << 20), b""):
                    h.update(block)
            finite = all(bool(torch.isfinite(x).all()) for x in hidden)
            payload = {"schema":"p880-first-scored-layer-raw-v1","status":"WAITING_EXTERNAL_VALIDATION","layer":3,"wins":list(wins),"hidden_all_finite":finite,"checkpoint_path":str(final),"checkpoint_sha256":h.hexdigest(),"binding":_binding(),"created_unix":time.time()}
            tmp_gate = gate.with_name(f".{gate.name}.{os.getpid()}.tmp")
            tmp_gate.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
            os.replace(tmp_gate, gate)
            print("[P880] first scored layer gate WAIT", flush=True)
            while not release.is_file():
                time.sleep(1)
            print("[P880] first scored layer gate RELEASED", flush=True)
    except Exception as exc:  # noqa: BLE001 - never kill the walk for a ckpt failure
        print(f"[P874] ckpt L{layer:03d} WRITE-FAILED (walk continues): {exc}", flush=True)


def load_newest() -> dict | None:
    """Return newest valid checkpoint payload, or None. Refuses binding mismatch."""
    d = _dir()
    if d is None:
        return None
    cands = sorted(d.glob("anchor_L*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    live_binding = _binding()
    for c in cands:
        try:
            payload = torch.load(c, map_location="cpu", weights_only=False)
            if payload.get("schema") != CKPT_SCHEMA:
                continue
            saved_binding = payload.get("binding", {})
            if live_binding and saved_binding and live_binding != saved_binding:
                print(f"[P874] RESUME REFUSED {c.name}: binding mismatch "
                      f"(saved {saved_binding} != live {live_binding})", flush=True)
                continue
            print(f"[P874] resume candidate {c.name} layer={payload['layer']}", flush=True)
            return payload
        except Exception as exc:  # noqa: BLE001
            print(f"[P874] resume candidate {c.name} unreadable: {exc}", flush=True)
    return None
