from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PARSER_ANCHOR = '''    ap.add_argument("--tag", default="")
    a = ap.parse_args()
'''
PARSER_REPLACEMENT = '''    ap.add_argument("--tag", default="")
    ap.add_argument("--capture-dir", default=None,
                    help="optional task-local MoE activation capture directory")
    ap.add_argument("--capture-layers", default="",
                    help="comma-separated routed-expert layers to capture")
    ap.add_argument("--capture-only", action="store_true",
                    help="seal captures without materializing teacher-logit outputs")
    a = ap.parse_args()
    _capture_layers = {int(x) for x in a.capture_layers.split(",") if x.strip()}
    _capture_dir = os.path.expanduser(a.capture_dir) if a.capture_dir else None
    if _capture_layers and not _capture_dir:
        raise ValueError("--capture-dir is required with --capture-layers")
    if _capture_dir:
        os.makedirs(_capture_dir, exist_ok=True)
    _capture_corpus_md5 = md5(a.corpus)
'''

FORWARD_ANCHOR = '''                for mi, s in enumerate(mbs):
                    hidden[mi] = lay(
                        hidden[mi], position_embeddings=pe, position_ids=pos,
                        attention_mask=masks[mi], input_ids=ids[s],
                        past_key_values=caches[mi])
'''
FORWARD_REPLACEMENT = '''                for mi, s in enumerate(mbs):
                    _capture = {}
                    _hook = None
                    if _capture_dir and L in _capture_layers:
                        def _capture_gate(_module, _inputs, _outputs):
                            _capture["x"] = _inputs[0].detach()
                            _capture["w"] = _outputs[1].detach()
                            _capture["topk"] = _outputs[2].detach()
                        _hook = lay.mlp.gate.register_forward_hook(_capture_gate)
                    hidden[mi] = lay(
                        hidden[mi], position_embeddings=pe, position_ids=pos,
                        attention_mask=masks[mi], input_ids=ids[s],
                        past_key_values=caches[mi])
                    if _hook is not None:
                        _hook.remove()
                        _x = _capture["x"]
                        _w = _capture["w"]
                        _topk = _capture["topk"]
                        if _x.ndim != 3:
                            raise RuntimeError(f"unexpected gate input shape {_x.shape}")
                        _w = _w.reshape(_x.shape[0], _x.shape[1], -1)
                        _topk = _topk.reshape(_x.shape[0], _x.shape[1], -1)
                        for _j in range(_x.shape[0]):
                            _win = wins[s.start + _j]
                            _rl = rlens[s.start + _j]
                            _path = os.path.join(
                                _capture_dir, f"xmoe_L{L:03d}_win{_win:04d}.pt")
                            _done = _path + ".DONE.json"
                            if os.path.exists(_path) and os.path.exists(_done):
                                _receipt = json.load(open(_done))
                                if (_receipt.get("md5") != md5(_path)
                                        or _receipt.get("corpus_md5") != _capture_corpus_md5):
                                    raise RuntimeError(f"capture receipt mismatch: {_path}")
                                continue
                            if os.path.exists(_path) or os.path.exists(_done):
                                raise RuntimeError(f"partial capture member: {_path}")
                            _obj = {
                                "x": _x[_j, :_rl].to(torch.bfloat16).cpu(),
                                "topk": _topk[_j, :_rl].to(torch.int16).cpu(),
                                "w": _w[_j, :_rl].to(torch.bfloat16).cpu(),
                                "win": int(_win),
                                "RL": int(_rl),
                                "layer": int(L),
                                "split": "disjoint_train",
                                "corpus_md5": _capture_corpus_md5,
                            }
                            _tmp = _path + ".tmp"
                            torch.save(_obj, _tmp)
                            with open(_tmp, "rb") as _handle:
                                os.fsync(_handle.fileno())
                            os.replace(_tmp, _path)
                            _receipt = {
                                "file": os.path.basename(_path),
                                "md5": md5(_path),
                                "bytes": os.path.getsize(_path),
                                "layer": int(L),
                                "win": int(_win),
                                "real_len": int(_rl),
                                "corpus_md5": _capture_corpus_md5,
                                "source_builder": os.path.abspath(__file__),
                                "source_builder_md5": md5(__file__),
                                "producer": "banana-smasher-public-capture-v1",
                            }
                            with open(_done + ".tmp", "w") as _handle:
                                json.dump(_receipt, _handle, indent=2, sort_keys=True)
                                _handle.write("\\n")
                                _handle.flush()
                                os.fsync(_handle.fileno())
                            os.replace(_done + ".tmp", _done)
                        del _x, _w, _topk, _capture
'''

READOUT_ANCHOR = '''            # readout
            for mi, s in enumerate(mbs):
'''
READOUT_REPLACEMENT = '''            # Capture-only is a public source-generation path: do not emit the
            # unused 3.2-GiB TRAIN-32 teacher-logit side product.
            if a.capture_only:
                del hidden, caches, masks
                torch.cuda.empty_cache()
                log(f"capture-only chunk done in {(time.time()-t_chunk)/60:.1f} min")
                continue

            # readout
            for mi, s in enumerate(mbs):
'''


def inject_builder(source: str) -> str:
    replacements = (
        (PARSER_ANCHOR, PARSER_REPLACEMENT, "parser"),
        (FORWARD_ANCHOR, FORWARD_REPLACEMENT, "forward"),
        (READOUT_ANCHOR, READOUT_REPLACEMENT, "readout"),
    )
    for anchor, replacement, label in replacements:
        if source.count(anchor) != 1:
            raise RuntimeError(f"sealed builder {label} anchor missing or non-unique")
        source = source.replace(anchor, replacement)
    return source


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_member(path: Path, *, layer: int, window: int, corpus_md5: str) -> dict[str, Any]:
    done = Path(f"{path}.DONE.json")
    if not path.is_file() or not done.is_file():
        raise RuntimeError(f"missing public capture member: {path}")
    value = json.loads(done.read_text())
    exact = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "layer": layer,
        "win": window,
        "corpus_md5": corpus_md5,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise RuntimeError(f"capture DONE binding mismatch: {done}")
    if value.get("md5") != _md5(path):
        raise RuntimeError(f"capture DONE byte hash mismatch: {done}")
    return value


def run_capture(
    *,
    run_root: Path,
    model_root: Path,
    meta_root: Path,
    corpus: Path,
    builder: Path,
    layers: list[int],
    windows: int,
    microbatch: int,
) -> dict[str, Any]:
    from . import solver_core as core
    from .workflow import artifact, atomic_json

    run_root = run_root.resolve()
    model_root = model_root.resolve()
    meta_root = meta_root.resolve()
    corpus = corpus.resolve()
    builder = builder.resolve()
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= layer < 43 for layer in layers):
        raise ValueError("capture layers must be unique values in 0..42")
    if windows not in (32, 64):
        raise ValueError("capture windows must be exactly 32 or 64")
    if microbatch < 1:
        raise ValueError("capture microbatch must be positive")
    for path in (model_root, meta_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (corpus, builder, meta_root / "model.safetensors.index.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    capture_root = run_root / "captures" / "pilot"
    capture_root.mkdir(parents=True, exist_ok=True)
    unused_teacher_root = run_root / "captures" / "teacher_unused"
    corpus_md5 = _md5(corpus)
    expected = [
        (layer, window, capture_root / f"xmoe_L{layer:03d}_win{window:04d}.pt")
        for layer in layers
        for window in range(windows)
    ]
    complete = True
    for layer, window, path in expected:
        try:
            _validate_member(path, layer=layer, window=window, corpus_md5=corpus_md5)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            complete = False
            break

    started = time.time()
    command = [
        sys.executable,
        "-m",
        "banana_smasher.capture_source",
        "--mode",
        "bf16",
        "--local-dir",
        str(model_root),
        "--meta-dir",
        str(meta_root),
        "--corpus",
        str(corpus),
        "--out",
        str(unused_teacher_root),
        "--start",
        "0",
        "--count",
        str(windows),
        "--chunk",
        str(windows),
        "--mb",
        str(microbatch),
        "--limit-layers",
        str(max(layers) + 1),
        "--capture-dir",
        str(capture_root),
        "--capture-layers",
        ",".join(str(layer) for layer in layers),
        "--capture-only",
        "--tag",
        "banana-smasher-public-train32",
    ]
    if not complete:
        environment = dict(os.environ)
        environment["BANANA_SMASHER_CAPTURE_BUILDER"] = str(builder)
        subprocess.run(command, check=True, env=environment)

    members: list[dict[str, Any]] = []
    for layer, window, path in expected:
        done_value = _validate_member(
            path, layer=layer, window=window, corpus_md5=corpus_md5
        )
        done = Path(f"{path}.DONE.json")
        capture_source = (
            f"banana-smasher-public-capture-v1:{_sha256(builder)}:"
            f"{_sha256(corpus)}:L{layer:03d}:W{window:04d}:capture"
        )
        done_source = capture_source.rsplit(":", 1)[0] + ":done"
        core.seal_staged_input(path, capture_source, min_size=1)
        core.seal_staged_input(done, done_source, min_size=1)
        members.append(
            {
                "layer": layer,
                "window": window,
                "capture": artifact(path),
                "capture_done": artifact(done),
                "capture_stage_receipt": artifact(core.staged_input_receipt_path(path)),
                "done_stage_receipt": artifact(core.staged_input_receipt_path(done)),
                "real_len": done_value["real_len"],
            }
        )

    manifest_path = run_root / "captures" / "MANIFEST.json"
    manifest = {
        "schema": "banana-smasher-public-capture-manifest-v1",
        "status": "PASS",
        "split": "disjoint_train",
        "run_root": str(run_root),
        "capture_root": str(capture_root),
        "layers": layers,
        "windows": windows,
        "members": members,
        "model_root": str(model_root),
        "model_index": artifact(meta_root / "model.safetensors.index.json"),
        "corpus": artifact(corpus),
        "corpus_md5": corpus_md5,
        "builder": artifact(builder),
        "command": command,
        "resumed": complete,
        "started_unix": started,
        "ended_unix": time.time(),
    }
    atomic_json(manifest_path, manifest)
    return {
        "schema": "banana-smasher-public-capture-receipt-v1",
        "status": "PASS",
        "command": "capture",
        "run_root": str(run_root),
        "layers": layers,
        "windows": windows,
        "members": len(members),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "resumed": complete,
    }


def _backend_main() -> None:
    builder_text = os.environ.get("BANANA_SMASHER_CAPTURE_BUILDER")
    if not builder_text:
        raise RuntimeError("BANANA_SMASHER_CAPTURE_BUILDER is required")
    builder = Path(builder_text).expanduser().resolve()
    source = inject_builder(builder.read_text())
    namespace = {
        "__name__": "__main__",
        "__file__": str(builder),
        "__package__": None,
    }
    exec(compile(source, str(builder), "exec"), namespace)


if __name__ == "__main__":
    _backend_main()
