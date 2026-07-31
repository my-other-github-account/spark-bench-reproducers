#!/usr/bin/env python3
"""Bind the sealed IQ3 wire pack and P1321 split admission to stock vLLM.

The source-hash-gated patch keeps ordinary ``vllm serve /model`` semantics.
DeepSeek-v4's registered FP8 quantization method remains the vLLM-selected
method; ``moe_quant_algo=IQ3_WIRE`` selects its vendored IQ3 MoE backend.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PREIMAGE = "02b504e355661a01f9ea1b60dda5db2c4203ae3ba4c7e65be66a78680ea058b9"
ATTENTION_PREIMAGE = "6667d75c72845e039648fc37530645d5098717e95f3052ff2b1cedfe3ca1db18"
PACK_MANIFEST_SHA256 = "4a4c15a52eaa8f87e4eb2f436da1580cb5e9addb15713d41bd9a74276731578a"

IMPORT_OLD = "from __future__ import annotations\n\nfrom typing import TYPE_CHECKING\n"
IMPORT_NEW = "from __future__ import annotations\n\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nfrom typing import TYPE_CHECKING\n"

BLOCK_OLD = '''    def _resolve_moe_overrides(self) -> None:
        if self._resolved_moe_quant_algo is not None:
            return
        try:
            hf_config = get_current_vllm_config().model_config.hf_config
        except Exception:
            return
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        algo = (quant_cfg.get("moe_quant_algo") or "").upper() or None
        self._resolved_moe_quant_algo = algo or ""
'''

BLOCK_NEW = '''    def _resolve_moe_overrides(self) -> None:
        if self._resolved_moe_quant_algo is not None:
            return
        try:
            vllm_config = get_current_vllm_config()
            hf_config = vllm_config.model_config.hf_config
        except Exception:
            return
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        algo = (quant_cfg.get("moe_quant_algo") or "").upper() or None
        self._resolved_moe_quant_algo = algo or ""
        if algo != "IQ3_WIRE":
            return

        model_root = Path(vllm_config.model_config.model).expanduser().resolve()
        relative_pack = str(quant_cfg.get("moe_pack_root", "wire_v4-step32"))
        pack_value = Path(relative_pack).expanduser()
        if pack_value.is_absolute() or ".." in pack_value.parts:
            raise ValueError("IQ3_WIRE moe_pack_root must be relative and contained in model root")
        pack_root = (model_root / pack_value).resolve()
        try:
            pack_root.relative_to(model_root)
        except ValueError as exc:
            raise ValueError("IQ3_WIRE pack escapes model root") from exc

        # Refuse malformed/incomplete pack roots before the expensive full
        # artifact authentication pass and before model allocation.
        for required in ("PACK_MANIFEST.json", "PACK_COMPLETE"):
            candidate = pack_root / required
            if not candidate.is_file() or candidate.is_symlink():
                raise ValueError(f"IQ3_WIRE pack is missing {required}: {pack_root}")

        # Authenticate the external artifact against the immutable box-6 seal,
        # then physically verify every payload row before model allocation.
        manifest_path = model_root / "BS_PACK_MANIFEST.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"IQ3_WIRE pack is missing regular BS_PACK_MANIFEST.json: {model_root}")
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        expected_manifest_sha = "4a4c15a52eaa8f87e4eb2f436da1580cb5e9addb15713d41bd9a74276731578a"
        if manifest_sha != expected_manifest_sha:
            raise ValueError(
                f"IQ3_WIRE manifest digest mismatch: {manifest_sha} != {expected_manifest_sha}"
            )
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != "bs-pack-v1" or manifest.get("status") != "PASS":
            raise ValueError("IQ3_WIRE BS_PACK_MANIFEST schema/status mismatch")
        rows = (manifest.get("payload") or {}).get("files")
        if not isinstance(rows, list) or not rows:
            raise ValueError("IQ3_WIRE BS_PACK_MANIFEST has no payload rows")
        for row in rows:
            rel = Path(str(row.get("path", "")))
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError(f"IQ3_WIRE unsafe manifest path: {rel}")
            physical = model_root / rel
            if not physical.is_file() or physical.is_symlink():
                raise ValueError(f"IQ3_WIRE payload is missing a regular file: {rel}")
            if physical.stat().st_size != int(row.get("bytes", -1)):
                raise ValueError(f"IQ3_WIRE payload size mismatch: {rel}")
            digest = hashlib.sha256()
            with physical.open("rb") as handle:
                for block in iter(lambda: handle.read(16 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != row.get("sha256"):
                raise ValueError(f"IQ3_WIRE payload digest mismatch: {rel}")

        # Exact product defaults. setdefault preserves ordinary expert env
        # overrides; no custom launcher or PYTHONPATH is required. P1321's
        # sealed split is scalar valid_m<4 and vector-M4 valid_m==4. MAX_M=4
        # admits both paths; VQ_WARP_M4_VECTOR selects the vector M4 kernel.
        defaults = {
            "DS4_DENSE_PATCH": str(model_root / "bs_runtime_assets/dense_patch.safetensors"),
            "VLLM_MOE_W2": "1",
            "VLLM_MOE_W2_NUM_LAYERS": "43",
            "VLLM_MOE_W2_PREPACKED_DIR": str(pack_root),
            "VLLM_MOE_W2_CUBIT_DIR": "/opt/genesis/runtime_cubins/cubins-sm120",
            "VLLM_MOE_W3_CUBIT_DIR": "/opt/genesis/runtime_cubins/cubins_e43",
            "VLLM_MOE_W2_FADVISE_GLOB": str(model_root / "*.safetensors"),
            "VLLM_MOE_VQ_D4_FAST": "1",
            "VLLM_MOE_VQ_GROUP_FAST": "1",
            "VLLM_MOE_VQ_FAST": "1",
            "VLLM_MOE_VQ_CUDA_WARP": "1",
            "VLLM_MOE_VQ_M1_FAST": "0",
            "VLLM_MOE_W2_DECODE_GRAPH": "1",
            "VLLM_MOE_W2_DECODE_GRAPH_MAX_T": "2",
            "VLLM_MOE_VQ_CUDA_WARP_MAX_M": "4",
            "VQ_WARP_M4_VECTOR": "1",
            "MALLOC_MMAP_THRESHOLD_": "65536",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
'''

ATTENTION_OLD = '''        if os.environ.get("DS4_DENSE_PATCH") and layer_id < config.num_hidden_layers:
'''
ATTENTION_NEW = '''        quant_cfg = getattr(config, "quantization_config", None) or {}
        iq3_wire = str(quant_cfg.get("moe_quant_algo", "")).upper() == "IQ3_WIRE"
        if iq3_wire and layer_id < config.num_hidden_layers:
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patched_bytes(target: Path) -> bytes:
    """Return the deterministic quant-method postimage without mutating target."""
    data = target.read_bytes()
    before = hashlib.sha256(data).hexdigest()
    if before != PREIMAGE:
        raise SystemExit(f"quant_config preimage drift: {before} != {PREIMAGE}")
    text = data.decode()
    if text.count(IMPORT_OLD) != 1 or text.count(BLOCK_OLD) != 1:
        raise SystemExit("quant_config patch anchors are not unique")
    text = text.replace(IMPORT_OLD, IMPORT_NEW).replace(BLOCK_OLD, BLOCK_NEW)
    compile(text, str(target), "exec")
    return text.encode()


def patched_attention_bytes(target: Path) -> bytes:
    """Register the IQ3 dense-sidecar scalar from model config, not env timing."""
    data = target.read_bytes()
    before = hashlib.sha256(data).hexdigest()
    if before != ATTENTION_PREIMAGE:
        raise SystemExit(
            f"attention.py preimage drift: {before} != {ATTENTION_PREIMAGE}"
        )
    text = data.decode()
    if text.count(ATTENTION_OLD) != 1:
        raise SystemExit("attention.py IQ3 sidecar patch anchor is not unique")
    text = text.replace(ATTENTION_OLD, ATTENTION_NEW)
    compile(text, str(target), "exec")
    return text.encode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--attention-target", type=Path)
    ap.add_argument("--receipt", type=Path)
    args = ap.parse_args()
    target = args.target
    before = sha256(target)
    target.write_bytes(patched_bytes(target))
    after = sha256(target)
    attention_before = attention_after = None
    if args.attention_target:
        attention_before = sha256(args.attention_target)
        args.attention_target.write_bytes(patched_attention_bytes(args.attention_target))
        attention_after = sha256(args.attention_target)
    if args.receipt:
        import json
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps({
            "schema": "p1321-iq3-wire-split-admission-defaults-patch-v3",
            "preimage_sha256": before,
            "postimage_sha256": after,
            "target": str(target),
            "attention_preimage_sha256": attention_before,
            "attention_postimage_sha256": attention_after,
            "attention_target": str(args.attention_target) if args.attention_target else None,
            "expected_pack_manifest_sha256": PACK_MANIFEST_SHA256,
            "full_payload_hash_verification_before_allocation": True,
            "defaults_live_in_quant_method": True,
            "environment_overrides_preserved": True,
            "p1321_split_admission": "scalar valid_m<4; vector valid_m==4",
            "p1321_winning_boot_config_sha256": "091e8eb3e4caa9793454f4a529d8c1f5fc0af0fcb4fa28cc89e34c8a4c314da2",
            "truth_label": "PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C",
        }, indent=2, sort_keys=True) + "\n")
    print(after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
