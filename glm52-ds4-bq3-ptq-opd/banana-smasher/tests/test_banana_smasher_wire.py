from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

from banana_smasher.contract import TIER_CODES, export_pack, verify_pack
from banana_smasher.loader import PackLoader
from banana_smasher.repack import repack_to_safetensors, verify_repack_roundtrip


def _write_file(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_banana_smasher_layer(root: Path) -> Path:
    root.mkdir(parents=True)
    tiers = {
        "d4_k256": list(range(3)),
        "d4_k1024": list(range(3, 91)),
        "d4_k2048": list(range(91, 251)),
        "d4_k4096": list(range(251, 256)),
    }
    rows: list[dict[str, object]] = []
    for tier_index, (tier, experts) in enumerate(tiers.items(), start=1):
        bits = int(tier.removeprefix("d4_k")).bit_length() - 1
        for projection in ("down", "fused13"):
            marker = tier_index + (0 if projection == "down" else 16)
            rows.extend(
                [
                    _write_file(
                        root / f"{tier}.{projection}.codebook.fp16.bin",
                        np.arange(4 * (tier_index + 1), dtype=np.float16).tobytes(),
                    ),
                    _write_file(
                        root / f"{tier}.{projection}.codes.le{bits}.bin",
                        bytes([marker]) * (len(experts) * (tier_index + 2)),
                    ),
                    _write_file(
                        root / f"{tier}.{projection}.expert_ids.i16.bin",
                        np.asarray(experts, dtype="<i2").tobytes(),
                    ),
                    _write_file(
                        root / f"{tier}.{projection}.scales.e8m0.bin",
                        bytes([marker + 1]) * (len(experts) * (tier_index + 1)),
                    ),
                ]
            )
    receipt = {
        "schema": "banana_smasher-materialized-layer-v1",
        "status": "PASS",
        "layer": 0,
        "files": rows,
    }
    (root / "LAYER_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-knapsack-input-index-v1",
                "status": "PASS",
                "intended_basis_sha256": "a" * 64,
                "intended_tiers": sorted(tiers),
                "envelope_bytes": 1,
                "source_receipts": [],
                "missing_inputs": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return root


def test_banana_smasher_wire_export_and_safetensors_roundtrip_are_byte_exact(
    tmp_path: Path,
) -> None:
    source = _write_banana_smasher_layer(tmp_path / "layer_000")
    pack = tmp_path / "pack"

    manifest = export_pack(
        source_root=source,
        output=pack,
        model_id="banana_smasher-fixture",
        instance_id="bs-pack-0001-banana_smasher",
        link_mode="hardlink",
    )
    receipt = verify_pack(pack)

    source_codes = source / "d4_k2048.down.codes.le11.bin"
    packed_codes = pack / "planes/layers/layer_000/truevq_d4" / source_codes.name
    assert os.stat(source_codes).st_ino == os.stat(packed_codes).st_ino
    assert manifest["source_format"] == "banana_smasher-materialized-layer-v1"
    assert (
        manifest["provenance"]["source_layer_receipt_sha256"]
        == hashlib.sha256((source / "LAYER_RECEIPT.json").read_bytes()).hexdigest()
    )
    assert receipt["tensor_count"] == 34

    tier_map = np.load(
        pack / "planes/layers/layer_000/experts/tier_map.npy",
        mmap_mode="r",
    )
    subtier_map = np.load(
        pack / "planes/layers/layer_000/experts/subtier_map.npy",
        mmap_mode="r",
    )
    assert np.all(tier_map == TIER_CODES["truevq_d4"])
    assert subtier_map[91] == 2048
    assert subtier_map[250] == 2048
    assert subtier_map[251] == 4096

    with PackLoader(pack, verify=True).open_layer(0, framework="np") as layer_view:
        raw_codes = layer_view.get("layers.0.truevq_d4.d4_k2048.down.codes")
        assert isinstance(raw_codes, np.memmap)
        assert raw_codes.tobytes() == source_codes.read_bytes()

    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.glob("*.bin")
    }
    repack_receipt = repack_to_safetensors(pack, drop_planes=True)
    roundtrip = verify_repack_roundtrip(pack)
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.glob("*.bin")
    }

    assert before == after
    assert repack_receipt["status"] == "PASS"
    assert roundtrip["byte_exact_tensors"] == 34
    with safe_open(pack / "bs-pack.safetensors", framework="np") as handle:
        key = "layers.0.truevq_d4.d4_k2048.down.codes"
        assert handle.get_tensor(key).tobytes() == source_codes.read_bytes()


def test_banana_smasher_wire_export_refuses_receipt_drift(tmp_path: Path) -> None:
    source = _write_banana_smasher_layer(tmp_path / "layer_000")
    (source / "d4_k2048.down.codes.le11.bin").write_bytes(b"drift")

    import pytest

    from banana_smasher.contract import PackValidationError

    with pytest.raises(PackValidationError, match="banana_smasher source byte count mismatch"):
        export_pack(
            source_root=source,
            output=tmp_path / "pack",
            model_id="banana_smasher-fixture",
            instance_id="bs-pack-0001-banana_smasher",
            link_mode="copy",
        )
