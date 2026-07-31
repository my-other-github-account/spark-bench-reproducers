from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors import safe_open
from test_contract import _write_qtip2_source

from banana_smasher.contract import export_pack, verify_pack
from banana_smasher.repack import repack_to_safetensors, verify_repack_roundtrip


def test_repack_streams_npy_payloads_to_named_safetensors_byte_exact(
    tmp_path: Path,
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    pack = tmp_path / "pack"
    export_pack(
        source_root=source,
        output=pack,
        model_id="fixture-model",
        instance_id="bs-pack-test-repack",
        link_mode="copy",
    )

    receipt = repack_to_safetensors(pack, drop_planes=True)
    verified = verify_pack(pack)
    roundtrip = verify_repack_roundtrip(pack)

    container = pack / "bs-pack.safetensors"
    assert receipt["status"] == "PASS"
    assert receipt["payload_bytes"] == sum(
        np.load(path, mmap_mode="r", allow_pickle=False).nbytes
        for path in source.rglob("*.npy")
    )
    assert not list((pack / "planes").rglob("*.npy"))
    assert verified["tensor_count"] == 6
    assert roundtrip["status"] == "PASS"
    assert roundtrip["byte_exact_tensors"] == 6
    assert roundtrip["mmap_container"] is True
    with safe_open(container, framework="np") as handle:
        assert set(handle.keys()) == {
            "layers.0.experts.tier_map",
            "layers.0.qtip2.codebooks",
            "layers.0.qtip2.codes",
            "layers.0.qtip2.expert_ids",
            "layers.0.qtip2.scales",
            "layers.0.qtip2.tensor_offsets",
        }
