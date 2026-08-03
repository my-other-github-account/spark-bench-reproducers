from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from test_contract import _write_qtip2_source

from banana_smasher.contract import KERNEL_MANIFEST_NAME, export_pack, layout_sha256
from banana_smasher.loader import PackLoader
from banana_smasher.repack import repack_to_safetensors


def test_shared_loader_opens_named_layer_tensors_from_verified_container(
    tmp_path: Path,
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    pack = tmp_path / "pack"
    export_pack(
        source_root=source,
        output=pack,
        model_id="fixture-model",
        instance_id="bs-pack-loader-0001",
        link_mode="copy",
    )
    repack_to_safetensors(pack, drop_planes=True)

    loader = PackLoader(pack, verify=True)
    assert loader.layers == [0]
    assert loader.tensor_names(0)[0] == "layers.0.experts.tier_map"
    with loader.open_layer(0, framework="np") as layer:
        tier_map = layer.get("layers.0.experts.tier_map")
        assert isinstance(tier_map, np.ndarray)
        assert tier_map.shape == (256,)
        assert np.all(tier_map == 0)
        assert set(layer.names) == set(loader.tensor_names(0))


def test_loader_imports_only_the_verified_runtime_adapter(tmp_path: Path) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    pack = tmp_path / "pack"
    export_pack(
        source_root=source,
        output=pack,
        model_id="fixture-model",
        instance_id="bs-pack-loader-0002",
        link_mode="copy",
    )
    cache = tmp_path / "kernel-cache"
    cache.mkdir()
    adapter = cache / "runtime_adapter.py"
    adapter.write_text(
        "class RuntimeAdapter:\n"
        "    API_VERSION = 1\n"
        "    def build_layer(self, **kwargs): return kwargs\n"
        "    def forward(self, state, **kwargs): return state\n"
    )
    manifest = {
        "schema": "bs-kernel-cache",
        "schema_version": 1,
        "quant_method": "banana_smasher",
        "pack_schema": "bs-pack",
        "pack_schema_version": 1,
        "tensor_layout_sha256": layout_sha256(),
        "families": ["qtip2"],
        "architectures": ["sm_120"],
        "runtime_adapter": {
            "path": adapter.name,
            "class": "RuntimeAdapter",
            "api_version": 1,
        },
        "files": [
            {
                "path": adapter.name,
                "bytes": adapter.stat().st_size,
                "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
            }
        ],
    }
    (cache / KERNEL_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    loader = PackLoader(
        pack,
        verify=True,
        kernel_cache_root=cache,
        architecture="sm_120",
    )
    adapter_class = loader.runtime_adapter_class()
    assert adapter_class.__name__ == "RuntimeAdapter"
    assert adapter_class.API_VERSION == 1
