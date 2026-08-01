from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from banana_smasher_plugin.contract import PackContractError, load_runtime_contract
from banana_smasher_plugin.repair import apply_dense_norm_repair, load_output_log_gains


def _pack(root: Path) -> Path:
    (root / "planes").mkdir(parents=True)
    (root / "repair").mkdir()
    repair_state = root / "repair/repair_state.safetensors"
    save_file(
        {
            "norms/model.norm": torch.arange(4, dtype=torch.float32),
            "outputs/model.layers.0.self_attn.o_b_proj.output_log_gain": torch.tensor(
                0.125, dtype=torch.float32
            ),
        },
        repair_state,
    )
    state_sha = hashlib.sha256(repair_state.read_bytes()).hexdigest()
    repair_manifest = {
        "schema": "bs-repair-materialization-v1",
        "status": "MATERIALIZED",
        "format": "bs-basic-repair-v1",
        "update": 12,
        "dense_state": {
            "path": "repair/repair_state.safetensors",
            "sha256": state_sha,
            "norms": 1,
            "outputs": 1,
            "tensors": [],
        },
    }
    repair_path = root / "repair/REPAIR_MANIFEST.json"
    repair_path.write_text(json.dumps(repair_manifest))
    repair_sha = hashlib.sha256(repair_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "bs-pack",
        "schema_version": 1,
        "source_format": "p1016-true-c-native-planes-v1",
        "quant_method": "banana_smasher",
        "instance_id": "fixture",
        "layers": [0],
        "tensor_layout_sha256": "0dae88283affb718f7b9cd7d6b2f9bd11016fb9b792ecf98ea96dce426ee4cc8",
        "repair": {
            "format": "bs-basic-repair-v1",
            "manifest": "repair/REPAIR_MANIFEST.json",
            "manifest_sha256": repair_sha,
            "state": "repair/repair_state.safetensors",
            "state_sha256": state_sha,
            "norms": 1,
            "outputs": 1,
            "update": 12,
        },
    }
    (root / "BANANA_PACK_MANIFEST.json").write_text(json.dumps(manifest))
    (root / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "banana_smasher",
                    "format": "bs-pack",
                    "format_version": 1,
                    "pack_manifest": "BANANA_PACK_MANIFEST.json",
                    "pack_root": ".",
                    "repair_format": "bs-basic-repair-v1",
                    "repair_manifest": "repair/REPAIR_MANIFEST.json",
                    "repair_state": "repair/repair_state.safetensors",
                    "repair_update": 12,
                }
            }
        )
    )
    return root


def test_runtime_contract_accepts_exact_native_repair_pack(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    contract = load_runtime_contract(pack)
    assert contract.pack_root == pack.resolve()
    assert contract.layers == (0,)
    assert contract.repair_update == 12


def test_runtime_contract_rejects_layout_lie(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    path = pack / "BANANA_PACK_MANIFEST.json"
    value = json.loads(path.read_text())
    value["source_format"] = "iq3-layout"
    path.write_text(json.dumps(value))
    with pytest.raises(PackContractError, match="source_format"):
        load_runtime_contract(pack)


def test_dense_repair_and_output_gains_are_exact(tmp_path: Path) -> None:
    contract = load_runtime_contract(_pack(tmp_path / "pack"))
    module = torch.nn.Module()
    module.model = torch.nn.Module()
    module.model.norm = torch.nn.LayerNorm(4, elementwise_affine=True)
    applied = apply_dense_norm_repair(module, contract)
    assert applied == ("model.norm.weight",)
    assert torch.equal(module.model.norm.weight, torch.arange(4, dtype=torch.float32))
    gains = load_output_log_gains(contract)
    assert gains == {"model.layers.0.self_attn.o_b_proj": pytest.approx(0.125)}
