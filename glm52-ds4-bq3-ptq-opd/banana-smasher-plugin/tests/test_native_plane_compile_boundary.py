from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import torch

import banana_smasher_plugin.native_planes as native_planes
from test_native_plane_runtime import _tiny_pack


def test_native_plane_forward_registers_opaque_vllm_compile_boundary(
    tmp_path: Path, monkeypatch,
) -> None:
    registrations: list[tuple[str, object, object, tuple[torch.Tag, ...]]] = []
    calls: list[tuple[int, int]] = []

    torch_utils = ModuleType("vllm.utils.torch_utils")

    def direct_register_custom_op(name, impl, *, fake_impl, tags=()):
        registrations.append((name, impl, fake_impl, tags))

        def invoke(x, expert_ids, layer_key, projection_key):
            calls.append((layer_key, projection_key))
            return impl(x, expert_ids, layer_key, projection_key)

        monkeypatch.setattr(torch.ops.vllm, name, invoke, raising=False)

    torch_utils.direct_register_custom_op = direct_register_custom_op
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.utils", ModuleType("vllm.utils"))
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    monkeypatch.setattr(
        native_planes, "_NATIVE_PLANE_CUSTOM_OP_REGISTERED", False, raising=False
    )
    monkeypatch.setattr(
        native_planes, "_NATIVE_PLANE_CUSTOM_OP_AVAILABLE", False, raising=False
    )
    monkeypatch.setattr(native_planes, "_NATIVE_PLANE_LAYER_REGISTRY", {}, raising=False)
    monkeypatch.setattr(native_planes, "_NATIVE_PLANE_NEXT_KEY", 1, raising=False)

    pack = native_planes.NativePlanePack.from_model_root(_tiny_pack(tmp_path / "model"))

    def dispatch(*, projection, x, expert_ids, state):
        del projection, expert_ids
        return torch.full(
            (x.shape[0], state.output_width), 7.0, dtype=torch.bfloat16
        )

    layer = native_planes.NativePlaneLayer(pack, 0, device="cpu", dispatch=dispatch)
    result = layer.forward(
        torch.ones((2, 4), dtype=torch.bfloat16),
        torch.tensor([0, 1]),
        "fused13",
    )

    assert registrations and registrations[0][0] == "banana_smasher_native_plane_forward", (
        "NativePlaneLayer.forward must register one opaque vLLM custom-op boundary "
        "before torch.compile traces Python state and custom kernels"
    )
    assert torch.Tag.cudagraph_unsafe in registrations[0][3], (
        "NativePlaneLayer.forward must register one opaque vLLM custom-op boundary "
        "tagged cudagraph_unsafe so vLLM partitions the stateful kernels out of capture"
    )
    assert calls == [(layer._custom_op_key, 0)]
    assert result.shape == (2, 4)
    assert torch.all(result == 7)

    second = native_planes.NativePlaneLayer(pack, 0, device="cpu", dispatch=dispatch)
    second.forward(torch.ones((1, 2)), torch.tensor([1]), "down")
    assert len(registrations) == 1
    assert calls[-1] == (second._custom_op_key, 1)
    assert second._custom_op_key != layer._custom_op_key
