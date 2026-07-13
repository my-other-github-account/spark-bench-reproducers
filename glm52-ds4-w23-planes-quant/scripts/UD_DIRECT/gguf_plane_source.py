"""Lazy GGUF expert dequantizer implementing the sealed rail PlaneSource API."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


class GGUFPlaneSource:
    """Expose split-GGUF routed experts as ``layer(L) -> expert(e, which)``.

    GGUF block 0 corresponds to native/HF layer ``first_moe_layer``.  Only
    routed expert tensors are read from the GGUF; the rail continues to load
    every non-expert tensor from its native checkpoint.
    """

    _SUFFIXES = {
        "gate": "ffn_gate_exps.weight",
        "up": "ffn_up_exps.weight",
        "down": "ffn_down_exps.weight",
    }

    def __init__(
        self,
        paths: Iterable[str | Path],
        *,
        reader_factory: Callable[[str | Path], Any] | None = None,
        dequant_fn: Callable[[np.ndarray, Any], np.ndarray] | None = None,
        device: str | torch.device = "cuda",
        first_moe_layer: int = 0,
        cache_dir: str | Path | None = None,
    ) -> None:
        if reader_factory is None or dequant_fn is None:
            from gguf import GGUFReader
            from gguf.quants import dequantize

            reader_factory = reader_factory or GGUFReader
            dequant_fn = dequant_fn or dequantize

        self.device = torch.device(device)
        self.first_moe_layer = first_moe_layer
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dequant_fn = dequant_fn
        self.readers = []
        self.tensors: dict[str, Any] = {}

        for path in paths:
            reader = reader_factory(path)
            self.readers.append(reader)  # retain memmap ownership
            for tensor in reader.tensors:
                if not any(tensor.name.endswith(s) for s in self._SUFFIXES.values()):
                    continue
                if tensor.name in self.tensors:
                    raise RuntimeError(f"duplicate GGUF tensor: {tensor.name}")
                self.tensors[tensor.name] = tensor

        blocks = sorted(
            {int(name.split(".", 2)[1]) for name in self.tensors if name.startswith("blk.")}
        )
        missing = []
        for block in blocks or [0]:
            for suffix in self._SUFFIXES.values():
                name = f"blk.{block}.{suffix}"
                if name not in self.tensors:
                    missing.append(name)
        if missing:
            raise RuntimeError(f"missing GGUF expert tensors: {missing[:8]}")

    def _tensor(self, block: int, which: str) -> Any:
        name = f"blk.{block}.{self._SUFFIXES[which]}"
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise RuntimeError(f"GGUF has no routed experts for block {block}: {name}") from exc

    @staticmethod
    def _logical_matrix_shape(tensor: Any) -> tuple[int, int, int]:
        # GGUF reports [K, N, E], while tensor.data is encoded as [E, N, bytes].
        k, n, experts = (int(x) for x in tensor.shape)
        return experts, n, k

    def _dequant_expert(self, tensor: Any, expert_index: int) -> torch.Tensor:
        array = self.dequant_fn(tensor.data[expert_index], tensor.tensor_type)
        array = np.ascontiguousarray(array, dtype=np.float32)
        return torch.from_numpy(array).to(device=self.device, dtype=torch.bfloat16)

    def layer(self, native_layer: int):
        if native_layer < self.first_moe_layer:
            raise ValueError(f"native layer {native_layer} is pre-MoE")
        block = native_layer - self.first_moe_layer
        gate = self._tensor(block, "gate")
        up = self._tensor(block, "up")
        down = self._tensor(block, "down")

        ge, gn, gk = self._logical_matrix_shape(gate)
        ue, un, uk = self._logical_matrix_shape(up)
        de, dn, dk = self._logical_matrix_shape(down)
        if ge != ue or ge != de or gk != uk:
            raise RuntimeError(
                f"incompatible GGUF expert shapes at block {block}: "
                f"gate={tuple(gate.shape)} up={tuple(up.shape)} down={tuple(down.shape)}"
            )

        def expert(expert_index: int, which: str) -> torch.Tensor:
            if not 0 <= expert_index < ge:
                raise IndexError(expert_index)
            if which == "13":
                return torch.cat(
                    [
                        self._dequant_expert(gate, expert_index),
                        self._dequant_expert(up, expert_index),
                    ],
                    dim=0,
                )
            if which == "2":
                return self._dequant_expert(down, expert_index)
            raise ValueError(f"unknown expert matrix selector: {which!r}")

        return expert, (ge, gn + un, gk, dn, dk)

    @staticmethod
    def _raw_bf16(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
        n = int(np.prod(shape))
        raw = torch.from_file(str(path), shared=False, size=n, dtype=torch.uint16)
        return raw.view(torch.bfloat16).reshape(shape)

    @staticmethod
    def _write_raw_bf16(path: Path, tensor: torch.Tensor) -> None:
        tmp = Path(str(path) + ".tmp")
        tensor.detach().cpu().contiguous().view(torch.uint16).numpy().tofile(tmp)
        os.replace(tmp, path)

    def full_layer(self, native_layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize a layer, caching raw bf16 to amortize rail chunks."""
        expert, (experts, n13, k13, n2, k2) = self.layer(native_layer)
        shapes = ((experts, n13, k13), (experts, n2, k2))
        if self.cache_dir:
            stem = self.cache_dir / f"layer_{native_layer:03d}"
            gu_path = Path(str(stem) + ".gate_up.bf16")
            dn_path = Path(str(stem) + ".down.bf16")
            meta_path = Path(str(stem) + ".json")
            if meta_path.exists() and gu_path.exists() and dn_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("shapes") == [list(x) for x in shapes]:
                    return (self._raw_bf16(gu_path, shapes[0]).to(self.device),
                            self._raw_bf16(dn_path, shapes[1]).to(self.device))

        gu = torch.empty(shapes[0], dtype=torch.bfloat16, device=self.device)
        dn = torch.empty(shapes[1], dtype=torch.bfloat16, device=self.device)
        for e in range(experts):
            gu[e] = expert(e, "13")
            dn[e] = expert(e, "2")

        if self.cache_dir:
            self._write_raw_bf16(gu_path, gu)
            self._write_raw_bf16(dn_path, dn)
            tmp_meta = Path(str(meta_path) + ".tmp")
            tmp_meta.write_text(json.dumps({"shapes": [list(x) for x in shapes]}))
            os.replace(tmp_meta, meta_path)
        return gu, dn
