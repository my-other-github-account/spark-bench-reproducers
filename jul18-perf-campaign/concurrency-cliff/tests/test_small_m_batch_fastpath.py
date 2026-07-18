from __future__ import annotations

import ast
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "runtime"


class FakeCuda:
    capturing = False

    @classmethod
    def is_current_stream_capturing(cls):
        return cls.capturing


FAKE_TORCH = types.SimpleNamespace(
    uint8=object(),
    float16=object(),
    Tensor=object,
    cuda=FakeCuda,
)


def load_functions(path: Path, names: set[str]) -> dict:
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    found = {node.name for node in nodes}
    missing = names - found
    if missing:
        raise AssertionError(f"missing required functions in {path.name}: {sorted(missing)}")
    module = ast.fix_missing_locations(
        ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
    )
    namespace = {"os": os, "torch": FAKE_TORCH, "Any": object}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class SmallMWarpDispatchTests(unittest.TestCase):
    def setUp(self):
        ns = load_functions(
            UTILS / "moe_vq_triton.py",
            {"fast_enabled", "cuda_warp_enabled", "dispatch_probe_label"},
        )
        self.enabled = ns["cuda_warp_enabled"]
        self.label = ns["dispatch_probe_label"]
        self.state = {
            "layer_key": 0,
            "scales": types.SimpleNamespace(dtype=FAKE_TORCH.uint8),
            "codebooks": types.SimpleNamespace(dtype=FAKE_TORCH.float16),
        }

    def test_m2_uses_warp_when_explicitly_enabled(self):
        env = {
            "VLLM_MOE_VQ_FAST": "1",
            "VLLM_MOE_VQ_CUDA_WARP": "1",
            "VLLM_MOE_VQ_CUDA_WARP_MAX_M": "4",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(self.enabled(self.state, mblock=4, valid_m=2))

    def test_default_preserves_m1_only_contract(self):
        env = {"VLLM_MOE_VQ_FAST": "1", "VLLM_MOE_VQ_CUDA_WARP": "1"}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(self.enabled(self.state, mblock=4, valid_m=1))
            self.assertFalse(self.enabled(self.state, mblock=4, valid_m=2))

    def test_m2_probe_reports_the_actual_warp_branch(self):
        env = {
            "VLLM_MOE_VQ_FAST": "1",
            "VLLM_MOE_VQ_CUDA_WARP": "1",
            "VLLM_MOE_VQ_CUDA_WARP_MAX_M": "4",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                self.label(self.state, mblock=4, valid_m=2), "cuda_warp_m2"
            )
        with patch.dict(os.environ, {**env, "VLLM_MOE_VQ_CUDA_WARP_MAX_M": "1"}, clear=True):
            self.assertEqual(
                self.label(self.state, mblock=4, valid_m=2), "fallback_m2"
            )

    def test_m5_never_uses_small_m_warp(self):
        env = {
            "VLLM_MOE_VQ_FAST": "1",
            "VLLM_MOE_VQ_CUDA_WARP": "1",
            "VLLM_MOE_VQ_CUDA_WARP_MAX_M": "8",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(self.enabled(self.state, mblock=4, valid_m=5))


class BatchGraphPolicyTests(unittest.TestCase):
    def setUp(self):
        FakeCuda.capturing = False
        ns = load_functions(
            UTILS / "moe_w2_cubit.py",
            {"_decode_graph_wanted", "decode_graph_key", "vq_valid_rows"},
        )
        self.graph_wanted = ns["_decode_graph_wanted"]
        self.graph_key = ns["decode_graph_key"]
        self.valid_rows = ns["vq_valid_rows"]

    def test_batch2_has_its_own_graph_shape(self):
        x = types.SimpleNamespace(shape=(2, 4096))
        env = {
            "VLLM_MOE_W2_DECODE_GRAPH": "1",
            "VLLM_MOE_W2_DECODE_GRAPH_MAX_T": "4",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(self.graph_wanted(x))

    def test_graph_policy_honors_max_t_and_outer_capture(self):
        x = types.SimpleNamespace(shape=(5, 4096))
        env = {
            "VLLM_MOE_W2_DECODE_GRAPH": "1",
            "VLLM_MOE_W2_DECODE_GRAPH_MAX_T": "4",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(self.graph_wanted(x))
            FakeCuda.capturing = True
            self.assertFalse(self.graph_wanted(types.SimpleNamespace(shape=(2, 4096))))

    def test_graph_cache_key_is_batch_shape_specific(self):
        topk = types.SimpleNamespace(shape=(2, 8))
        self.assertEqual(self.graph_key(0, types.SimpleNamespace(shape=(2, 4096)), topk), (0, 2, 8))
        self.assertNotEqual(
            self.graph_key(0, types.SimpleNamespace(shape=(1, 4096)), topk),
            self.graph_key(0, types.SimpleNamespace(shape=(2, 4096)), topk),
        )

    def test_valid_rows_tracks_batch_up_to_mblock(self):
        self.assertEqual(self.valid_rows(tokens=1, mblock=4), 1)
        self.assertEqual(self.valid_rows(tokens=2, mblock=4), 2)
        self.assertEqual(self.valid_rows(tokens=4, mblock=4), 4)
        self.assertEqual(self.valid_rows(tokens=8, mblock=4), 4)
        self.assertEqual(self.valid_rows(tokens=2, mblock=16), 16)


if __name__ == "__main__":
    unittest.main()
