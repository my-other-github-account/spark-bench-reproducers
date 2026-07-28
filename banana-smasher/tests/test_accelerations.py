from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AccelerationTests(unittest.TestCase):
    def test_recovered_accelerations_have_working_source_and_gate_receipts(self) -> None:
        expected = json.loads((ROOT / "configs/EXPECTED_PERF.json").read_text())
        receipts = json.loads((ROOT / "receipts/ACCELERATION_RECEIPTS.json").read_text())
        rows = {row["id"]: row for row in receipts["rows"]}
        recovered = {
            "p963_true_c_mb2_bulk_stage",
            "p959_terminal_seed_rebuild",
            "p486_full164_eval",
            "p968_audit_and_batching_gates",
        }
        self.assertTrue(recovered <= set(expected["required_accelerations"]))
        self.assertTrue(recovered <= set(rows))
        for identifier in recovered:
            row = rows[identifier]
            self.assertTrue((ROOT / row["implementation"]).is_file(), identifier)
            self.assertTrue((ROOT / row["gate"]).is_file(), identifier)
            if "negative_gate" in row:
                self.assertTrue((ROOT / row["negative_gate"]).is_file(), identifier)

    def test_perf_gate_ready_and_degraded(self) -> None:
        module = load("perf_gate", ROOT / "vendor/runtime/perf_gate.py")
        expected = json.loads((ROOT / "configs/EXPECTED_PERF.json").read_text())
        metrics = {
            "validity": "fresh-measurement",
            "prompt_tokens": 2048,
            "prefill_tok_s": 1100,
            "decode_tok_s": 17,
            "ttft_seconds": 2.0,
            "decode_kernel_classes": expected["required_decode_kernel_classes"],
            "cache_verified": True,
            "resident_envelope_verified": True,
        }
        self.assertEqual(module.evaluate(metrics, expected)["status"], "READY")
        failure_cases = (
            ("prefill_tok_s", 999, "prefill_tok_s"),
            ("decode_tok_s", 14.99, "decode_tok_s"),
            ("ttft_seconds", 2.51, "ttft_seconds"),
            ("prompt_tokens", 1024, "prompt_tokens"),
            ("validity", "historical-measurement", "fresh_measurement"),
        )
        for field, value, failed_gate in failure_cases:
            with self.subTest(field=field):
                candidate = dict(metrics)
                candidate[field] = value
                result = module.evaluate(candidate, expected)
                self.assertEqual(result["status"], "DEGRADED")
                self.assertIn(failed_gate, result["failed"])

    def test_container_memory_gate_uses_lowest_visible_ceiling(self) -> None:
        runtime = ROOT / "vendor" / "runtime"
        sys.path.insert(0, str(runtime))
        try:
            module = load("container_entrypoint", runtime / "entrypoint.py")
        finally:
            sys.path.remove(str(runtime))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meminfo = root / "meminfo"
            cgroup_v2 = root / "memory.max"
            cgroup_v1 = root / "memory.limit_in_bytes"
            meminfo.write_text("MemTotal:       200000000 kB\n")
            cgroup_v2.write_text("150000000000\n")
            self.assertEqual(
                module.detected_memory_limit_bytes(meminfo, cgroup_v2, cgroup_v1),
                150000000000,
            )
            cgroup_v2.write_text("max\n")
            self.assertEqual(
                module.detected_memory_limit_bytes(meminfo, cgroup_v2, cgroup_v1),
                200000000 * 1024,
            )
            meminfo.unlink()
            with self.assertRaises(module.PackValidationError):
                module.detected_memory_limit_bytes(meminfo, cgroup_v2, cgroup_v1)

    def test_pipeline_accelerations_execute(self) -> None:
        module = load("accelerated_pipeline", ROOT / "vendor/pipeline/accelerated_pipeline.py")
        self.assertEqual(module.shard_ranges(10), [(0, 2), (2, 5), (5, 7), (7, 10)])
        self.assertEqual(module.replay_gate([1.0, 2.0], [1.0, 2.0 + 1e-13])["status"], "PASS")
        events = []
        streamed = module.stream_codebooks(
            ["a", "b"],
            lambda name: events.append("produce:" + name) or "ok",
            lambda name: events.append("rebuild:" + name) or "ok",
            lambda name: events.append("stream:" + name) or "ok",
        )
        self.assertEqual(len(streamed), 2)
        self.assertEqual(events[:3], ["produce:a", "rebuild:a", "stream:a"])
        revoked = module.speculative_publish("old", lambda: "new", lambda: "warm", lambda value: value)
        self.assertEqual(revoked["status"], "REVOKED")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for index in range(8):
                source = root / ("s" + str(index))
                destination = root / "out" / ("d" + str(index))
                source.write_bytes((str(index) * 32).encode())
                rows.append((source, destination))
            receipts = module.bulk_move(rows, streams=8)
            self.assertEqual(len(receipts), 8)
            self.assertTrue(all(row["status"] == "PASS" for row in receipts))


if __name__ == "__main__":
    unittest.main()
