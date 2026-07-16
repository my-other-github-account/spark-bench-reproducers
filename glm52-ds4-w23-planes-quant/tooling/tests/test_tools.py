from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

TOOLING = Path(__file__).resolve().parents[1]


def run_tool(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOLING / name), *args],
        text=True,
        capture_output=True,
        check=False,
    )


class SweepsTests(unittest.TestCase):
    def test_reports_mean_window_and_pooled_kld_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "arm.jsonl"
            ledger.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"event": "probe", "step": 8, "win": 1, "baseline": 1.0, "kld": 0.9, "delta_pct": 10.0},
                        {"event": "probe", "step": 8, "win": 2, "baseline": 3.0, "kld": 2.4, "delta_pct": 20.0},
                    )
                )
                + "\n"
            )
            result = run_tool("sweeps.py", "--json", str(ledger))
            self.assertEqual(result.returncode, 0, result.stderr)
            panel = json.loads(result.stdout)["trajectory"][0]
            self.assertAlmostEqual(panel["mean_window_delta_pct"], 15.0)
            self.assertAlmostEqual(panel["pooled_kld_delta_pct"], 17.5)


class AggregateRailTests(unittest.TestCase):
    def write_shard(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def test_complete_coverage_emits_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "shard.jsonl"
            self.write_shard(
                shard,
                [
                    {"event": "baseline", "win": 0, "kld": 0.9, "ledger_ref": 1.0},
                    {"event": "baseline", "win": 1, "kld": 1.8, "ledger_ref": 2.0},
                ],
            )
            result = run_tool("agg_rail.py", str(shard), "--expected-windows", "2")
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertTrue(data["complete"])
            self.assertAlmostEqual(data["pooled_kld_delta_pct"], 10.0)

    def test_partial_coverage_suppresses_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "shard.jsonl"
            self.write_shard(
                shard,
                [{"event": "baseline", "win": 0, "kld": 0.9, "ledger_ref": 1.0}],
            )
            result = run_tool(
                "agg_rail.py",
                str(shard),
                "--expected-windows",
                "2",
                "--allow-partial",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertFalse(data["complete"])
            self.assertNotIn("measured_kld_mean", data)
            self.assertNotIn("pooled_kld_delta_pct", data)

    def test_conflicting_duplicate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.jsonl"
            second = Path(tmp) / "b.jsonl"
            self.write_shard(first, [{"event": "baseline", "win": 0, "kld": 0.9, "ledger_ref": 1.0}])
            self.write_shard(second, [{"event": "baseline", "win": 0, "kld": 0.8, "ledger_ref": 1.0}])
            result = run_tool("agg_rail.py", str(first), str(second), "--expected-windows", "1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("conflicting duplicate window 0", result.stderr)

    def test_mixed_manifest_identity_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.jsonl"
            second = Path(tmp) / "b.jsonl"
            self.write_shard(
                first,
                [
                    {"event": "start", "manifest_md5": "manifest-a"},
                    {"event": "baseline", "win": 0, "kld": 0.9, "ledger_ref": 1.0},
                ],
            )
            self.write_shard(
                second,
                [
                    {"event": "start", "manifest_md5": "manifest-b"},
                    {"event": "baseline", "win": 1, "kld": 1.8, "ledger_ref": 2.0},
                ],
            )
            result = run_tool(
                "agg_rail.py",
                str(first),
                str(second),
                "--expected-windows",
                "2",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mixed manifest_md5 identities", result.stderr)


class RunPilotPatcherTests(unittest.TestCase):
    def test_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "run_pilot.sh"
            script.write_text(
                "#!/bin/sh\n"
                "export BR_VQ3B_DIR=\"$ROOT/planes\"\n"
                "export BR_TRAIN=1,2\n"
                "export BR_PROBE=3,4\n"
                "export BR_REF_KLD=\"$ROOT/ref.json\"\n"
                "export BR_OUTDIR=\"$ROOT/out\"\n"
            )
            first = run_tool("fix_runpilot_env.py", str(script))
            second = run_tool("fix_runpilot_env.py", str(script))
            check = run_tool("fix_runpilot_env.py", "--check", str(script))
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("NOCHANGE", second.stdout)
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)

    def test_preserves_trailing_export_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "run_pilot.sh"
            script.write_text(
                "#!/bin/sh\n"
                "export BR_VQ3B_DIR=\"$ROOT/planes\"\n"
                "export BR_TRAIN=1,2\n"
                "export BR_PROBE=3,4\n"
                "export BR_REF_KLD=\"$ROOT/ref.json\"\n"
                "export BR_OUTDIR=\"$ROOT/out\" BR_TAG=${BR_TAG:-pilot1}\n"
            )
            result = run_tool("fix_runpilot_env.py", str(script))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                'export BR_OUTDIR="${BR_OUTDIR:-$ROOT/out}" '
                "BR_TAG=${BR_TAG:-pilot1}",
                script.read_text(),
            )


class RailSourceReceiptTests(unittest.TestCase):
    def load_module(self):
        fake_torch = types.ModuleType("torch")
        fake_v3 = types.ModuleType("t8192_ds4_build_v3")
        setattr(fake_v3, "PlaneSource", object)
        setattr(fake_v3, "DEV", "cpu")
        path = TOOLING / "vq3u_rail_source_fixed.py"
        spec = importlib.util.spec_from_file_location("vq3u_rail_source_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {"torch": fake_torch, "t8192_ds4_build_v3": fake_v3},
        ):
            spec.loader.exec_module(module)
        return module

    def test_accepts_export_and_canonical_receipt_schemas(self) -> None:
        module = self.load_module()
        self.assertEqual(module._receipt_md5({"md5": "export-md5"}), "export-md5")
        self.assertEqual(
            module._receipt_md5({"canonical_md5": "canonical-md5"}),
            "canonical-md5",
        )
        self.assertEqual(module._receipt_source_label({}, True), "local_override")
        self.assertEqual(
            module._receipt_source_label(
                {"canonical_source_host": "spark-N"}, False
            ),
            "spark-N",
        )


class LedgerExtractTests(unittest.TestCase):
    def test_filters_and_projects_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            ledger.write_text(
                json.dumps({"row": "IQ3_BIN", "kl_mean": 0.1, "secret": "drop"})
                + "\n"
                + json.dumps({"row": "OTHER", "kl_mean": 0.2})
                + "\n"
            )
            result = run_tool(
                "ledger_extract.py",
                str(ledger),
                "--contains",
                "BIN",
                "--fields",
                "row,kl_mean",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"row": "IQ3_BIN", "kl_mean": 0.1})


if __name__ == "__main__":
    unittest.main()
