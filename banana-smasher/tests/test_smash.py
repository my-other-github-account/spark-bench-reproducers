from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SMASH = ROOT / "smash"
STAGES = (
    "init",
    "capture",
    "anchors",
    "anchor-mix",
    "grid",
    "solve",
    "retrodict",
    "build",
    "measure",
    "calibrate",
    "resolve",
    "repair",
    "pack",
    "serve",
    "eval",
    "status",
    "verify",
)


def run_smash(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SMASH), *args],
        cwd=cwd or ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SmashAcceptanceTests(unittest.TestCase):
    def test_help_names_every_stage(self) -> None:
        result = run_smash("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for stage in STAGES:
            self.assertIn(stage, result.stdout)

    def test_every_stage_dry_run_is_offline_and_write_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            for stage in STAGES:
                result = run_smash(
                    "--workspace",
                    str(workspace),
                    stage,
                    "--dry-run",
                )
                self.assertEqual(result.returncode, 0, f"{stage}: {result.stderr}")
                plan = json.loads(result.stdout)
                self.assertEqual(plan["stage"], stage)
                self.assertEqual(plan["mode"], "dry-run")
                self.assertTrue(plan["offline"])
                self.assertIn("receipt", plan)
            self.assertFalse(workspace.exists(), "dry-run must not create the workspace")

    def test_stage_contracts_cover_all_stages(self) -> None:
        document = json.loads((ROOT / "contracts" / "STAGE_CONTRACTS.json").read_text())
        self.assertEqual(document["schema"], "banana-smasher-stage-contracts-v1")
        self.assertEqual(set(document["stages"]), set(STAGES))
        for name, contract in document["stages"].items():
            self.assertEqual(contract["write_scope"], f"workspace/{name}")
            self.assertIn("receipt", contract)
            self.assertIsInstance(contract["inputs"], list)
            self.assertIsInstance(contract["outputs"], list)

    def test_init_profiles_local_config_atomically_and_idempotently(self) -> None:
        config = {
            "architectures": ["ExampleForCausalLM"],
            "model_type": "example_moe",
            "num_hidden_layers": 4,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps(config))
            workspace = root / "workspace"
            args = (
                "--workspace",
                str(workspace),
                "init",
                "--model",
                str(model),
                "--budget-bytes",
                "1048576",
                "--node-ram",
                "32",
            )
            first = run_smash(*args)
            self.assertEqual(first.returncode, 0, first.stderr)
            profile = workspace / "init" / "MODEL_PROFILE.json"
            menu = workspace / "init" / "MENU_TEMPLATE.json"
            receipt = workspace / "init" / "RECEIPT.json"
            self.assertTrue(profile.is_file())
            self.assertTrue(menu.is_file())
            self.assertTrue(receipt.is_file())
            before = {path.name: sha256(path) for path in (profile, menu, receipt)}
            payload = json.loads(profile.read_text())
            self.assertEqual(payload["architecture"]["layers"], 4)
            self.assertEqual(payload["architecture"]["experts"], 8)
            second = run_smash(*args)
            self.assertEqual(second.returncode, 0, second.stderr)
            after = {path.name: sha256(path) for path in (profile, menu, receipt)}
            self.assertEqual(before, after, "sealed idempotent rerun must not rewrite outputs")
            self.assertEqual(
                {p.relative_to(workspace).parts[0] for p in workspace.rglob("*") if p.is_file()},
                {"init"},
            )

    def test_live_stage_refuses_missing_upstream_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smash("--workspace", str(Path(tmp) / "workspace"), "capture")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing prerequisite receipt", result.stderr.lower())

    def test_vendor_index_has_required_capabilities_and_runtime_source_identities(self) -> None:
        index = json.loads((ROOT / "vendor" / "VENDOR_INDEX.json").read_text())
        required = {
            "qtip_rep16",
            "vq_d4",
            "mxfp4",
            "fortress_measure",
            "solver_scip",
            "calibration_p930",
            "repair_p959",
            "packers",
            "runtime",
            "kernel",
            "evalplus",
            "receipt_schemas",
            "pipeline_acceleration",
            "recovered_research",
        }
        self.assertTrue(required <= set(index["capabilities"]))
        upstream = index["runtime_upstream_sha256"]
        self.assertTrue(upstream["mixed_tier_backend.py"].startswith("db14f360"))
        self.assertTrue(upstream["mixed_tier_patch.py"].startswith("80696f62"))
        self.assertTrue(upstream["mixed_prefill_server.py"].startswith("ffe52247"))
        for row in index["files"]:
            path = ROOT / row["path"]
            self.assertTrue(path.is_file(), row["path"])
            self.assertEqual(sha256(path), row["sha256"], row["path"])

    def test_offline_verifier_exercises_all_commands_with_network_blocked(self) -> None:
        result = subprocess.run(
            ["python3", str(ROOT / "tools" / "offline_dry_run.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(set(payload["stages"]), set(STAGES))
        self.assertTrue(payload["network_blocked"])

    def test_recovered_sources_have_code_receipts_gates_and_sha_closure(self) -> None:
        result = subprocess.run(
            ["python3", str(ROOT / "tools" / "verify_recovered_sources.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["recovered_source_entries"], 89)
        self.assertIn("P963", payload["promotable_families"])
        self.assertIn("P526", payload["hold_only_families"])

    def test_manifest_and_self_containment_verifiers_pass(self) -> None:
        manifest = run_smash("verify", "--manifest")
        self.assertEqual(manifest.returncode, 0, manifest.stderr)
        no_external = run_smash("verify", "--self-contained")
        self.assertEqual(no_external.returncode, 0, no_external.stderr)
        self.assertIn("PASS", no_external.stdout)


if __name__ == "__main__":
    unittest.main()
