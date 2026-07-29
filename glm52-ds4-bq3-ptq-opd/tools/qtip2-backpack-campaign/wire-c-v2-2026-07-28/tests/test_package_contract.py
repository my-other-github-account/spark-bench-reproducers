from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wire_c_verify_package", ROOT / "code/verify_package.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class PublicationMutationTests(unittest.TestCase):
    def load(self, relative: str):
        return json.loads((ROOT / relative).read_text())

    def test_canonical_p931_contract_passes(self):
        self.assertEqual(
            VERIFY.validate_p931_document(self.load("artifacts/P931_V3_DEFINITIVE.public.json")),
            [],
        )

    def test_p931_rejects_one_ulp_objective_drift(self):
        document = self.load("artifacts/P931_V3_DEFINITIVE.public.json")
        document["solver"]["objective_reweighted"] = 0.03507863303949007
        self.assertTrue(any("solver" in failure for failure in VERIFY.validate_p931_document(document)))

    def test_p931_rejects_measured_or_physical_mislabel(self):
        document = self.load("artifacts/P931_V3_DEFINITIVE.public.json")
        document["public_validity"]["measured"] = True
        document["public_validity"]["physical_checkpoint_scored"] = True
        self.assertTrue(any("public validity" in failure for failure in VERIFY.validate_p931_document(document)))

    def test_p931_rejects_wall_time_and_source_map_drift(self):
        document = self.load("artifacts/P931_V3_DEFINITIVE.public.json")
        document["solver"]["wall_seconds"] += 1.0
        document["verification"]["source_output_shas"].pop(next(iter(document["verification"]["source_output_shas"])))
        failures = VERIFY.validate_p931_document(document)
        self.assertTrue(any("solver" in failure for failure in failures))
        self.assertTrue(any("source-output map" in failure for failure in failures))

    def test_p931_rejects_missing_alias_map(self):
        document = self.load("artifacts/P931_V3_DEFINITIVE.public.json")
        document["verification"]["source_to_reviewed_manifest_aliases"] = {}
        self.assertTrue(any("source aliases" in failure for failure in VERIFY.validate_p931_document(document)))

    def test_canonical_p963_contract_passes(self):
        self.assertEqual(
            VERIFY.validate_p963_document(self.load("artifacts/P963_EXACT_ACCELERATION_SEAL.public.json")),
            [],
        )

    def test_p963_rejects_output_or_timing_drift(self):
        document = self.load("artifacts/P963_EXACT_ACCELERATION_SEAL.public.json")
        document["accelerated"]["output_set_sha256"] = "0" * 64
        document["comparison"]["speedup"] += 0.1
        failures = VERIFY.validate_p963_document(document)
        self.assertTrue(any("output SHA" in failure for failure in failures))
        self.assertTrue(any("speedup arithmetic" in failure for failure in failures))

    def test_binding_evaluation_counts_are_n5_and_three_greedy(self):
        p967 = self.load("evaluation/P967_INFERENCE_PROTOCOL.public.json")
        p968 = self.load("evaluation/P968_AUTHORITY_MAP.public.json")
        self.assertEqual(p967["sampled_extension"]["n_per_task"], 5)
        self.assertEqual(p967["greedy_instability"]["repeats"], 3)
        self.assertEqual(p968["protocol_preregistration"]["sampled"]["n_per_task"], 5)
        self.assertEqual(p968["protocol_preregistration"]["greedy_instability"]["repeats"], 3)
        self.assertFalse(p968["public_validity"]["true_c_paired_results_available"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
