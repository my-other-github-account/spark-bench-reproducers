import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from authority.authority_guard import (
    AuthorityStore,
    GuardViolation,
    assert_reclaim_allowed,
    assert_seal_dependencies,
    build_protected_index,
    resolve_codebook_binding,
    resolve_plan_codebook,
)


class AuthorityStoreTests(unittest.TestCase):
    def test_ingest_survives_source_path_deletion_and_resolves_only_by_sha(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "mission" / "layer.codebook.fp16.bin"
            source.parent.mkdir()
            payload = b"sealed-codebook-bytes"
            source.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()

            store = AuthorityStore(root / "authority_store")
            actual = store.ingest(source, metadata={"role": "codebook"})
            source.unlink()

            self.assertEqual(actual, expected)
            resolved = store.resolve(expected)
            self.assertEqual(resolved, store.root / "store" / f"{expected}.bin")
            self.assertEqual(resolved.read_bytes(), payload)
            rows = [json.loads(line) for line in (store.root / "index.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sha256"], expected)

    def test_estimate_only_substitution_waiver_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = AuthorityStore(root / "authority_store")
            expected_file = root / "expected.bin"
            substitute_file = root / "substitute.bin"
            expected_file.write_bytes(b"expected")
            substitute_file.write_bytes(b"substitute")
            expected = store.ingest(expected_file)
            substitute = store.ingest(substitute_file)
            plan = root / "mission" / "PLAN.json"
            plan.parent.mkdir()
            plan.write_text("{}")
            waiver = {
                "expected_codebook_sha256": expected,
                "substitute_codebook_sha256": substitute,
                "measured_delta": 0.002,
                "ci95": [-0.001, 0.005],
                "measurement_receipt_sha256": "0" * 64,
                "windows": 64,
            }
            (plan.parent / "SUBSTITUTION_WAIVER.json").write_text(json.dumps(waiver))

            with self.assertRaisesRegex(GuardViolation, "measurement receipt"):
                resolve_codebook_binding(store, plan, expected, substitute)

    def test_measured_waiver_resolves_exact_substitute_from_store(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = AuthorityStore(root / "authority_store")
            expected_file = root / "expected.bin"
            substitute_file = root / "substitute.bin"
            expected_file.write_bytes(b"expected")
            substitute_file.write_bytes(b"substitute")
            expected = store.ingest(expected_file)
            substitute = store.ingest(substitute_file)
            plan = root / "mission" / "PLAN.json"
            plan.parent.mkdir()
            plan.write_text("{}")
            measurement = {
                "schema": "p936-codebook-substitution-measurement-v1",
                "status": "PASS",
                "expected_codebook_sha256": expected,
                "substitute_codebook_sha256": substitute,
                "measured_delta": 0.029,
                "ci95": [0.021, 0.037],
                "windows": 64,
            }
            measurement_path = root / "measurement.json"
            measurement_path.write_text(json.dumps(measurement, sort_keys=True))
            measurement_sha = store.ingest(measurement_path, metadata={"role": "measurement_receipt"})
            waiver = {
                "expected_codebook_sha256": expected,
                "substitute_codebook_sha256": substitute,
                "measured_delta": measurement["measured_delta"],
                "ci95": measurement["ci95"],
                "measurement_receipt_sha256": measurement_sha,
                "windows": measurement["windows"],
            }
            (plan.parent / "SUBSTITUTION_WAIVER.json").write_text(json.dumps(waiver))

            resolved = resolve_codebook_binding(store, plan, expected, substitute)
            self.assertEqual(resolved, store.path_for(substitute))

    def test_measured_waiver_rejects_additional_estimate_claim(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = AuthorityStore(root / "authority_store")
            expected_file = root / "expected.bin"
            substitute_file = root / "substitute.bin"
            expected_file.write_bytes(b"expected")
            substitute_file.write_bytes(b"substitute")
            expected = store.ingest(expected_file)
            substitute = store.ingest(substitute_file)
            plan = root / "mission" / "PLAN.json"
            plan.parent.mkdir()
            plan.write_text("{}")
            measurement = {
                "schema": "p936-codebook-substitution-measurement-v1",
                "status": "PASS",
                "expected_codebook_sha256": expected,
                "substitute_codebook_sha256": substitute,
                "measured_delta": 0.029,
                "ci95": [0.021, 0.037],
                "windows": 64,
            }
            measurement_path = root / "measurement.json"
            measurement_path.write_text(json.dumps(measurement, sort_keys=True))
            measurement_sha = store.ingest(measurement_path)
            waiver = {
                "expected_codebook_sha256": expected,
                "substitute_codebook_sha256": substitute,
                "measured_delta": measurement["measured_delta"],
                "ci95": measurement["ci95"],
                "measurement_receipt_sha256": measurement_sha,
                "windows": measurement["windows"],
                "estimate": "sub-noise ~0.1-0.2%",
            }
            (plan.parent / "SUBSTITUTION_WAIVER.json").write_text(json.dumps(waiver))

            with self.assertRaisesRegex(GuardViolation, "schema"):
                resolve_codebook_binding(store, plan, expected, substitute)

    def test_measured_waiver_rejects_non_finite_measurement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = AuthorityStore(root / "authority_store")
            expected_file = root / "expected.bin"
            substitute_file = root / "substitute.bin"
            expected_file.write_bytes(b"expected")
            substitute_file.write_bytes(b"substitute")
            expected = store.ingest(expected_file)
            substitute = store.ingest(substitute_file)
            plan = root / "mission" / "PLAN.json"
            plan.parent.mkdir()
            plan.write_text("{}")
            measurement = {
                "schema": "p936-codebook-substitution-measurement-v1",
                "status": "PASS",
                "expected_codebook_sha256": expected,
                "substitute_codebook_sha256": substitute,
                "measured_delta": float("inf"),
                "ci95": [float("inf"), float("inf")],
                "windows": 64,
            }
            measurement_path = root / "measurement.json"
            measurement_path.write_text(json.dumps(measurement))
            measurement_sha = store.ingest(measurement_path)
            waiver = {
                "expected_codebook_sha256": expected,
                "substitute_codebook_sha256": substitute,
                "measured_delta": measurement["measured_delta"],
                "ci95": measurement["ci95"],
                "measurement_receipt_sha256": measurement_sha,
                "windows": measurement["windows"],
            }
            (plan.parent / "SUBSTITUTION_WAIVER.json").write_text(json.dumps(waiver))

            with self.assertRaisesRegex(GuardViolation, "finite"):
                resolve_codebook_binding(store, plan, expected, substitute)

    def test_reclaim_refuses_sha_referenced_by_sealed_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            protected_file = root / "codebook.bin"
            protected_file.write_bytes(b"protected-codebook")
            digest = hashlib.sha256(protected_file.read_bytes()).hexdigest()
            sealed_manifest = root / "SEALED_MANIFEST.json"
            sealed_manifest.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "rows": [{"codebook": {"sha256": digest}}],
                    }
                )
            )
            ignored_manifest = root / "FAILED_MANIFEST.json"
            ignored_manifest.write_text(
                json.dumps({"status": "FAIL_CLOSED", "sha256": "f" * 64})
            )
            index_path = root / "protected_sha_index.jsonl"
            index = build_protected_index([sealed_manifest, ignored_manifest], index_path)

            self.assertIn(digest, index["entries"])
            self.assertNotIn("f" * 64, index["entries"])
            rows = [json.loads(line) for line in index_path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sha256"], digest)
            self.assertEqual(rows[0]["schema"], "p936-protected-sha-index-v1")
            with self.assertRaisesRegex(GuardViolation, "protected SHA"):
                assert_reclaim_allowed([protected_file], index_path)

    def test_reclaim_allows_protected_sha_after_exact_archive_readback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            protected_file = root / "codebook.bin"
            protected_file.write_bytes(b"protected-codebook")
            digest = hashlib.sha256(protected_file.read_bytes()).hexdigest()
            manifest = root / "SEALED_MANIFEST.json"
            manifest.write_text(json.dumps({"status": "PASS", "codebook_sha256": digest}))
            index_path = root / "protected_sha_index.json"
            build_protected_index([manifest], index_path)
            archived = root / "nas" / f"{digest}.bin"
            archived.parent.mkdir()
            archived.write_bytes(protected_file.read_bytes())
            archive_receipt = root / "ARCHIVE_FIRST.json"
            archive_receipt.write_text(
                json.dumps(
                    {
                        "schema": "p936-archive-first-v1",
                        "status": "PASS",
                        "entries": [
                            {
                                "source_path": str(protected_file.resolve()),
                                "source_sha256": digest,
                                "nas_path": str(archived.resolve()),
                                "archive_bytes": archived.stat().st_size,
                                "readback_sha256": digest,
                            }
                        ],
                    }
                )
            )

            assert_reclaim_allowed([protected_file], index_path, archive_receipt)

    def test_seal_requires_two_host_copies_of_every_dependency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "codebook.bin"
            source.write_bytes(b"dependency")
            store3 = AuthorityStore(root / "compute-node-a")
            store4 = AuthorityStore(root / "compute-node-b")
            digest = store3.ingest(source)
            dependency = {"sha256": digest, "bytes": source.stat().st_size, "role": "codebook"}
            locations = {"compute-node-a": store3.root, "compute-node-b": store4.root}

            with self.assertRaisesRegex(GuardViolation, "copy census"):
                assert_seal_dependencies([dependency], locations, min_copies=2)

            store4.ingest(source)
            census = assert_seal_dependencies([dependency], locations, min_copies=2)
            self.assertEqual(census["status"], "PASS")
            self.assertEqual(census["dependencies"][0]["copy_count"], 2)

            with self.assertRaisesRegex(GuardViolation, "at least one codebook"):
                assert_seal_dependencies([], locations, min_copies=2)

    def test_plan_build_resolves_sha_after_mission_path_disappears(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "mission" / "codebook.bin"
            source.parent.mkdir()
            source.write_bytes(b"codebook-from-sealed-plan")
            store = AuthorityStore(root / "authority_store")
            digest = store.ingest(source)
            plan = root / "PLAN.json"
            plan.write_text("{}")
            row = {
                "codebook": {
                    "path": str(source),
                    "sha256": digest,
                    "bytes": source.stat().st_size,
                }
            }
            source.unlink()

            resolved = resolve_plan_codebook(store, plan, row)
            self.assertEqual(resolved, store.path_for(digest))


if __name__ == "__main__":
    unittest.main()
