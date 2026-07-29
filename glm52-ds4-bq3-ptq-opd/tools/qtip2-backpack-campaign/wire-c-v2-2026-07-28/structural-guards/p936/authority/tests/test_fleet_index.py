import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from authority.scripts import build_fleet_index


class FleetIndexDiscoveryTests(unittest.TestCase):
    def test_discovers_k1_archives_and_sealed_manifests_without_filename_allowlist(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            k1_sha = hashlib.sha256(b"k1-l006-archive").hexdigest()
            codebook_sha = hashlib.sha256(b"codebook").hexdigest()
            k1 = root / "K1_L006_ARCHIVE.json"
            ordinary = root / "opaque.json"
            failed = root / "FAILED.json"
            ledger = root / "sealed_events.jsonl"
            ledger_sha = hashlib.sha256(b"ledger-codebook").hexdigest()
            k1.write_text(json.dumps({"status": "PASS", "archive_sha256": k1_sha}))
            ordinary.write_text(json.dumps({"status": "SEALED", "codebook_sha256": codebook_sha}))
            failed.write_text(json.dumps({"status": "FAIL", "sha256": "f" * 64}))
            ledger.write_text(
                json.dumps({"status": "RUNNING", "sha256": "e" * 64}) + "\n"
                + json.dumps({"status": "PASS", "codebook_sha256": ledger_sha}) + "\n"
            )

            selected, stats = build_fleet_index.discover_sealed_manifests([root], max_bytes=0)

            self.assertEqual(selected, [k1.resolve(), ordinary.resolve(), ledger.resolve()])
            self.assertEqual(stats["scanned_candidates"], 4)
            self.assertEqual(stats["selected_manifests"], 3)
            self.assertEqual(stats["parse_skips"], [])


if __name__ == "__main__":
    unittest.main()
