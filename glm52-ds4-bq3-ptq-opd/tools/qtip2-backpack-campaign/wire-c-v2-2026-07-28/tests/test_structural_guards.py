from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P953 = ROOT / "structural-guards/p953"
sys.path.insert(0, str(P953))

from immutable_sha_authority import (  # noqa: E402
    AuthorityError,
    ImmutableSHAIndex,
    resume_layer_plan,
    sha256_file,
)


class ImmutableAuthorityTests(unittest.TestCase):
    def test_expected_sha_resolves_after_path_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "human-name.bin"
            moved = root / "renamed.bin"
            original.write_bytes(b"content-addressed authority")
            digest = sha256_file(original)
            original.rename(moved)
            document = {
                "schema": "true-c-immutable-sha-index-v1",
                "status": "SEALED",
                "objects": [{
                    "sha256": digest,
                    "bytes": moved.stat().st_size,
                    "host": "compute-node-a",
                    "path": str(moved),
                    "role": "codebook",
                }],
            }
            resolved = ImmutableSHAIndex.from_document(document).resolve(digest)
            self.assertEqual(resolved.path, str(moved))

    def test_duplicate_and_wrong_bytes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.bin"
            payload.write_bytes(b"expected")
            digest = sha256_file(payload)
            row = {
                "sha256": digest,
                "bytes": payload.stat().st_size,
                "host": "compute-node-a",
                "path": str(payload),
            }
            document = {"schema": "true-c-immutable-sha-index-v1", "status": "SEALED", "objects": [row, dict(row)]}
            with self.assertRaisesRegex(AuthorityError, "duplicate"):
                ImmutableSHAIndex.from_document(document)
            payload.write_bytes(b"wrong!!!")
            single = {"schema": "true-c-immutable-sha-index-v1", "status": "SEALED", "objects": [row]}
            with self.assertRaisesRegex(AuthorityError, "SHA drift"):
                ImmutableSHAIndex.from_document(single).resolve(digest)

    def test_resume_starts_at_first_unfinished_layer_only(self):
        progress = {
            "completed_layers": list(range(14)),
            "mmap_completed_layers": list(range(14)),
            "active_layer": None,
            "local_stage_retired": True,
            "mmap_loader_mode": "torch-mmap",
        }
        binding = "1" * 64
        checkpoint = {
            "schema": "p874-anchor-walk-ckpt-sidecar-v2",
            "status": "SEALED",
            "layer": 13,
            "wins": list(range(64)),
            "binding_sha256": binding,
            "checkpoint_sha256": "2" * 64,
            "checkpoint_bytes": 4096,
        }
        self.assertEqual(
            resume_layer_plan(progress, checkpoint, expected_binding_sha256=binding),
            list(range(14, 43)),
        )

    def test_resume_rejects_noncontiguous_prefix(self):
        progress = {
            "completed_layers": [0, 1, 3],
            "mmap_completed_layers": [0, 1, 3],
            "active_layer": None,
            "local_stage_retired": True,
            "mmap_loader_mode": "torch-mmap",
        }
        with self.assertRaisesRegex(AuthorityError, "contiguous"):
            resume_layer_plan(progress, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
