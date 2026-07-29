#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from immutable_sha_authority import (
    AuthorityError,
    ImmutableSHAIndex,
    bind_stage_specs,
    resume_layer_plan,
    sha256_file,
    validate_inherited_prefix,
)


def sealed_index(*objects: dict[str, object]) -> dict[str, object]:
    return {"schema": "true-c-immutable-sha-index-v1", "status": "SEALED", "objects": list(objects)}


def authority_row(path: Path, digest: str) -> dict[str, object]:
    return {
        "sha256": digest,
        "bytes": path.stat().st_size,
        "host": "fixture-node",
        "path": str(path),
        "role": "payload",
    }


class ImmutableAuthorityTests(unittest.TestCase):
    def test_renamed_provenance_path_resolves_expected_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = root / "renamed-object.bin"
            authority.write_bytes(b"immutable-scoring-input")
            digest = sha256_file(authority)
            row = {
                "kind": "genesis_vq",
                "artifact": "/public-provenance/deleted-object.bin",
                "artifact_sha256": digest,
                "artifact_bytes": authority.stat().st_size,
            }
            index = ImmutableSHAIndex.from_document(sealed_index(authority_row(authority, digest)))
            specs = bind_stage_specs([row], index)
            resolved = index.resolve(digest, expected_bytes=authority.stat().st_size)
            self.assertEqual(specs[0]["source"], str(authority))
            self.assertEqual(sha256_file(Path(resolved.path)), digest)
            self.assertNotEqual(specs[0]["source"], row["artifact_provenance_path"])

    def test_missing_duplicate_and_wrong_bytes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.bin"
            valid.write_bytes(b"valid")
            digest = sha256_file(valid)
            row = authority_row(valid, digest)
            with self.assertRaisesRegex(AuthorityError, "duplicate"):
                ImmutableSHAIndex.from_document(sealed_index(row, dict(row)))
            index = ImmutableSHAIndex.from_document(sealed_index(row))
            with self.assertRaisesRegex(AuthorityError, "missing"):
                index.resolve("0" * 64)
            wrong = root / "wrong.bin"
            wrong.write_bytes(b"wrong")
            with self.assertRaisesRegex(AuthorityError, "object SHA drift"):
                ImmutableSHAIndex.from_document(sealed_index(authority_row(wrong, digest))).resolve(digest)

    def test_index_bytes_are_pinned_before_parse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.bin"
            payload.write_bytes(b"payload")
            index_path = root / "index.json"
            index_path.write_text(json.dumps(sealed_index(authority_row(payload, sha256_file(payload)))))
            with self.assertRaisesRegex(AuthorityError, "index SHA drift"):
                ImmutableSHAIndex.load(index_path, expected_index_sha256="f" * 64)


class PrefixAndResumeTests(unittest.TestCase):
    def test_inherited_prefix_uses_hashes_not_paths_or_producer_names(self) -> None:
        codebook_sha = "1" * 64
        rows = [
            {
                "identity": [7, expert, "down"],
                "artifact_sha256": f"{expert + 2:064x}",
                "codebook_sha256": codebook_sha,
                "status": "PASS",
            }
            for expert in range(3)
        ]
        contract = {
            "codebook_group": ["down", 2048],
            "rows": copy.deepcopy(rows),
            "expected_rows": 512,
            "codebook_sha256": codebook_sha,
        }
        receipt = {
            "codebook_group": ["down", 2048],
            "status": "PARTIAL",
            "completed_rows": 3,
            "expected_rows": 512,
            "codebook_sha256": codebook_sha,
            "rows": copy.deepcopy(rows),
            "task_id": "renamed-producer",
            "codebook": "/public-provenance/renamed-codebook.bin",
        }
        for index, row in enumerate(receipt["rows"]):
            row["task_id"] = f"producer-{index}"
            row["artifact"] = f"/public-provenance/payload-{index}.pt"
        result = validate_inherited_prefix(receipt, contract)
        self.assertEqual(result["status"], "PASS_EXACT_HASH_BOUND_PREFIX")
        self.assertFalse(result["producer_task_id_coupled"])
        self.assertFalse(result["path_coupled"])
        bad = copy.deepcopy(receipt)
        bad["rows"][0]["artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(AuthorityError, "payload SHA"):
            validate_inherited_prefix(bad, contract)

    def test_completed_l000_l013_resume_starts_at_l014(self) -> None:
        progress = {
            "completed_layers": list(range(14)),
            "mmap_completed_layers": list(range(14)),
            "active_layer": None,
            "local_stage_retired": True,
            "mmap_loader_mode": "torch-mmap",
        }
        binding = "2" * 64
        sidecar = {
            "schema": "p874-anchor-walk-ckpt-sidecar-v2",
            "status": "SEALED",
            "layer": 13,
            "wins": [3, 11, 19],
            "binding_sha256": binding,
            "checkpoint_sha256": "3" * 64,
            "checkpoint_bytes": 4096,
        }
        self.assertEqual(
            resume_layer_plan(progress, sidecar, expected_binding_sha256=binding),
            list(range(14, 43)),
        )

    def test_resume_without_checkpoint_never_recomputes_prefix(self) -> None:
        progress = {
            "completed_layers": list(range(14)),
            "mmap_completed_layers": list(range(14)),
            "active_layer": None,
            "local_stage_retired": True,
            "mmap_loader_mode": "torch-mmap",
        }
        with self.assertRaisesRegex(AuthorityError, "no checkpoint"):
            resume_layer_plan(progress, None)

    def test_noncontiguous_resume_prefix_fails_closed(self) -> None:
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
