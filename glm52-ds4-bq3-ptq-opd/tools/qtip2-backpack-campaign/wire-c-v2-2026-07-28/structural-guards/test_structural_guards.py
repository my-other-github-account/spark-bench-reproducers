#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from immutable_sha_authority import (
    AuthorityError,
    ImmutableSHAIndex,
    bind_stage_specs,
    canonical_sha256,
    resume_layer_plan,
    validate_completed_layer_receipts,
)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def object_row(path: Path, digest: str) -> dict[str, object]:
    return {
        "sha256": digest,
        "bytes": path.stat().st_size,
        "host": "fixture-host",
        "path": str(path),
        "role": "payload",
    }


def index_document(*rows: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "true-c-immutable-sha-index-v1",
        "status": "SEALED",
        "objects": list(rows),
    }


class ImmutableAuthorityTests(unittest.TestCase):
    def test_deleted_provenance_path_resolves_by_expected_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = root / "authority" / "renamed-object.bin"
            authority.parent.mkdir()
            authority.write_bytes(b"immutable-scoring-input")
            digest = hashlib.sha256(authority.read_bytes()).hexdigest()
            provenance = root / "deleted-provenance.bin"
            row = {
                "kind": "qtip2_exact",
                "artifact": str(provenance),
                "artifact_sha256": digest,
                "artifact_bytes": authority.stat().st_size,
            }
            index = ImmutableSHAIndex.from_document(
                index_document(object_row(authority, digest))
            )
            specifications = bind_stage_specs([row], index)
            resolved = index.resolve(digest, expected_bytes=authority.stat().st_size)
            score_input_sha = hashlib.sha256(Path(resolved.path).read_bytes()).hexdigest()
            self.assertFalse(provenance.exists())
            self.assertEqual(specifications[0]["source"], str(authority))
            self.assertEqual(score_input_sha, digest)

    def test_missing_duplicate_and_wrong_bytes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.bin"
            valid.write_bytes(b"valid")
            digest = hashlib.sha256(valid.read_bytes()).hexdigest()
            row = object_row(valid, digest)
            with self.assertRaisesRegex(AuthorityError, "duplicate"):
                ImmutableSHAIndex.from_document(index_document(row, dict(row)))
            index = ImmutableSHAIndex.from_document(index_document(row))
            with self.assertRaisesRegex(AuthorityError, "missing"):
                index.resolve("0" * 64)
            wrong = root / "wrong.bin"
            wrong.write_bytes(b"wrong")
            with self.assertRaisesRegex(AuthorityError, "object SHA drift"):
                ImmutableSHAIndex.from_document(
                    index_document(object_row(wrong, digest))
                ).resolve(digest)


class ResumeReceiptTests(unittest.TestCase):
    def _preseed(self, root: Path, tamper_layer: int | None = None):
        wins = [3, 11, 19]
        binding_sha = "1" * 64
        receipt_references = []
        previous_sha = None
        checkpoint = root / "anchor_L013.pt"
        checkpoint.write_bytes(b"exact-checkpoint")
        for layer in range(14):
            statistics = {
                "hidden_tensor_count": 2,
                "hidden_numel": 128 + layer,
                "hidden_sum": float(layer),
                "hidden_sum_squares": float(layer * layer),
            }
            receipt = {
                "schema": "p874-anchor-walk-layer-receipt-v3",
                "status": "SEALED",
                "layer": layer,
                "wins": wins,
                "binding_sha256": binding_sha,
                "previous_receipt_sha256": previous_sha,
                "sufficient_statistics": statistics,
                "sufficient_statistics_sha256": canonical_sha256(statistics),
            }
            if layer == 13:
                receipt.update(
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_bytes": checkpoint.stat().st_size,
                        "checkpoint_sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                    }
                )
            path = root / f"anchor_L{layer:03d}.json"
            raw = json_bytes(receipt)
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            receipt_references.append(
                {"layer": layer, "path": str(path), "sha256": digest}
            )
            previous_sha = digest
        progress = {
            "completed_layers": list(range(14)),
            "mmap_completed_layers": list(range(14)),
            "active_layer": None,
            "local_stage_retired": True,
            "mmap_loader_mode": "torch-mmap",
            "completed_layer_receipts": receipt_references,
        }
        if tamper_layer is not None:
            target = root / f"anchor_L{tamper_layer:03d}.json"
            value = json.loads(target.read_bytes())
            value["sufficient_statistics"]["hidden_sum"] += 1.0
            target.write_bytes(json_bytes(value))
        return progress, binding_sha, wins

    def test_l000_through_l013_resume_starts_at_l014(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            progress, binding_sha, wins = self._preseed(Path(directory))
            latest = validate_completed_layer_receipts(
                progress,
                expected_binding_sha256=binding_sha,
                expected_wins=wins,
            )
            pending = resume_layer_plan(
                progress, latest, expected_binding_sha256=binding_sha
            )
            self.assertEqual(pending, list(range(14, 43)))
            self.assertTrue(set(pending).isdisjoint(range(14)))

    def test_tampered_completed_layer_statistics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            progress, binding_sha, wins = self._preseed(Path(directory), 7)
            with self.assertRaisesRegex(AuthorityError, "receipt SHA drift"):
                validate_completed_layer_receipts(
                    progress,
                    expected_binding_sha256=binding_sha,
                    expected_wins=wins,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
