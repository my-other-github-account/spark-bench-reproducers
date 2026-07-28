from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from banana_smasher import core
from banana_smasher.self_containment import scan_package


class NegativeContractTests(unittest.TestCase):
    def test_self_containment_rejects_parent_reference_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.py").write_text("open('" + ".." + "/outside')\n")
            (root / "target").write_text("x")
            (root / "link").symlink_to(root / "target")
            result = scan_package(root)
            self.assertEqual(result["status"], "FAIL")
            kinds = {row["kind"] for row in result["failures"]}
            self.assertIn("parent-directory-reference", kinds)
            self.assertIn("symlink", kinds)

    def test_manifest_rejects_tamper_and_missing_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "payload.txt"
            payload.write_text("changed")
            manifest = root / "TOOLS_MANIFEST.json"
            manifest.write_text(json.dumps({
                "aggregate_sha256": "a" * 64,
                "files": [{"path": "payload.txt", "bytes": 1, "sha256": "b" * 64}],
            }))
            package = root / "PACKAGE_MANIFEST.json"
            package.write_text(json.dumps({
                "tools_manifest_sha256": core.sha256_file(manifest),
                "tools_aggregate_sha256": "a" * 64,
            }))
            with patch.object(core, "PACKAGE_ROOT", root), patch.object(core, "MANIFEST_PATH", manifest):
                result = core.verify_manifest()
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(result["failures"])

    def test_privacy_rejects_private_address_and_task_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "receipt.json").write_text(
                '{"host":"192' + '.168.1.9","task":"t_' + 'deadbeef' + '"}\n'
            )
            result = scan_package(root)
            self.assertEqual(result["status"], "FAIL")
            kinds = {row["kind"] for row in result["failures"]}
            self.assertIn("private-ipv4", kinds)
            self.assertIn("kanban-task-id", kinds)


if __name__ == "__main__":
    unittest.main()
