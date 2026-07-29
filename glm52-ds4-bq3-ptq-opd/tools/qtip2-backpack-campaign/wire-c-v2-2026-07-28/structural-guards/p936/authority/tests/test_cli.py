import json
import tempfile
import unittest
from pathlib import Path

from authority import cli
from authority.authority_guard import AuthorityStore


class AuthorityCliTests(unittest.TestCase):
    def test_seal_check_accepts_top_level_dependency_list(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = root / "codebook.bin"
            payload.write_bytes(b"exact-codebook")
            first = AuthorityStore(root / "store-a")
            second = AuthorityStore(root / "store-b")
            digest = first.ingest(payload)
            second.ingest(payload)
            dependencies = root / "dependencies.json"
            locations = root / "locations.json"
            output = root / "census.json"
            dependencies.write_text(json.dumps([
                {"sha256": digest, "bytes": payload.stat().st_size, "role": "codebook"}
            ]))
            locations.write_text(json.dumps({
                "host-a": {"mode": "local", "root": str(first.root)},
                "host-b": {"mode": "local", "root": str(second.root)},
            }))

            result = cli.main([
                "seal-check", "--dependencies", str(dependencies),
                "--locations", str(locations), "--min-copies", "2",
                "--output", str(output),
            ])

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output.read_text())["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
