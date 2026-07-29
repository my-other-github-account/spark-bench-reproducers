from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publication_safety", ROOT / "code/publication_safety.py")
assert SPEC and SPEC.loader
SAFETY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAFETY)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def row(path: str, data: bytes, *, source_sha: str | None = None) -> dict:
    public_sha = digest(data)
    source_sha = source_sha or public_sha
    return {
        "path": path,
        "privacy_substitution_applied": source_sha != public_sha,
        "provenance_type": "sealed_source_public_copy",
        "public_copy_bytes": len(data),
        "public_copy_sha256": public_sha,
        "role": "test text",
        "source_sha256": source_sha,
        "source_verification": "test fixture",
    }


def build_package(base: Path, files: dict[str, bytes]) -> Path:
    root = base / "package"
    root.mkdir()
    rows = []
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append(row(relative, data))
    (root / "PACKAGE_MANIFEST.json").write_text(json.dumps({"files": rows}) + "\n")
    return root


class PublicationSafetyTests(unittest.TestCase):
    def test_clean_tree_returns_machine_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root = build_package(Path(td), {"a.json": b"{}\n", "rows/data.jsonl": b'{"x":1}\n'})
            receipt = SAFETY.scan_package(root)
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["scanned_file_count"], 2)
            self.assertEqual(receipt["privacy_scan_status"], "PASS_NO_PRIVATE_OR_SECRET_MATERIAL")

    def test_manifest_path_and_binary_rejections(self):
        invalid = (
            "../x.json", "/x.json", "a\\b.json", "a/./b.json", "a/../b.json",
            "a\x00.json", ".hidden.json", "cache/.git/x.json", "payload.bin", "payload.txt",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                SAFETY.validate_relative_path(value)

    def test_symlink_file_and_directory_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            outside = base / "outside.json"
            outside.write_text("{}\n")
            root = build_package(base, {"safe.json": b"{}\n"})
            (root / "escape.json").symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "symlink_file"):
                SAFETY.scan_package(root)
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            outside = base / "outside"
            outside.mkdir()
            root = build_package(base, {"safe.json": b"{}\n"})
            (root / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink_directory"):
                SAFETY.scan_package(root)

    def test_nul_invalid_utf8_hidden_and_fifo_rejected(self):
        fixtures = (("nul.json", b"{}\x00"), ("bad.json", b"\xff"), (".hidden.json", b"{}\n"))
        for name, data in fixtures:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                root = build_package(Path(td), {name: data})
                with self.assertRaises(RuntimeError):
                    SAFETY.scan_package(root)
        if hasattr(os, "mkfifo"):
            with tempfile.TemporaryDirectory() as td:
                root = build_package(Path(td), {"safe.json": b"{}\n"})
                os.mkfifo(root / "pipe.json")
                with self.assertRaisesRegex(RuntimeError, "nonregular_file"):
                    SAFETY.scan_package(root)

    def test_manifest_closure_and_provenance_semantics_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = build_package(Path(td), {"safe.json": b"{}\n"})
            (root / "extra.json").write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "unmanifested"):
                SAFETY.scan_package(root)
        with tempfile.TemporaryDirectory() as td:
            root = build_package(Path(td), {"safe.json": b"{}\n"})
            manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
            manifest["files"][0]["source_sha256"] = "f" * 64
            manifest["files"][0]["privacy_substitution_applied"] = False
            (root / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "privacy_substitution_semantics"):
                SAFETY.scan_package(root)

    def test_private_and_secret_families_rejected(self):
        samples = {
            "unix_home": "/" + "Users" + "/alice/project/file.json",
            "windows_home": "C:" + "\\Users" + "\\alice" + "\\secret.txt",
            "host": "spa" + "rk-3",
            "private_dns": "node.corp." + "internal",
            "task": "t_" + "deadbeef",
            "rfc1918": "192" + ".168.1.2",
            "cgnat": "100" + ".100.1.2",
            "identity": "d" + "nola",
            "url_creds": "https://alice:" + "hunter2@example.com/repo",
            "pem": "-----BEGIN " + "PRIVATE KEY-----",
            "github": "gh" + "p_" + "A" * 24,
            "openai": "s" + "k-" + "A" * 24,
            "hf": "h" + "f_" + "A" * 24,
            "slack": "xo" + "xb-" + "A" * 20,
            "aws": "AK" + "IA" + "A" * 16,
            "google": "AI" + "za" + "A" * 32,
            "assigned": "api_" + "key = " + '"live-secret-value"',
        }
        for label, text in samples.items():
            with self.subTest(label=label):
                self.assertTrue(SAFETY.privacy_findings(text), text)

    def test_hashes_and_approved_placeholders_pass(self):
        samples = (
            "sha256=" + "a" * 64,
            'token = "REDACTED"',
            'api_key = "PUBLIC_API_KEY"',
            'password = os.environ["PASSWORD"]',
            'secret = "${SECRET}"',
            'private_key = "<required>"',
            "TOKEN_PATTERNS = (re.compile(r'example'),)",
        )
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(SAFETY.privacy_findings(text), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
