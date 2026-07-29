from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLKIT))

from p968_common import ARMS, MAX_TOKENS, PREFIX, conversation, row_path, valid_row


class ProtocolTests(unittest.TestCase):
    def test_binding_sampled_arm(self) -> None:
        self.assertEqual(
            ARMS["sampled"],
            {"n": 5, "temperature": 0.2, "top_p": 0.95, "seed_start": 10000},
        )
        self.assertEqual(list(range(ARMS["sampled"]["seed_start"], ARMS["sampled"]["seed_start"] + ARMS["sampled"]["n"])), list(range(10000, 10005)))

    def test_greedy_instability_arm(self) -> None:
        self.assertEqual(
            ARMS["greedy"],
            {"n": 3, "temperature": 0.0, "top_p": 1.0, "seed_start": 20000},
        )
        self.assertEqual(MAX_TOKENS, 4096)

    def test_prompt_template_is_exact(self) -> None:
        messages = conversation({"prompt": "def answer():\n    pass\n"})
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertTrue(messages[0]["content"].startswith(PREFIX + "\n```python\n"))
        self.assertTrue(messages[0]["content"].endswith("\n```"))

    def test_row_path_is_deterministic_and_sanitized(self) -> None:
        path = row_path(Path("results"), "true_c", "humaneval", "sampled", 4, "HumanEval/7")
        self.assertEqual(path.as_posix(), "results/true_c/humaneval/sampled/s004/HumanEval_7.json")

    def test_valid_row_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.json"
            text = "```python\nprint(1)\n```"
            row = {
                "model_arm": "true_c",
                "output": {"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()},
            }
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertTrue(valid_row(path, {"model_arm": "true_c"}))
            row["output"]["text_sha256"] = "0" * 64
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertFalse(valid_row(path, {"model_arm": "true_c"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
