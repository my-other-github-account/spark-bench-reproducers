import unittest

from reclamation import safe_reclaim


class SafeReclaimDefaultsTests(unittest.TestCase):
    def test_default_protected_index_is_required_jsonl_path(self):
        self.assertEqual(
            safe_reclaim.DEFAULT_PROTECTED_INDEX,
            "~/authority_store/protected_sha_index.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
