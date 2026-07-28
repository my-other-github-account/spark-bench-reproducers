from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vendor" / "repair" / "sealed_wire_seed.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sealed_wire_seed", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SealedWireSeedTests(unittest.TestCase):
    def test_seed_reuses_exact_inventory_sources(self) -> None:
        module = load_module()
        inventory = [
            {"cell": "l1/e1/down", "tier": "qtip2", "payload_sha256": "a" * 64},
            {"cell": "l1/e2/down", "tier": "qtip2", "payload_sha256": "b" * 64},
            {"cell": "l2/e1/down", "tier": "qtip3", "payload_sha256": "c" * 64},
            {"cell": "l2/e2/down", "tier": "qtip3", "payload_sha256": "d" * 64},
        ]
        updates = [
            {"target": "l1/e1/down", "source": "l1/e2/down"},
            {"target": "l2/e1/down", "source": "l2/e2/down"},
        ]
        seed = module.build_seed(inventory, updates, expected_updates=2)
        self.assertEqual([row["target"] for row in seed], ["l1/e1/down", "l2/e1/down"])
        self.assertEqual(seed[0]["source_payload_sha256"], "b" * 64)
        self.assertEqual(seed[1]["source_payload_sha256"], "d" * 64)
        self.assertTrue(all(row["source_tier"] == row["target_tier"] for row in seed))

    def test_seed_rejects_cross_tier_and_duplicate_targets(self) -> None:
        module = load_module()
        inventory = [
            {"cell": "a", "tier": "qtip2", "payload_sha256": "a" * 64},
            {"cell": "b", "tier": "qtip3", "payload_sha256": "b" * 64},
        ]
        with self.assertRaisesRegex(ValueError, "tier mismatch"):
            module.build_seed(inventory, [{"target": "a", "source": "b"}], expected_updates=1)
        with self.assertRaisesRegex(ValueError, "duplicate target"):
            module.build_seed(
                inventory,
                [{"target": "a", "source": "a"}, {"target": "a", "source": "a"}],
                expected_updates=2,
            )


if __name__ == "__main__":
    unittest.main()
