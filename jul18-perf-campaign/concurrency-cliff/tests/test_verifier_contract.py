from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools/verify_golden_env.py"


class VerifierContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not VERIFIER.exists():
            raise AssertionError("tools/verify_golden_env.py is missing")
        cls.text = VERIFIER.read_text()

    def test_pins_task_runtime_hashes(self):
        self.assertIn("2cc85159", self.text)
        self.assertIn("7b25bf83", self.text)
        self.assertIn("2b0baea3", self.text)

    def test_requires_small_m_and_batch_graph_env(self):
        self.assertIn('"VLLM_MOE_VQ_CUDA_WARP_MAX_M": "4"', self.text)
        self.assertIn('"VLLM_MOE_W2_DECODE_GRAPH_MAX_T": "4"', self.text)

    def test_postflight_requires_m2_warp_and_graph_replay_probes(self):
        self.assertIn("VQ DISPATCH PROBE path=cuda_warp_m2", self.text)
        self.assertIn("DECODE-GRAPH REPLAY PROBE", self.text)


if __name__ == "__main__":
    unittest.main()
