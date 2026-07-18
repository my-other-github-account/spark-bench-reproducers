from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_server.sh"


class LauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not LAUNCHER.exists():
            raise AssertionError("launch_server.sh is missing")
        cls.text = LAUNCHER.read_text()

    def test_opts_into_small_m_and_batch_shape_graphs(self):
        self.assertIn("export VLLM_MOE_VQ_CUDA_WARP_MAX_M=4", self.text)
        self.assertIn("export VLLM_MOE_W2_DECODE_GRAPH_MAX_T=4", self.text)

    def test_uses_public_runtime_kernel_and_requires_local_wire(self):
        self.assertIn('PYTHONPATH="$ROOT/runtime:$ROOT/src/vq_warp_m4', self.text)
        self.assertIn('WIRE=${WIRE:?set WIRE to a local complete learned-VQ wire directory}', self.text)
        self.assertIn('test -f "$WIRE/PACK_COMPLETE"', self.text)

    def test_keeps_outer_runtime_eager_and_private_graphs_enabled(self):
        self.assertIn("export VLLM_MOE_W2_DECODE_GRAPH=1", self.text)
        self.assertIn("--enforce-eager", self.text)

    def test_refuses_busy_gpu_or_port(self):
        self.assertIn('REFUSE GPU busy', self.text)
        self.assertIn('REFUSE port {port} already open', self.text)


if __name__ == "__main__":
    unittest.main()
