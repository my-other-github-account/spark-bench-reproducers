# SPDX-License-Identifier: Apache-2.0
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("iq3_vq_wire.py")
spec = importlib.util.spec_from_file_location("iq3_vq_wire", MODULE_PATH)
iq3_vq_wire = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(iq3_vq_wire)


class BitPackingTest(unittest.TestCase):
    def test_round_trip_each_supported_index_width(self):
        rng = np.random.default_rng(7)
        for bits in (8, 10, 11, 12):
            values = rng.integers(0, 1 << bits, size=(5, 37), dtype=np.uint16)
            packed = iq3_vq_wire.pack_index_rows(values, bits)
            self.assertEqual(packed.shape, (5, math.ceil(37 * bits / 8)))
            got = iq3_vq_wire.unpack_index_rows(packed, bits, 37)
            np.testing.assert_array_equal(got, values)

    def test_pack_rejects_out_of_range_indices(self):
        values = np.array([[0, 1024]], dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "10-bit"):
            iq3_vq_wire.pack_index_rows(values, 10)


class VqDecodeTest(unittest.TestCase):
    def test_d4_and_d8_wire_decode_is_exact(self):
        rng = np.random.default_rng(11)
        for d, k in ((4, 1024), (4, 4096), (8, 256), (8, 2048)):
            n, width = 3, 64
            codes = rng.integers(0, k, size=(n, width // d), dtype=np.uint16)
            codebook = rng.standard_normal((k, d)).astype(np.float16)
            scales = rng.integers(120, 132, size=(n, width // 32), dtype=np.uint8)
            packed = iq3_vq_wire.pack_index_rows(codes, math.ceil(math.log2(k)))
            got = iq3_vq_wire.decode_vq_rows(packed, scales, codebook, width)
            expected = codebook[codes].reshape(n, width).astype(np.float32)
            expected *= np.exp2(scales.astype(np.float32) - 127).repeat(32, axis=1)
            np.testing.assert_array_equal(got, expected.astype(np.float16))

    def test_decode_rejects_non_power_of_two_width(self):
        packed = np.zeros((1, 1), dtype=np.uint8)
        scales = np.zeros((1, 1), dtype=np.uint8)
        codebook = np.zeros((256, 4), dtype=np.float16)
        with self.assertRaisesRegex(ValueError, "multiple of 32"):
            iq3_vq_wire.decode_vq_rows(packed, scales, codebook, 31)


class TierSchemaTest(unittest.TestCase):
    def test_parse_supported_vq_tiers(self):
        expected = {
            "vqa": (4, 256),
            "d4_k1024": (4, 1024),
            "d4_k2048": (4, 2048),
            "d4_k4096": (4, 4096),
            "d8_k256": (8, 256),
            "d8_k512": (8, 512),
            "d8_k1024": (8, 1024),
            "d8_k2048": (8, 2048),
            "d8_k4096": (8, 4096),
        }
        for tier, shape in expected.items():
            self.assertEqual(iq3_vq_wire.parse_vq_tier(tier), shape)

    def test_non_vq_tier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported VQ tier"):
            iq3_vq_wire.parse_vq_tier("fp4")


class ProjectionPackingTest(unittest.TestCase):
    def test_projection_pack_routes_multiple_dimensions_without_drift(self):
        rng = np.random.default_rng(23)
        outputs, width = 3, 64
        source = {}
        for tier in ("d4_k1024", "d8_k256"):
            d, k = iq3_vq_wire.parse_vq_tier(tier)
            source[tier] = {
                "codes": rng.integers(
                    0, k, size=(2, outputs, width // d), dtype=np.uint16
                ),
                "scales": rng.integers(
                    122, 130, size=(2, outputs, width // 32), dtype=np.uint8
                ),
                "codebook": rng.standard_normal((k, d)).astype(np.float16),
            }
        repaired = (
            source["d4_k1024"]["codebook"].astype(np.float32) + 0.125
        ).astype(np.float16)
        packed = iq3_vq_wire.pack_vq_projection(
            assignments=["d4_k1024", "d8_k256", "d4_k1024"],
            source_rows={
                "d4_k1024": np.array([1, 0], dtype=np.int32),
                "d8_k256": np.array([0], dtype=np.int32),
            },
            sources=source,
            width=width,
            codebook_overrides={"d4_k1024": repaired},
        )

        self.assertEqual(packed["dimension"].tolist(), [4, 8, 4])
        self.assertEqual(packed["bits"].tolist(), [10, 8, 10])
        self.assertEqual(packed["tier_names"], ["d4_k1024", "d8_k256"])
        expected_rows = (("d4_k1024", 1), ("d8_k256", 0), ("d4_k1024", 0))
        for expert, (tier, row) in enumerate(expected_rows):
            got = iq3_vq_wire.decode_projection_expert(packed, expert, width)
            cb = repaired if tier == "d4_k1024" else source[tier]["codebook"]
            expected = cb[source[tier]["codes"][row]].reshape(outputs, width).astype(np.float32)
            expected *= np.exp2(
                source[tier]["scales"][row].astype(np.float32) - 127
            ).repeat(32, axis=1)
            np.testing.assert_array_equal(got, expected.astype(np.float16))

    def test_projection_pack_is_deterministic(self):
        source = {
            "vqa": {
                "codes": np.arange(32, dtype=np.uint16).reshape(1, 2, 16),
                "scales": np.array([[[127, 127], [127, 127]]], dtype=np.uint8),
                "codebook": np.arange(1024, dtype=np.float16).reshape(256, 4),
            }
        }
        kwargs = dict(
            assignments=["vqa"],
            source_rows={"vqa": np.array([0], dtype=np.int32)},
            sources=source,
            width=64,
        )
        first = iq3_vq_wire.pack_vq_projection(**kwargs)
        second = iq3_vq_wire.pack_vq_projection(**kwargs)
        for key in ("codes", "scales", "code_offset", "scale_offset", "codebooks"):
            np.testing.assert_array_equal(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
