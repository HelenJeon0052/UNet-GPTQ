from __future__ import annotations

import unittest
import numpy as np

from unet_gptq.metrics.quality import slice_quality


class QualityContractTests(unittest.TestCase):
    def test_phase_sensitive_and_magnitude_nmse_are_distinct(self):
        target = np.ones((2, 2), dtype=np.complex64)
        result = slice_quality(-target, target)
        self.assertEqual(result["magnitude_nmse"], 0.0)
        self.assertEqual(result["complex_nmse"], 4.0)
        self.assertIsNone(result["psnr_db"])
        self.assertTrue(result["psnr_is_posinf"])

    def test_known_values_and_reference_range(self):
        target = np.ones((2, 2), dtype=np.complex64)
        result = slice_quality(target * 0.5, target)
        self.assertAlmostEqual(result["magnitude_nmse"], 0.25)
        self.assertAlmostEqual(result["complex_nmse"], 0.25)
        self.assertAlmostEqual(result["psnr_db"], 6.020599913279624)

    def test_invalid_inputs_and_zero_reference(self):
        with self.assertRaises(ValueError):
            slice_quality(np.ones((2, 2)), np.ones((3, 2)))
        with self.assertRaises(ValueError):
            slice_quality(np.full((2, 2), np.nan), np.ones((2, 2)))
        result = slice_quality(np.ones((2, 2)), np.zeros((2, 2)))
        self.assertIsNone(result["magnitude_nmse"])
        self.assertTrue(result["psnr_is_neginf"])


if __name__ == "__main__":
    unittest.main()
