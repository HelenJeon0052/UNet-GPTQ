from __future__ import annotations

import unittest
import numpy as np

from unet_gptq.diffusion.aggregation import overlapping_pair_components


class OverlapContractTests(unittest.TestCase):
    def test_correct_channels_contribute_to_each_middle_slice(self):
        pairs = [np.full((1, 4, 2, 2), value, dtype=float) for value in (1, 3, 9)]
        left, right = overlapping_pair_components(pairs)
        np.testing.assert_array_equal(left, np.full((1, 2, 2, 2), 2))
        np.testing.assert_array_equal(right, np.full((1, 2, 2, 2), 6))

    def test_rejects_missing_or_malformed_pairs(self):
        with self.assertRaises(ValueError):
            overlapping_pair_components([np.zeros((1, 4, 2, 2))])
        with self.assertRaises(ValueError):
            overlapping_pair_components([np.zeros((1, 2, 2, 2))] * 3)


if __name__ == "__main__":
    unittest.main()
