import math
import unittest

from calibrate_gate64 import average_ranks, fit_rows, pearson


class PredictorTests(unittest.TestCase):
    def test_average_ranks_with_ties(self):
        self.assertEqual(average_ranks([3.0, 1.0, 1.0, 2.0]), [4.0, 1.5, 1.5, 3.0])

    def test_perfect_linear_fit(self):
        fit = fit_rows([(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)])
        self.assertAlmostEqual(fit["slope"], 2.0)
        self.assertAlmostEqual(fit["intercept"], 1.0)
        self.assertAlmostEqual(fit["pearson_r"], 1.0)
        self.assertAlmostEqual(fit["spearman_rho"], 1.0)

    def test_negative_correlation(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [3, 2, 1]), -1.0)

    def test_constant_column_rejected(self):
        with self.assertRaises(ValueError):
            fit_rows([(1.0, 2.0), (1.0, 3.0)])


if __name__ == "__main__":
    unittest.main()
