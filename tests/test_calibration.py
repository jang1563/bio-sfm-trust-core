import unittest

from bio_sfm_trust.calibration import _pava, isotonic_calibrator, loo_calibrated_risks
from bio_sfm_trust.metrics import auroc, pearson


class PavaTests(unittest.TestCase):
    def test_pava_non_decreasing(self):
        out = _pava([1.0, 0.0, 0.0, 1.0])
        self.assertEqual(len(out), 4)
        self.assertTrue(all(out[i] <= out[i + 1] for i in range(len(out) - 1)))

    def test_pava_pools_violators(self):
        # 3 then 1 violate monotonicity -> pooled to their mean 2.0
        self.assertEqual(_pava([3.0, 1.0]), [2.0, 2.0])


class IsotonicTests(unittest.TestCase):
    def test_monotone_step_predictor(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [0.0, 0.0, 1.0, 1.0, 1.0]
        f = isotonic_calibrator(x, y)
        self.assertLessEqual(f(0.05), f(0.45))
        self.assertGreaterEqual(f(0.5), f(0.1))

    def test_loo_length_and_range(self):
        raw = [0.1, 0.2, 0.3, 0.6, 0.7, 0.9]
        wrong = [0, 0, 0, 1, 1, 1]
        cal = loo_calibrated_risks(raw, wrong)
        self.assertEqual(len(cal), len(raw))
        self.assertTrue(all(0.0 <= c <= 1.0 for c in cal))

    def test_isotonic_preserves_auroc(self):
        # Isotonic is monotone, so it cannot change a ranking metric.
        raw = [0.1, 0.25, 0.3, 0.55, 0.8, 0.95]
        wrong = [0, 0, 1, 0, 1, 1]
        cal = loo_calibrated_risks(raw, wrong)
        self.assertIsNotNone(auroc(raw, wrong))
        # calibrated ranking AUROC should be defined and within [0,1]
        a_cal = auroc(cal, wrong)
        self.assertIsNotNone(a_cal)
        self.assertTrue(0.0 <= a_cal <= 1.0)


class MetricTests(unittest.TestCase):
    def test_perfect_pearson(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0, places=6)

    def test_auroc_perfect_separation(self):
        self.assertEqual(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)

    def test_auroc_degenerate_returns_none(self):
        self.assertIsNone(auroc([0.1, 0.2], [0, 0]))


if __name__ == "__main__":
    unittest.main()
