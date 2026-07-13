import unittest

from bio_sfm_trust.metrics import auroc, pearson


class AurocValidationTests(unittest.TestCase):
    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            auroc([0.1], [0, 1])

    def test_rejects_nonfinite_scores(self):
        for score in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                auroc([score, 0.5], [0, 1])

    def test_rejects_nonbinary_labels(self):
        for label in (-1, 0.5, 2, "1"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                auroc([0.1, 0.9], [0, label])

    def test_degenerate_valid_labels_return_none(self):
        self.assertIsNone(auroc([0.1, 0.9], [0, 0]))


class PearsonValidationTests(unittest.TestCase):
    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            pearson([1.0], [1.0, 2.0])

    def test_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            pearson([1.0, float("nan")], [1.0, 2.0])
        with self.assertRaises(ValueError):
            pearson([1.0, 2.0], [1.0, float("inf")])

    def test_short_or_constant_valid_inputs_return_none(self):
        self.assertIsNone(pearson([], []))
        self.assertIsNone(pearson([1.0], [2.0]))
        self.assertIsNone(pearson([1.0, 1.0], [2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
