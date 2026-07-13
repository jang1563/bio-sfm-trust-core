import itertools
import math
import unittest

from bio_sfm_trust.calibration import isotonic_calibrator, loo_calibrated_risks


class IsotonicInvariantTests(unittest.TestCase):
    def test_duplicate_predictors_are_permutation_invariant(self):
        rows = [(0.2, 0.0), (0.2, 1.0), (0.7, 1.0), (0.7, 0.0)]
        expected = None
        for order in itertools.permutations(rows):
            calibrator = isotonic_calibrator(
                [risk for risk, _ in order],
                [target for _, target in order],
            )
            predictions = tuple(calibrator(value) for value in (0.1, 0.2, 0.5, 0.7, 0.9))
            if expected is None:
                expected = predictions
            self.assertEqual(predictions, expected)

    def test_duplicate_predictors_keep_their_sample_weights(self):
        calibrator = isotonic_calibrator(
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        )
        # The tied x=0 group has mean 0.5 and weight two. Pooling it with the
        # violating x=1 observation therefore gives (2*0.5 + 1*0) / 3.
        self.assertAlmostEqual(calibrator(0.0), 1.0 / 3.0)
        self.assertAlmostEqual(calibrator(1.0), 1.0 / 3.0)

    def test_fit_rejects_empty_or_mismatched_inputs(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            isotonic_calibrator([], [])
        with self.assertRaisesRegex(ValueError, "equal length"):
            isotonic_calibrator([0.1], [])

    def test_fit_rejects_nonfinite_or_nonprobability_values(self):
        for invalid_x in (math.nan, math.inf, "0.1"):
            with self.subTest(invalid_x=invalid_x):
                with self.assertRaisesRegex(ValueError, "finite number"):
                    isotonic_calibrator([invalid_x], [0.0])
        for invalid_y in (math.nan, math.inf, -0.1, 1.1, "0.5"):
            with self.subTest(invalid_y=invalid_y):
                with self.assertRaisesRegex(ValueError, "probability|between"):
                    isotonic_calibrator([0.1], [invalid_y])

    def test_predict_rejects_nonfinite_query(self):
        calibrator = isotonic_calibrator([0.1, 0.9], [0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "finite number"):
            calibrator(math.nan)


class LeaveOneOutValidationTests(unittest.TestCase):
    def test_loo_rejects_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            loo_calibrated_risks([], [])

    def test_loo_rejects_mismatched_inputs(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            loo_calibrated_risks([0.1], [])

    def test_loo_rejects_invalid_targets_before_refitting(self):
        for invalid in (0.5, 2, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "binary"):
                    loo_calibrated_risks([0.1, 0.9], [0, invalid])

    def test_single_observation_preserves_zero_fallback(self):
        self.assertEqual(loo_calibrated_risks([0.1], [1]), [0.0])


if __name__ == "__main__":
    unittest.main()
