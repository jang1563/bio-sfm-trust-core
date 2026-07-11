"""RCPS / conformal trust-threshold: the false-accept guarantee, the refuse-when-weak behavior,
and monotonicity in alpha. Pure stdlib (seeded random for reproducibility)."""

import random
import unittest

from bio_sfm_trust import (
    clopper_pearson_upper_bound,
    false_accept_rate,
    hoeffding_upper_bound,
    rcps_threshold,
    split_ltt_threshold,
    validate_fixed_threshold,
)


def _draw(n, rng):
    # calibrated risk == P(wrong): wrong ~ Bernoulli(risk). The trusted set {risk<=tau} then has
    # true false-accept rate ~ mean risk over accepted (<= tau).
    risks = [rng.random() for _ in range(n)]
    wrong = [1 if rng.random() < r else 0 for r in risks]
    return risks, wrong


class ConformalTests(unittest.TestCase):
    def test_false_accept_rate(self):
        risks, wrong = [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]
        self.assertEqual(false_accept_rate(risks, wrong, 0.5), 0.0)   # accept {0.1,0.2}: both right
        self.assertEqual(false_accept_rate(risks, wrong, 1.0), 0.5)   # accept all: 2/4 wrong
        self.assertIsNone(false_accept_rate(risks, wrong, 0.0))       # empty accept set

    def test_rcps_controls_held_out_false_accept_rate(self):
        rng = random.Random(0)
        cr, cw = _draw(600, rng)
        tau = rcps_threshold(cr, cw, alpha=0.2, delta=0.1)
        self.assertIsNotNone(tau)
        self.assertTrue(0.0 < tau < 1.0)
        # the guarantee transfers to fresh data: the trusted set's actual false-accept rate <= alpha
        hr, hw = _draw(4000, rng)
        rate = false_accept_rate(hr, hw, tau)
        self.assertIsNotNone(rate)
        self.assertLessEqual(rate, 0.2, f"held-out false-accept {rate:.3f} must respect alpha=0.2")

    def test_rcps_refuses_when_signal_is_weak(self):
        # wrongness independent of risk at base rate ~0.5 -> cannot certify a 0.1 false-accept set
        rng = random.Random(1)
        risks = [rng.random() for _ in range(400)]
        wrong = [rng.randint(0, 1) for _ in range(400)]
        self.assertIsNone(rcps_threshold(risks, wrong, alpha=0.1, delta=0.1))

    def test_rcps_lenient_alpha_is_at_least_as_permissive(self):
        rng = random.Random(2)
        r, w = _draw(500, rng)
        t_strict = rcps_threshold(r, w, alpha=0.2, delta=0.1)
        t_lenient = rcps_threshold(r, w, alpha=0.45, delta=0.1)
        self.assertIsNotNone(t_lenient)
        if t_strict is not None:
            self.assertGreaterEqual(t_lenient, t_strict)   # more risk tolerance -> larger tau

    def test_fixed_threshold_certificate_reports_ucb(self):
        report = validate_fixed_threshold(
            [0.1] * 40 + [0.9] * 10,
            [0] * 40 + [1] * 10,
            tau=0.1,
            alpha=0.2,
            delta=0.1,
        )
        self.assertTrue(report["certified"])
        self.assertEqual(report["n_accepted"], 40)
        self.assertAlmostEqual(report["ucb"], hoeffding_upper_bound(0.0, 40, 0.1))

    def test_clopper_pearson_upper_bound_matches_exact_reference_values(self):
        self.assertAlmostEqual(
            clopper_pearson_upper_bound(0, 22, 0.0125),
            0.18060009421415313,
            places=12,
        )
        self.assertAlmostEqual(
            clopper_pearson_upper_bound(5, 60, 0.0125),
            0.19930937545542193,
            places=12,
        )
        self.assertAlmostEqual(
            clopper_pearson_upper_bound(99, 100, 0.001),
            0.999989995046714,
            places=12,
        )

    def test_fixed_threshold_supports_predeclared_exact_bound(self):
        report = validate_fixed_threshold(
            [0.1] * 22 + [0.9] * 8,
            [0] * 22 + [1] * 8,
            tau=0.1,
            alpha=0.2,
            delta=0.0125,
            bound="clopper_pearson",
        )
        self.assertTrue(report["certified"])
        self.assertEqual(report["bound"], "clopper_pearson")
        self.assertLessEqual(report["ucb"], 0.2)

    def test_split_ltt_uses_independent_certification_split(self):
        fit_risks = [0.1] * 80 + [0.9] * 80
        fit_wrong = [0] * 80 + [1] * 80
        cert_risks = [0.1] * 40 + [0.9] * 40
        cert_wrong = [0] * 40 + [1] * 40
        report = split_ltt_threshold(
            fit_risks,
            fit_wrong,
            cert_risks,
            cert_wrong,
            alpha=0.2,
            delta=0.1,
        )
        self.assertTrue(report["certified"])
        self.assertEqual(report["method"], "split_learn_then_test_hoeffding")
        self.assertEqual(report["n_fit"], 160)
        self.assertEqual(report["n_certification"], 80)
        self.assertEqual(report["certification"]["n_accepted"], 40)

    def test_split_ltt_refuses_when_certification_disagrees_with_fit(self):
        report = split_ltt_threshold(
            [0.1] * 80 + [0.9] * 80,
            [0] * 80 + [1] * 80,
            [0.1] * 40 + [0.9] * 40,
            [1] * 40 + [0] * 40,
            alpha=0.2,
            delta=0.1,
        )
        self.assertFalse(report["certified"])
        self.assertIsNone(report["tau"])
        self.assertEqual(report["reason"], "hoeffding_ucb_exceeds_alpha")

    def test_split_ltt_reports_exact_method_when_predeclared(self):
        report = split_ltt_threshold(
            [0.1] * 30 + [0.9] * 30,
            [0] * 30 + [1] * 30,
            [0.1] * 22 + [0.9] * 8,
            [0] * 22 + [1] * 8,
            alpha=0.2,
            delta=0.0125,
            bound="clopper_pearson",
        )
        self.assertTrue(report["certified"])
        self.assertEqual(report["method"], "split_learn_then_test_clopper_pearson")

    def test_length_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            validate_fixed_threshold([0.1], [], tau=0.1, alpha=0.2)


if __name__ == "__main__":
    unittest.main()
