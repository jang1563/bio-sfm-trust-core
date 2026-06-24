"""RCPS / conformal trust-threshold: the false-accept guarantee, the refuse-when-weak behavior,
and monotonicity in alpha. Pure stdlib (seeded random for reproducibility)."""

import random
import unittest

from bio_sfm_trust import false_accept_rate, rcps_threshold


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


if __name__ == "__main__":
    unittest.main()
