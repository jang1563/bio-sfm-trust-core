import unittest

from bio_sfm_trust.scoring import action_outcome, net_reward, summarize_actions


class ActionOutcomeTests(unittest.TestCase):
    def test_verify_always_correct_costs_assay(self):
        self.assertEqual(action_outcome("verify_assay", sfm_correct=False, baseline_correct=False), (1, 1))

    def test_trust_follows_sfm(self):
        self.assertEqual(action_outcome("trust_sfm", sfm_correct=True, baseline_correct=False), (1, 0))
        self.assertEqual(action_outcome("trust_sfm", sfm_correct=False, baseline_correct=True), (0, 0))

    def test_baseline_follows_baseline(self):
        self.assertEqual(action_outcome("default_baseline", sfm_correct=False, baseline_correct=True), (1, 0))

    def test_defer_zero(self):
        self.assertEqual(action_outcome("defer", sfm_correct=True, baseline_correct=True), (0, 0))


class NetRewardTests(unittest.TestCase):
    def test_net_reward_formula(self):
        # one verify (1,1) + one correct trust (1,0); lam=0.5 -> (2 - 0.5*1)/2 = 0.75
        self.assertAlmostEqual(net_reward([(1, 1), (1, 0)], lam=0.5), 0.75, places=6)

    def test_empty(self):
        self.assertEqual(net_reward([], lam=0.5), 0.0)


class SummarizeTests(unittest.TestCase):
    def test_rates_sum_to_one(self):
        rows = [
            {"action": "trust_sfm", "sfm_correct": True, "baseline_correct": False},
            {"action": "verify_assay", "sfm_correct": False, "baseline_correct": False},
            {"action": "default_baseline", "sfm_correct": False, "baseline_correct": True},
            {"action": "defer", "sfm_correct": False, "baseline_correct": False},
        ]
        s = summarize_actions(rows, lam=0.5)
        self.assertEqual(s["n"], 4)
        total = s["trust_rate"] + s["verify_rate"] + s["default_rate"] + s["defer_rate"]
        self.assertAlmostEqual(total, 1.0, places=6)
        # 3 correct (trust ok, verify, baseline ok), 1 assay -> (3 - 0.5)/4 = 0.625
        self.assertAlmostEqual(s["net_reward_per_item"], 0.625, places=6)

    def test_trust_error_counted(self):
        rows = [{"action": "trust_sfm", "sfm_correct": False, "baseline_correct": False}]
        s = summarize_actions(rows, lam=0.5)
        self.assertEqual(s["trust_error_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
