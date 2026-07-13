import unittest

from bio_sfm_trust.gate import (
    _policy_net_loo,
    calibrated_gate,
    confidence_to_risk,
    phase2_calibration_gate,
    risk_threshold_policy_net,
)


def _synthetic_records():
    """40 monomers (pLDDT tightly tracks quality) + 40 complexes (pLDDT decoupled
    from quality, i.e. overconfident) -> a designed monomer>complex calibration gap."""
    records = []
    for i in range(40):
        plddt = 60 + i              # 60..99
        quality = round(plddt / 100.0, 4)   # perfectly correlated with pLDDT
        records.append({
            "target_id": f"mono_{i}",
            "regime": "monomer",
            "mean_plddt": plddt,
            "iptm": None,
            "template_baseline_correct": quality >= 0.95,
            "truth": {"correct": quality >= 0.9, "quality": quality},
        })
    for i in range(40):
        plddt = 60 + i              # same pLDDT spread...
        quality = round(min(1.0, 0.70 + ((i * 17) % 40) / 100.0), 4)  # ...but decoupled
        iptm = round(0.4 + ((i * 7) % 40) / 100.0, 4)
        records.append({
            "target_id": f"cplx_{i}",
            "regime": "complex",
            "mean_plddt": plddt,
            "iptm": iptm,
            "template_baseline_correct": quality >= 0.95,
            "truth": {"correct": quality >= 0.9, "quality": quality},
        })
    return records


VALID_DECISIONS = {
    "eligible_for_phase2_interface_pilot",
    "eligible_pending_more_targets",
    "redesign_policy_before_pilot",
    "do_not_run_signal_not_calibrated",
}


class ConfidenceToRiskTests(unittest.TestCase):
    def test_monomer_risk(self):
        self.assertAlmostEqual(confidence_to_risk({"regime": "monomer", "mean_plddt": 90}), 0.1, places=6)

    def test_valid_plddt_bounds(self):
        self.assertEqual(confidence_to_risk({"regime": "monomer", "mean_plddt": 100}), 0.0)
        self.assertEqual(confidence_to_risk({"regime": "monomer", "mean_plddt": 0}), 1.0)

    def test_rejects_missing_or_unknown_regime(self):
        for record in (
            {"mean_plddt": 90},
            {"regime": "", "mean_plddt": 90},
            {"regime": "other", "mean_plddt": 90},
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                confidence_to_risk(record)

    def test_rejects_missing_or_out_of_range_plddt(self):
        for plddt in (None, -0.1, 100.1):
            with self.subTest(plddt=plddt), self.assertRaises(ValueError):
                confidence_to_risk({"regime": "monomer", "mean_plddt": plddt})

    def test_rejects_nonfinite_signals(self):
        records = (
            {"regime": "monomer", "mean_plddt": float("nan")},
            {"regime": "monomer", "mean_plddt": float("inf")},
            {"regime": "complex", "mean_plddt": 80, "iptm": float("nan")},
            {"regime": "complex", "mean_plddt": 80, "pae_interaction": float("inf")},
        )
        for record in records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                confidence_to_risk(record)

    def test_rejects_invalid_iptm(self):
        for iptm in (-0.01, 1.01):
            with self.subTest(iptm=iptm), self.assertRaises(ValueError):
                confidence_to_risk({"regime": "complex", "mean_plddt": 80, "iptm": iptm})

    def test_pae_interaction_is_nonnegative_and_clamped(self):
        with self.assertRaises(ValueError):
            confidence_to_risk({"regime": "complex", "mean_plddt": 80, "pae_interaction": -0.1})
        self.assertEqual(
            confidence_to_risk({"regime": "complex", "mean_plddt": 80, "pae_interaction": 60}),
            1.0,
        )
        self.assertAlmostEqual(
            confidence_to_risk({"regime": "complex", "mean_plddt": 80, "pae_interaction": 15}),
            0.5,
        )

    def test_complex_blends_iptm(self):
        r = confidence_to_risk({"regime": "complex", "mean_plddt": 80, "iptm": 0.6})
        self.assertAlmostEqual(r, 1.0 - 0.5 * 0.8 - 0.5 * 0.6, places=6)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.records = _synthetic_records()

    def test_phase2_gate_runs_and_reports_calibration_gap(self):
        gate = phase2_calibration_gate(self.records, lam=0.5)
        self.assertEqual(gate["scope"]["n_targets"], 80)
        self.assertIn(gate["decision"], VALID_DECISIONS)
        self.assertIsNotNone(gate["signal_validity"]["wrong_risk_auroc"])
        gap = gate["regime_calibration"]["monomer_minus_complex"]
        self.assertIsNotNone(gap)
        self.assertGreater(gap, 0.0)  # monomers calibrate better than complexes

    def test_calibrated_gate_runs(self):
        gate = calibrated_gate(self.records, lam=0.5, correct_lddt=0.9)
        self.assertIn(gate["decision"], VALID_DECISIONS)
        self.assertEqual(gate["calibration"], "leave_one_out_isotonic_raw_risk_to_p_wrong")

    def test_phase2_gate_requires_boolean_template_baseline_outcomes(self):
        missing = dict(self.records[0])
        missing.pop("template_baseline_correct")
        with self.assertRaisesRegex(ValueError, "template_baseline_correct"):
            phase2_calibration_gate([missing])

        for invalid in (None, 0, 1, "false"):
            with self.subTest(invalid=invalid):
                record = dict(self.records[0], template_baseline_correct=invalid)
                with self.assertRaisesRegex(ValueError, "template_baseline_correct"):
                    phase2_calibration_gate([record])

    def test_calibrated_gate_rejects_missing_or_invalid_truth_quality(self):
        missing = dict(self.records[0], truth={"correct": True})
        with self.assertRaisesRegex(ValueError, "truth.quality"):
            calibrated_gate([missing])

        for invalid in (float("nan"), float("inf"), float("-inf"), -0.01, 1.01):
            with self.subTest(invalid=invalid):
                record = dict(
                    self.records[0],
                    truth={"correct": True, "quality": invalid},
                )
                with self.assertRaisesRegex(ValueError, "truth.quality"):
                    calibrated_gate([record])

    def test_calibrated_gate_rejects_invalid_correct_lddt_cutoff(self):
        for invalid in (float("nan"), float("inf"), float("-inf"), -0.01, 1.01):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "correct_lddt"):
                    calibrated_gate(self.records, correct_lddt=invalid)

    def test_policy_net_trust_all_when_risk_low(self):
        # all risk below lambda -> never verify -> equals trust-all accuracy, no assay cost
        recs = [{"truth": {"correct": True}}, {"truth": {"correct": False}}]
        net = risk_threshold_policy_net(recs, [0.1, 0.1], lam=0.5)
        self.assertAlmostEqual(net, 0.5, places=6)  # 1 correct / 2, 0 assays

    def test_policy_helpers_reject_length_mismatch(self):
        with self.assertRaises(ValueError):
            risk_threshold_policy_net([{"truth": {"correct": True}}], [], lam=0.5)
        with self.assertRaises(ValueError):
            _policy_net_loo([0], [], lam=0.5)

    def test_policy_helpers_reject_nonfinite_inputs(self):
        records = [{"truth": {"correct": True}}]
        with self.assertRaises(ValueError):
            risk_threshold_policy_net(records, [float("nan")], lam=0.5)
        with self.assertRaises(ValueError):
            risk_threshold_policy_net(records, [0.1], lam=float("inf"))
        with self.assertRaises(ValueError):
            _policy_net_loo([0], [float("inf")], lam=0.5)

    def test_policy_helpers_reject_invalid_risks_or_costs(self):
        records = [{"truth": {"correct": True}}]
        for risk in (-0.1, 1.1):
            with self.subTest(risk=risk), self.assertRaises(ValueError):
                risk_threshold_policy_net(records, [risk], lam=0.5)
        with self.assertRaises(ValueError):
            _policy_net_loo([0], [0.1], lam=-0.1)

    def test_policy_helpers_reject_nonbinary_truth(self):
        with self.assertRaises(ValueError):
            risk_threshold_policy_net([{"truth": {"correct": 2}}], [0.1], lam=0.5)
        with self.assertRaises(ValueError):
            _policy_net_loo([2], [0.1], lam=0.5)


if __name__ == "__main__":
    unittest.main()
