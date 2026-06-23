import unittest

from bio_sfm_trust.gate import (
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

    def test_bounds(self):
        self.assertEqual(confidence_to_risk({"regime": "monomer", "mean_plddt": 120}), 0.0)
        self.assertEqual(confidence_to_risk({"regime": "monomer", "mean_plddt": -10}), 1.0)

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

    def test_policy_net_trust_all_when_risk_low(self):
        # all risk below lambda -> never verify -> equals trust-all accuracy, no assay cost
        recs = [{"truth": {"correct": True}}, {"truth": {"correct": False}}]
        net = risk_threshold_policy_net(recs, [0.1, 0.1], lam=0.5)
        self.assertAlmostEqual(net, 0.5, places=6)  # 1 correct / 2, 0 assays


if __name__ == "__main__":
    unittest.main()
