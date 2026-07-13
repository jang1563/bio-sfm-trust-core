import unittest

from bio_sfm_trust.actions import ACTIONS, normalize_action, parse_action_record


class ActionTests(unittest.TestCase):
    def test_canonical_actions(self):
        self.assertEqual(ACTIONS, ("trust_sfm", "verify_assay", "default_baseline", "defer"))

    def test_aliases_normalize(self):
        self.assertEqual(normalize_action("trust"), "trust_sfm")
        self.assertEqual(normalize_action("VERIFY"), "verify_assay")
        self.assertEqual(normalize_action("baseline"), "default_baseline")
        self.assertEqual(normalize_action("abstain"), "defer")

    def test_unknown_action_raises(self):
        with self.assertRaises(ValueError):
            normalize_action("nonsense")

    def test_parse_from_json(self):
        rec = parse_action_record('{"action": "trust", "confidence": "0.8"}')
        self.assertEqual(rec["action"], "trust_sfm")
        self.assertEqual(rec["confidence"], 0.8)

    def test_parse_from_freeform_text(self):
        rec = parse_action_record("I would verify_assay this one.")
        self.assertEqual(rec["action"], "verify_assay")

    def test_freeform_action_uses_token_boundaries(self):
        with self.assertRaises(ValueError):
            parse_action_record("I distrust this output.")

    def test_freeform_rejects_ambiguous_actions(self):
        with self.assertRaises(ValueError):
            parse_action_record("Either trust_sfm or verify_assay.")

    def test_freeform_allows_repeated_aliases_for_same_action(self):
        rec = parse_action_record("I would trust, meaning trust_sfm.")
        self.assertEqual(rec["action"], "trust_sfm")

    def test_json_action_record_must_be_an_object(self):
        with self.assertRaises(ValueError):
            parse_action_record('["trust_sfm"]')

    def test_parse_from_dict(self):
        rec = parse_action_record({"action": "defer"})
        self.assertEqual(rec["action"], "defer")


if __name__ == "__main__":
    unittest.main()
