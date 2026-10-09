import unittest

import audit_glt_build_03_template_controls as audit


class TemplateControlTests(unittest.TestCase):
    def test_seen_and_held_out_templates_are_declared(self):
        self.assertEqual(
            audit.TEMPLATES,
            ("canonical", "article_suffix", "suffix_only"),
        )

    def test_control_pairs_cover_noncommuting_and_commuting_cases(self):
        self.assertIn(("R", "M"), audit.CONTROL_PAIRS)
        self.assertIn(("T", "N"), audit.CONTROL_PAIRS)


if __name__ == "__main__":
    unittest.main()
