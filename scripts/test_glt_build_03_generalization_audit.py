import unittest

import run_glt_build_03_generalization_audit as experiment


class GeneralizationAuditTests(unittest.TestCase):
    def test_templates_have_same_semantic_operations(self):
        state = experiment.State("mira", 0, "nora", template="suffix_only")
        self.assertNotEqual(
            experiment.render_state(state),
            experiment.render_state(experiment.apply_op(state, "R")),
        )
        self.assertEqual(
            experiment.apply_ops(state, ("R", "R")),
            state,
        )

    def test_eval_uses_unseen_template_combination(self):
        train, test = experiment.scene_split()
        train_rows = list(experiment.training_examples("single_only", train))
        eval_rows = list(experiment.eval_examples(test))
        self.assertTrue(train_rows)
        self.assertTrue(eval_rows)
        self.assertTrue(all(row.source_state.template == "suffix_only" for row in eval_rows))
        self.assertIn("the", experiment.fixed_vocab()[0])
        self.assertIn("today", experiment.fixed_vocab()[0])

    def test_pair_exposed_has_only_declared_pair_prefixes(self):
        train, _ = experiment.scene_split()
        rows = list(experiment.training_examples("pair_exposed", train))
        pair_names = {row.op_name for row in rows if len(row.op_seq) == 2}
        self.assertEqual(pair_names, {"RM", "TN"})


if __name__ == "__main__":
    unittest.main()
