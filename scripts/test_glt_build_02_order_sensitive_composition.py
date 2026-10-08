import unittest

import run_glt_build_02_order_sensitive_composition as experiment


class OrderSensitiveCompositionTests(unittest.TestCase):
    def test_role_swap_and_subject_mark_are_noncommuting(self):
        state = experiment.State("mira", 0, "nora")
        self.assertNotEqual(experiment.apply_ops(state, ("M", "R")), experiment.apply_ops(state, ("R", "M")))

    def test_tense_and_negation_are_commuting_control(self):
        state = experiment.State("mira", 0, "nora")
        self.assertEqual(experiment.apply_ops(state, ("T", "N")), experiment.apply_ops(state, ("N", "T")))

    def test_role_swap_and_subject_mark_are_involutions(self):
        state = experiment.State("mira", 0, "nora")
        self.assertEqual(experiment.apply_ops(state, ("R", "R")), state)
        self.assertEqual(experiment.apply_ops(state, ("M", "M")), state)

    def test_training_arms_have_expected_pair_support(self):
        train, _ = experiment.scene_split()
        single = list(experiment.training_examples("single_only", train))
        exposed = list(experiment.training_examples("pair_exposed", train))
        self.assertTrue(all(len(row.op_seq) < 2 for row in single))
        self.assertTrue(any(row.op_seq == ("R", "M") for row in exposed))
        self.assertTrue(any(row.op_seq == ("T", "N") for row in exposed))

    def test_fixed_length_batch_and_vocab(self):
        train, _ = experiment.scene_split()
        token_to_id, vocab = experiment.fixed_vocab()
        self.assertIn("<R>", token_to_id)
        self.assertIn("marked", token_to_id)
        x, y = experiment.sample_batch("single_only", train, __import__("random").Random(3), token_to_id, 8)
        self.assertEqual(tuple(x.shape), (8, experiment.CONTEXT_LENGTH))
        self.assertEqual(tuple(y.shape), (8, experiment.CONTEXT_LENGTH))
        self.assertTrue(bool((y == -100).any()))

    def test_training_labels_start_with_first_target_token(self):
        token_to_id, _ = experiment.fixed_vocab()
        state = experiment.State("mira", 0, "nora")
        row = experiment.example(0, state, (), "train")
        x, y = experiment.encode_example(row, token_to_id)
        first_target_index = len(["<bos>", "<sep>"] + list(row.source_tokens))
        self.assertEqual(y[first_target_index], token_to_id[row.target_tokens[0]])
        self.assertEqual(y[first_target_index - 1], -100)
        self.assertEqual(len(x), len(y))
        self.assertEqual(y[-1], token_to_id["<eos>"])


if __name__ == "__main__":
    unittest.main()
