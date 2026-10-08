import unittest

import torch

import run_glt_build_01_pair_prefix_holdout as experiment
import run_glt_build_01_explicit_operator_smoke as base


class PairPrefixHoldoutTests(unittest.TestCase):
    def test_single_only_pool_contains_no_pair_prefix(self):
        train, _ = experiment.support.scene_split()
        rows = list(experiment.training_examples("single_only", train))
        self.assertEqual(len(rows), 34560)
        self.assertTrue(all(len(row.op_seq) < 2 for row in rows))

    def test_expanded_pool_contains_pair_prefixes(self):
        train, _ = experiment.support.scene_split()
        rows = list(experiment.training_examples("expanded_pairs", train))
        self.assertEqual(len(rows), 36000)
        self.assertTrue(any(len(row.op_seq) == 2 for row in rows))

    def test_both_arms_have_the_same_split(self):
        train, test = experiment.support.scene_split()
        train_groups = {experiment.support.group_id(s) for _, s in train}
        test_groups = {experiment.support.group_id(s) for _, s in test}
        self.assertEqual((len(train), len(test)), (360, 88))
        self.assertEqual((len(train_groups), len(test_groups)), (180, 44))
        self.assertFalse(train_groups & test_groups)

    def test_fixed_vocab_covers_all_evaluation_forms(self):
        train, test = experiment.support.scene_split()
        groups = experiment.support.evaluation_groups(test)
        token_to_id, vocab = experiment.support.fixed_vocab()
        self.assertEqual(len(vocab), 50)
        for rows in groups.values():
            for row in rows:
                for token in row.source_tokens + row.target_tokens:
                    self.assertIn(token, token_to_id)

    def test_batches_are_padded_to_a_matched_shape(self):
        train, _ = experiment.support.scene_split()
        token_to_id, _ = experiment.support.fixed_vocab()
        x, y, plans = experiment.sample_batch("single_only", train, __import__('random').Random(4), token_to_id, 16)
        self.assertEqual(tuple(x.shape), (16, experiment.support.CONTEXT_LENGTH))
        self.assertEqual(tuple(y.shape), (16, experiment.support.CONTEXT_LENGTH))
        self.assertEqual(len(plans), 16)
        self.assertTrue(bool((y == -100).any()))


if __name__ == "__main__":
    unittest.main()
