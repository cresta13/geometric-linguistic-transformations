"""Checks for the paired input-support design and its split invariants."""

import random
import unittest

import numpy as np
import pandas as pd
import torch

import run_glt_build_01_input_support as experiment
import run_glt_build_01_explicit_operator_smoke as base


class InputSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.train, cls.test = experiment.scene_split()
        cls.tokens, cls.vocab = experiment.fixed_vocab()

    def test_split_is_closed_under_every_operation(self):
        train_groups = {experiment.group_id(s) for _, s in self.train}
        test_groups = {experiment.group_id(s) for _, s in self.test}
        self.assertEqual((len(self.train), len(self.test)), (360, 88))
        self.assertEqual((len(train_groups), len(test_groups)), (180, 44))
        self.assertFalse(train_groups & test_groups)
        for _, state in self.train + self.test:
            for variant in range(16):
                source = experiment.source_variant(state, variant)
                for op in base.OPS:
                    self.assertEqual(experiment.group_id(base.apply_op(source, op)), experiment.group_id(state))
        self.assertEqual({s.subject for _, s in self.test}, set(base.ENTITIES))

    def test_variants_round_trip_and_render_distinct_states(self):
        state = self.train[0][1]
        variants = [experiment.source_variant(state, i) for i in range(16)]
        self.assertEqual([experiment.variant_id(s) for s in variants], list(range(16)))
        self.assertEqual(len({base.render_state(s) for s in variants}), 16)
        for source in variants:
            for op in base.OPS:
                self.assertEqual(base.apply_op(base.apply_op(source, op), op), source)

    def test_training_pools_keep_unseen_pairs_absent(self):
        expected = {"base_only": 3600, "expanded_sources": 36000}
        for arm in experiment.ARMS:
            count = 0
            for ex in experiment.training_examples(arm, self.train):
                count += 1
                self.assertIn(ex.op_seq, experiment.SEQUENCES)
                self.assertNotIn(ex.op_seq, base.UNSEEN_PAIR_OPS)
                if len(ex.op_seq) == 2 or arm == "base_only":
                    self.assertEqual(experiment.variant_id(ex.source_state), 0)
                self.assertTrue(set(ex.source_tokens + ex.target_tokens).issubset(self.tokens))
            self.assertEqual(count, expected[arm])

    def test_training_and_heldout_surface_forms_do_not_overlap(self):
        train_forms = {base.render_state(experiment.source_variant(state, v)) for _, state in self.train for v in range(16)}
        test_forms = {base.render_state(experiment.source_variant(state, v)) for _, state in self.test for v in range(16)}
        self.assertFalse(train_forms & test_forms)
        for _, state in self.train:
            for op in base.OPS:
                self.assertIn(base.render_state(base.apply_op(state, op)), train_forms)

    def test_schedule_and_training_shapes_are_paired(self):
        x0, y0, plan0 = experiment.sample_batch("base_only", self.train, random.Random(81), self.tokens, 128)
        x1, y1, plan1 = experiment.sample_batch("expanded_sources", self.train, random.Random(81), self.tokens, 128)
        np.testing.assert_array_equal(plan0, plan1)
        self.assertEqual(tuple(x0.shape), (128, 32))
        self.assertEqual(tuple(x0.shape), tuple(x1.shape))
        pairs = plan0[:, 1] >= 6
        self.assertTrue(torch.equal(x0[pairs], x1[pairs]))
        self.assertTrue(torch.equal(y0[pairs], y1[pairs]))
        varied = (plan0[:, 1] < 6) & (plan0[:, 2] != 0)
        self.assertTrue(bool((x0[varied] != x1[varied]).any()))
        self.assertTrue(bool((y0 == -100).any()))

    def test_evaluation_counts_and_unique_keys(self):
        groups = experiment.evaluation_groups(self.test)
        self.assertEqual({name: len(rows) for name, rows in groups.items()},
                         dict(base_single=440, nonbase_single=6600, nonbase_identity=1320,
                              seen_pair=352, reverse_seen_pair=352, unseen_pair=264))
        keys = [(family, ex.scene_id, experiment.variant_id(ex.source_state), ex.op_seq)
                for family, rows in groups.items() for ex in rows]
        self.assertEqual(len(keys), len(set(keys)))

    def test_paired_effect_resamples_semantic_groups(self):
        rows = []
        for group in range(4):
            for seed in range(3):
                for arm in experiment.ARMS:
                    for _ in range(10):
                        rows.append(dict(family="test", group_id=group, seed=seed, arm=arm,
                                         exact_match=int(arm == "expanded_sources")))
        result = experiment.paired_effect(pd.DataFrame(rows), ["family"]).iloc[0]
        self.assertEqual(result.difference, 1)
        self.assertEqual(result.n_scene_groups, 4)
        self.assertEqual(result.n_group_seed_pairs, 12)
        with self.assertRaises(ValueError):
            experiment.paired_effect(pd.DataFrame(rows[:-10]), ["family"])


if __name__ == "__main__":
    unittest.main()
