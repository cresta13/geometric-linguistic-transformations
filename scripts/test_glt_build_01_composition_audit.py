"""Protocol and inference regression checks for the corrected GLT-BUILD audit."""

import random
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_glt_build_01_composition_audit as audit
import run_glt_build_01_explicit_operator_smoke as base


class CompositionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.train, cls.groups, cls.scenes = audit.build_dataset()
        cls.token_to_id, cls.vocab = base.build_vocab(cls.train, cls.groups)

    def test_all_heldout_scenes_and_operations_are_covered(self):
        self.assertEqual(len(self.train), 3600)
        self.assertEqual(len(self.scenes), 88)
        self.assertEqual({state.subject for _, state in self.scenes}, set(base.ENTITIES))
        self.assertEqual(len(self.groups["heldout_scene_single"]), 88 * 5)
        for group in ("heldout_scene_seen_pair", "heldout_scene_reverse_seen_pair"):
            self.assertEqual(len(self.groups[group]), 88 * 4)
        self.assertEqual(len(self.groups["heldout_unseen_pair"]), 88 * 3)
        self.assertEqual({ex.source_state.subject for ex in self.groups["train_seen"]}, set(base.ENTITIES))
        for subject in base.ENTITIES:
            seen = {ex.op_seq for ex in self.groups["train_seen"] if ex.source_state.subject == subject}
            self.assertEqual(len(seen), 10)

    def test_training_and_vocabulary_match_pilot(self):
        old_train, old_groups = base.build_examples(96)
        self.assertEqual(old_train, self.train)
        _, old_vocab = base.build_vocab(old_train, old_groups)
        self.assertEqual(old_vocab, self.vocab)
        keys = {(ex.op_seq, ex.source_tokens) for ex in self.train}
        for family, group in self.groups.items():
            if family != "train_seen":
                self.assertFalse(keys & {(ex.op_seq, ex.source_tokens) for ex in group})

    def test_batched_generation_matches_scalar_with_and_without_intervention(self):
        base.set_seed(731)
        model = base.TinyDecoderOnlyTransformer(len(self.vocab), 16, 2, 2, 64, 0)
        examples = self.groups["heldout_scene_single"][:5] + self.groups["heldout_scene_seen_pair"][:3]
        prompts = [base.encode_prompt(ex, self.token_to_id) for ex in examples]
        for injection in (None, {2: torch.linspace(-1, 1, 16)}):
            batched = audit.generate_batch(model, prompts, self.vocab, self.token_to_id, 8, 3, injection)
            scalar = [base.generate(model, p, self.vocab, self.token_to_id["<eos>"],
                                    self.token_to_id["<pad>"], 8, torch.device("cpu"), injection) for p in prompts]
            self.assertEqual(batched, scalar)

    def test_intermediate_is_generated_output_not_oracle(self):
        requests = []

        def always_wrong(batch):
            requests.extend(batch)
            return [("mira", ".")] * len(batch)

        frame = audit.composition_audit(always_wrong, self.scenes[:1])
        self.assertIn((("N",), ("mira", ".")), requests)
        self.assertTrue((frame.direct_order_agreement == 1).all())
        self.assertTrue((frame.direct_both_correct == 0).all())
        self.assertTrue((frame.sequential_both_correct == 0).all())

    def test_correct_symbolic_execution_passes_composition(self):
        sid, state = self.scenes[0]
        lookup = {}
        for seq in [()] + [(op,) for op in base.OPS]:
            transformed = base.apply_ops(state, seq)
            lookup[base.render_state(transformed)] = transformed

        def symbolic(batch):
            return [base.render_state(base.apply_ops(lookup[source], ops)) for ops, source in batch]

        frame = audit.composition_audit(symbolic, [(sid, state)])
        self.assertTrue((frame.filter(regex="_correct$") == 1).all().all())

    def test_hidden_readouts_ignore_right_padding(self):
        base.set_seed(23)
        model = base.TinyDecoderOnlyTransformer(len(self.vocab), 16, 2, 2, 64, 0)
        examples = self.groups["heldout_scene_single"][:5]
        sentences = [ex.target_tokens for ex in examples]
        one = audit.extract_hidden(model, sentences, self.token_to_id, 1)
        many = audit.extract_hidden(model, sentences, self.token_to_id, 5)
        self.assertEqual(len(many), 6)
        for key in one:
            np.testing.assert_allclose(one[key], many[key], atol=2e-6, rtol=2e-6)

    def test_checkpoint_restores_weights_rng_and_optimizer(self):
        base.set_seed(41)
        model = base.TinyDecoderOnlyTransformer(len(self.vocab), 16, 2, 2, 64, 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        rng = random.Random(41)
        encoded = [base.encode_example(ex, self.token_to_id) for ex in self.train[:20]]

        def train_once(net, opt, batch_rng):
            x, y = base.sample_batch(encoded, 4, self.token_to_id["<pad>"], batch_rng)
            opt.zero_grad(set_to_none=True)
            logits, _ = net(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(self.vocab)), y.reshape(-1), ignore_index=-100)
            loss.backward()
            opt.step()

        train_once(model, optimizer, rng)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            audit.save_checkpoint(path, model, optimizer, rng, {}, 41, 1)
            saved = torch.load(path, weights_only=True)
            restored = base.TinyDecoderOnlyTransformer(len(self.vocab), 16, 2, 2, 64, 0)
            restored.load_state_dict(saved["model_state"])
            opt2 = torch.optim.AdamW(restored.parameters(), lr=0.003)
            opt2.load_state_dict(saved["optimizer_state"])
            rng2 = random.Random()
            rng2.setstate(saved["batch_rng_state"])
            train_once(model, optimizer, rng)
            torch.set_rng_state(saved["torch_rng_state"])
            train_once(restored, opt2, rng2)
            for left, right in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_bootstrap_counts_scenes_not_repeated_rows(self):
        frame = pd.DataFrame([dict(family="test", scene_id=scene, seed=seed, exact_match=scene % 2)
                              for scene in range(4) for seed in range(3) for _ in range(5)])
        result = audit.bootstrap_rates(frame, ["family"], draws=100)
        self.assertEqual(result.iloc[0].n_sources, 4)
        self.assertEqual(result.iloc[0].n_rows, 60)
        self.assertEqual(result.iloc[0].rate, 0.5)

    def test_json_write_retries_temporary_windows_lock(self):
        replace = Path.replace
        attempts = []

        def temporarily_locked(source, target):
            attempts.append(target)
            if len(attempts) < 3:
                raise PermissionError("sharing violation")
            return replace(source, target)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            path.write_text('{"status": "old"}', encoding="utf-8")
            with patch.object(Path, "replace", temporarily_locked), patch.object(audit.time, "sleep"):
                audit.write_json(path, {"status": "new"})
            self.assertEqual(json.loads(path.read_text())["status"], "new")
            self.assertEqual(len(attempts), 3)

    def test_json_write_reports_persistent_lock_without_destroying_previous_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            path.write_text('{"status": "old"}', encoding="utf-8")
            with patch.object(Path, "replace", side_effect=PermissionError("persistent lock")), patch.object(audit.time, "sleep"):
                with self.assertRaises(PermissionError):
                    audit.write_json(path, {"status": "new"})
            self.assertEqual(json.loads(path.read_text())["status"], "old")


if __name__ == "__main__":
    unittest.main()
