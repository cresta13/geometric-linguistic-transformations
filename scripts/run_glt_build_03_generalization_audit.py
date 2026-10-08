"""Stress-test GLT-BUILD-02 composition outside the training surface template."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import random
from pathlib import Path
import sys

import torch

import run_glt_build_02_order_sensitive_composition as exp
import run_glt_build_01_explicit_operator_smoke as base
import run_glt_build_01_input_support as support


PROTOCOL = "glt-build-03-template-and-lexical-generalization-v1"
ARMS = exp.ARMS
OPS = exp.OPS
PAIRS = exp.PAIRS
PAIR_TRAIN = exp.PAIR_TRAIN
SOURCE_VARIANTS = exp.SOURCE_VARIANTS
CONTEXT_LENGTH = 32
ORIGINAL_SUMMARY = exp.summary


@dataclass(frozen=True)
class State:
    subject: str
    verb_idx: int
    object: str
    tense: str = "present"
    negated: bool = False
    marked: frozenset[str] = frozenset()
    template: str = "canonical"


@dataclass(frozen=True)
class Example:
    split: str
    family: str
    op_seq: tuple[str, ...]
    scene_id: int
    source_state: State
    target_state: State
    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]

    @property
    def op_name(self) -> str:
        return "I" if not self.op_seq else "".join(self.op_seq)


def apply_op(state: State, op: str) -> State:
    if op == "R":
        return replace(state, subject=state.object, object=state.subject)
    if op == "M":
        marked = set(state.marked)
        if state.subject in marked:
            marked.remove(state.subject)
        else:
            marked.add(state.subject)
        return replace(state, marked=frozenset(marked))
    if op == "T":
        return replace(state, tense="past" if state.tense == "present" else "present")
    if op == "N":
        return replace(state, negated=not state.negated)
    raise ValueError(f"Unknown operation: {op}")


def apply_ops(state: State, ops: tuple[str, ...]) -> State:
    for op in ops:
        state = apply_op(state, op)
    return state


def group_id(state: State) -> int:
    return support.group_id(base.State(state.subject, state.verb_idx, state.object))


def scene_split():
    train, test = support.scene_split()
    convert = lambda rows: [(sid, State(s.subject, s.verb_idx, s.object)) for sid, s in rows]
    return convert(train), convert(test)


def source_variant(state: State, variant: int, template: str | None = None) -> State:
    tense, negated, mark = SOURCE_VARIANTS[variant]
    marked = frozenset(() if mark == "none" else (state.subject if mark == "subject" else state.object,))
    return replace(
        state,
        tense=tense,
        negated=negated,
        marked=marked,
        template=template or state.template,
    )


def variant_id(state: State) -> int:
    mark = "none"
    if state.subject in state.marked:
        mark = "subject"
    elif state.object in state.marked:
        mark = "object"
    return SOURCE_VARIANTS.index((state.tense, state.negated, mark))


def render_state(state: State) -> tuple[str, ...]:
    verb = base.VERBS[state.verb_idx]
    subject = ["marked", state.subject] if state.subject in state.marked else [state.subject]
    obj = ["marked", state.object] if state.object in state.marked else [state.object]
    if state.negated:
        verb_tokens = ["does" if state.tense == "present" else "did", "not", verb.base]
    else:
        verb_tokens = [verb.present if state.tense == "present" else verb.past]
    if state.template == "canonical":
        return tuple(subject + verb_tokens + obj + ["."])
    if state.template == "article_suffix":
        return tuple(["the"] + subject + verb_tokens + obj + ["today", "."])
    if state.template == "suffix_only":
        return tuple(subject + verb_tokens + obj + ["today", "."])
    raise ValueError(f"Unknown template: {state.template}")


def example(sid: int, state: State, seq: tuple[str, ...], family: str, split: str) -> Example:
    target = apply_ops(state, seq)
    return Example(split, family, seq, sid, state, target, render_state(state), render_state(target))


def training_examples(arm: str, scenes):
    sequences = PAIR_TRAIN if arm == "pair_exposed" else ()
    all_sequences = [()] + [(op,) for op in OPS] + list(sequences)
    for sid, raw in scenes:
        for seq in all_sequences:
            variants = range(len(SOURCE_VARIANTS)) if len(seq) <= 1 else (0,)
            templates = ("canonical", "article_suffix") if len(seq) <= 1 else ("canonical",)
            for template in templates:
                for variant in variants:
                    state = source_variant(
                        State(raw.subject, raw.verb_idx, raw.object, template=template),
                        variant,
                        template,
                    )
                    yield example(sid, state, seq, "train", "train")


def eval_examples(test_scenes):
    rows = []
    for sid, raw in test_scenes:
        state = State(raw.subject, raw.verb_idx, raw.object, template="suffix_only")
        for variant in range(len(SOURCE_VARIANTS)):
            source = source_variant(state, variant, "suffix_only")
            rows.append(example(sid, source, (), "identity", "eval"))
            for op in OPS:
                rows.append(example(sid, source, (op,), "single", "eval"))
    return rows


def sample_batch(arm, scenes, rng, token_to_id, batch_size):
    inputs = torch.full((batch_size, CONTEXT_LENGTH), token_to_id["<pad>"], dtype=torch.long)
    labels = torch.full((batch_size, CONTEXT_LENGTH), -100, dtype=torch.long)
    sequences = [()] + [(op,) for op in OPS]
    if arm == "pair_exposed":
        sequences += list(PAIR_TRAIN)
    for row in range(batch_size):
        sid, raw = scenes[rng.randrange(len(scenes))]
        seq = sequences[rng.randrange(len(sequences))]
        template = rng.choice(("canonical", "article_suffix")) if len(seq) <= 1 else "canonical"
        variant = rng.randrange(len(SOURCE_VARIANTS)) if len(seq) <= 1 else 0
        state = source_variant(State(raw.subject, raw.verb_idx, raw.object, template=template), variant, template)
        ex = example(sid, state, seq, "train", "train")
        prefix = ["<bos>"] + [f"<{op}>" for op in seq] + ["<sep>"] + list(ex.source_tokens) + ["<sep>"]
        target = list(ex.target_tokens) + ["<eos>"]
        ids = [token_to_id[t] for t in prefix + target]
        x, y = ids[:-1], [-100] * (len(prefix) - 1) + ids[len(prefix):]
        if len(x) > CONTEXT_LENGTH:
            raise ValueError("Training sequence exceeds matched context length")
        inputs[row, :len(x)] = torch.tensor(x)
        labels[row, :len(y)] = torch.tensor(y)
    return inputs, labels


def fixed_vocab():
    ordinary = set(base.ENTITIES) | {"does", "did", "not", "marked", ".", "the", "today"}
    for verb in base.VERBS:
        ordinary.update((verb.base, verb.present, verb.past))
    vocab = ["<pad>", "<bos>", "<sep>", "<eos>"] + [f"<{op}>" for op in OPS] + sorted(ordinary)
    return {token: i for i, token in enumerate(vocab)}, vocab


def prompt(seq, source, token_to_id):
    return [token_to_id[t] for t in ["<bos>"] + [f"<{op}>" for op in seq] + ["<sep>"] + list(source) + ["<sep>"]]


def composition_audit(generator, test_scenes):
    rows = []
    base_states = [State(s.subject, s.verb_idx, s.object, template="suffix_only") for _, s in test_scenes]
    sources = [render_state(state) for state in base_states]
    for a, b in PAIRS:
        first_a = generator([((a,), source) for source in sources])
        first_b = generator([((b,), source) for source in sources])
        direct_ab = generator([((a, b), source) for source in sources])
        direct_ba = generator([((b, a), source) for source in sources])
        seq_ab = generator([((b,), out) for out in first_a])
        seq_ba = generator([((a,), out) for out in first_b])
        oracle_a = [render_state(apply_op(state, a)) for state in base_states]
        oracle_b = [render_state(apply_op(state, b)) for state in base_states]
        oracle_ab = generator([((b,), out) for out in oracle_a])
        oracle_ba = generator([((a,), out) for out in oracle_b])
        for i, (scene_id, _) in enumerate(test_scenes):
            target_ab = render_state(apply_ops(base_states[i], (a, b)))
            target_ba = render_state(apply_ops(base_states[i], (b, a)))
            row = dict(
                scene_id=scene_id,
                group_id=group_id(base_states[i]),
                pair=a + b,
                pair_seen_in_training=int((a, b) in PAIR_TRAIN),
                order_distinct=int(target_ab != target_ba),
                source_text=base.token_text(sources[i]),
                target_ab_text=base.token_text(target_ab),
                target_ba_text=base.token_text(target_ba),
                first_a_text=base.token_text(first_a[i]),
                first_b_text=base.token_text(first_b[i]),
                first_a_correct=int(first_a[i] == oracle_a[i]),
                first_b_correct=int(first_b[i] == oracle_b[i]),
            )
            for name, outputs, target in (
                ("direct_ab", direct_ab, target_ab),
                ("direct_ba", direct_ba, target_ba),
                ("sequential_ab", seq_ab, target_ab),
                ("sequential_ba", seq_ba, target_ba),
                ("oracle_ab", oracle_ab, target_ab),
                ("oracle_ba", oracle_ba, target_ba),
            ):
                row[name + "_text"] = base.token_text(outputs[i])
                row[name + "_correct"] = int(outputs[i] == target)
            row["direct_order_distinct_output"] = int(direct_ab[i] != direct_ba[i])
            row["sequential_order_distinct_output"] = int(seq_ab[i] != seq_ba[i])
            rows.append(row)
    return __import__("pandas").DataFrame(rows)


def summary(out_dir, config):
    ORIGINAL_SUMMARY(out_dir, config)
    path = out_dir / "SUMMARY.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("# GLT-BUILD-02: Order-Sensitive Composition", "# GLT-BUILD-03: Template and Lexical Generalization Audit", 1)
    text += (
        "\n## Generalization Audit\n\n"
        "Training uses canonical and article-suffix templates for identity and "
        "single operations. Evaluation uses the unseen suffix-only combination "
        "on held-out scene groups. The test therefore probes surface-template "
        "recombination and lexical scene-group generalization together.\n\n"
        "This is a stress test of the bounded GLT-BUILD-02 behavior. It does not "
        "turn the result into evidence for a universal operator or a Lie algebra.\n"
    )
    path.write_text(text, encoding="utf-8")


def validate(out_dir, config):
    behavior = __import__("pandas").read_csv(out_dir / "csv/behavior.csv")
    composition = __import__("pandas").read_csv(out_dir / "csv/composition.csv")
    if not ((behavior.generated_text == behavior.target_text).astype(int) == behavior.exact_match).all():
        raise ValueError("Behavior labels do not match saved text")
    for mode in ("direct_ab", "direct_ba", "sequential_ab", "sequential_ba", "oracle_ab", "oracle_ba"):
        target = "target_ab_text" if mode.endswith("ab") else "target_ba_text"
        if not ((composition[mode + "_text"] == composition[target]).astype(int) == composition[mode + "_correct"]).all():
            raise ValueError(f"Composition labels do not match saved text for {mode}")
    expected_behavior = 2 * len(config["seeds"]) * len(config["checkpoints"]) * config["test_scenes"] * len(SOURCE_VARIANTS) * (1 + len(OPS))
    expected_comp = 2 * len(config["seeds"]) * len(config["checkpoints"]) * config["test_scenes"] * len(PAIRS)
    if len(behavior) != expected_behavior or len(composition) != expected_comp:
        raise ValueError(f"Unexpected rows: behavior={len(behavior)} composition={len(composition)}")
    if behavior.duplicated(["arm", "seed", "checkpoint_step", "scene_id", "source_variant", "family", "op_name"]).any():
        raise ValueError("Duplicate behavior key")
    if composition.duplicated(["arm", "seed", "checkpoint_step", "scene_id", "pair"]).any():
        raise ValueError("Duplicate composition key")
    manifest = json.loads((out_dir / "artifacts.json").read_text(encoding="utf-8"))
    for item in manifest:
        path = out_dir / item["path"]
        if path.stat().st_size != item["bytes"] or exp.audit.sha256(path) != item["sha256"]:
            raise ValueError("Artifact hash mismatch")


def install_overrides():
    exp.__file__ = __file__
    exp.PROTOCOL = PROTOCOL
    exp.State = State
    exp.Example = Example
    exp.PAIRS = PAIRS
    exp.PAIR_TRAIN = PAIR_TRAIN
    exp.SOURCE_VARIANTS = SOURCE_VARIANTS
    exp.scene_split = scene_split
    exp.group_id = group_id
    exp.apply_op = apply_op
    exp.apply_ops = apply_ops
    exp.render_state = render_state
    exp.source_variant = source_variant
    exp.variant_id = variant_id
    exp.example = example
    exp.training_examples = training_examples
    exp.eval_examples = eval_examples
    exp.sample_batch = sample_batch
    exp.fixed_vocab = fixed_vocab
    exp.prompt = prompt
    exp.composition_audit = composition_audit
    exp.summary = summary
    exp.validate = validate


def main():
    install_overrides()
    exp.main()


if __name__ == "__main__":
    main()
