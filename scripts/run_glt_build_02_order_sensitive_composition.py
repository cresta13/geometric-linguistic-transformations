"""Test order-sensitive composition in a controlled synthetic language."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

import run_glt_build_01_composition_audit as audit
import run_glt_build_01_explicit_operator_smoke as base
import run_glt_build_01_input_support as support


PROTOCOL = "glt-build-02-order-sensitive-composition-v1"
ARMS = ("pair_exposed", "single_only")
OPS = ("R", "M", "T", "N")
PAIR_TRAIN = (("R", "M"), ("T", "N"))
PAIRS = (
    ("R", "M"), ("M", "R"),
    ("T", "N"), ("N", "T"),
    ("R", "N"), ("N", "R"),
    ("M", "T"), ("T", "M"),
)
SINGLE_SEQUENCES = [()] + [(op,) for op in OPS]
PAIR_SEQUENCES = SINGLE_SEQUENCES + list(PAIR_TRAIN)
CONTEXT_LENGTH = 32


@dataclass(frozen=True)
class State:
    subject: str
    verb_idx: int
    object: str
    tense: str = "present"
    negated: bool = False
    marked: frozenset[str] = frozenset()


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


MARK_VARIANTS = ("none", "subject", "object")
SOURCE_VARIANTS = tuple(
    (tense, negated, mark)
    for tense in ("present", "past")
    for negated in (False, True)
    for mark in MARK_VARIANTS
)


def now() -> str:
    return base.now()


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


def render_state(state: State) -> tuple[str, ...]:
    verb = base.VERBS[state.verb_idx]
    subject = ["marked", state.subject] if state.subject in state.marked else [state.subject]
    object_tokens = ["marked", state.object] if state.object in state.marked else [state.object]
    if state.negated:
        verb_tokens = ["does" if state.tense == "present" else "did", "not", verb.base]
    else:
        verb_tokens = [verb.present if state.tense == "present" else verb.past]
    return tuple(subject + verb_tokens + object_tokens + ["."])


def source_variant(state: State, variant: int) -> State:
    tense, negated, mark = SOURCE_VARIANTS[variant]
    marked = frozenset(() if mark == "none" else (state.subject if mark == "subject" else state.object,))
    return replace(state, tense=tense, negated=negated, marked=marked)


def variant_id(state: State) -> int:
    mark = "none"
    if state.subject in state.marked:
        mark = "subject"
    elif state.object in state.marked:
        mark = "object"
    return SOURCE_VARIANTS.index((state.tense, state.negated, mark))


def group_id(state: base.State) -> int:
    return support.group_id(state)


def scene_split() -> tuple[list[tuple[int, base.State]], list[tuple[int, base.State]]]:
    return support.scene_split()


def example(sid: int, state: State, seq: tuple[str, ...], family: str, split: str = "train") -> Example:
    target = apply_ops(state, seq)
    return Example(split, family, seq, sid, state, target, render_state(state), render_state(target))


def training_examples(arm: str, scenes: list[tuple[int, base.State]]):
    sequences = PAIR_SEQUENCES if arm == "pair_exposed" else SINGLE_SEQUENCES
    for sid, raw_state in scenes:
        state = State(raw_state.subject, raw_state.verb_idx, raw_state.object)
        for seq in sequences:
            variants = range(len(SOURCE_VARIANTS)) if len(seq) <= 1 else (0,)
            for variant in variants:
                yield example(sid, source_variant(state, variant), seq, "train")


def eval_examples(test_scenes: list[tuple[int, base.State]]) -> list[Example]:
    rows = []
    for sid, raw_state in test_scenes:
        state = State(raw_state.subject, raw_state.verb_idx, raw_state.object)
        for variant in range(len(SOURCE_VARIANTS)):
            source = source_variant(state, variant)
            rows.append(example(sid, source, (), "identity", "eval"))
            for op in OPS:
                rows.append(example(sid, source, (op,), "single", "eval"))
    return rows


def sample_batch(arm, scenes, rng, token_to_id, batch_size):
    inputs = torch.full((batch_size, CONTEXT_LENGTH), token_to_id["<pad>"], dtype=torch.long)
    labels = torch.full((batch_size, CONTEXT_LENGTH), -100, dtype=torch.long)
    for row in range(batch_size):
        sid, raw_state = scenes[rng.randrange(len(scenes))]
        seq = (PAIR_SEQUENCES if arm == "pair_exposed" else SINGLE_SEQUENCES)[rng.randrange(len(PAIR_SEQUENCES) if arm == "pair_exposed" else len(SINGLE_SEQUENCES))]
        variant = rng.randrange(len(SOURCE_VARIANTS)) if len(seq) <= 1 else 0
        ex = example(sid, source_variant(State(raw_state.subject, raw_state.verb_idx, raw_state.object), variant), seq, "train")
        x, y = encode_example(ex, token_to_id)
        if len(x) > CONTEXT_LENGTH:
            raise ValueError("Training sequence exceeds matched context length")
        inputs[row, :len(x)] = torch.tensor(x)
        labels[row, :len(y)] = torch.tensor(y)
    return inputs, labels


def op_tokens(seq):
    return [f"<{op}>" for op in seq]


def encode_example(ex: Example, token_to_id: dict[str, int]):
    prefix = ["<bos>"] + op_tokens(ex.op_seq) + ["<sep>"] + list(ex.source_tokens) + ["<sep>"]
    target = list(ex.target_tokens) + ["<eos>"]
    full = prefix + target
    ids = [token_to_id[token] for token in full]
    split = len(prefix)
    return ids[:-1], [-100] * (split - 1) + ids[split:]


def fixed_vocab():
    specials = ["<pad>", "<bos>", "<sep>", "<eos>"] + op_tokens(OPS)
    words = [".", "marked"] + list(base.ENTITIES)
    for verb in base.VERBS:
        words.extend([verb.base, verb.present, verb.past])
    words.extend(["does", "did", "not"])
    vocab = list(dict.fromkeys(specials + words))
    return {token: i for i, token in enumerate(vocab)}, vocab


def prompt(seq, source, token_to_id):
    return [token_to_id[token] for token in ["<bos>"] + op_tokens(seq) + ["<sep>"] + list(source) + ["<sep>"]]


class Generator:
    def __init__(self, model, vocab, token_to_id, batch_size, max_new_tokens):
        self.model = model
        self.vocab = vocab
        self.token_to_id = token_to_id
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.cache = {}

    def __call__(self, requests):
        missing = list(dict.fromkeys(request for request in requests if request not in self.cache))
        if missing:
            outputs = audit.generate_batch(
                self.model,
                [prompt(seq, source, self.token_to_id) for seq, source in missing],
                self.vocab,
                self.token_to_id,
                self.max_new_tokens,
                self.batch_size,
            )
            self.cache.update(zip(missing, outputs))
        return [self.cache[request] for request in requests]


def write_pool(path: Path, examples):
    fields = ["scene_id", "group_id", "source_variant", "op_name", "source", "target", "family"]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ex in examples:
            writer.writerow(dict(scene_id=ex.scene_id, group_id=group_id(ex.source_state),
                                 source_variant=variant_id(ex.source_state), op_name=ex.op_name,
                                 source=base.token_text(ex.source_tokens), target=base.token_text(ex.target_tokens), family=ex.family))
            count += 1
    return count


def behavior_audit(generator, examples):
    rows = []
    outputs = generator([(ex.op_seq, ex.source_tokens) for ex in examples])
    for ex, output in zip(examples, outputs):
        rows.append(dict(scene_id=ex.scene_id, group_id=group_id(ex.source_state),
                         source_variant=variant_id(ex.source_state), family=ex.family,
                         op_name=ex.op_name, source_text=base.token_text(ex.source_tokens),
                         target_text=base.token_text(ex.target_tokens), generated_text=base.token_text(output),
                         exact_match=int(output == ex.target_tokens)))
    return pd.DataFrame(rows)


def composition_audit(generator, test_scenes):
    rows = []
    base_states = [State(s.subject, s.verb_idx, s.object) for _, s in test_scenes]
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
            row = dict(scene_id=scene_id, group_id=group_id(base.State(base_states[i].subject, base_states[i].verb_idx, base_states[i].object)),
                       pair=a + b, pair_seen_in_training=int((a, b) in PAIR_TRAIN),
                       order_distinct=int(target_ab != target_ba), source_text=base.token_text(sources[i]),
                       target_ab_text=base.token_text(target_ab), target_ba_text=base.token_text(target_ba),
                       first_a_text=base.token_text(first_a[i]), first_b_text=base.token_text(first_b[i]),
                       first_a_correct=int(first_a[i] == oracle_a[i]), first_b_correct=int(first_b[i] == oracle_b[i]))
            for name, outputs, target in (("direct_ab", direct_ab, target_ab), ("direct_ba", direct_ba, target_ba),
                                          ("sequential_ab", seq_ab, target_ab), ("sequential_ba", seq_ba, target_ba),
                                          ("oracle_ab", oracle_ab, target_ab), ("oracle_ba", oracle_ba, target_ba)):
                row[name + "_text"] = base.token_text(outputs[i])
                row[name + "_correct"] = int(outputs[i] == target)
            row["direct_order_distinct_output"] = int(direct_ab[i] != direct_ba[i])
            row["sequential_order_distinct_output"] = int(seq_ab[i] != seq_ba[i])
            rows.append(row)
    return pd.DataFrame(rows)


def append_frame(path, frame):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def summary(out_dir, config):
    csv_dir = out_dir / "csv"
    behavior = pd.read_csv(csv_dir / "behavior.csv")
    composition = pd.read_csv(csv_dir / "composition.csv")
    final_behavior = behavior[behavior.checkpoint_step == config["steps"]]
    final_comp = composition[composition.checkpoint_step == config["steps"]]
    behavior_summary = final_behavior.groupby(["arm", "family"], as_index=False).exact_match.mean()
    pair_summary = final_comp.groupby(["arm", "pair"], as_index=False)[
        ["order_distinct", "direct_ab_correct", "direct_ba_correct", "sequential_ab_correct", "sequential_ba_correct",
         "oracle_ab_correct", "oracle_ba_correct", "direct_order_distinct_output", "sequential_order_distinct_output"]
    ].mean()
    primary = final_comp.groupby("arm", as_index=False)[["direct_ab_correct", "direct_ba_correct", "sequential_ab_correct", "sequential_ba_correct"]].mean()
    noncomm = final_comp[final_comp.pair.isin(["RM", "MR"])].groupby("arm", as_index=False)[["sequential_ab_correct", "sequential_ba_correct", "sequential_order_distinct_output"]].mean()
    comm = final_comp[final_comp.pair.isin(["TN", "NT"])].groupby("arm", as_index=False)[["sequential_ab_correct", "sequential_ba_correct", "sequential_order_distinct_output"]].mean()
    single_only_unseen = final_comp[(final_comp.arm == "single_only") & (final_comp.pair_seen_in_training == 0)]
    direct_unseen = float(single_only_unseen[["direct_ab_correct", "direct_ba_correct"]].to_numpy().mean())
    decision = dict(
        primary_metric="order-sensitive sequential correctness on RM/MR with TN/NT commuting control",
        noncommuting_sequential={row.arm: {"ab": float(row.sequential_ab_correct), "ba": float(row.sequential_ba_correct), "distinct_output": float(row.sequential_order_distinct_output)} for row in noncomm.itertuples()},
        commuting_sequential={row.arm: {"ab": float(row.sequential_ab_correct), "ba": float(row.sequential_ba_correct), "distinct_output": float(row.sequential_order_distinct_output)} for row in comm.itertuples()},
        single_only_direct_unseen_rate=direct_unseen,
        interpretation="order_sensitive_composition_supported" if all(noncomm.sequential_ab_correct > 0.8) and all(noncomm.sequential_ba_correct > 0.8) and all(noncomm.sequential_order_distinct_output > 0.8) else "order_sensitive_composition_not_supported",
        no_claim="This is a finite discrete noncommuting-action test; it is not a Lie-algebra proof",
    )
    behavior_summary.to_csv(csv_dir / "final_behavior_summary.csv", index=False)
    pair_summary.to_csv(csv_dir / "final_pair_summary.csv", index=False)
    primary.to_csv(csv_dir / "final_primary_summary.csv", index=False)
    write_json(out_dir / "decision.json", decision)
    lines = [
        "# GLT-BUILD-02: Order-Sensitive Composition", "",
        "Completed exploratory run testing a genuinely noncommuting pair in a controlled synthetic language.", "",
        "## Design", "",
        "`R` swaps subject and object. `M` toggles a mark on the current subject. Both are involutions, but `R` and `M` do not commute: marking then swapping marks the original subject as object, while swapping then marking marks the original object as subject. `T` toggles tense and `N` toggles negation; they are an independent commuting control.", "",
        "The `single_only` arm sees identity and all four single operations, but no pair prefix. The `pair_exposed` arm additionally sees RM and TN pair prefixes on base sources. Both arms use the same 360/88 split, 12 source-state variants for identity/singles, three seeds, two layers, four heads, width 64, batch 64 and 1600 updates.", "",
        "## Final Behavioral Results", "", base.df_to_markdown(behavior_summary), "", base.df_to_markdown(pair_summary), "",
        "## Decision", "", base.df_to_markdown(primary), "", base.df_to_markdown(noncomm), "", base.df_to_markdown(comm), "",
        "The primary order-sensitive contrast is RM versus MR. A successful result requires both sequential orders to be correct and to produce distinct outputs, while the TN/NT control should remain order-invariant. Direct prefix accuracy is reported separately because a model can execute two single commands sequentially without constructing a new pair prefix.", "",
        "Decision: **" + decision["interpretation"] + "**.", "",
        "## Limits", "",
        "This is a discrete finite-state composition test. It does not establish a Lie algebra, continuous generators, or a universal latent operator. Rates are descriptive across three seeds and 44 held-out groups. The language is synthetic and the pair-exposed arm has a different command alphabet, so the arms are matched by seed and architecture but not claimed to have identical command sampling.", "",
        "![Order-sensitive outcomes](figures/order_sensitive_outcomes.png)", "",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    plot = final_comp.groupby(["arm", "pair"], as_index=False)[["sequential_ab_correct", "sequential_ba_correct"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(PAIRS))
    for arm in ARMS:
        data = plot[plot.arm == arm].set_index("pair").reindex([a + b for a, b in PAIRS])
        ax.plot(list(x), data.sequential_ab_correct, marker="o", label=f"{arm}: AB")
        ax.plot(list(x), data.sequential_ba_correct, marker="x", linestyle="--", label=f"{arm}: BA")
    ax.set_xticks(list(x), [a + b for a, b in PAIRS])
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("Sequential exact-match rate")
    ax.set_xlabel("Ordered pair")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "figures/order_sensitive_outcomes.png", dpi=160)
    plt.close(fig)


def validate(out_dir, config):
    behavior = pd.read_csv(out_dir / "csv/behavior.csv")
    composition = pd.read_csv(out_dir / "csv/composition.csv")
    if not ((behavior.generated_text == behavior.target_text).astype(int) == behavior.exact_match).all():
        raise ValueError("Behavior labels do not match saved text")
    for mode, target in (("direct_ab", "target_ab_text"), ("direct_ba", "target_ba_text"),
                         ("sequential_ab", "target_ab_text"), ("sequential_ba", "target_ba_text"),
                         ("oracle_ab", "target_ab_text"), ("oracle_ba", "target_ba_text")):
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
    manifest = json.loads((out_dir / "artifacts.json").read_text())
    for item in manifest:
        path = out_dir / item["path"]
        if path.stat().st_size != item["bytes"] or audit.sha256(path) != item["sha256"]:
            raise ValueError("Artifact hash mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--checkpoints", default="0,400,800,1600")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    checkpoints = sorted({0, args.steps} | {int(x) for x in args.checkpoints.split(",")})
    if len(set(seeds)) != len(seeds) or max(checkpoints) > args.steps or args.threads < 1 or args.eval_batch_size < 1:
        parser.error("Invalid run arguments")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("csv", "figures", "checkpoints"):
        (out_dir / directory).mkdir()
    status = dict(status="running", phase="initializing", pid=os.getpid(), started_at=now(), completed_models=0, total_models=len(seeds) * len(ARMS))

    def update(**values):
        status.update(values, updated_at=now())
        audit.write_json(out_dir / "run_status.json", status)

    update()
    try:
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        train_scenes, test_scenes = scene_split()
        token_to_id, vocab = fixed_vocab()
        pool_counts = {arm: write_pool(out_dir / "csv" / f"training_pool_{arm}.csv", training_examples(arm, train_scenes)) for arm in ARMS}
        write_pool(out_dir / "csv/evaluation_examples.csv", eval_examples(test_scenes))
        config = vars(args) | dict(protocol=PROTOCOL, seeds=seeds, checkpoints=checkpoints, arms=list(ARMS),
                                    train_scenes=len(train_scenes), test_scenes=len(test_scenes), train_groups=180,
                                    test_groups=len({group_id(s) for _, s in test_scenes}), available_training_examples=pool_counts,
                                    d_model=64, heads=4, layers=2, batch_size=64, context_length=CONTEXT_LENGTH,
                                    lr=0.003, weight_decay=0.01, dropout=0.0, max_new_tokens=12, vocab=vocab,
                                    device="cpu", noncommuting_pair="R/M", commuting_control="T/N",
                                    source_sha256={Path(p).name: audit.sha256(Path(p)) for p in (__file__, audit.__file__, support.__file__, base.__file__)},
                                    git_head=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip())
        write_json(out_dir / "metadata.json", config)
        artifacts = []
        completed = 0
        for seed in seeds:
            for arm in ARMS:
                base.set_seed(seed)
                rng = random.Random(seed + (12000 if arm == "pair_exposed" else 13000))
                model = base.TinyDecoderOnlyTransformer(len(vocab), 64, 4, 2, 64, 0.0)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss_value = None
                for step in range(args.steps + 1):
                    if step:
                        model.train()
                        inputs, labels = sample_batch(arm, train_scenes, rng, token_to_id, 64)
                        optimizer.zero_grad(set_to_none=True)
                        logits, _ = model(inputs)
                        loss = loss_fn(logits.reshape(-1, len(vocab)), labels.reshape(-1))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        loss_value = float(loss.item())
                    if step % 25 == 0:
                        update(phase="training", current_seed=seed, arm=arm, current_step=step, latest_loss=loss_value)
                    if step not in checkpoints:
                        continue
                    update(phase="checkpoint", current_seed=seed, arm=arm, current_step=step, latest_loss=loss_value)
                    stem = f"{arm}_seed_{seed}_step_{step:06d}"
                    checkpoint_path = out_dir / "checkpoints" / f"{stem}.pt"
                    audit.save_checkpoint(checkpoint_path, model, optimizer, rng, config | dict(arm=arm), seed, step)
                    generator = Generator(model, vocab, token_to_id, args.eval_batch_size, 12)
                    behavior = behavior_audit(generator, eval_examples(test_scenes))
                    composition = composition_audit(generator, test_scenes)
                    for name, frame in (("behavior", behavior), ("composition", composition)):
                        frame.insert(0, "checkpoint_step", step)
                        frame.insert(0, "seed", seed)
                        frame.insert(0, "arm", arm)
                        append_frame(out_dir / "csv" / f"{name}.csv", frame)
                    artifacts.append(dict(path=checkpoint_path.relative_to(out_dir).as_posix(), bytes=checkpoint_path.stat().st_size, sha256=audit.sha256(checkpoint_path)))
                    write_json(out_dir / "artifacts.json", artifacts)
                    print(f"{arm} seed={seed} step={step}: saved outputs and checkpoint", flush=True)
                completed += 1
                update(completed_models=completed)
                del generator, model, optimizer
        update(phase="validating")
        validate(out_dir, config)
        update(phase="reporting")
        summary(out_dir, config)
        update(status="finished", phase="complete", finished_at=now())
    except BaseException as error:
        update(status="failed", phase="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
