"""Paired training-input support experiment for GLT-BUILD-01A."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import torch

import run_glt_build_01_composition_audit as audit
import run_glt_build_01_explicit_operator_smoke as base


ARMS = ("base_only", "expanded_sources")
SEQUENCES = [()] + [(op,) for op in base.OPS] + base.TRAIN_PAIR_OPS
PAIRS = list(combinations(range(len(base.ENTITIES)), 2))
PROTOCOL = "glt-build-01a-paired-input-support-v1"
CONTEXT_LENGTH = 32


def group_id(state):
    pair = tuple(sorted((base.ENTITIES.index(state.subject), base.ENTITIES.index(state.object))))
    return state.verb_idx * len(PAIRS) + PAIRS.index(pair)


def scene_split():
    rng = random.Random(1731)
    heldout_groups = set()
    for verb in range(len(base.VERBS)):
        heldout_groups.update(verb * len(PAIRS) + p for p in rng.sample(range(len(PAIRS)), 6 if verb < 4 else 5))
    train, test = [], []
    for sid, state in base.all_base_scenes():
        (test if group_id(state) in heldout_groups else train).append((sid, state))
    return train, test


def source_variant(state, variant):
    if variant not in range(16):
        raise ValueError("Source variant must be in [0, 15]")
    return replace(state, tense="past" if variant & 1 else "present", negated=bool(variant & 2),
                   voice="passive" if variant & 4 else "active", mood="question" if variant & 8 else "statement")


def variant_id(state):
    return int(state.tense == "past") + 2 * int(state.negated) + 4 * int(state.voice == "passive") + 8 * int(state.mood == "question")


def example(sid, state, seq, family="train", split="train"):
    target = base.apply_ops(state, seq)
    return base.Example(split, family, seq, sid, state, target, base.render_state(state), base.render_state(target))


def effective_variant(arm, seq_index, variant):
    if arm not in ARMS:
        raise ValueError(arm)
    return variant if arm == "expanded_sources" and seq_index < 6 else 0


def training_examples(arm, scenes):
    for sid, state in scenes:
        for seq_index, seq in enumerate(SEQUENCES):
            variants = range(16) if arm == "expanded_sources" and seq_index < 6 else [0]
            for variant in variants:
                yield example(sid, source_variant(state, variant), seq)


def sample_batch(arm, scenes, rng, token_to_id, batch_size):
    plan = [(rng.randrange(len(scenes)), rng.randrange(len(SEQUENCES)), rng.randrange(16)) for _ in range(batch_size)]
    inputs = torch.full((batch_size, CONTEXT_LENGTH), token_to_id["<pad>"], dtype=torch.long)
    labels = torch.full((batch_size, CONTEXT_LENGTH), -100, dtype=torch.long)
    for row, (scene_index, seq_index, variant) in enumerate(plan):
        sid, state = scenes[scene_index]
        state = source_variant(state, effective_variant(arm, seq_index, variant))
        ex = example(sid, state, SEQUENCES[seq_index])
        x, y, _ = base.encode_example(ex, token_to_id)
        if len(x) > CONTEXT_LENGTH:
            raise ValueError("Training sequence exceeds matched context length")
        inputs[row, :len(x)] = torch.tensor(x)
        labels[row, :len(y)] = torch.tensor(y)
    return inputs, labels, np.asarray(plan, dtype=np.int32)


def evaluation_groups(scenes):
    groups = {name: [] for name in ("base_single", "nonbase_single", "nonbase_identity", "seen_pair", "reverse_seen_pair", "unseen_pair")}
    for sid, state in scenes:
        for variant in range(16):
            source = source_variant(state, variant)
            family = "nonbase_single" if variant else "base_single"
            groups[family].extend(example(sid, source, (op,), family, "test") for op in base.OPS)
            if variant:
                groups["nonbase_identity"].append(example(sid, source, (), "nonbase_identity", "test"))
        for family, sequences in (("seen_pair", base.TRAIN_PAIR_OPS),
                                  ("reverse_seen_pair", [tuple(reversed(seq)) for seq in base.TRAIN_PAIR_OPS]),
                                  ("unseen_pair", base.UNSEEN_PAIR_OPS)):
            groups[family].extend(example(sid, state, seq, family, "test") for seq in sequences)
    return groups


def fixed_vocab():
    ordinary = set(base.ENTITIES) | {"does", "did", "not", "is", "was", "by", ".", "?"}
    for verb in base.VERBS:
        ordinary.update((verb.base, verb.present, verb.past, verb.participle))
    vocab = list(base.SPECIAL_TOKENS) + sorted(ordinary)
    return {token: i for i, token in enumerate(vocab)}, vocab


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_examples(path, examples):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_id", "group_id", "source_variant", "op_name", "source", "target", "family"])
        writer.writeheader()
        count = 0
        for ex in examples:
            writer.writerow(dict(scene_id=ex.scene_id, group_id=group_id(ex.source_state), source_variant=variant_id(ex.source_state),
                                 op_name=ex.op_name, source=base.token_text(ex.source_tokens), target=base.token_text(ex.target_tokens), family=ex.family))
            count += 1
    return count


def evaluate(generator, groups, scene_to_group):
    rows = []
    for family, examples in groups.items():
        outputs = generator([(ex.op_seq, ex.source_tokens) for ex in examples])
        for ex, output in zip(examples, outputs):
            rows.append(dict(scene_id=ex.scene_id, group_id=scene_to_group[ex.scene_id], family=family,
                             source_variant=variant_id(ex.source_state), op_name=ex.op_name,
                             source_text=base.token_text(ex.source_tokens), target_text=base.token_text(ex.target_tokens),
                             generated_text=base.token_text(output), exact_match=int(output == ex.target_tokens)))
    return pd.DataFrame(rows)


def clustered_summary(frame, columns):
    clustered = frame.rename(columns={"scene_id": "ordered_scene_id", "group_id": "scene_id"})
    return audit.bootstrap_rates(clustered, columns).rename(columns={"n_sources": "n_scene_groups"})


def paired_effect(frame, columns):
    aggregate = frame.groupby(columns + ["group_id", "seed", "arm"], as_index=False).exact_match.mean()
    paired = aggregate.pivot(index=columns + ["group_id", "seed"], columns="arm", values="exact_match").reset_index()
    if paired[list(ARMS)].isna().any().any():
        raise ValueError("Incomplete paired evaluation")
    paired["exact_match"] = paired.expanded_sources - paired.base_only
    differences = audit.bootstrap_rates(paired.rename(columns={"group_id": "scene_id"}), columns)
    means = paired.groupby(columns, as_index=False)[list(ARMS)].mean()
    return means.merge(differences.rename(columns={"rate": "difference", "n_sources": "n_scene_groups", "n_rows": "n_group_seed_pairs"}), on=columns)


def table(frame):
    return base.df_to_markdown(frame.astype(object).where(frame.notna(), "undefined"))


def reports(out_dir, config):
    csv_dir = out_dir / "csv"
    behavior = pd.read_csv(csv_dir / "behavior.csv")
    comp = pd.read_csv(csv_dir / "composition.csv")
    modes = [key for key in comp if key.endswith("_correct") or key.endswith("_order_agreement")]
    long = comp.melt(id_vars=["arm", "seed", "checkpoint_step", "scene_id", "group_id", "pair"],
                     value_vars=modes, var_name="metric", value_name="exact_match")
    behavior_summary = clustered_summary(behavior, ["arm", "checkpoint_step", "family"])
    comp_summary = clustered_summary(long, ["arm", "checkpoint_step", "pair", "metric"])
    primitive_effect = paired_effect(behavior, ["checkpoint_step", "family"])
    primary = long[(long.metric == "sequential_both_correct") & (long.pair != "SQ")]
    primary_effect = paired_effect(primary, ["checkpoint_step", "metric"])
    diagnostic_effect = paired_effect(long, ["checkpoint_step", "pair", "metric"])
    for name, frame in (("behavior_summary", behavior_summary), ("composition_summary", comp_summary),
                        ("behavior_paired_effect", primitive_effect), ("primary_paired_effect", primary_effect),
                        ("composition_paired_effect", diagnostic_effect)):
        frame.to_csv(csv_dir / (name + ".csv"), index=False)
    final_step = config["steps"]
    final_primary = primary_effect[primary_effect.checkpoint_step == final_step].iloc[0]
    final_behavior = behavior_summary[behavior_summary.checkpoint_step == final_step]

    def rate(arm, family):
        return float(final_behavior[(final_behavior.arm == arm) & (final_behavior.family == family)].iloc[0].rate)

    competent = rate("base_only", "base_single") >= 0.95 and rate("expanded_sources", "base_single") >= 0.95 and rate("expanded_sources", "nonbase_single") >= 0.95
    rescue = bool(competent and final_primary.difference >= 0.20 and final_primary.ci_low > 0)
    decision = dict(primary_checkpoint=final_step, both_order_gain=float(final_primary.difference),
                    ci_low=float(final_primary.ci_low), ci_high=float(final_primary.ci_high),
                    primitive_competence_gate=bool(competent), rescue_criterion_met=rescue,
                    interpretation="input_support_rescue_under_fixed_protocol" if rescue else "no_clean_rescue_under_fixed_budget",
                    no_claim="No latent algebra or universal compositionality is established by this text-mediated task")
    audit.write_json(out_dir / "decision.json", decision)
    display_metrics = ["direct_ab_correct", "direct_ba_correct", "sequential_ab_correct", "sequential_ba_correct", "oracle_ab_correct", "oracle_ba_correct"]
    by_pair = long[(long.checkpoint_step == final_step) & long.metric.isin(display_metrics)].pivot_table(
        index=["arm", "pair"], columns="metric", values="exact_match", aggfunc="mean").reset_index()
    by_pair.to_csv(csv_dir / "final_composition_by_pair.csv", index=False)
    lines = ["# GLT-BUILD-01A: Paired Input-Support Experiment", "",
             "Status: completed exploratory experiment. Read decision.json together with the competency and per-pair tables.", "",
             "## Design", "",
             "Two arms share the initial weights, scene/command draw schedule, optimizer, updates, batch size and fixed padded "
             "training shape. In expanded_sources only identity/single-operation examples vary the four binary source-state "
             "features. Taught pair commands retain base inputs in both arms. Prediction-token counts differ with sentence "
             "length and are recorded; this is not an equal-target-token intervention.", "",
             "The split groups both participant orders with the same verb together: 180 training and 44 held-out semantic "
             "groups (360 and 88 directed scenes). No source state or operation endpoint can cross this group split. "
             "The split and vocabulary order differ from the earlier pilot, so the paired base_only arm is the comparator.", "",
             "## Primary Result", "", table(primary_effect), "",
             "Primary metric: both sequential orders correct, macro-averaged over TN/TQ/NQ/VQ/NV/TV; SQ is a separate "
             "positive-control pair. The predeclared rescue rule requires a gain of at least 0.20, a positive lower "
             "paired-bootstrap bound, and at least 95% base single-step competence in both arms plus 95% nonbase "
             "single-step competence in expanded_sources. The final fixed checkpoint is primary; earlier scores do not "
             "select a stopping point. Rule outcome: **" + decision["interpretation"] + "**.", "",
             "## Primitive And Direct-Command Checks", "", table(final_behavior.drop(columns="checkpoint_step")), "",
             "## Composition By Pair", "", table(by_pair), "",
             "The second sequential step receives the generated intermediate, without repair. Oracle rows use the "
             "correct intermediate. Direct unseen pair commands are absent from both training pools; successful "
             "text-mediated composition need not imply understanding a novel two-command prefix.", "",
             "## Limits And Artifacts", "",
             "Intervals use 1000 crossed group/seed bootstrap draws, with paired arm differences. Participant reversals, "
             "source variants and operations remain within a resampled group. Three seeds give limited uncertainty "
             "resolution; degenerate intervals are not population certainty. Expanding support intentionally increases "
             "the available unique examples and changes target lengths; equal updates are not equal epochs or tokens. "
             "The result concerns this controlled grammar and explicit commands. No geometry fit or intervention sweep "
             "is added in this experiment.", "",
             "Local checkpoints include weights, optimizer and RNG states; local hidden/ stores a fixed panel's two "
             "readouts at layers 0, 1 and 2. CSVs preserve outputs, source states, training pools, coverage and paired "
             "summaries. artifacts.json verifies every binary; metadata.json records code hashes and the exact budget.", "",
             "![Paired sequential execution](figures/paired_sequential_execution.png)", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = audit.plt.subplots(figsize=(8, 4.5))
    for arm in ARMS:
        ax.plot(primary_effect.checkpoint_step, primary_effect[arm], marker="o", label=arm)
    ax.set(xlabel="Training step", ylabel="Both sequential orders correct (six pairs)", ylim=(-0.04, 1.04))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures/paired_sequential_execution.png", dpi=160)
    audit.plt.close(fig)


def validate_outputs(out_dir, config, expected_rows, artifacts):
    for name, n_per_checkpoint in expected_rows.items():
        frame = pd.read_csv(out_dir / "csv" / (name + ".csv"))
        expected = n_per_checkpoint * len(ARMS) * len(config["seeds"]) * len(config["checkpoints"])
        if len(frame) != expected:
            raise ValueError(f"{name}: expected {expected}, got {len(frame)}")
        if name == "behavior":
            keys = ["arm", "seed", "checkpoint_step", "scene_id", "source_variant", "family", "op_name"]
            if not ((frame.generated_text == frame.target_text).astype(int) == frame.exact_match).all():
                raise ValueError("Incorrect behavior labels")
        else:
            keys = ["arm", "seed", "checkpoint_step", "scene_id", "pair"]
            for mode in ("direct_ab", "direct_ba", "sequential_ab", "sequential_ba", "oracle_ab", "oracle_ba"):
                if not ((frame[mode + "_text"] == frame.target_text).astype(int) == frame[mode + "_correct"]).all():
                    raise ValueError("Incorrect composition labels")
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate evaluation keys")
    for item in artifacts:
        path = out_dir / item["path"]
        if path.stat().st_size != item["bytes"] or audit.sha256(path) != item["sha256"]:
            raise ValueError("Artifact verification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--checkpoints", default="0,400,800,1600")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--debug-test-groups", type=int, default=0)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    checkpoints = sorted({0, args.steps} | {int(value) for value in args.checkpoints.split(",")})
    if len(set(seeds)) != len(seeds) or min(seeds) < 0 or min(checkpoints) < 0 or max(checkpoints) > args.steps or args.threads < 1 or args.eval_batch_size < 1 or args.debug_test_groups < 0:
        parser.error("Invalid seeds, checkpoints or resource settings")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in ("csv", "figures", "checkpoints", "hidden"):
        (out_dir / name).mkdir()
    status = dict(status="running", phase="initializing", pid=os.getpid(), started_at=base.now(), completed_models=0, total_models=len(seeds) * len(ARMS))

    def update(**values):
        status.update(values, updated_at=base.now())
        audit.write_json(out_dir / "run_status.json", status)

    update()
    try:
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        train_scenes, test_scenes = scene_split()
        if args.debug_test_groups:
            selected = sorted({group_id(state) for _, state in test_scenes})[:args.debug_test_groups]
            test_scenes = [(sid, state) for sid, state in test_scenes if group_id(state) in selected]
        scene_to_group = {sid: group_id(state) for sid, state in train_scenes + test_scenes}
        groups = evaluation_groups(test_scenes)
        token_to_id, vocab = fixed_vocab()
        counts = {arm: write_examples(out_dir / "csv" / ("training_pool_" + arm + ".csv"), training_examples(arm, train_scenes)) for arm in ARMS}
        write_examples(out_dir / "csv/evaluation_examples.csv", (ex for examples in groups.values() for ex in examples))
        pd.DataFrame([dict(split=split, scene_id=sid, group_id=group_id(state), subject=state.subject, object=state.object, verb=base.VERBS[state.verb_idx].base)
                      for split, scenes in (("train", train_scenes), ("test", test_scenes)) for sid, state in scenes]).to_csv(out_dir / "csv/split_scenes.csv", index=False)
        panel_train_ids = set()
        for subject in base.ENTITIES:
            panel_train_ids.update([sid for sid, state in train_scenes if state.subject == subject][:2])
        panel_scenes = [(sid, state) for sid, state in train_scenes if sid in panel_train_ids] + test_scenes
        sentences = sorted({base.render_state(source_variant(state, variant)) for _, state in panel_scenes for variant in range(16)})
        pd.DataFrame([dict(row_id=i, sentence=base.token_text(tokens), tokens_json=json.dumps(tokens)) for i, tokens in enumerate(sentences)]).to_csv(out_dir / "csv/hidden_panel_sentences.csv", index=False)
        config = vars(args) | dict(protocol=PROTOCOL, seeds=seeds, checkpoints=checkpoints, arms=list(ARMS),
                                  train_scenes=len(train_scenes), test_scenes=len(test_scenes), test_groups=len({group_id(s) for _, s in test_scenes}),
                                  available_training_examples=counts, train_scene_groups=180, batch_size=64,
                                  d_model=64, heads=4, layers=2, max_seq_len=64, padded_training_length=CONTEXT_LENGTH,
                                  lr=0.003, weight_decay=0.01, dropout=0.0, max_new_tokens=12, bootstrap_draws=1000,
                                  supervised_token_budget="measured separately; not matched", vocab=vocab, device="cpu",
                                  primary_pairs=["TN", "TQ", "NQ", "VQ", "NV", "TV"], rescue_min_gain=0.20, competence_min=0.95,
                                  python_version=sys.version, torch_version=str(torch.__version__), numpy_version=np.__version__,
                                  git_head=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
                                  source_sha256={Path(p).name: audit.sha256(Path(p)) for p in (__file__, audit.__file__, base.__file__)})
        audit.write_json(out_dir / "metadata.json", config)
        artifacts, pairing = [], []
        expected_initial, expected_schedule = {}, {}
        completed = 0
        for seed in seeds:
            for arm in ARMS:
                base.set_seed(seed)
                rng = random.Random(seed + 73_000)
                model = base.TinyDecoderOnlyTransformer(len(vocab), 64, 4, 2, 64, 0.0)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
                initial_digest = model_digest(model)
                if seed in expected_initial and initial_digest != expected_initial[seed]:
                    raise ValueError("Paired arms have different initial weights")
                expected_initial[seed] = initial_digest
                schedule_digest = hashlib.sha256()
                supervised_tokens, loss_value = 0, None
                command_counts = Counter()
                update(current_seed=seed, arm=arm, current_step=0)
                for step in range(args.steps + 1):
                    if step:
                        model.train()
                        inputs, labels, plan = sample_batch(arm, train_scenes, rng, token_to_id, 64)
                        schedule_digest.update(plan.tobytes())
                        command_counts.update(int(value) for value in plan[:, 1])
                        supervised_tokens += int(labels.ne(-100).sum())
                        optimizer.zero_grad(set_to_none=True)
                        logits, _ = model(inputs)
                        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), labels.reshape(-1), ignore_index=-100)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        loss_value = float(loss.item())
                    if step % 25 == 0:
                        update(phase="training", current_step=step, latest_loss=loss_value)
                    if step not in checkpoints:
                        continue
                    update(phase="checkpoint", current_step=step, latest_loss=loss_value)
                    stem = f"{arm}_seed_{seed}_step_{step:06d}"
                    checkpoint_path = out_dir / "checkpoints" / (stem + ".pt")
                    audit.save_checkpoint(checkpoint_path, model, optimizer, rng, config | dict(arm=arm), seed, step)
                    generator = audit.Generator(model, vocab, token_to_id, args.eval_batch_size, 12)
                    update(phase="behavior")
                    behavior = evaluate(generator, groups, scene_to_group)
                    update(phase="composition")
                    composition = audit.composition_audit(generator, test_scenes)
                    composition["group_id"] = composition.scene_id.map(scene_to_group)
                    update(phase="saving_readouts")
                    hidden = audit.extract_hidden(model, sentences, token_to_id, args.eval_batch_size)
                    hidden_path = out_dir / "hidden" / (stem + ".npz")
                    np.savez_compressed(hidden_path, **hidden)
                    for name, frame in (("behavior", behavior), ("composition", composition)):
                        frame.insert(0, "checkpoint_step", step)
                        frame.insert(0, "seed", seed)
                        frame.insert(0, "arm", arm)
                        path = out_dir / "csv" / (name + ".csv")
                        frame.to_csv(path, mode="a", header=not path.exists(), index=False)
                    train_path = out_dir / "csv/training_log.csv"
                    pd.DataFrame([dict(arm=arm, seed=seed, checkpoint_step=step, loss=loss_value,
                                       examples_seen=step * 64, supervised_tokens=supervised_tokens,
                                       schedule_sha256=schedule_digest.hexdigest())]).to_csv(train_path, mode="a", header=not train_path.exists(), index=False)
                    for path in (checkpoint_path, hidden_path):
                        artifacts.append(dict(path=path.relative_to(out_dir).as_posix(), bytes=path.stat().st_size, sha256=audit.sha256(path)))
                    audit.write_json(out_dir / "artifacts.json", artifacts)
                    print(f"{arm} seed={seed} step={step}: saved outputs, weights and readouts", flush=True)
                digest = schedule_digest.hexdigest()
                if seed in expected_schedule and digest != expected_schedule[seed]:
                    raise ValueError("Paired training draw schedules differ")
                expected_schedule[seed] = digest
                pairing.append(dict(arm=arm, seed=seed, initial_weights_sha256=initial_digest,
                                    draw_schedule_sha256=digest, command_counts=dict(command_counts), supervised_tokens=supervised_tokens))
                audit.write_json(out_dir / "pairing_checks.json", pairing)
                completed += 1
                update(completed_models=completed)
                del generator, model, optimizer, hidden
        update(phase="validating")
        validate_outputs(out_dir, config, {"behavior": sum(map(len, groups.values())), "composition": len(test_scenes) * len(audit.PAIRS)}, artifacts)
        update(phase="reporting")
        reports(out_dir, config)
        update(status="finished", phase="complete", finished_at=base.now())
    except BaseException as error:
        update(status="failed", phase="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
