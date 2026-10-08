"""Test whether explicit operation prefixes compose without pair-prefix training."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

import pandas as pd
import torch

import run_glt_build_01_composition_audit as audit
import run_glt_build_01_explicit_operator_smoke as base
import run_glt_build_01_input_support as support


ARMS = ("expanded_pairs", "single_only")
SINGLE_SEQUENCES = [()] + [(op,) for op in base.OPS]
PROTOCOL = "glt-build-01a-pair-prefix-holdout-v1"


def training_examples(arm, scenes):
    if arm == "expanded_pairs":
        yield from support.training_examples("expanded_sources", scenes)
        return
    for sid, state in scenes:
        for seq in SINGLE_SEQUENCES:
            for variant in range(16):
                yield support.example(sid, support.source_variant(state, variant), seq)


def sample_batch(arm, scenes, rng, token_to_id, batch_size):
    inputs = torch.full((batch_size, support.CONTEXT_LENGTH), token_to_id["<pad>"], dtype=torch.long)
    labels = torch.full((batch_size, support.CONTEXT_LENGTH), -100, dtype=torch.long)
    plans = []
    for row in range(batch_size):
        scene_index = rng.randrange(len(scenes))
        seq_index = rng.randrange(10 if arm == "expanded_pairs" else 6)
        variant = rng.randrange(16)
        sid, state = scenes[scene_index]
        seq = support.SEQUENCES[seq_index] if arm == "expanded_pairs" else SINGLE_SEQUENCES[seq_index]
        source_variant = variant
        if len(seq) == 2:
            source_variant = 0
        ex = support.example(sid, support.source_variant(state, source_variant), seq)
        x, y, _ = base.encode_example(ex, token_to_id)
        inputs[row, :len(x)] = torch.tensor(x)
        labels[row, :len(y)] = torch.tensor(y)
        plans.append((scene_index, seq_index, variant if len(seq) < 2 else 0))
    return inputs, labels, plans


def append_frame(path, frame):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def write_pool(path, examples):
    fields = ["scene_id", "group_id", "source_variant", "op_name", "source", "target", "family"]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ex in examples:
            writer.writerow(dict(scene_id=ex.scene_id, group_id=support.group_id(ex.source_state),
                                 source_variant=support.variant_id(ex.source_state), op_name=ex.op_name,
                                 source=base.token_text(ex.source_tokens), target=base.token_text(ex.target_tokens), family=ex.family))
            count += 1
    return count


def write_summary(out_dir, config):
    csv_dir = out_dir / "csv"
    behavior = pd.read_csv(csv_dir / "behavior.csv")
    composition = pd.read_csv(csv_dir / "composition.csv")
    final = behavior[behavior.checkpoint_step == config["steps"]]
    final_comp = composition[composition.checkpoint_step == config["steps"]]
    behavior_summary = final.groupby(["arm", "family"], as_index=False).exact_match.mean()
    comp_summary = final_comp.groupby(["arm", "pair"], as_index=False)[
        ["direct_ab_correct", "direct_ba_correct", "sequential_ab_correct", "sequential_ba_correct",
         "oracle_ab_correct", "oracle_ba_correct", "direct_both_correct", "sequential_both_correct"]
    ].mean()
    pair_group = final_comp[final_comp.pair != "SQ"].groupby(["arm", "pair"], as_index=False).sequential_both_correct.mean()
    primary = pair_group.groupby("arm", as_index=False).sequential_both_correct.mean()
    direct = final_comp[final_comp.pair != "SQ"].groupby(["arm", "pair_seen_in_training"], as_index=False).direct_ab_correct.mean()
    unseen_direct = float(direct[(direct.arm == "single_only") & (direct.pair_seen_in_training == 0)].direct_ab_correct.iloc[0])
    expanded_direct = float(direct[(direct.arm == "expanded_pairs") & (direct.pair_seen_in_training == 0)].direct_ab_correct.iloc[0])
    decision = dict(primary_metric="both sequential orders correct on TN/TQ/NQ/VQ/NV/TV",
                    primary_rates={row.arm: float(row.sequential_both_correct) for row in primary.itertuples()},
                    direct_unseen_single_only=unseen_direct,
                    direct_unseen_expanded_pairs=expanded_direct,
                    interpretation="prefix_composition_supported" if unseen_direct >= 0.20 else "prefix_composition_not_supported",
                    no_claim="Sequential text transformation and explicit prefix composition are separate capabilities; no latent algebra is established")
    audit.write_json(out_dir / "decision.json", decision)
    behavior_summary.to_csv(csv_dir / "final_behavior_summary.csv", index=False)
    comp_summary.to_csv(csv_dir / "final_composition_by_pair.csv", index=False)
    primary.to_csv(csv_dir / "final_primary_summary.csv", index=False)
    direct.to_csv(csv_dir / "final_direct_pair_summary.csv", index=False)
    lines = ["# GLT-BUILD-01A: Pair-Prefix Holdout", "",
             "Completed exploratory run. This tests explicit command-prefix composition under a fixed synthetic grammar.", "",
             "## Design", "",
             "`expanded_pairs` receives identity, five singles and four taught pair prefixes. `single_only` receives identity and five singles only; no two-operation prefix is present in its training pool. Both arms use all 16 source-state variants for identity and singles, the same 360/88 scene split, three seeds, two layers, four heads, width 64, batch 64 and 1600 updates. Pair training is the only intended arm difference. The schedule is not paired because the arms have different command alphabets; initial seeds and model configuration are matched.", "",
             "## Final Primitive And Pair Results", "", base.df_to_markdown(behavior_summary), "", base.df_to_markdown(comp_summary), "",
             "## Primary Decision", "", base.df_to_markdown(primary), "", base.df_to_markdown(direct), "",
             "The primary metric averages `sequential_both_correct` over TN, TQ, NQ, VQ, NV and TV, excluding SQ. Direct pair accuracy is reported separately for pairs seen and absent from the training pool. `single_only` is the critical test: it must execute a two-operation prefix without ever seeing a pair prefix during training.", "",
             "Decision: **" + decision["interpretation"] + "**.", "",
             "## Limits", "",
             "All external operations commute in the generator. This run therefore tests composition of command tokens and the ability to reuse learned single-operation behavior; it does not test noncommuting algebra. The model remains small, the language is synthetic, and the direct pair test uses explicit operation tokens. Rates are descriptive across three seeds and 44 held-out groups; no p-value is claimed. Saved checkpoints and CSVs are indexed by `artifacts.json`.", "",
             "![Pair-prefix outcomes](figures/pair_prefix_outcomes.png)", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = audit.plt.subplots(figsize=(8, 4.5))
    plot = pd.read_csv(csv_dir / "behavior.csv").groupby(["arm", "checkpoint_step", "family"], as_index=False).exact_match.mean()
    for arm in ARMS:
        data = plot[(plot.arm == arm) & plot.family.isin(["seen_pair", "unseen_pair", "nonbase_single"])]
        for family, group in data.groupby("family"):
            ax.plot(group.checkpoint_step, group.exact_match, marker="o", label=f"{arm}: {family}")
    ax.set(xlabel="Training step", ylabel="Exact-match rate", ylim=(-0.04, 1.04))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures/pair_prefix_outcomes.png", dpi=160)
    audit.plt.close(fig)


def validate(out_dir, config):
    behavior = pd.read_csv(out_dir / "csv/behavior.csv")
    composition = pd.read_csv(out_dir / "csv/composition.csv")
    if not ((behavior.generated_text == behavior.target_text).astype(int) == behavior.exact_match).all():
        raise ValueError("Behavior labels do not match saved text")
    for mode in ("direct_ab", "direct_ba", "sequential_ab", "sequential_ba", "oracle_ab", "oracle_ba"):
        if not ((composition[mode + "_text"] == composition.target_text).astype(int) == composition[mode + "_correct"]).all():
            raise ValueError("Composition labels do not match saved text")
    per_scene_behavior = 5 + 15 * 5 + 15 + 4 + 4 + 3
    expected_behavior = 2 * len(config["seeds"]) * len(config["checkpoints"]) * config["test_scenes"] * per_scene_behavior
    expected_comp = 2 * len(config["seeds"]) * len(config["checkpoints"]) * config["test_scenes"] * len(audit.PAIRS)
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
    parser.add_argument("--debug-test-groups", type=int, default=0)
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    checkpoints = sorted({0, args.steps} | {int(x) for x in args.checkpoints.split(",")})
    if len(set(seeds)) != len(seeds) or max(checkpoints) > args.steps or args.threads < 1 or args.eval_batch_size < 1 or args.debug_test_groups < 0:
        parser.error("Invalid run arguments")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in ("csv", "figures", "checkpoints"):
        (out_dir / name).mkdir()
    status = dict(status="running", phase="initializing", pid=os.getpid(), started_at=base.now(), completed_models=0, total_models=len(seeds) * len(ARMS))

    def update(**values):
        status.update(values, updated_at=base.now())
        audit.write_json(out_dir / "run_status.json", status)

    update()
    try:
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        train_scenes, test_scenes = support.scene_split()
        if args.debug_test_groups:
            groups = sorted({support.group_id(s) for _, s in test_scenes})[:args.debug_test_groups]
            test_scenes = [(sid, s) for sid, s in test_scenes if support.group_id(s) in groups]
        eval_groups = support.evaluation_groups(test_scenes)
        scene_to_group = {sid: support.group_id(s) for sid, s in train_scenes + test_scenes}
        token_to_id, vocab = support.fixed_vocab()
        pool_counts = {}
        for arm in ARMS:
            pool_counts[arm] = write_pool(out_dir / "csv" / f"training_pool_{arm}.csv", training_examples(arm, train_scenes))
        write_pool(out_dir / "csv/evaluation_examples.csv", (ex for group in eval_groups.values() for ex in group))
        config = vars(args) | dict(protocol=PROTOCOL, seeds=seeds, checkpoints=checkpoints, arms=list(ARMS),
                                  train_scenes=len(train_scenes), test_scenes=len(test_scenes), train_groups=180,
                                  test_groups=len({support.group_id(s) for _, s in test_scenes}), available_training_examples=pool_counts,
                                  d_model=64, heads=4, layers=2, batch_size=64, padded_training_length=support.CONTEXT_LENGTH,
                                  lr=0.003, weight_decay=0.01, dropout=0.0, max_new_tokens=12, vocab=vocab,
                                  device="cpu", no_pair_prefix_in_single_only=True,
                                  source_sha256={Path(p).name: audit.sha256(Path(p)) for p in (__file__, audit.__file__, support.__file__, base.__file__)},
                                  git_head=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip())
        audit.write_json(out_dir / "metadata.json", config)
        artifacts=[]
        completed=0
        for seed in seeds:
            for arm in ARMS:
                base.set_seed(seed)
                rng=random.Random(seed + (8100 if arm == "expanded_pairs" else 9100))
                model=base.TinyDecoderOnlyTransformer(len(vocab),64,4,2,64,0.0)
                optimizer=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.01)
                loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss_value=None
                for step in range(args.steps+1):
                    if step:
                        model.train()
                        inputs,labels,_=sample_batch(arm,train_scenes,rng,token_to_id,64)
                        optimizer.zero_grad(set_to_none=True)
                        logits,_=model(inputs)
                        loss=loss_fn(logits.reshape(-1,len(vocab)),labels.reshape(-1))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                        optimizer.step()
                        loss_value=float(loss.item())
                    if step % 25 == 0:
                        update(phase="training",current_seed=seed,arm=arm,current_step=step,latest_loss=loss_value)
                    if step not in checkpoints:
                        continue
                    update(phase="checkpoint",current_seed=seed,arm=arm,current_step=step,latest_loss=loss_value)
                    stem=f"{arm}_seed_{seed}_step_{step:06d}"
                    checkpoint_path=out_dir/"checkpoints"/(stem+".pt")
                    audit.save_checkpoint(checkpoint_path,model,optimizer,rng,config|dict(arm=arm),seed,step)
                    generator=audit.Generator(model,vocab,token_to_id,args.eval_batch_size,12)
                    behavior=support.evaluate(generator,eval_groups,scene_to_group)
                    composition=audit.composition_audit(generator,test_scenes)
                    composition["group_id"]=composition.scene_id.map(scene_to_group)
                    for name,frame in (("behavior",behavior),("composition",composition)):
                        frame.insert(0,"checkpoint_step",step); frame.insert(0,"seed",seed); frame.insert(0,"arm",arm)
                        append_frame(out_dir/"csv"/(name+".csv"),frame)
                    for path in (checkpoint_path,):
                        artifacts.append(dict(path=path.relative_to(out_dir).as_posix(),bytes=path.stat().st_size,sha256=audit.sha256(path)))
                    audit.write_json(out_dir/"artifacts.json",artifacts)
                    print(f"{arm} seed={seed} step={step}: saved outputs and checkpoint",flush=True)
                completed += 1
                update(completed_models=completed)
                del generator,model,optimizer
        update(phase="validating")
        validate(out_dir,config)
        update(phase="reporting")
        write_summary(out_dir,config)
        update(status="finished",phase="complete",finished_at=base.now())
    except BaseException as error:
        update(status="failed",phase="failed",error=repr(error),traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
