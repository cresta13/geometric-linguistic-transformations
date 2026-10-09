"""Evaluate GLT-BUILD-03 checkpoints on seen and unseen surface templates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

import run_glt_build_03_generalization_audit as exp


TEMPLATES = ("canonical", "article_suffix", "suffix_only")
CONTROL_PAIRS = (("R", "M"), ("M", "R"), ("T", "N"), ("N", "T"))


def behavior_rows(model, vocab, token_to_id, test_scenes, template, seed, arm, step):
    examples = []
    for sid, raw in test_scenes:
        state = exp.State(raw.subject, raw.verb_idx, raw.object, template=template)
        for variant in range(len(exp.SOURCE_VARIANTS)):
            source = exp.source_variant(state, variant, template)
            examples.append(exp.example(sid, source, (), "identity", "eval"))
            for op in exp.OPS:
                examples.append(exp.example(sid, source, (op,), "single", "eval"))
    generator = exp.exp.Generator(model, vocab, token_to_id, 32, 12)
    outputs = generator([(row.op_seq, row.source_tokens) for row in examples])
    return [
        dict(
            seed=seed,
            arm=arm,
            checkpoint_step=step,
            template=template,
            scene_id=row.scene_id,
            source_variant=exp.variant_id(row.source_state),
            family=row.family,
            op_name=row.op_name,
            exact_match=int(output == row.target_tokens),
        )
        for row, output in zip(examples, outputs)
    ]


def composition_rows(model, vocab, token_to_id, test_scenes, template, seed, arm, step):
    states = [exp.State(raw.subject, raw.verb_idx, raw.object, template=template) for _, raw in test_scenes]
    sources = [exp.render_state(state) for state in states]
    generator = exp.exp.Generator(model, vocab, token_to_id, 32, 12)
    rows = []
    for a, b in CONTROL_PAIRS:
        first_a = generator([((a,), source) for source in sources])
        first_b = generator([((b,), source) for source in sources])
        seq_ab = generator([((b,), output) for output in first_a])
        seq_ba = generator([((a,), output) for output in first_b])
        for i, (scene_id, _) in enumerate(test_scenes):
            target_ab = exp.render_state(exp.apply_ops(states[i], (a, b)))
            target_ba = exp.render_state(exp.apply_ops(states[i], (b, a)))
            rows.extend(
                [
                    dict(seed=seed, arm=arm, checkpoint_step=step, template=template,
                         scene_id=scene_id, pair=a + b, order="ab",
                         exact_match=int(seq_ab[i] == target_ab),
                         order_distinct=int(target_ab != target_ba),
                         output_distinct=int(seq_ab[i] != seq_ba[i])),
                    dict(seed=seed, arm=arm, checkpoint_step=step, template=template,
                         scene_id=scene_id, pair=a + b, order="ba",
                         exact_match=int(seq_ba[i] == target_ba),
                         order_distinct=int(target_ab != target_ba),
                         output_distinct=int(seq_ab[i] != seq_ba[i])),
                ]
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    checkpoint_dir = result_dir / "checkpoints"
    csv_dir = result_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    train_scenes, test_scenes = exp.scene_split()
    token_to_id, vocab = exp.fixed_vocab()
    behavior = []
    composition = []
    for arm in exp.ARMS:
        for seed in (0, 1, 2):
            path = checkpoint_dir / f"{arm}_seed_{seed}_step_001600.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = exp.exp.base.TinyDecoderOnlyTransformer(len(vocab), 64, 4, 2, 64, 0.0)
            model.load_state_dict(payload["model_state"])
            model.eval()
            for template in TEMPLATES:
                behavior.extend(behavior_rows(model, vocab, token_to_id, test_scenes, template, seed, arm, 1600))
                composition.extend(composition_rows(model, vocab, token_to_id, test_scenes, template, seed, arm, 1600))
            del model
    behavior_df = pd.DataFrame(behavior)
    composition_df = pd.DataFrame(composition)
    behavior_df.to_csv(csv_dir / "template_control_behavior.csv", index=False)
    composition_df.to_csv(csv_dir / "template_control_composition.csv", index=False)
    behavior_summary = behavior_df.groupby(["arm", "template", "family"], as_index=False).exact_match.mean()
    comp_summary = composition_df.groupby(["arm", "template", "pair", "order"], as_index=False).exact_match.mean()
    behavior_summary.to_csv(csv_dir / "template_control_behavior_summary.csv", index=False)
    comp_summary.to_csv(csv_dir / "template_control_composition_summary.csv", index=False)
    lines = [
        "# GLT-BUILD-03: Template Control Audit",
        "",
        "This post-hoc control evaluates the same final checkpoints on two templates seen during training and the suffix-only combination held out by the protocol.",
        "",
        "## Behavior",
        "",
        exp.exp.base.df_to_markdown(behavior_summary),
        "",
        "## Sequential Composition",
        "",
        exp.exp.base.df_to_markdown(comp_summary),
        "",
        "Interpretation: canonical and article_suffix are seen training templates; suffix_only is the held-out surface combination. These controls separate failure of the operation itself from failure to recombine surface structure.",
        "",
    ]
    (result_dir / "CONTROL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print((result_dir / "CONTROL_SUMMARY.md").as_posix())


if __name__ == "__main__":
    main()
