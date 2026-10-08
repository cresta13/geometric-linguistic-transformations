"""Corrected, bounded follow-up to the GLT-BUILD-01A smoke test."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_glt_build_01_explicit_operator_smoke as base


PROTOCOL = "glt-build-01a-composition-audit-v2"
PAIRS = base.TRAIN_PAIR_OPS + base.UNSEEN_PAIR_OPS
READOUTS = ("last_sep", "content_mean")


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Windows readers can briefly deny replacement of an otherwise writable file.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.8))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_scenes(per_subject: int = 0) -> list[tuple[int, base.State]]:
    scenes = [(sid, s) for sid, s in base.all_base_scenes() if base.scene_is_heldout(sid, s)]
    if per_subject == 0:
        return scenes
    chosen = []
    for subject in base.ENTITIES:
        chosen.extend([(sid, s) for sid, s in scenes if s.subject == subject][:per_subject])
    return chosen


def build_dataset(per_subject: int = 0):
    train, groups = base.build_examples(100_000)
    scenes = select_scenes(per_subject)
    selected_ids = {sid for sid, _ in scenes}
    # A small training sanity check covers every subject and every taught operation.
    sanity_ids = set()
    for subject in base.ENTITIES:
        ids = sorted({ex.scene_id for ex in train if ex.source_state.subject == subject})
        sanity_ids.update(ids[:2])
    groups["train_seen"] = [ex for ex in train if ex.scene_id in sanity_ids]
    for family in groups:
        if family != "train_seen":
            groups[family] = [ex for ex in groups[family] if ex.scene_id in selected_ids]
    train_keys = {(ex.op_seq, ex.source_tokens) for ex in train}
    test_keys = {(ex.op_seq, ex.source_tokens) for k, g in groups.items() if k != "train_seen" for ex in g}
    if train_keys & test_keys:
        raise ValueError("Training/evaluation input overlap")
    return train, groups, scenes


def prompt(ops: tuple[str, ...], source: tuple[str, ...], token_to_id: dict[str, int]) -> list[int]:
    tokens = ["<bos>"] + base.op_tokens(ops) + ["<sep>"] + list(source) + ["<sep>"]
    return [token_to_id[token] for token in tokens]


@torch.no_grad()
def generate_batch(model, prompts, vocab, token_to_id, max_new_tokens=12, batch_size=64, injection=None):
    """Bucket equal-length inputs so padding cannot change absolute positions."""
    model.eval()
    device = next(model.parameters()).device
    buckets = defaultdict(list)
    for index, ids in enumerate(prompts):
        buckets[len(ids)].append((index, ids))
    output = [None] * len(prompts)
    eos_id = token_to_id["<eos>"]
    for bucket in buckets.values():
        for offset in range(0, len(bucket), batch_size):
            chunk = bucket[offset:offset + batch_size]
            sequence = torch.tensor([ids for _, ids in chunk], dtype=torch.long, device=device)
            ended = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            generated = [[] for _ in chunk]
            for _ in range(max_new_tokens):
                logits, _ = model(sequence, injection=injection)
                next_ids = logits[:, -1].argmax(dim=-1)
                for row, (next_id, was_ended) in enumerate(zip(next_ids.tolist(), ended.tolist())):
                    if not was_ended:
                        generated[row].append(next_id)
                ended |= next_ids.eq(eos_id)
                if bool(ended.all()):
                    break
                next_ids = torch.where(ended, eos_id, next_ids)
                sequence = torch.cat([sequence, next_ids[:, None]], dim=1)
            for (index, _), tokens in zip(chunk, generated):
                output[index] = base.decode_tokens(tokens, vocab, eos_id)
    return output


class Generator:
    def __init__(self, model, vocab, token_to_id, batch_size, max_new_tokens):
        self.model, self.vocab, self.token_to_id = model, vocab, token_to_id
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens
        self.cache = {}

    def __call__(self, requests):
        missing = list(dict.fromkeys(key for key in requests if key not in self.cache))
        if missing:
            generated = generate_batch(
                self.model, [prompt(ops, src, self.token_to_id) for ops, src in missing],
                self.vocab, self.token_to_id, self.max_new_tokens, self.batch_size,
            )
            self.cache.update(zip(missing, generated))
        return [self.cache[key] for key in requests]


def behavior_audit(generator, groups):
    rows = []
    for family, examples in groups.items():
        outputs = generator([(ex.op_seq, ex.source_tokens) for ex in examples])
        for ex, output in zip(examples, outputs):
            rows.append(dict(family=family, scene_id=ex.scene_id, op_name=ex.op_name,
                             source_text=base.token_text(ex.source_tokens),
                             target_text=base.token_text(ex.target_tokens),
                             generated_text=base.token_text(output), exact_match=int(output == ex.target_tokens)))
    return pd.DataFrame(rows)


def composition_audit(generator, scenes):
    rows = []
    sources = [base.render_state(state) for _, state in scenes]
    for a, b in PAIRS:
        first_a = generator([((a,), src) for src in sources])
        first_b = generator([((b,), src) for src in sources])
        direct_ab = generator([((a, b), src) for src in sources])
        direct_ba = generator([((b, a), src) for src in sources])
        # The second command consumes the actual first output, including mistakes.
        seq_ab = generator([((b,), out) for out in first_a])
        seq_ba = generator([((a,), out) for out in first_b])
        oracle_a = [base.render_state(base.apply_op(state, a)) for _, state in scenes]
        oracle_b = [base.render_state(base.apply_op(state, b)) for _, state in scenes]
        oracle_ab = generator([((b,), out) for out in oracle_a])
        oracle_ba = generator([((a,), out) for out in oracle_b])
        for index, (scene_id, state) in enumerate(scenes):
            target = base.render_state(base.apply_ops(state, (a, b)))
            if target != base.render_state(base.apply_ops(state, (b, a))):
                raise ValueError("This protocol only tests externally commuting operations")
            row = dict(scene_id=scene_id, pair=a + b,
                       pair_seen_in_training=int((a, b) in base.TRAIN_PAIR_OPS),
                       source_text=base.token_text(sources[index]), target_text=base.token_text(target),
                       first_a_text=base.token_text(first_a[index]), first_b_text=base.token_text(first_b[index]),
                       first_a_correct=int(first_a[index] == oracle_a[index]),
                       first_b_correct=int(first_b[index] == oracle_b[index]))
            for name, outputs in (("direct_ab", direct_ab), ("direct_ba", direct_ba),
                                  ("sequential_ab", seq_ab), ("sequential_ba", seq_ba),
                                  ("oracle_ab", oracle_ab), ("oracle_ba", oracle_ba)):
                row[name + "_text"] = base.token_text(outputs[index])
                row[name + "_correct"] = int(outputs[index] == target)
            for name, left, right in (("direct", direct_ab, direct_ba), ("sequential", seq_ab, seq_ba)):
                row[name + "_order_agreement"] = int(left[index] == right[index])
                row[name + "_both_correct"] = int(left[index] == target and right[index] == target)
            rows.append(row)
    return pd.DataFrame(rows)


def representation_data(scenes):
    train, test, pairs = base.representation_examples()
    selected = {sid for sid, _ in scenes}
    test = [ex for ex in test if ex.scene_id in selected]
    pairs = [ex for ex in pairs if ex.scene_id in selected]
    tokens = sorted({tokens for ex in train + test + pairs for tokens in (ex.source_tokens, ex.target_tokens)})
    index = {tokens: i for i, tokens in enumerate(tokens)}
    return train, test, tokens, index


@torch.no_grad()
def extract_hidden(model, tokens, token_to_id, batch_size):
    model.eval()
    device = next(model.parameters()).device
    layers = list(range(model.n_layers + 1))
    collected = {(mode, layer): [] for mode in READOUTS for layer in layers}
    prompts = [prompt((), row, token_to_id) for row in tokens]
    for offset in range(0, len(prompts), batch_size):
        ids, lengths = base.pad_batch(prompts[offset:offset + batch_size], token_to_id["<pad>"])
        ids, lengths = ids.to(device), lengths.to(device)
        _, hidden = model(ids, return_hidden=True)
        positions = torch.arange(ids.shape[1], device=device)[None, :]
        hidden[0] = model.token_embedding(ids) + model.position_embedding(positions)
        for layer, values in hidden.items():
            last = values[torch.arange(len(ids), device=device), lengths - 1]
            # Exclude BOS, operator, separators and padding from the content mean.
            mask = (positions >= 3) & (positions < lengths[:, None] - 1)
            pooled = (values * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1)[:, None]
            collected[("last_sep", layer)].append(last.cpu().numpy())
            collected[("content_mean", layer)].append(pooled.cpu().numpy())
    return {f"{mode}_layer_{layer}": np.concatenate(parts) for (mode, layer), parts in collected.items()}


def cosine_rows(left, right):
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = norms > 1e-10
    if not valid.any():
        return float("nan")
    return float(np.mean(np.sum(left[valid] * right[valid], axis=1) / norms[valid]))


def fit_audit(hidden, train, test, index, seed):
    rows = []
    for readout, values in hidden.items():
        values = values.astype(np.float64)
        centroids = {}
        data = {}
        for op in base.OPS:
            tr = [ex for ex in train if ex.op_seq == (op,)]
            te = [ex for ex in test if ex.op_seq == (op,)]
            arrays = [np.vstack([values[index[getattr(ex, attr)]] for ex in group])
                      for group in (tr, te) for attr in ("source_tokens", "target_tokens")]
            data[op] = arrays
            centroids[op] = (arrays[1] - arrays[0]).mean(axis=0)
        for op_index, op in enumerate(base.OPS):
            x_train, y_train, x_test, y_test = data[op]
            mean_target = y_train.mean(axis=0)
            rng = np.random.default_rng(seed + 3100 + op_index)
            shuffled = y_train[rng.permutation(len(y_train))]
            random_vector = rng.normal(size=values.shape[1])
            random_vector *= np.linalg.norm(centroids[op]) / max(np.linalg.norm(random_vector), 1e-12)
            predictions = {
                "identity": x_test,
                "target_mean": np.broadcast_to(mean_target, y_test.shape),
                "additive": x_test + centroids[op],
                "wrong_additive": x_test + centroids[base.OPS[(op_index + 1) % len(base.OPS)]],
                "random_additive": x_test + random_vector,
            }
            for name, targets in (("", y_train), ("shuffled_", shuffled)):
                for affine in (False, True):
                    weights, is_affine = base.fit_ridge_map(x_train, targets, 1.0, affine)
                    predictions[name + ("affine" if affine else "linear")] = base.apply_ridge_map(x_test, weights, is_affine)
            target_variance = float(np.mean(np.sum((y_test - mean_target) ** 2, axis=1)))
            for method, prediction in predictions.items():
                mse = float(np.mean(np.sum((prediction - y_test) ** 2, axis=1)))
                rows.append(dict(readout=readout, operation=op, method=method,
                                 n_train=len(x_train), n_sources=len(x_test),
                                 target_variance=target_variance, raw_cosine=cosine_rows(prediction, y_test),
                                 centered_cosine=cosine_rows(prediction - mean_target, y_test - mean_target),
                                 relative_rmse=np.sqrt(mse / target_variance) if target_variance > 1e-12 else np.nan))
    return pd.DataFrame(rows)


def causal_audit(model, tokens, token_to_id, hidden, index, train, test, args, seed):
    values = hidden[f"last_sep_layer_{args.layers}"]
    centroids = {}
    for op in base.OPS:
        examples = [ex for ex in train if ex.op_seq == (op,)]
        centroids[op] = np.mean([values[index[ex.target_tokens]] - values[index[ex.source_tokens]] for ex in examples], axis=0)
    rows = []
    for op_index, op in enumerate(base.OPS):
        examples = [ex for ex in test if ex.op_seq == (op,)]
        rng = np.random.default_rng(seed + 4200 + op_index)
        random_vector = rng.normal(size=args.d_model)
        random_vector *= np.linalg.norm(centroids[op]) / max(np.linalg.norm(random_vector), 1e-12)
        conditions = dict(none=None, target_vector=centroids[op], negative_vector=-centroids[op],
                          wrong_vector=centroids[base.OPS[(op_index + 1) % len(base.OPS)]], random_norm=random_vector)
        for condition, vector in conditions.items():
            injection = None if vector is None else {args.layers: torch.tensor(vector, dtype=torch.float32)}
            generated = generate_batch(model, [prompt((), ex.source_tokens, token_to_id) for ex in examples],
                                       tokens, token_to_id, args.max_new_tokens, args.eval_batch_size, injection)
            for ex, output in zip(examples, generated):
                rows.append(dict(scene_id=ex.scene_id, operation=op, condition=condition,
                                 target_text=base.token_text(ex.target_tokens), generated_text=base.token_text(output),
                                 exact_match=int(output == ex.target_tokens), identity_match=int(output == ex.source_tokens)))
    return pd.DataFrame(rows)


def save_checkpoint(path, model, optimizer, rng, config, seed, step):
    torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                    torch_rng_state=torch.get_rng_state(), batch_rng_state=rng.getstate(),
                    config=config, seed=seed, step=step), path)


def bootstrap_rates(frame, group_columns, value_column="exact_match", draws=1000):
    """Crossed bootstrap: resample scene IDs and seeds, never individual generations."""
    output = []
    rng = np.random.default_rng(70419)
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        matrix = group.pivot_table(index="scene_id", columns="seed", values=value_column, aggfunc="mean")
        if matrix.isna().any().any():
            raise ValueError("Unbalanced seed-by-scene evaluation")
        values = matrix.to_numpy()
        scene_draws = rng.integers(len(matrix), size=(draws, len(matrix)))
        seed_draws = rng.integers(len(matrix.columns), size=(draws, len(matrix.columns)))
        estimates = values[scene_draws[:, :, None], seed_draws[:, None, :]].mean(axis=(1, 2))
        output.append(dict(zip(group_columns, keys)) | dict(
            rate=float(values.mean()), n_rows=len(group), n_sources=len(matrix), n_seeds=len(matrix.columns),
            seed_std=float(np.std(values.mean(axis=0), ddof=1)) if len(matrix.columns) > 1 else 0.0,
            ci_low=float(np.quantile(estimates, 0.025)), ci_high=float(np.quantile(estimates, 0.975))))
    return pd.DataFrame(output)


def finish_reports(out_dir):
    csv_dir = out_dir / "csv"
    behavior = pd.read_csv(csv_dir / "behavior.csv")
    composition = pd.read_csv(csv_dir / "composition.csv")
    causal = pd.read_csv(csv_dir / "causal.csv")
    fits = pd.read_csv(csv_dir / "representation_fit.csv")
    behavior_summary = bootstrap_rates(behavior, ["checkpoint_step", "family"])
    modes = [key for key in composition if key.endswith("_correct") or key.endswith("_order_agreement")]
    comp_long = composition.melt(id_vars=["seed", "checkpoint_step", "scene_id", "pair", "pair_seen_in_training"],
                                 value_vars=modes, var_name="metric", value_name="exact_match")
    composition_summary = bootstrap_rates(comp_long, ["checkpoint_step", "pair_seen_in_training", "metric"])
    causal_summary = bootstrap_rates(causal, ["checkpoint_step", "operation", "condition"])
    for name, frame in (("behavior_summary", behavior_summary), ("composition_summary", composition_summary), ("causal_summary", causal_summary)):
        frame.to_csv(csv_dir / (name + ".csv"), index=False)
    final_step = int(behavior.checkpoint_step.max())
    final_behavior = behavior_summary[behavior_summary.checkpoint_step == final_step]
    final_comp = composition_summary[(composition_summary.checkpoint_step == final_step) & ~composition_summary.metric.str.startswith("first_")]
    final_causal = causal_summary[(causal_summary.checkpoint_step == final_step) & (causal_summary.condition == "target_vector")]
    fit_summary = fits.groupby(["checkpoint_step", "readout", "method"], as_index=False)[["raw_cosine", "centered_cosine", "relative_rmse"]].mean()
    fit_summary.to_csv(csv_dir / "representation_summary.csv", index=False)
    fit_display = fit_summary[(fit_summary.checkpoint_step.isin([0, final_step])) &
                              (fit_summary.readout == "last_sep_layer_2") &
                              (fit_summary.method.isin(["target_mean", "linear", "shuffled_linear", "additive"]))]
    lines = ["# GLT-BUILD-01A: Corrected Composition Audit", "",
             "Completed exploratory follow-up; three training seeds are not a definitive population estimate.", "",
             "## Protocol", "",
             "The training grammar, scene split, operation examples and architecture match the 2026-09-11 pilot. "
             "Default evaluation uses every held-out scene. Training sanity checks cover all eight subjects. "
             "All five operations commute in the external generator. No endpoint-equality measurement is counted as learned commutativity.", "",
             "AB means apply A first, then B. Sequential AB feeds the generated A output into command B; "
             "oracle AB feeds the correct A intermediate into B. Transformed sources were not training inputs, "
             "so sequential/oracle failure also tests this input-distribution boundary. Order agreement alone can reflect two wrong outputs.", "",
             "## Behavioral Exact Match", "", base.df_to_markdown(final_behavior.drop(columns="checkpoint_step")), "",
             "## Actual Model Composition", "", base.df_to_markdown(final_comp.drop(columns="checkpoint_step")), "",
             "## Mean-Delta Interventions", "", base.df_to_markdown(final_causal.drop(columns="checkpoint_step")), "",
             "Interventions repeat at the last position of every full-prefix forward pass, at the final layer, gain 1. "
             "Only this intervention recipe is tested. All control rows are in csv/causal_summary.csv.", "",
             "## Initialization And Geometry Controls", "", base.df_to_markdown(fit_display), "",
             "Raw cosine must be compared to checkpoint zero and the per-operation mean-target baseline. "
             "Centered cosine subtracts the training target mean. Relative RMSE is normalized by the mean-target prediction error; "
             "1 equals that baseline and lower is better. Undefined centered metrics for constant embedding readouts are left blank. "
             "Layer 0 is token-plus-position embedding only; content_mean excludes prefix tokens and separators. "
             "The shuffled linear/affine maps use one fixed training-target permutation per seed/operation, a diagnostic control, not a p-value. "
             "Both pooling choices are fixed in advance; no layer, gain or regularizer selection uses evaluation scores.", "",
             "## Uncertainty And Artifacts", "",
             "Rates average scenes equally and retain repeated seeds within scene. Intervals are 95% percentile intervals "
             "from 1000 crossed scene/seed bootstrap draws. With only three seeds or all-zero/all-one observations they "
             "can be narrow or degenerate; they do not assert population certainty. The scene holdout uses known words and "
             "one grammar; it is not a lexical or template holdout.", "",
             "Weights, optimizer and RNG states are in local checkpoints/. Sentence readouts (both modes, every layer including "
             "layer 0) are in local hidden/ and indexed by csv/representation_sentences.csv. These are selected readouts, "
             "not full token activation tensors. artifacts.json records sizes and SHA-256 hashes. The script reconstructs "
             "them from metadata.json; binary artifacts are excluded from Git to keep the repository compact.", "",
             "![Behavior over training](figures/behavior_over_training.png)", "",
             "![Geometry controls](figures/geometry_controls.png)", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 5))
    for family, group in behavior_summary.groupby("family"):
        ax.plot(group.checkpoint_step, group.rate, marker="o", label=family)
    ax.set(xlabel="Training step", ylabel="Exact-match rate", ylim=(-0.04, 1.04))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures/behavior_over_training.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ("raw_cosine", "relative_rmse")):
        for method, group in fit_summary[(fit_summary.readout == "last_sep_layer_2") &
                                         fit_summary.method.isin(["target_mean", "linear", "shuffled_linear", "additive"])].groupby("method"):
            ax.plot(group.checkpoint_step, group[metric], marker="o", label=method)
        ax.set(xlabel="Training step", ylabel=metric)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures/geometry_controls.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--checkpoints", default="0,100,250,500,800")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--scene-limit-per-subject", type=int, default=0, help="Debug only; 0 evaluates all held-out scenes")
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    checkpoints = sorted({0, args.steps} | {int(step) for step in args.checkpoints.split(",")})
    if not seeds or len(set(seeds)) != len(seeds) or args.steps < 0 or min(checkpoints) < 0 or max(checkpoints) > args.steps:
        parser.error("Require unique seeds and checkpoints within [0, steps]")
    if min(args.d_model, args.layers, args.heads, args.batch_size, args.eval_batch_size, args.threads, args.max_new_tokens) < 1 or args.scene_limit_per_subject < 0:
        parser.error("Invalid model, batch, thread or evaluation dimensions")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("csv", "figures", "checkpoints", "hidden"):
        (out_dir / directory).mkdir()
    status = dict(status="running", phase="initializing", pid=os.getpid(), started_at=base.now(), completed_seeds=0, total_seeds=len(seeds))

    def update(**values):
        status.update(values, updated_at=base.now())
        write_json(out_dir / "run_status.json", status)

    update()
    try:
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        train, groups, scenes = build_dataset(args.scene_limit_per_subject)
        token_to_id, vocab = base.build_vocab(train, groups)
        base.save_examples(out_dir, train, groups)
        rep_train, rep_test, sentences, sentence_index = representation_data(scenes)
        pd.DataFrame([dict(row_id=i, sentence=base.token_text(tokens), tokens_json=json.dumps(tokens),
                           prompt_length=len(tokens) + 4) for i, tokens in enumerate(sentences)]).to_csv(out_dir / "csv/representation_sentences.csv", index=False)
        coverage = [dict(scene_id=sid, subject=state.subject, verb=base.VERBS[state.verb_idx].base,
                         object=state.object) for sid, state in scenes]
        pd.DataFrame(coverage).to_csv(out_dir / "csv/evaluation_scenes.csv", index=False)
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        config = vars(args) | dict(protocol=PROTOCOL, started_at=status["started_at"], seeds=seeds, checkpoints=checkpoints,
                                  learning_rate=0.003, weight_decay=0.01, dropout=0.0, ridge_alpha=1.0,
                                  bootstrap_draws=1000, device="cpu", torch_version=str(torch.__version__),
                                  numpy_version=np.__version__, python_version=sys.version, vocab=vocab,
                                  train_examples=len(train), train_scenes=len({ex.scene_id for ex in train}),
                                  eval_scenes=len(scenes), max_seq_len=64, git_head=git_head,
                                  source_sha256={Path(path).name: sha256(Path(path)) for path in (__file__, base.__file__)})
        write_json(out_dir / "metadata.json", config)
        write_json(out_dir / "vocab.json", vocab)
        encoded = [base.encode_example(ex, token_to_id) for ex in train]
        artifact_rows = []
        for seed_index, seed in enumerate(seeds):
            base.set_seed(seed)
            rng = random.Random(seed)
            model = base.TinyDecoderOnlyTransformer(len(vocab), args.d_model, args.heads, args.layers, 64, 0.0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss_value = None
            for step in range(args.steps + 1):
                if step:
                    model.train()
                    inputs, labels = base.sample_batch(encoded, args.batch_size, token_to_id["<pad>"], rng)
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = model(inputs)
                    loss = loss_fn(logits.reshape(-1, len(vocab)), labels.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    loss_value = float(loss.item())
                if step % 25 == 0:
                    update(phase="training", current_seed=seed, current_step=step, latest_loss=loss_value)
                if step not in checkpoints:
                    continue
                update(phase="checkpoint", current_seed=seed, current_step=step, latest_loss=loss_value)
                stem = f"seed_{seed}_step_{step:06d}"
                checkpoint_path = out_dir / "checkpoints" / (stem + ".pt")
                save_checkpoint(checkpoint_path, model, optimizer, rng, config, seed, step)
                generator = Generator(model, vocab, token_to_id, args.eval_batch_size, args.max_new_tokens)
                update(phase="behavior")
                behavior = behavior_audit(generator, groups)
                update(phase="composition")
                composition = composition_audit(generator, scenes)
                update(phase="representations")
                hidden = extract_hidden(model, sentences, token_to_id, args.eval_batch_size)
                hidden_path = out_dir / "hidden" / (stem + ".npz")
                np.savez_compressed(hidden_path, **hidden)
                fits = fit_audit(hidden, rep_train, rep_test, sentence_index, seed)
                update(phase="causal")
                causal = causal_audit(model, vocab, token_to_id, hidden, sentence_index, rep_train, rep_test, args, seed)
                for name, frame in (("behavior", behavior), ("composition", composition), ("representation_fit", fits), ("causal", causal)):
                    frame.insert(0, "checkpoint_step", step)
                    frame.insert(0, "seed", seed)
                    path = out_dir / "csv" / (name + ".csv")
                    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
                training_path = out_dir / "csv/training_log.csv"
                pd.DataFrame([dict(seed=seed, checkpoint_step=step, loss=loss_value)]).to_csv(training_path, mode="a", header=not training_path.exists(), index=False)
                for path in (checkpoint_path, hidden_path):
                    artifact_rows.append(dict(path=path.relative_to(out_dir).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
                write_json(out_dir / "artifacts.json", artifact_rows)
                print(f"seed={seed} step={step} complete; saved weights, hidden readouts and CSVs", flush=True)
            update(completed_seeds=seed_index + 1)
        update(phase="reporting")
        finish_reports(out_dir)
        update(status="finished", phase="complete", finished_at=base.now())
    except BaseException as error:
        update(status="failed", phase="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
