import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

from run_upat_large import UPATAudit
from upat_dataset import UPATDataset, UPATDatasetConfig


OUT_DIR = Path("results/experiments/upat_large_results")
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run held-out anchor alignment-size curves for UPAT cross-model Procrustes transfer."
    )
    parser.add_argument(
        "--sizes",
        default="25,50,100,250,500,1000",
        help="Comma-separated held-out alignment anchor sizes.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Repeats per source-target direction and alignment size.",
    )
    parser.add_argument(
        "--anchor-target-count",
        type=int,
        default=1200,
        help="Number of unique held-out anchor texts to generate before subsampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=31415,
        help="Base random seed.",
    )
    return parser.parse_args()


def collect_texts(df):
    return set(df["source"]) | set(df["target"])


def build_heldout_anchor_texts(main_texts, target_count, seed):
    anchor_texts = []
    seen = set(main_texts)

    for offset in range(1000):
        cfg = UPATDatasetConfig(
            train_templates=200,
            test_templates=200,
            seed_train=seed + 2 * offset,
            seed_test=seed + 2 * offset + 1,
        )
        df = UPATDataset(cfg).build()

        for text in sorted(collect_texts(df)):
            if text in seen:
                continue
            seen.add(text)
            anchor_texts.append(text)
            if len(anchor_texts) >= target_count:
                return anchor_texts

    raise RuntimeError(
        f"Only generated {len(anchor_texts)} held-out anchor texts; requested {target_count}."
    )


def fit_procrustes(source_anchor, target_anchor, source_all, target_all):
    source_mean = source_anchor.mean(axis=0, keepdims=True)
    target_mean = target_anchor.mean(axis=0, keepdims=True)

    source_anchor_c = source_anchor - source_mean
    target_anchor_c = target_anchor - target_mean
    source_all_c = source_all - source_mean
    target_all_c = target_all - target_mean

    min_dim = min(
        source_anchor_c.shape[1],
        target_anchor_c.shape[1],
        source_all_c.shape[1],
        target_all_c.shape[1],
    )
    source_anchor_c = source_anchor_c[:, :min_dim]
    target_anchor_c = target_anchor_c[:, :min_dim]
    source_all_c = source_all_c[:, :min_dim]
    target_all_c = target_all_c[:, :min_dim]

    u, _, vt = svd(source_anchor_c.T @ target_anchor_c, full_matrices=False, check_finite=False)
    r = u @ vt
    return source_all_c @ r, target_all_c


def evaluate_direction(audit, train_model, test_model, source_anchor, target_anchor, seed):
    source_sp = audit.spaces[train_model]
    target_sp = audit.spaces[test_model]

    source_aligned, target_aligned = fit_procrustes(
        source_anchor=source_anchor,
        target_anchor=target_anchor,
        source_all=source_sp["raw"],
        target_all=target_sp["raw"],
    )

    source_deltas = normalize(audit.make_raw_deltas(train_model, source_aligned))
    target_deltas = normalize(audit.make_raw_deltas(test_model, target_aligned))

    pred = audit.fit_predict(
        source_deltas[audit.train_mask],
        audit.y[audit.train_mask],
        target_deltas[audit.test_mask],
        seed=seed,
    )

    return {
        "acc": accuracy_score(audit.y[audit.test_mask], pred),
        "f1": f1_score(audit.y[audit.test_mask], pred, average="macro"),
    }


def run_curve(audit, anchor_spaces, anchor_texts, sizes, repeats, seed):
    observed = pd.read_csv(CSV_DIR / "cross_model_procrustes.csv")
    observed = observed[observed["train_model"] != observed["test_model"]]
    rng = np.random.default_rng(seed)
    rows = []

    for obs in observed.itertuples(index=False):
        train_model = obs.train_model
        test_model = obs.test_model
        print(f"\nHeld-out alignment: {train_model} -> {test_model}", flush=True)

        source_anchor_all = anchor_spaces[train_model]
        target_anchor_all = anchor_spaces[test_model]

        for size in sizes:
            real_size = min(size, len(anchor_texts))
            for repeat in range(repeats):
                sample = rng.choice(len(anchor_texts), size=real_size, replace=False)
                metrics = evaluate_direction(
                    audit=audit,
                    train_model=train_model,
                    test_model=test_model,
                    source_anchor=source_anchor_all[sample],
                    target_anchor=target_anchor_all[sample],
                    seed=seed + 1000 * repeat + real_size,
                )
                rows.append({
                    "train_model": train_model,
                    "test_model": test_model,
                    "alignment_size": real_size,
                    "repeat": repeat,
                    "heldout_anchor_pool_size": len(anchor_texts),
                    "observed_full_anchor_f1": obs.aligned_f1,
                    "observed_raw_f1": obs.raw_f1,
                    "heldout_acc": metrics["acc"],
                    "heldout_f1": metrics["f1"],
                    "heldout_minus_raw_f1": metrics["f1"] - obs.raw_f1,
                    "heldout_minus_full_anchor_f1": metrics["f1"] - obs.aligned_f1,
                })

            print(f"  size {real_size}: {repeats}/{repeats}", flush=True)

        pd.DataFrame(rows).to_csv(CSV_DIR / "heldout_alignment_curve_raw.partial.csv", index=False)

    raw_df = pd.DataFrame(rows)
    summary_df = (
        raw_df
        .groupby("alignment_size")
        .agg(
            mean_f1=("heldout_f1", "mean"),
            std_f1=("heldout_f1", "std"),
            mean_acc=("heldout_acc", "mean"),
            mean_full_anchor_f1=("observed_full_anchor_f1", "mean"),
            mean_raw_f1=("observed_raw_f1", "mean"),
            mean_gap_to_full_anchor=("heldout_minus_full_anchor_f1", "mean"),
            mean_gain_over_raw=("heldout_minus_raw_f1", "mean"),
            n=("heldout_f1", "count"),
        )
        .reset_index()
    )
    direction_summary_df = (
        raw_df
        .groupby(["train_model", "test_model", "alignment_size"])
        .agg(
            mean_f1=("heldout_f1", "mean"),
            std_f1=("heldout_f1", "std"),
            mean_acc=("heldout_acc", "mean"),
            observed_full_anchor_f1=("observed_full_anchor_f1", "first"),
            observed_raw_f1=("observed_raw_f1", "first"),
            mean_gap_to_full_anchor=("heldout_minus_full_anchor_f1", "mean"),
            mean_gain_over_raw=("heldout_minus_raw_f1", "mean"),
            n=("heldout_f1", "count"),
        )
        .reset_index()
    )

    raw_df.to_csv(CSV_DIR / "heldout_alignment_curve_raw.csv", index=False)
    summary_df.to_csv(CSV_DIR / "heldout_alignment_curve_summary.csv", index=False)
    direction_summary_df.to_csv(CSV_DIR / "heldout_alignment_curve_by_direction.csv", index=False)
    (CSV_DIR / "heldout_alignment_curve_raw.partial.csv").unlink(missing_ok=True)
    return raw_df, summary_df, direction_summary_df


def plot_curve(summary_df, direction_summary_df):
    plt.figure(figsize=(8, 5))
    ordered = summary_df.sort_values("alignment_size")
    plt.errorbar(
        ordered["alignment_size"],
        ordered["mean_f1"],
        yerr=ordered["std_f1"],
        marker="o",
        capsize=4,
        label="held-out anchor Procrustes",
    )
    plt.plot(
        ordered["alignment_size"],
        ordered["mean_raw_f1"],
        linestyle="--",
        label="raw cross-model mean",
    )
    plt.plot(
        ordered["alignment_size"],
        ordered["mean_full_anchor_f1"],
        linestyle=":",
        label="full-anchor Procrustes mean",
    )
    plt.xscale("log")
    plt.xlabel("Held-out alignment anchor texts")
    plt.ylabel("Macro F1")
    plt.title("UPAT held-out alignment-size curve")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_heldout_alignment_size_curve.png", dpi=220, bbox_inches="tight")
    plt.close()

    largest_size = direction_summary_df["alignment_size"].max()
    largest = direction_summary_df[direction_summary_df["alignment_size"] == largest_size].copy()
    labels = [
        f"{row.train_model.split('/')[-1]}\n->{row.test_model.split('/')[-1]}"
        for row in largest.itertuples(index=False)
    ]
    x = np.arange(len(largest))
    plt.figure(figsize=(max(12, len(largest) * 0.45), 6))
    plt.bar(x, largest["observed_raw_f1"], label="raw F1")
    plt.bar(x, largest["mean_f1"], alpha=0.75, label=f"held-out alignment F1, n={largest_size}")
    plt.bar(x, largest["observed_full_anchor_f1"], alpha=0.45, label="full-anchor alignment F1")
    plt.xticks(x, labels, rotation=70, ha="right", fontsize=7)
    plt.ylabel("Macro F1")
    plt.title("UPAT held-out alignment by direction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_heldout_alignment_by_direction.png", dpi=220, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    if not sizes:
        raise ValueError("At least one alignment size is required.")

    audit = UPATAudit()
    audit.build_dataset()

    main_texts = collect_texts(audit.df)
    anchor_target_count = max(args.anchor_target_count, max(sizes))
    anchor_texts = build_heldout_anchor_texts(
        main_texts=main_texts,
        target_count=anchor_target_count,
        seed=args.seed,
    )
    pd.DataFrame({"text": anchor_texts}).to_csv(CSV_DIR / "heldout_alignment_anchor_texts.csv", index=False)
    print(f"Held-out anchor texts: {len(anchor_texts)}", flush=True)

    audit.extract_all_spaces()

    cross_path = CSV_DIR / "cross_model_procrustes.csv"
    if not cross_path.exists():
        audit.run_cross_model_procrustes()

    anchor_spaces = {}
    for model_name in audit.procrustes_models:
        print(f"\nEXTRACTING HELD-OUT ANCHORS {model_name}", flush=True)
        anchor_spaces[model_name] = audit.get_embeddings(model_name, anchor_texts)

    _, summary_df, direction_summary_df = run_curve(
        audit=audit,
        anchor_spaces=anchor_spaces,
        anchor_texts=anchor_texts,
        sizes=sizes,
        repeats=args.repeats,
        seed=args.seed,
    )
    plot_curve(summary_df, direction_summary_df)
    print(f"\nSaved: {CSV_DIR / 'heldout_alignment_curve_raw.csv'}")
    print(f"Saved: {CSV_DIR / 'heldout_alignment_curve_summary.csv'}")
    print(f"Saved: {CSV_DIR / 'heldout_alignment_curve_by_direction.csv'}")


if __name__ == "__main__":
    sys.exit(main())
