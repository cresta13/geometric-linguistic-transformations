import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

from run_upat_large import UPATAudit


OUT_DIR = Path("results/experiments/upat_large_results")
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_NULL = int(os.getenv("UPAT_PROCRUSTES_NULL_REPEATS", "30"))
SEED = int(os.getenv("UPAT_PROCRUSTES_NULL_SEED", "1729"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UPAT cross-model Procrustes null controls."
    )
    parser.add_argument(
        "--n-null",
        type=int,
        default=N_NULL,
        help="Number of null repeats per source-target direction and null type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base random seed for null controls.",
    )
    return parser.parse_args()


def make_direction_context(audit, train_model, test_model):
    source_sp = audit.spaces[train_model]
    target_sp = audit.spaces[test_model]

    common_texts = sorted(set(source_sp["texts"]) & set(target_sp["texts"]))
    source_idx = np.array([source_sp["idx"][text] for text in common_texts])
    target_idx = np.array([target_sp["idx"][text] for text in common_texts])

    min_dim = min(source_sp["raw"].shape[1], target_sp["raw"].shape[1])
    source_anchor = source_sp["raw"][source_idx, :min_dim]
    target_anchor = target_sp["raw"][target_idx, :min_dim]
    source_all = source_sp["raw"][:, :min_dim]
    target_all = target_sp["raw"][:, :min_dim]

    source_mean = source_anchor.mean(axis=0, keepdims=True)
    target_mean = target_anchor.mean(axis=0, keepdims=True)

    source_anchor_c = source_anchor - source_mean
    target_anchor_c = target_anchor - target_mean
    source_all_c = source_all - source_mean
    target_all_c = target_all - target_mean

    return {
        "source_anchor_c": source_anchor_c,
        "target_anchor_c": target_anchor_c,
        "source_all_c": source_all_c,
        "target_all_c": target_all_c,
    }


def apply_fast_procrustes(context, target_order=None):
    target_anchor_c = context["target_anchor_c"]
    if target_order is not None:
        target_anchor_c = target_anchor_c[target_order]

    u, _, vt = svd(context["source_anchor_c"].T @ target_anchor_c, full_matrices=False, check_finite=False)
    r = u @ vt
    return context["source_all_c"] @ r, context["target_all_c"]


def apply_fast_random_orthogonal(context, seed):
    rng = np.random.default_rng(seed)
    dim = context["source_all_c"].shape[1]
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return context["source_all_c"] @ q, context["target_all_c"]


def evaluate_transfer(audit, train_model, test_model, context, *, pairing="matched", shuffle_labels=False, random_orthogonal=False, seed=0):
    rng = np.random.default_rng(seed)

    if pairing == "random_pairing":
        target_order = rng.permutation(len(context["target_anchor_c"]))
    elif pairing != "matched":
        raise ValueError(f"Unknown pairing mode: {pairing}")
    else:
        target_order = None

    if random_orthogonal:
        source_aligned, target_aligned = apply_fast_random_orthogonal(context, seed=seed)
    else:
        source_aligned, target_aligned = apply_fast_procrustes(context, target_order=target_order)

    source_deltas = normalize(audit.make_raw_deltas(train_model, source_aligned))
    target_deltas = normalize(audit.make_raw_deltas(test_model, target_aligned))

    y_train = audit.y[audit.train_mask].copy()
    if shuffle_labels:
        y_train = rng.permutation(y_train)

    pred = audit.fit_predict(
        source_deltas[audit.train_mask],
        y_train,
        target_deltas[audit.test_mask],
        seed=seed,
    )

    return {
        "acc": accuracy_score(audit.y[audit.test_mask], pred),
        "f1": f1_score(audit.y[audit.test_mask], pred, average="macro"),
    }


def run_nulls(audit):
    observed = pd.read_csv(CSV_DIR / "cross_model_procrustes.csv")
    rows = []
    summary_rows = []

    for obs in observed.itertuples(index=False):
        train_model = obs.train_model
        test_model = obs.test_model
        if train_model == test_model:
            continue

        print(f"\nNulls: {train_model} -> {test_model}", flush=True)
        context = make_direction_context(audit, train_model, test_model)

        for null_type in ["random_pairing", "random_labels", "random_orthogonal"]:
            f1_values = []
            acc_values = []

            for repeat in range(N_NULL):
                seed = SEED + repeat
                if null_type == "random_pairing":
                    metrics = evaluate_transfer(
                        audit,
                        train_model,
                        test_model,
                        context,
                        pairing="random_pairing",
                        shuffle_labels=False,
                        seed=seed,
                    )
                else:
                    metrics = evaluate_transfer(
                        audit,
                        train_model,
                        test_model,
                        context,
                        pairing="matched",
                        shuffle_labels=null_type == "random_labels",
                        random_orthogonal=null_type == "random_orthogonal",
                        seed=seed,
                    )

                f1_values.append(metrics["f1"])
                acc_values.append(metrics["acc"])
                rows.append({
                    "train_model": train_model,
                    "test_model": test_model,
                    "null_type": null_type,
                    "repeat": repeat,
                    "null_acc": metrics["acc"],
                    "null_f1": metrics["f1"],
                    "observed_aligned_f1": obs.aligned_f1,
                    "observed_raw_f1": obs.raw_f1,
                    "observed_gain_f1": obs.gain_f1,
                })

                if (repeat + 1) % 100 == 0 or repeat + 1 == N_NULL:
                    print(f"  {null_type}: {repeat + 1}/{N_NULL}", flush=True)

            f1_arr = np.array(f1_values)
            acc_arr = np.array(acc_values)
            summary_rows.append({
                "train_model": train_model,
                "test_model": test_model,
                "null_type": null_type,
                "n_null": N_NULL,
                "observed_raw_f1": obs.raw_f1,
                "observed_aligned_f1": obs.aligned_f1,
                "observed_gain_f1": obs.gain_f1,
                "null_mean_f1": f1_arr.mean(),
                "null_std_f1": f1_arr.std(ddof=1) if len(f1_arr) > 1 else 0.0,
                "null_max_f1": f1_arr.max(),
                "null_mean_acc": acc_arr.mean(),
                "p_empirical_null_ge_observed": ((f1_arr >= obs.aligned_f1).sum() + 1) / (N_NULL + 1),
                "observed_minus_null_mean_f1": obs.aligned_f1 - f1_arr.mean(),
            })

        pd.DataFrame(rows).to_csv(CSV_DIR / "procrustes_null_raw.partial.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(CSV_DIR / "procrustes_null_summary.partial.csv", index=False)

    raw_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)

    raw_df.to_csv(CSV_DIR / "procrustes_null_raw.csv", index=False)
    summary_df.to_csv(CSV_DIR / "procrustes_null_summary.csv", index=False)
    (CSV_DIR / "procrustes_null_raw.partial.csv").unlink(missing_ok=True)
    (CSV_DIR / "procrustes_null_summary.partial.csv").unlink(missing_ok=True)
    return raw_df, summary_df


def plot_summary(summary_df):
    if summary_df.empty:
        return

    for null_type, df in summary_df.groupby("null_type"):
        labels = [
            f"{row.train_model.split('/')[-1]}\n->{row.test_model.split('/')[-1]}"
            for row in df.itertuples(index=False)
        ]
        x = np.arange(len(df))

        plt.figure(figsize=(max(12, len(df) * 0.45), 6))
        plt.bar(x, df["observed_aligned_f1"], label="observed aligned F1")
        plt.bar(x, df["null_mean_f1"], alpha=0.65, label=f"{null_type} mean F1")
        plt.xticks(x, labels, rotation=70, ha="right", fontsize=7)
        plt.ylabel("Macro F1")
        plt.title(f"UPAT Procrustes null baseline: {null_type}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"10_procrustes_null_{null_type}.png", dpi=220, bbox_inches="tight")
        plt.close()


def main():
    global N_NULL, SEED
    args = parse_args()
    N_NULL = args.n_null
    SEED = args.seed
    print(f"Procrustes null repeats: {N_NULL}; seed: {SEED}", flush=True)

    audit = UPATAudit()
    audit.build_dataset()
    audit.extract_all_spaces()

    cross_path = CSV_DIR / "cross_model_procrustes.csv"
    if not cross_path.exists():
        audit.run_cross_model_procrustes()

    _, summary_df = run_nulls(audit)
    plot_summary(summary_df)
    print(f"\nSaved: {CSV_DIR / 'procrustes_null_raw.csv'}")
    print(f"Saved: {CSV_DIR / 'procrustes_null_summary.csv'}")


if __name__ == "__main__":
    sys.exit(main())
