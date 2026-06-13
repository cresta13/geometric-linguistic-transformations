import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def evaluate_transfer(audit, train_model, test_model, *, pairing="matched", shuffle_labels=False, seed=0):
    source_sp = audit.spaces[train_model]
    target_sp = audit.spaces[test_model]

    common_texts = sorted(set(source_sp["texts"]) & set(target_sp["texts"]))
    source_idx = np.array([source_sp["idx"][text] for text in common_texts])
    target_idx = np.array([target_sp["idx"][text] for text in common_texts])

    rng = np.random.default_rng(seed)

    if pairing == "random_pairing":
        target_idx = rng.permutation(target_idx)
    elif pairing != "matched":
        raise ValueError(f"Unknown pairing mode: {pairing}")

    source_aligned, target_aligned = audit.align_source_to_target(
        source_anchor=source_sp["raw"][source_idx],
        target_anchor=target_sp["raw"][target_idx],
        source_all=source_sp["raw"],
        target_all=target_sp["raw"],
    )

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

        for null_type in ["random_pairing", "random_labels"]:
            f1_values = []
            acc_values = []

            for repeat in range(N_NULL):
                seed = SEED + repeat
                if null_type == "random_pairing":
                    metrics = evaluate_transfer(
                        audit,
                        train_model,
                        test_model,
                        pairing="random_pairing",
                        shuffle_labels=False,
                        seed=seed,
                    )
                else:
                    metrics = evaluate_transfer(
                        audit,
                        train_model,
                        test_model,
                        pairing="matched",
                        shuffle_labels=True,
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

    raw_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)

    raw_df.to_csv(CSV_DIR / "procrustes_null_raw.csv", index=False)
    summary_df.to_csv(CSV_DIR / "procrustes_null_summary.csv", index=False)
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
