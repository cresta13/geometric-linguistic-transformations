"""
Multi-seed replication of the y-only / x-only / delta / concat ablation
(see lie_llm_y_only_ablation_experiment.py for the single-seed version and
PAPER.md Section 5 for context).

For each seed in SEEDS:
  - rebuild the dataset with that seed (different sampling of subjects,
    objects, templates -> different train/test split content, though the
    *structure* of the full-semantic-holdout split, i.e. which entity/template
    pools go to train vs test, is fixed by the script's design)
  - extract embeddings (cached per-seed)
  - compute delta / y_only / x_only / concat accuracies for logreg + linear SVC
  - run McNemar's test comparing delta vs y_only predictions (SVC) on the
    shared y_test items

Output:
  - results/ablation_multiseed_summary.csv         (one row per seed/model/classifier/representation)
  - results/ablation_multiseed_aggregated.csv      (mean +/- std across seeds, per model/classifier/representation)
  - results/ablation_multiseed_mcnemar.csv         (per seed/model: McNemar statistic + p-value for delta vs y_only, SVC)

Requires: torch, transformers, scikit-learn, pandas, numpy, statsmodels (for McNemar).
If statsmodels is unavailable, the script falls back to a manual McNemar
implementation (no external dependency needed beyond numpy/scipy).

Usage:
    python lie_llm_y_only_ablation_multiseed.py
    # optionally: LIE_SEEDS="42,43,44,45,46" python lie_llm_y_only_ablation_multiseed.py
"""

import os
import gc
import importlib
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEEDS = [int(s) for s in os.getenv("LIE_SEEDS", "42,43,44,45,46").split(",")]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def mcnemar_test(correct_a, correct_b):
    """
    Manual McNemar's test (with continuity correction) comparing two
    classifiers' correctness vectors (boolean arrays) on the same test items.

    Returns (statistic, p_value).
    """
    from scipy.stats import chi2

    # b = a correct, b wrong ; c = a wrong, b correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return 0.0, 1.0

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p)


def run_classifiers_with_preds(X, y_labels, split):
    train_mask = split == "train"
    test_mask = split == "test"

    X_train, y_train = X[train_mask], y_labels[train_mask]
    X_test, y_test = X[test_mask], y_labels[test_mask]

    classes = sorted(set(y_labels))
    random_baseline = 1 / len(classes)

    classifiers = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=-1),
        ),
        "linear_svc": make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", max_iter=20000),
        ),
    }

    out = {}
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        correct = (pred == y_test)
        out[clf_name] = {
            "accuracy": float(acc),
            "random_baseline": float(random_baseline),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "correct": correct,  # boolean array, same order as y_test
        }
    return out


def main():
    all_rows = []
    mcnemar_rows = []

    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED = {seed}\n{'='*60}")

        # Set seed and re-import the experiment module fresh so module-level
        # random.seed/np.random.seed/torch.manual_seed and OUT_DIR (cache dir)
        # take effect for this seed.
        os.environ["LIE_SEED"] = str(seed)
        os.environ["LIE_OUT_DIR_SUFFIX"] = f"_seed{seed}"

        if "lie_llm_full_semantic_holdout_experiment" in globals():
            mod = importlib.reload(globals()["lie_llm_full_semantic_holdout_experiment"])
        else:
            import lie_llm_full_semantic_holdout_experiment as mod
            globals()["lie_llm_full_semantic_holdout_experiment"] = mod

        # Use a per-seed cache directory so embeddings aren't reused across seeds
        # (dataset content differs per seed -> different prompts -> must recompute).
        mod.OUT_DIR = mod.OUT_DIR.rstrip("/") + f"_seed{seed}"
        os.makedirs(mod.OUT_DIR, exist_ok=True)

        pair_df = mod.build_dataset()
        prompts, pair_df = mod.index_prompts(pair_df)

        labels = pair_df["class"].astype(str).to_numpy()
        split = pair_df["split"].astype(str).to_numpy()
        src_idx = pair_df["source_idx"].to_numpy()
        tgt_idx = pair_df["target_idx"].to_numpy()

        for model_name in mod.MODELS:
            print(f"\n--- MODEL: {model_name} (seed {seed}) ---")
            vectors = mod.get_vectors(model_name, prompts)

            X_src = vectors[src_idx]
            X_tgt = vectors[tgt_idx]
            X_delta = X_tgt - X_src
            X_concat = np.concatenate([X_src, X_tgt], axis=1)

            representations = {
                "delta": X_delta,
                "y_only": X_tgt,
                "x_only": X_src,
                "concat": X_concat,
            }

            results_by_rep = {}
            for rep_name, X in representations.items():
                res = run_classifiers_with_preds(X, labels, split)
                results_by_rep[rep_name] = res
                for clf_name, r in res.items():
                    print(f"  {rep_name:8s} {clf_name:10s} acc={r['accuracy']:.4f}")
                    all_rows.append({
                        "seed": seed,
                        "model": model_name,
                        "representation": rep_name,
                        "classifier": clf_name,
                        "accuracy": r["accuracy"],
                        "random_baseline": r["random_baseline"],
                        "n_train": r["n_train"],
                        "n_test": r["n_test"],
                    })

            # McNemar: delta vs y_only, linear SVC
            correct_delta = results_by_rep["delta"]["linear_svc"]["correct"]
            correct_y = results_by_rep["y_only"]["linear_svc"]["correct"]
            stat, p = mcnemar_test(correct_delta, correct_y)
            mcnemar_rows.append({
                "seed": seed,
                "model": model_name,
                "classifier": "linear_svc",
                "delta_acc": results_by_rep["delta"]["linear_svc"]["accuracy"],
                "y_only_acc": results_by_rep["y_only"]["linear_svc"]["accuracy"],
                "diff": results_by_rep["delta"]["linear_svc"]["accuracy"]
                        - results_by_rep["y_only"]["linear_svc"]["accuracy"],
                "mcnemar_stat": stat,
                "mcnemar_p": p,
            })
            print(f"  McNemar (delta vs y_only, SVC): stat={stat:.3f} p={p:.4f}")

            del vectors, X_src, X_tgt, X_delta, X_concat, results_by_rep
            gc.collect()

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "ablation_multiseed_summary.csv"), index=False)
    print("\nSaved:", os.path.join(RESULTS_DIR, "ablation_multiseed_summary.csv"))

    agg = (
        summary.groupby(["model", "classifier", "representation"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.to_csv(os.path.join(RESULTS_DIR, "ablation_multiseed_aggregated.csv"), index=False)
    print("Saved:", os.path.join(RESULTS_DIR, "ablation_multiseed_aggregated.csv"))

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    mcnemar_df.to_csv(os.path.join(RESULTS_DIR, "ablation_multiseed_mcnemar.csv"), index=False)
    print("Saved:", os.path.join(RESULTS_DIR, "ablation_multiseed_mcnemar.csv"))

    print("\n=== Aggregated accuracy (mean ± std across seeds) ===")
    print(agg.pivot_table(index=["model", "classifier"], columns="representation", values="mean"))

    print("\n=== McNemar delta vs y_only (SVC), per seed ===")
    print(mcnemar_df)


if __name__ == "__main__":
    main()
