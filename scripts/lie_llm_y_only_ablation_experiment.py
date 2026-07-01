"""
Ablation: is the delta-vector classifier doing something the target embedding
alone (or source embedding alone) couldn't already do?

This script reuses the exact same dataset construction, embedding extraction,
and full-semantic-holdout train/test split as
`lie_llm_full_semantic_holdout_experiment.py`, but trains the same linear
classifiers (logreg, linear SVC) on four different representations of each
(source, target) pair:

  - "delta"   : target_emb - source_emb           (original experiment)
  - "y_only"  : target_emb                        (style-of-y baseline)
  - "x_only"  : source_emb                        (style-of-x baseline)
  - "concat"  : [source_emb, target_emb]          (upper-bound baseline)

Rationale (the standard endpoint-control objection): if "y_only" performs comparably
to "delta", the classifier may simply be learning to recognize the surface
style/register of the transformed sentence (e.g. "questions sound like
questions") rather than anything about the *transformation* (the geometric
relationship between x and y). "x_only" should be near chance, since the
source sentences are template-matched across classes and carry no information
about which transformation will be applied to them. "concat" gives an upper
bound: if delta already matches concat, the subtraction is not losing
information relative to having both vectors available.

Output: results/ablation_y_vs_delta_summary.csv with one row per
(model, classifier, representation).

Requires the same environment as lie_llm_full_semantic_holdout_experiment.py
(torch, transformers, scikit-learn, pandas, numpy). Vector extraction is
cached under OUT_DIR exactly as in that script, so if you have already run
the full semantic holdout experiment for a model, this script will reuse the
cached `vectors_<model>.npy` file instead of recomputing embeddings.
"""

import os
import gc
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Reuse dataset construction + embedding extraction from the main experiment.
from lie_llm_full_semantic_holdout_experiment import (
    build_dataset,
    index_prompts,
    get_vectors,
    OUT_DIR,
    MODELS,
)

ABLATION_OUT = "results/ablation_y_vs_delta_summary.csv"


def run_classifiers(X, y_labels, split):
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

    out = []
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        out.append((clf_name, acc, random_baseline, len(y_train), len(y_test)))
    return out


def main():
    pair_df = build_dataset()
    prompts, pair_df = index_prompts(pair_df)

    labels = pair_df["class"].astype(str).to_numpy()
    split = pair_df["split"].astype(str).to_numpy()
    src_idx = pair_df["source_idx"].to_numpy()
    tgt_idx = pair_df["target_idx"].to_numpy()

    all_rows = []

    for model_name in MODELS:
        print(f"\n=== MODEL: {model_name} ===")
        vectors = get_vectors(model_name, prompts)

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

        for rep_name, X in representations.items():
            results = run_classifiers(X, labels, split)
            for clf_name, acc, baseline, n_train, n_test in results:
                print(f"  {rep_name:8s} {clf_name:10s} acc={acc:.4f}")
                all_rows.append({
                    "model": model_name,
                    "representation": rep_name,
                    "classifier": clf_name,
                    "accuracy": float(acc),
                    "random_baseline": float(baseline),
                    "n_train": int(n_train),
                    "n_test": int(n_test),
                })

        del vectors, X_src, X_tgt, X_delta, X_concat
        gc.collect()

    summary = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(ABLATION_OUT), exist_ok=True)
    summary.to_csv(ABLATION_OUT, index=False)
    print("\nSaved:", ABLATION_OUT)
    print(summary.pivot_table(index=["model", "classifier"], columns="representation", values="accuracy"))


if __name__ == "__main__":
    main()
