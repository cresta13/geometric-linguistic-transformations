import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import lie_llm_full_semantic_holdout_experiment as full


OUT_DIR = Path(os.getenv("SPOTCHECK_OUT_DIR", "track1_spotcheck_results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [m.strip() for m in os.getenv("SPOTCHECK_MODELS", "bert-large-uncased").split(",") if m.strip()]
N_BASE = int(os.getenv("SPOTCHECK_N_BASE", "100"))


def run_classifiers(X, labels, split):
    train_mask = split == "train"
    test_mask = split == "test"
    classifiers = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=-1)),
        "linear_svc": make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", max_iter=20000)),
    }
    rows = []
    for clf_name, clf in classifiers.items():
        clf.fit(X[train_mask], labels[train_mask])
        pred = clf.predict(X[test_mask])
        rows.append({
            "classifier": clf_name,
            "accuracy": float(accuracy_score(labels[test_mask], pred)),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        })
    return rows


def main():
    full.OUT_DIR = str(OUT_DIR)
    full.N_BASE = N_BASE
    os.makedirs(full.OUT_DIR, exist_ok=True)

    pair_df = full.build_dataset()
    prompts, pair_df = full.index_prompts(pair_df)
    pair_df.to_csv(OUT_DIR / "transformation_pairs.csv", index=False)
    pd.DataFrame({"prompt_id": range(len(prompts)), "prompt": prompts}).to_csv(OUT_DIR / "prompts.csv", index=False)

    labels = pair_df["class"].astype(str).to_numpy()
    split = pair_df["split"].astype(str).to_numpy()
    src_idx = pair_df["source_idx"].to_numpy()
    tgt_idx = pair_df["target_idx"].to_numpy()

    rows = []
    for model_name in MODELS:
        print(f"\n=== MODEL: {model_name} ===")
        vectors = full.get_vectors(model_name, prompts)
        x_src = vectors[src_idx]
        x_tgt = vectors[tgt_idx]
        reps = {
            "x_only": x_src,
            "y_only": x_tgt,
            "delta": x_tgt - x_src,
            "concat": np.concatenate([x_src, x_tgt], axis=1),
        }
        for rep_name, X in reps.items():
            for row in run_classifiers(X, labels, split):
                row.update({
                    "model": model_name,
                    "representation": rep_name,
                    "random_baseline": 1 / len(set(labels)),
                    "n_base": N_BASE,
                })
                print(model_name, rep_name, row["classifier"], f"{row['accuracy']:.4f}")
                rows.append(row)
        del vectors, reps
        gc.collect()

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "spotcheck_representation_ablation.csv", index=False)
    pivot = result.pivot_table(index=["model", "classifier"], columns="representation", values="accuracy").reset_index()
    pivot.to_csv(OUT_DIR / "spotcheck_representation_ablation_pivot.csv", index=False)
    print(pivot)


if __name__ == "__main__":
    main()
