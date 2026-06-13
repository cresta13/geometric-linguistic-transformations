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

import lie_llm_syntax_holdout_experiment as syntax


OUT_DIR = Path(os.getenv("SYNTAX_ABLATION_OUT_DIR", "results/experiments/syntax_representation_ablation_results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [m.strip() for m in os.getenv(
    "SYNTAX_ABLATION_MODELS",
    "bert-base-uncased,roberta-base,distilroberta-base,gpt2,distilgpt2",
).split(",") if m.strip()]

N_BASE = int(os.getenv("SYNTAX_ABLATION_N_BASE", str(syntax.N_BASE)))


def run_classifiers(X, labels, train_mask, test_mask):
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
    syntax.OUT_DIR = str(OUT_DIR)
    os.makedirs(syntax.OUT_DIR, exist_ok=True)

    base_df = syntax.make_base_rows(N_BASE)
    pair_df = syntax.build_pairs(base_df)
    prompts, pair_df = syntax.index_prompts(pair_df)

    base_df.to_csv(OUT_DIR / "base_sentences.csv", index=False)
    pair_df.to_csv(OUT_DIR / "transformation_pairs.csv", index=False)
    pd.DataFrame({"prompt_id": range(len(prompts)), "prompt": prompts}).to_csv(
        OUT_DIR / "prompts.csv",
        index=False,
    )

    labels = pair_df["class"].astype(str).to_numpy()
    syntax_family = pair_df["syntax_family"].astype(str).to_numpy()
    train_mask = np.isin(syntax_family, ["simple_svo", "with_context"])
    test_mask = np.isin(syntax_family, ["relative_clause", "temporal_clause", "reported_statement", "conditional"])
    src_idx = pair_df["source_idx"].to_numpy()
    tgt_idx = pair_df["target_idx"].to_numpy()

    all_rows = []

    for model_name in MODELS:
        print(f"\n=== MODEL: {model_name} ===")
        vectors = syntax.get_vectors(model_name, prompts)

        x_src = vectors[src_idx]
        x_tgt = vectors[tgt_idx]
        representations = {
            "x_only": x_src,
            "y_only": x_tgt,
            "delta": x_tgt - x_src,
            "concat": np.concatenate([x_src, x_tgt], axis=1),
        }

        for rep_name, X in representations.items():
            for row in run_classifiers(X, labels, train_mask, test_mask):
                row.update({
                    "model": model_name,
                    "representation": rep_name,
                    "random_baseline": 1 / len(set(labels)),
                    "n_base": N_BASE,
                })
                print(model_name, rep_name, row["classifier"], f"{row['accuracy']:.4f}")
                all_rows.append(row)

        del vectors, x_src, x_tgt, representations
        gc.collect()

    result = pd.DataFrame(all_rows)
    result.to_csv(OUT_DIR / "syntax_representation_ablation.csv", index=False)

    pivot = result.pivot_table(
        index=["model", "classifier"],
        columns="representation",
        values="accuracy",
    ).reset_index()
    pivot.to_csv(OUT_DIR / "syntax_representation_ablation_pivot.csv", index=False)

    print("\nSUMMARY")
    print(pivot)
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
