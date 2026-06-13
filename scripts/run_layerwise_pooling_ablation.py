import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

import lie_llm_syntax_holdout_experiment as syntax


OUT_DIR = Path(os.getenv("LAYERWISE_OUT_DIR", "results/experiments/layerwise_pooling_ablation_results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("LAYERWISE_MODEL", "bert-base-uncased")
N_BASE = int(os.getenv("LAYERWISE_N_BASE", "300"))
BATCH_SIZE = int(os.getenv("LAYERWISE_BATCH_SIZE", "16"))


def pool(hidden, mask, mode):
    if mode == "mean":
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-9)
    if mode == "cls":
        return hidden[:, 0, :]
    if mode == "last_token":
        idx = mask.sum(dim=1).clamp_min(1) - 1
        return hidden[torch.arange(hidden.shape[0]), idx]
    raise ValueError(mode)


def extract_layer_vectors(prompts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"

    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()

    rows_by_key = {}
    started = time.time()

    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors="pt")

        with torch.no_grad():
            out = model(**enc)

        for layer_idx, hidden in enumerate(out.hidden_states):
            for pooling in ["mean", "cls", "last_token"]:
                key = (layer_idx, pooling)
                rows_by_key.setdefault(key, []).append(pool(hidden, enc["attention_mask"], pooling).cpu().numpy())

        done = start + len(batch)
        print(f"{MODEL_NAME}: {done}/{len(prompts)}, {time.time() - started:.1f}s")

    result = {key: np.vstack(parts) for key, parts in rows_by_key.items()}

    del model
    gc.collect()
    return result


def fit_score(X, labels, train_mask, test_mask):
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
        })
    return rows


def main():
    base_df = syntax.make_base_rows(N_BASE)
    pair_df = syntax.build_pairs(base_df)
    prompts, pair_df = syntax.index_prompts(pair_df)

    pair_df.to_csv(OUT_DIR / "transformation_pairs.csv", index=False)
    pd.DataFrame({"prompt_id": range(len(prompts)), "prompt": prompts}).to_csv(OUT_DIR / "prompts.csv", index=False)

    labels = pair_df["class"].astype(str).to_numpy()
    syntax_family = pair_df["syntax_family"].astype(str).to_numpy()
    train_mask = np.isin(syntax_family, ["simple_svo", "with_context"])
    test_mask = np.isin(syntax_family, ["relative_clause", "temporal_clause", "reported_statement", "conditional"])
    src_idx = pair_df["source_idx"].to_numpy()
    tgt_idx = pair_df["target_idx"].to_numpy()

    vectors_by_key = extract_layer_vectors(prompts)

    all_rows = []
    for (layer_idx, pooling), vectors in vectors_by_key.items():
        x_src = vectors[src_idx]
        x_tgt = vectors[tgt_idx]

        for rep_name, X in {
            "delta": x_tgt - x_src,
            "y_only": x_tgt,
        }.items():
            for row in fit_score(X, labels, train_mask, test_mask):
                row.update({
                    "model": MODEL_NAME,
                    "layer": layer_idx,
                    "pooling": pooling,
                    "representation": rep_name,
                    "n_base": N_BASE,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                })
                print(row)
                all_rows.append(row)

    result = pd.DataFrame(all_rows)
    result.to_csv(OUT_DIR / "layerwise_pooling_ablation.csv", index=False)

    best = (
        result[result["classifier"] == "linear_svc"]
        .sort_values("accuracy", ascending=False)
        .head(20)
    )
    best.to_csv(OUT_DIR / "layerwise_pooling_ablation_top20.csv", index=False)
    print(best)
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
