import gc
import os
import random
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

import lie_llm_full_semantic_holdout_experiment as full


OUT_DIR = Path(os.getenv("POOLING_OUT_DIR", "results/experiments/full_semantic_pooling_ablation_results"))
MODELS = [m.strip() for m in os.getenv("POOLING_MODELS", "bert-base-uncased,roberta-base,gpt2").split(",") if m.strip()]
N_BASE = int(os.getenv("POOLING_N_BASE", "150"))
BATCH_SIZE = int(os.getenv("POOLING_BATCH_SIZE", "32"))
SEED = int(os.getenv("POOLING_SEED", "42"))


def reset_full_dataset_seed():
    full.N_BASE = N_BASE
    full.random.seed(SEED)
    full.np.random.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def safe_name(model_name):
    return model_name.replace("/", "__")


def last_token_pool(hidden, attention_mask):
    lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_idx, lengths]


def get_pooled_vectors(model_name, prompts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    pooled = {"mean": [], "cls": [], "last_token": []}

    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=180,
            return_tensors="pt",
        )

        with torch.no_grad():
            out = model(**inputs)

        hidden = out.hidden_states[-1]
        mask = inputs["attention_mask"]

        pooled["mean"].append(full.mean_pool(hidden, mask).cpu().numpy())
        pooled["cls"].append(hidden[:, 0, :].cpu().numpy())
        pooled["last_token"].append(last_token_pool(hidden, mask).cpu().numpy())

        done = start + len(batch)
        print(f"  {model_name}: {done}/{len(prompts)}", flush=True)

    del model
    gc.collect()

    return {k: np.vstack(v) for k, v in pooled.items()}


def build_features(vectors, pair_df, representation):
    src = vectors[pair_df["source_idx"].to_numpy()]
    tgt = vectors[pair_df["target_idx"].to_numpy()]

    if representation == "x_only":
        return src
    if representation == "y_only":
        return tgt
    if representation == "delta":
        return tgt - src
    if representation == "concat":
        return np.hstack([src, tgt])
    raise ValueError(representation)


def run_one(model_name, pooling, representation, vectors, pair_df):
    X = build_features(vectors, pair_df, representation)
    y = pair_df["class"].astype(str).to_numpy()
    split = pair_df["split"].astype(str).to_numpy()

    train = split == "train"
    test = split == "test"

    classifiers = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        ),
        "linear_svc": make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", max_iter=20000),
        ),
    }

    rows = []
    for clf_name, clf in classifiers.items():
        clf.fit(X[train], y[train])
        pred = clf.predict(X[test])
        rows.append({
            "model": model_name,
            "pooling": pooling,
            "representation": representation,
            "classifier": clf_name,
            "accuracy": float(accuracy_score(y[test], pred)),
            "n_base": N_BASE,
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reset_full_dataset_seed()

    pair_df = full.build_dataset()
    prompts, pair_df = full.index_prompts(pair_df)
    pair_df.to_csv(OUT_DIR / "transformation_pairs.csv", index=False)
    pd.DataFrame({"prompt_id": range(len(prompts)), "prompt": prompts}).to_csv(OUT_DIR / "prompts.csv", index=False)

    rows = []
    for model_name in MODELS:
        print(f"\n=== MODEL: {model_name} ===", flush=True)
        pooled_vectors = get_pooled_vectors(model_name, prompts)

        for pooling, vectors in pooled_vectors.items():
            for representation in ["x_only", "y_only", "concat", "delta"]:
                rows.extend(run_one(model_name, pooling, representation, vectors, pair_df))

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "full_semantic_pooling_ablation.csv", index=False)
    pivot = result.pivot_table(
        index=["model", "pooling", "classifier"],
        columns="representation",
        values="accuracy",
    ).reset_index()
    pivot.to_csv(OUT_DIR / "full_semantic_pooling_ablation_pivot.csv", index=False)
    print(pivot)


if __name__ == "__main__":
    main()
