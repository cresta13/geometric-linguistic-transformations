from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_lie_multilingual_max_audit import LANGS, MODELS, build_dataset, embed_texts, fit_pca  # noqa: E402


OUT_DIR = Path(os.getenv("LIE_TRIPLE_CONTROL_OUT_DIR", "results/experiments/lie_multilingual_triple_endpoint_controls_results"))
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, FIG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


ENDPOINT_NAMES = ["abc_text", "bca_text", "cab_text", "acb_text", "cba_text", "bac_text"]
CYCLIC_NAMES = ["abc_text", "bca_text", "cab_text"]
ANTICYCLIC_NAMES = ["acb_text", "cba_text", "bac_text"]


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_")


def make_features(df: pd.DataFrame, vecs_by_text: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    source = np.vstack([vecs_by_text[row.source] for row in df.itertuples(index=False)])
    endpoints = {
        name: np.vstack([vecs_by_text[getattr(row, name)] for row in df.itertuples(index=False)])
        for name in ENDPOINT_NAMES
    }
    cyclic = [endpoints[name] for name in CYCLIC_NAMES]
    anticyclic = [endpoints[name] for name in ANTICYCLIC_NAMES]
    endpoint_concat = np.hstack([endpoints[name] for name in ENDPOINT_NAMES])
    delta_concat = np.hstack([endpoints[name] - source for name in ENDPOINT_NAMES])
    signed_endpoint = sum(cyclic) - sum(anticyclic)
    return {
        "source_only": source,
        "endpoint_mean": sum(endpoints.values()) / len(ENDPOINT_NAMES),
        "cyclic_endpoint_mean": sum(cyclic) / len(cyclic),
        "anticyclic_endpoint_mean": sum(anticyclic) / len(anticyclic),
        "endpoint_concat": endpoint_concat,
        "delta_concat": delta_concat,
        "signed_endpoint_sum": signed_endpoint,
        "signed_delta_sum": signed_endpoint,
    }


def heldout_controls(model_name: str, df: pd.DataFrame, vecs_by_text: dict[str, np.ndarray]) -> pd.DataFrame:
    triple_df = df[df["kind"].eq("triple")].copy().reset_index(drop=True)
    labels = sorted(triple_df["label"].unique())
    y_map = {label: idx for idx, label in enumerate(labels)}
    y = triple_df["label"].map(y_map).to_numpy()
    features = make_features(triple_df, vecs_by_text)
    rows = []
    languages = [spec.code for spec in LANGS]
    for heldout in languages:
        train_mask = ~triple_df["language"].eq(heldout).to_numpy()
        test_mask = triple_df["language"].eq(heldout).to_numpy()
        for feature_name, x in features.items():
            clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
            clf.fit(normalize(x[train_mask]), y[train_mask])
            pred = clf.predict(normalize(x[test_mask]))
            rows.append({
                "model": model_name,
                "feature": feature_name,
                "heldout_language": heldout,
                "accuracy": accuracy_score(y[test_mask], pred),
                "macro_f1": f1_score(y[test_mask], pred, average="macro"),
                "chance": 1.0 / len(labels),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            })
    return pd.DataFrame(rows)


def analyze_model(model_name: str, df: pd.DataFrame, args: dict) -> None:
    safe = safe_model_name(model_name)
    done_marker = CHECKPOINT_DIR / f"{safe}.done.json"
    if done_marker.exists() and not args["force"]:
        print(f"[SKIP] {model_name}", flush=True)
        return

    print(f"\n=== MODEL {model_name} ===", flush=True)
    text_cols = ["source", *ENDPOINT_NAMES]
    texts = sorted({str(value) for col in text_cols for value in df[col].dropna().tolist()})
    print(f"texts={len(texts)}", flush=True)
    raw = embed_texts(model_name, texts, args["device"], args["batch_size"])
    vecs = fit_pca(raw, args["pca_dim"])
    vecs_by_text = {text: vecs[idx] for idx, text in enumerate(texts)}
    controls = heldout_controls(model_name, df, vecs_by_text)
    controls.to_csv(CSV_DIR / f"triple_endpoint_controls_{safe}.csv", index=False)
    done_marker.write_text(json.dumps({"model": model_name, "finished_at": time.ctime()}, indent=2), encoding="utf-8")
    del raw, vecs, vecs_by_text
    gc.collect()
    if args["device"] == "cuda":
        torch.cuda.empty_cache()


def read_many() -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in CSV_DIR.glob("triple_endpoint_controls_*.csv")
        if "all_models" not in path.name and "summary" not in path.name
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate() -> None:
    controls = read_many()
    if controls.empty:
        return
    controls.to_csv(CSV_DIR / "triple_endpoint_controls_all_models.csv", index=False)
    summary = controls.groupby(["feature"]).agg(
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        min_macro_f1=("macro_f1", "min"),
        max_macro_f1=("macro_f1", "max"),
        mean_accuracy=("accuracy", "mean"),
        n=("macro_f1", "count"),
    ).reset_index().sort_values("mean_macro_f1")
    summary.to_csv(CSV_DIR / "triple_endpoint_controls_summary.csv", index=False)
    by_model = controls.groupby(["model", "feature"]).agg(
        mean_macro_f1=("macro_f1", "mean"),
        min_macro_f1=("macro_f1", "min"),
        max_macro_f1=("macro_f1", "max"),
        n=("macro_f1", "count"),
    ).reset_index()
    by_model.to_csv(CSV_DIR / "triple_endpoint_controls_by_model.csv", index=False)
    by_language = controls.groupby(["heldout_language", "feature"]).agg(
        mean_macro_f1=("macro_f1", "mean"),
        min_macro_f1=("macro_f1", "min"),
        max_macro_f1=("macro_f1", "max"),
        n=("macro_f1", "count"),
    ).reset_index()
    by_language.to_csv(CSV_DIR / "triple_endpoint_controls_by_language.csv", index=False)

    plt.figure(figsize=(9, 5))
    ordered = summary.sort_values("mean_macro_f1")
    plt.barh(ordered["feature"], ordered["mean_macro_f1"], xerr=ordered["std_macro_f1"], capsize=3)
    plt.axvline(0.25, color="black", linestyle="--", linewidth=1, label="chance")
    plt.xlabel("Held-out-language macro F1")
    plt.title("Third-order endpoint-control leakage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "triple_endpoint_control_macro_f1.png", dpi=220, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = {
        "seed": int(os.getenv("LIE_TRIPLE_CONTROL_SEED", "20260627")),
        "n_templates_per_language": int(os.getenv("LIE_TRIPLE_CONTROL_TEMPLATES_PER_LANGUAGE", "96")),
        "pca_dim": int(os.getenv("LIE_TRIPLE_CONTROL_PCA_DIM", "128")),
        "batch_size": int(os.getenv("LIE_TRIPLE_CONTROL_BATCH_SIZE", "8")),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "force": os.getenv("LIE_TRIPLE_CONTROL_FORCE", "0") == "1",
    }
    model_env = os.getenv("LIE_TRIPLE_CONTROL_MODELS", "")
    models = [model.strip() for model in model_env.split(",") if model.strip()] or MODELS
    run_config = {**args, "models": models, "languages": [spec.code for spec in LANGS]}
    print("LIE MULTILINGUAL TRIPLE ENDPOINT CONTROLS")
    print(json.dumps(run_config, indent=2), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    df = build_dataset(args["n_templates_per_language"], args["seed"])
    df.to_csv(CSV_DIR / "triple_endpoint_control_dataset.csv", index=False)
    failures = []
    for model_name in models:
        try:
            analyze_model(model_name, df, args)
            aggregate()
        except Exception as exc:
            print(f"[FAIL] {model_name}: {exc}", flush=True)
            traceback.print_exc()
            failures.append({"model": model_name, "error": repr(exc), "traceback": traceback.format_exc()})
            (OUT_DIR / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    aggregate()
    status = {
        "finished_at": time.ctime(),
        "failures": failures,
        "completed_markers": sorted(path.name for path in CHECKPOINT_DIR.glob("*.done.json")),
    }
    (OUT_DIR / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("DONE")
    print(json.dumps(status, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
