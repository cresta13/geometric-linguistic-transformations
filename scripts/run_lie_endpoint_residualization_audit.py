import gc
import itertools
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
from scipy.stats import percentileofscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, normalize

sys.path.append(str(Path(__file__).resolve().parent))
from run_lie_multilingual_max_audit import (  # noqa: E402
    LANGS,
    TRIPLE_OPS,
    build_dataset,
    embed_texts,
    relative_norm,
)


OUT_DIR = Path("results/experiments/lie_endpoint_residualization_results")
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, FIG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


DEFAULT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "distilbert-base-multilingual-cased",
]

ENDPOINTS = ["abc", "bca", "cab", "acb", "cba", "bac"]
ENDPOINT_TEXT_COLS = [f"{name}_text" for name in ENDPOINTS]
OBSERVED_POSITIVE = frozenset([0, 1, 2])
OBSERVED_NEGATIVE = frozenset([3, 4, 5])
EXACT_SIGN_MASKS = [
    frozenset(mask)
    for mask in itertools.combinations(range(6), 3)
    if frozenset(mask) not in {OBSERVED_POSITIVE, OBSERVED_NEGATIVE}
]


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_")


def fit_pca(raw: np.ndarray, pca_dim: int) -> np.ndarray:
    dim = min(pca_dim, raw.shape[0] - 1, raw.shape[1])
    return PCA(n_components=dim, random_state=42).fit_transform(raw)


def sign_sum(deltas: list[np.ndarray], positive_mask: frozenset[int]) -> np.ndarray:
    out = np.zeros_like(deltas[0])
    for idx, delta in enumerate(deltas):
        out += delta if idx in positive_mask else -delta
    return out


def exact_null_stats(deltas: list[np.ndarray]) -> dict:
    scale = np.mean([np.linalg.norm(delta) for delta in deltas]) + 1e-12
    observed_vec = sign_sum(deltas, OBSERVED_POSITIVE)
    observed = float(np.linalg.norm(observed_vec) / scale)
    nulls = np.asarray([
        float(np.linalg.norm(sign_sum(deltas, mask)) / scale)
        for mask in EXACT_SIGN_MASKS
    ])
    return {
        "relative_signed_permutation_norm": observed,
        "exact_null_mean": float(nulls.mean()),
        "exact_null_std": float(nulls.std(ddof=1)),
        "ratio_to_exact_null_mean": float(observed / (nulls.mean() + 1e-12)),
        "exact_null_percentile_smaller_is_better": float(percentileofscore(nulls, observed, kind="mean")),
        "exact_empirical_p_smaller_or_equal_null": float(((nulls <= observed).sum() + 1) / (len(nulls) + 1)),
    }


def project_out_direction(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return x.copy()
    unit = direction / norm
    return x - np.outer(x @ unit, unit)


def endpoint_rows(triple_df: pd.DataFrame, vecs_by_text: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for row_idx, row in enumerate(triple_df.itertuples(index=False)):
        source = vecs_by_text[row.source]
        for pos, endpoint in enumerate(ENDPOINTS):
            text = getattr(row, f"{endpoint}_text")
            delta = vecs_by_text[text] - source
            rows.append(
                {
                    "row_idx": row_idx,
                    "language": row.language,
                    "template_id": row.template_id,
                    "triple": row.label,
                    "endpoint": endpoint,
                    "sign": 1 if pos < 3 else 0,
                    "position": pos,
                    "text": text,
                    "delta": delta,
                    "endpoint_vec": vecs_by_text[text],
                    "source_vec": source,
                }
            )
    return pd.DataFrame(rows)


def endpoint_feature_controls(model_name: str, endpoint_df: pd.DataFrame, triple_df: pd.DataFrame, vecs_by_text: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    langs = sorted(triple_df["language"].unique())
    label_encoder = LabelEncoder()
    triple_labels = label_encoder.fit_transform(triple_df["label"])

    source_x = np.vstack([vecs_by_text[row.source] for row in triple_df.itertuples(index=False)])
    endpoint_mean_x = []
    endpoint_concat_x = []
    signed_x = []
    for row in triple_df.itertuples(index=False):
        source = vecs_by_text[row.source]
        deltas = [vecs_by_text[getattr(row, col)] - source for col in ENDPOINT_TEXT_COLS]
        endpoints = [vecs_by_text[getattr(row, col)] for col in ENDPOINT_TEXT_COLS]
        endpoint_mean_x.append(np.mean(endpoints, axis=0))
        endpoint_concat_x.append(np.concatenate(endpoints))
        signed_x.append(sign_sum(deltas, OBSERVED_POSITIVE))

    feature_sets = {
        "source_only": source_x,
        "endpoint_mean": np.vstack(endpoint_mean_x),
        "endpoint_concat": np.vstack(endpoint_concat_x),
        "signed_vector": np.vstack(signed_x),
    }

    for heldout in langs:
        train_mask = ~triple_df["language"].eq(heldout).to_numpy()
        test_mask = triple_df["language"].eq(heldout).to_numpy()
        for feature, x in feature_sets.items():
            clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
            clf.fit(normalize(x[train_mask]), triple_labels[train_mask])
            pred = clf.predict(normalize(x[test_mask]))
            rows.append(
                {
                    "model": model_name,
                    "control_task": "triple_label_from_row_features",
                    "feature": feature,
                    "heldout_language": heldout,
                    "accuracy": accuracy_score(triple_labels[test_mask], pred),
                    "macro_f1": f1_score(triple_labels[test_mask], pred, average="macro"),
                    "chance": 1.0 / len(label_encoder.classes_),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                }
            )

    endpoint_y = endpoint_df["sign"].to_numpy()
    endpoint_position_y = endpoint_df["position"].to_numpy()
    triple_y_per_endpoint = label_encoder.transform(endpoint_df["triple"])
    endpoint_delta_x = np.vstack(endpoint_df["delta"])
    endpoint_vec_x = np.vstack(endpoint_df["endpoint_vec"])

    endpoint_tasks = [
        ("cyclic_vs_anticyclic_from_endpoint_delta", endpoint_delta_x, endpoint_y, 0.5),
        ("cyclic_vs_anticyclic_from_endpoint_vec", endpoint_vec_x, endpoint_y, 0.5),
        ("endpoint_position_from_endpoint_delta", endpoint_delta_x, endpoint_position_y, 1.0 / 6.0),
        ("triple_label_from_single_endpoint_delta", endpoint_delta_x, triple_y_per_endpoint, 1.0 / len(label_encoder.classes_)),
    ]
    for heldout in langs:
        train_mask = ~endpoint_df["language"].eq(heldout).to_numpy()
        test_mask = endpoint_df["language"].eq(heldout).to_numpy()
        for task, x, y, chance in endpoint_tasks:
            clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
            clf.fit(normalize(x[train_mask]), y[train_mask])
            pred = clf.predict(normalize(x[test_mask]))
            rows.append(
                {
                    "model": model_name,
                    "control_task": task,
                    "feature": "endpoint",
                    "heldout_language": heldout,
                    "accuracy": accuracy_score(y[test_mask], pred),
                    "macro_f1": f1_score(y[test_mask], pred, average="macro"),
                    "chance": chance,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def residualized_signed_permutation(model_name: str, endpoint_df: pd.DataFrame, triple_df: pd.DataFrame, vecs_by_text: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    sign_probe_rows = []
    langs = sorted(triple_df["language"].unique())
    endpoint_delta_x = np.vstack(endpoint_df["delta"])
    endpoint_y = endpoint_df["sign"].to_numpy()

    for heldout in langs:
        train_mask = ~endpoint_df["language"].eq(heldout).to_numpy()
        test_mask = endpoint_df["language"].eq(heldout).to_numpy()
        clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
        clf.fit(normalize(endpoint_delta_x[train_mask]), endpoint_y[train_mask])
        pred = clf.predict(normalize(endpoint_delta_x[test_mask]))
        sign_probe_rows.append(
            {
                "model": model_name,
                "heldout_language": heldout,
                "accuracy": accuracy_score(endpoint_y[test_mask], pred),
                "macro_f1": f1_score(endpoint_y[test_mask], pred, average="macro"),
                "chance": 0.5,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
        )
        direction = clf.coef_[0]

        sub = triple_df[triple_df["language"].eq(heldout)]
        for row in sub.itertuples(index=False):
            source = vecs_by_text[row.source]
            deltas = [vecs_by_text[getattr(row, col)] - source for col in ENDPOINT_TEXT_COLS]
            raw_stats = exact_null_stats(deltas)
            residual_deltas = list(project_out_direction(np.vstack(deltas), direction))
            residual_stats = exact_null_stats(residual_deltas)
            rows.append(
                {
                    "model": model_name,
                    "language": row.language,
                    "template_id": row.template_id,
                    "triple": row.label,
                    "raw_relative_signed_permutation_norm": raw_stats["relative_signed_permutation_norm"],
                    "raw_ratio_to_exact_null_mean": raw_stats["ratio_to_exact_null_mean"],
                    "raw_exact_empirical_p": raw_stats["exact_empirical_p_smaller_or_equal_null"],
                    "residual_relative_signed_permutation_norm": residual_stats["relative_signed_permutation_norm"],
                    "residual_ratio_to_exact_null_mean": residual_stats["ratio_to_exact_null_mean"],
                    "residual_exact_empirical_p": residual_stats["exact_empirical_p_smaller_or_equal_null"],
                    "residual_minus_raw_ratio": residual_stats["ratio_to_exact_null_mean"] - raw_stats["ratio_to_exact_null_mean"],
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(sign_probe_rows)


def analyze_model(model_name: str, df: pd.DataFrame, args: dict):
    safe = safe_model_name(model_name)
    done_marker = CHECKPOINT_DIR / f"{safe}.done.json"
    if done_marker.exists() and not args["force"]:
        print(f"[SKIP] {model_name} already done", flush=True)
        return

    print(f"\n=== MODEL {model_name} ===", flush=True)
    triple_df = df[df["kind"].eq("triple")].copy().reset_index(drop=True)
    text_cols = ["source", *ENDPOINT_TEXT_COLS]
    texts = sorted({str(v) for col in text_cols for v in triple_df[col].dropna().tolist()})
    print(f"triple_rows={len(triple_df)} texts={len(texts)}", flush=True)

    raw = embed_texts(model_name, texts, args["device"], args["batch_size"])
    vecs = fit_pca(raw, args["pca_dim"])
    vecs_by_text = {text: vecs[i] for i, text in enumerate(texts)}

    endpoint_df = endpoint_rows(triple_df, vecs_by_text)
    controls = endpoint_feature_controls(model_name, endpoint_df, triple_df, vecs_by_text)
    residual_raw, sign_probe = residualized_signed_permutation(model_name, endpoint_df, triple_df, vecs_by_text)

    controls.to_csv(CSV_DIR / f"endpoint_controls_{safe}.csv", index=False)
    residual_raw.to_csv(CSV_DIR / f"residualized_signed_raw_{safe}.csv", index=False)
    sign_probe.to_csv(CSV_DIR / f"sign_probe_{safe}.csv", index=False)

    summary = residual_raw.groupby(["model", "language", "triple"]).agg(
        mean_raw_ratio=("raw_ratio_to_exact_null_mean", "mean"),
        mean_residual_ratio=("residual_ratio_to_exact_null_mean", "mean"),
        mean_residual_minus_raw_ratio=("residual_minus_raw_ratio", "mean"),
        frac_raw_below_null=("raw_ratio_to_exact_null_mean", lambda s: float((s < 1.0).mean())),
        frac_residual_below_null=("residual_ratio_to_exact_null_mean", lambda s: float((s < 1.0).mean())),
        n=("triple", "count"),
    ).reset_index()
    summary.to_csv(CSV_DIR / f"residualized_signed_summary_{safe}.csv", index=False)

    done_marker.write_text(json.dumps({"model": model_name, "finished_at": time.ctime()}, indent=2), encoding="utf-8")
    del raw, vecs, vecs_by_text, endpoint_df
    gc.collect()
    if args["device"] == "cuda":
        torch.cuda.empty_cache()


def read_many(pattern: str) -> pd.DataFrame:
    paths = [path for path in sorted(CSV_DIR.glob(pattern)) if "all_models" not in path.name and "global" not in path.name]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def aggregate_outputs():
    outputs = {
        "endpoint_controls_all_models.csv": read_many("endpoint_controls_*.csv"),
        "sign_probe_all_models.csv": read_many("sign_probe_*.csv"),
        "residualized_signed_raw_all_models.csv": read_many("residualized_signed_raw_*.csv"),
        "residualized_signed_summary_all_models.csv": read_many("residualized_signed_summary_*.csv"),
    }
    for filename, data in outputs.items():
        if not data.empty:
            data.to_csv(CSV_DIR / filename, index=False)

    summary = outputs["residualized_signed_summary_all_models.csv"]
    if not summary.empty:
        global_summary = summary.groupby("triple").agg(
            mean_raw_ratio=("mean_raw_ratio", "mean"),
            mean_residual_ratio=("mean_residual_ratio", "mean"),
            mean_residual_minus_raw_ratio=("mean_residual_minus_raw_ratio", "mean"),
            mean_frac_raw_below_null=("frac_raw_below_null", "mean"),
            mean_frac_residual_below_null=("frac_residual_below_null", "mean"),
            n_cells=("triple", "count"),
        ).reset_index()
        global_summary.to_csv(CSV_DIR / "residualized_signed_global_summary.csv", index=False)

        x = np.arange(len(global_summary))
        width = 0.38
        plt.figure(figsize=(7.2, 4.2))
        plt.bar(x - width / 2, global_summary["mean_raw_ratio"], width, label="raw")
        plt.bar(x + width / 2, global_summary["mean_residual_ratio"], width, label="sign-residualized")
        plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
        plt.xticks(x, global_summary["triple"])
        plt.ylabel("Mean ratio to exact sign-null")
        plt.title("Signed-permutation effect before/after endpoint sign residualization")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "01_raw_vs_residualized_signed_ratio.png", dpi=220, bbox_inches="tight")
        plt.close()

    controls = outputs["endpoint_controls_all_models.csv"]
    if not controls.empty:
        control_summary = controls.groupby(["control_task", "feature"]).agg(
            mean_macro_f1=("macro_f1", "mean"),
            min_macro_f1=("macro_f1", "min"),
            max_macro_f1=("macro_f1", "max"),
            mean_chance=("chance", "mean"),
            n=("macro_f1", "count"),
        ).reset_index()
        control_summary.to_csv(CSV_DIR / "endpoint_control_global_summary.csv", index=False)


def main():
    args = {
        "seed": int(os.getenv("LIE_ENDPOINT_SEED", "20260623")),
        "n_templates_per_language": int(os.getenv("LIE_ENDPOINT_TEMPLATES_PER_LANGUAGE", "96")),
        "pca_dim": int(os.getenv("LIE_ENDPOINT_PCA_DIM", "96")),
        "batch_size": int(os.getenv("LIE_ENDPOINT_BATCH_SIZE", "8")),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "force": os.getenv("LIE_ENDPOINT_FORCE", "0") == "1",
    }
    model_env = os.getenv("LIE_ENDPOINT_MODELS", "")
    models = [m.strip() for m in model_env.split(",") if m.strip()] or DEFAULT_MODELS

    config = {**args, "models": models, "languages": [spec.code for spec in LANGS], "triples": ["".join(t) for t in TRIPLE_OPS]}
    print("LIE ENDPOINT RESIDUALIZATION AUDIT")
    print(json.dumps(config, indent=2), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    df = build_dataset(args["n_templates_per_language"], args["seed"])
    df.to_csv(CSV_DIR / "endpoint_residualization_dataset.csv", index=False)
    print("dataset rows", len(df), "triple rows", int(df["kind"].eq("triple").sum()), flush=True)

    failures = []
    for model in models:
        try:
            analyze_model(model, df, args)
            aggregate_outputs()
        except Exception as exc:
            print(f"[FAIL] {model}: {exc}", flush=True)
            traceback.print_exc()
            failures.append({"model": model, "error": repr(exc), "traceback": traceback.format_exc()})
            (OUT_DIR / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
            continue

    aggregate_outputs()
    status = {
        "finished_at": time.ctime(),
        "failures": failures,
        "completed_markers": [p.name for p in CHECKPOINT_DIR.glob("*.done.json")],
    }
    (OUT_DIR / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("DONE")
    print(json.dumps(status, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
