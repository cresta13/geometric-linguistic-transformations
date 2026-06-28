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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, normalize

sys.path.append(str(Path(__file__).resolve().parent))
from run_lie_endpoint_residualization_audit import (  # noqa: E402
    DEFAULT_MODELS,
    ENDPOINT_TEXT_COLS,
    CSV_DIR as PREV_CSV_DIR,
    endpoint_rows,
    exact_null_stats,
    sign_sum,
)
from run_lie_multilingual_max_audit import LANGS, TRIPLE_OPS, build_dataset, embed_texts  # noqa: E402


OUT_DIR = Path(os.getenv("LIE_SUBSPACE_OUT_DIR", "results/experiments/lie_endpoint_subspace_residualization_results"))
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, FIG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_")


def fit_pca(raw: np.ndarray, pca_dim: int) -> np.ndarray:
    dim = min(pca_dim, raw.shape[0] - 1, raw.shape[1])
    return PCA(n_components=dim, random_state=42).fit_transform(raw)


def rowspace_basis(coef: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    coef = np.asarray(coef, dtype=float)
    _, singular_values, vt = np.linalg.svd(coef, full_matrices=False)
    if len(singular_values) == 0:
        return np.zeros((0, coef.shape[1]))
    keep = singular_values > (tol * max(singular_values[0], 1.0))
    return vt[keep]


def combine_bases(*bases: np.ndarray) -> np.ndarray:
    pieces = [basis for basis in bases if basis.size]
    if not pieces:
        return np.zeros((0, 0))
    stacked = np.vstack(pieces)
    _, singular_values, vt = np.linalg.svd(stacked, full_matrices=False)
    keep = singular_values > (1e-8 * max(singular_values[0], 1.0))
    return vt[keep]


def project_out_basis(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return x.copy()
    return x - (x @ basis.T) @ basis


def fit_probe(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray, chance: float) -> tuple[LogisticRegression, dict]:
    clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    clf.fit(normalize(x[train_mask]), y[train_mask])
    pred = clf.predict(normalize(x[test_mask]))
    return clf, {
        "accuracy": accuracy_score(y[test_mask], pred),
        "macro_f1": f1_score(y[test_mask], pred, average="macro"),
        "chance": chance,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


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

    endpoint_x = np.vstack(endpoint_df["delta"])
    triple_encoder = LabelEncoder()
    triple_y = triple_encoder.fit_transform(endpoint_df["triple"])
    position_y = endpoint_df["position"].to_numpy()
    sign_y = endpoint_df["sign"].to_numpy()
    langs = sorted(triple_df["language"].unique())

    residual_rows = []
    probe_rows = []

    for heldout in langs:
        train_mask = ~endpoint_df["language"].eq(heldout).to_numpy()
        test_mask = endpoint_df["language"].eq(heldout).to_numpy()

        triple_clf, triple_metrics = fit_probe(endpoint_x, triple_y, train_mask, test_mask, 1.0 / len(triple_encoder.classes_))
        position_clf, position_metrics = fit_probe(endpoint_x, position_y, train_mask, test_mask, 1.0 / 6.0)
        sign_clf, sign_metrics = fit_probe(endpoint_x, sign_y, train_mask, test_mask, 0.5)

        for task, metrics, rank in [
            ("triple_label_from_single_endpoint_delta", triple_metrics, rowspace_basis(triple_clf.coef_).shape[0]),
            ("endpoint_position_from_endpoint_delta", position_metrics, rowspace_basis(position_clf.coef_).shape[0]),
            ("cyclic_vs_anticyclic_from_endpoint_delta", sign_metrics, rowspace_basis(sign_clf.coef_).shape[0]),
        ]:
            probe_rows.append({"model": model_name, "heldout_language": heldout, "probe": task, "subspace_rank": rank, **metrics})

        triple_basis = rowspace_basis(triple_clf.coef_)
        position_basis = rowspace_basis(position_clf.coef_)
        sign_basis = rowspace_basis(sign_clf.coef_)
        combined_basis = combine_bases(triple_basis, position_basis, sign_basis)

        bases = {
            "raw": np.zeros((0, endpoint_x.shape[1])),
            "remove_sign": sign_basis,
            "remove_triple_label": triple_basis,
            "remove_endpoint_position": position_basis,
            "remove_triple_plus_position_plus_sign": combined_basis,
        }

        sub = triple_df[triple_df["language"].eq(heldout)]
        for row in sub.itertuples(index=False):
            source = vecs_by_text[row.source]
            deltas = np.vstack([vecs_by_text[getattr(row, col)] - source for col in ENDPOINT_TEXT_COLS])
            for residualization, basis in bases.items():
                residual_deltas = list(project_out_basis(deltas, basis))
                stats = exact_null_stats(residual_deltas)
                residual_rows.append(
                    {
                        "model": model_name,
                        "language": row.language,
                        "template_id": row.template_id,
                        "triple": row.label,
                        "residualization": residualization,
                        "subspace_rank": int(basis.shape[0]),
                        "relative_signed_permutation_norm": stats["relative_signed_permutation_norm"],
                        "ratio_to_exact_null_mean": stats["ratio_to_exact_null_mean"],
                        "exact_empirical_p": stats["exact_empirical_p_smaller_or_equal_null"],
                    }
                )

    residual = pd.DataFrame(residual_rows)
    probes = pd.DataFrame(probe_rows)
    residual.to_csv(CSV_DIR / f"subspace_residualized_raw_{safe}.csv", index=False)
    probes.to_csv(CSV_DIR / f"endpoint_subspace_probes_{safe}.csv", index=False)
    summary = residual.groupby(["model", "language", "triple", "residualization"]).agg(
        mean_ratio_to_exact_null=("ratio_to_exact_null_mean", "mean"),
        frac_below_null=("ratio_to_exact_null_mean", lambda s: float((s < 1.0).mean())),
        mean_subspace_rank=("subspace_rank", "mean"),
        n=("triple", "count"),
    ).reset_index()
    summary.to_csv(CSV_DIR / f"subspace_residualized_summary_{safe}.csv", index=False)
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
    raw = read_many("subspace_residualized_raw_*.csv")
    summary = read_many("subspace_residualized_summary_*.csv")
    probes = read_many("endpoint_subspace_probes_*.csv")
    if not raw.empty:
        raw.to_csv(CSV_DIR / "subspace_residualized_raw_all_models.csv", index=False)
    if not summary.empty:
        summary.to_csv(CSV_DIR / "subspace_residualized_summary_all_models.csv", index=False)
        global_summary = summary.groupby(["triple", "residualization"]).agg(
            mean_ratio_to_exact_null=("mean_ratio_to_exact_null", "mean"),
            mean_frac_below_null=("frac_below_null", "mean"),
            mean_subspace_rank=("mean_subspace_rank", "mean"),
            n_cells=("triple", "count"),
        ).reset_index()
        global_summary.to_csv(CSV_DIR / "subspace_residualized_global_summary.csv", index=False)

        pivot = global_summary.pivot(index="triple", columns="residualization", values="mean_ratio_to_exact_null")
        order = ["raw", "remove_sign", "remove_triple_label", "remove_endpoint_position", "remove_triple_plus_position_plus_sign"]
        pivot = pivot[[col for col in order if col in pivot.columns]]
        plt.figure(figsize=(11, 4.8))
        x = np.arange(len(pivot.index))
        width = 0.16
        for idx, col in enumerate(pivot.columns):
            plt.bar(x + (idx - (len(pivot.columns) - 1) / 2) * width, pivot[col], width, label=col)
        plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
        plt.xticks(x, pivot.index)
        plt.ylabel("Mean ratio to exact sign-null")
        plt.title("Signed-permutation ratios after endpoint-derived subspace removal")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "01_subspace_residualization_ratios.png", dpi=220, bbox_inches="tight")
        plt.close()
    if not probes.empty:
        probes.to_csv(CSV_DIR / "endpoint_subspace_probes_all_models.csv", index=False)
        probe_summary = probes.groupby("probe").agg(
            mean_macro_f1=("macro_f1", "mean"),
            min_macro_f1=("macro_f1", "min"),
            max_macro_f1=("macro_f1", "max"),
            mean_chance=("chance", "mean"),
            mean_subspace_rank=("subspace_rank", "mean"),
            n=("macro_f1", "count"),
        ).reset_index()
        probe_summary.to_csv(CSV_DIR / "endpoint_subspace_probe_summary.csv", index=False)


def main():
    args = {
        "seed": int(os.getenv("LIE_SUBSPACE_SEED", "20260624")),
        "n_templates_per_language": int(os.getenv("LIE_SUBSPACE_TEMPLATES_PER_LANGUAGE", "96")),
        "pca_dim": int(os.getenv("LIE_SUBSPACE_PCA_DIM", "96")),
        "batch_size": int(os.getenv("LIE_SUBSPACE_BATCH_SIZE", "8")),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "force": os.getenv("LIE_SUBSPACE_FORCE", "0") == "1",
    }
    model_env = os.getenv("LIE_SUBSPACE_MODELS", "")
    models = [m.strip() for m in model_env.split(",") if m.strip()] or DEFAULT_MODELS
    config = {**args, "models": models, "languages": [spec.code for spec in LANGS], "triples": ["".join(t) for t in TRIPLE_OPS]}
    print("LIE ENDPOINT SUBSPACE RESIDUALIZATION AUDIT")
    print(json.dumps(config, indent=2), flush=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    df = build_dataset(args["n_templates_per_language"], args["seed"])
    df.to_csv(CSV_DIR / "endpoint_subspace_residualization_dataset.csv", index=False)
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
