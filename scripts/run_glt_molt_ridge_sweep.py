from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_glt_molt_affine_operator_audit import all_texts, model_language_audit  # noqa: E402
from run_lie_endpoint_residualization_audit import DEFAULT_MODELS, safe_model_name  # noqa: E402
from run_lie_multilingual_max_audit import LANGS, embed_texts, fit_pca  # noqa: E402
from run_lie_structure_constants_audit import OPS, build_structure_dataset  # noqa: E402


OUT_DIR = Path(os.getenv("GLT_MOLT_RIDGE_OUT_DIR", "results/experiments/glt_molt_ridge_sweep_results"))
CSV_DIR = OUT_DIR / "csv"
CKPT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, CKPT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def parse_alphas() -> list[float]:
    raw = os.getenv("GLT_MOLT_RIDGE_ALPHAS", "0.1,1,10,100")
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def alpha_label(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def add_alpha(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = df.copy()
    out.insert(1, "ridge_alpha", alpha)
    return out


def read_many(pattern: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in CSV_DIR.glob(pattern)
        if "all_models" not in path.name and "summary" not in path.name
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate() -> None:
    fit = read_many("operator_fit_*.csv")
    comp = read_many("composition_prediction_*.csv")
    closure = read_many("matrix_closure_*.csv")
    jacobi = read_many("matrix_jacobi_*.csv")

    if not fit.empty:
        fit.to_csv(CSV_DIR / "operator_fit_all_models.csv", index=False)
        fit.groupby(["ridge_alpha", "method", "operator"]).agg(
            mean_target_cosine=("target_cosine", "mean"),
            std_target_cosine=("target_cosine", "std"),
            mean_mse=("mse", "mean"),
            cells=("target_cosine", "count"),
        ).reset_index().to_csv(CSV_DIR / "operator_fit_summary.csv", index=False)
        fit.groupby(["ridge_alpha", "method"]).agg(
            mean_target_cosine=("target_cosine", "mean"),
            mean_mse=("mse", "mean"),
            cells=("target_cosine", "count"),
        ).reset_index().to_csv(CSV_DIR / "operator_fit_by_alpha_method.csv", index=False)

    if not comp.empty:
        comp.to_csv(CSV_DIR / "composition_prediction_all_models.csv", index=False)
        comp.groupby(["ridge_alpha", "method", "pair"]).agg(
            mean_ab_target_cosine=("ab_target_cosine", "mean"),
            mean_ba_target_cosine=("ba_target_cosine", "mean"),
            mean_commutator_cosine=("commutator_cosine", "mean"),
            cells=("ab_target_cosine", "count"),
        ).reset_index().to_csv(CSV_DIR / "composition_prediction_summary.csv", index=False)

    if not closure.empty:
        closure.to_csv(CSV_DIR / "matrix_closure_all_models.csv", index=False)
        closure.groupby(["ridge_alpha", "method", "pair"]).agg(
            mean_closure_residual_ratio=("closure_residual_ratio", "mean"),
            std_closure_residual_ratio=("closure_residual_ratio", "std"),
            mean_random_subspace_null=("random_subspace_null_mean", "mean"),
            mean_empirical_p=("empirical_p_below_random_subspace", "mean"),
            cells=("closure_residual_ratio", "count"),
        ).reset_index().to_csv(CSV_DIR / "matrix_closure_summary.csv", index=False)
        closure.groupby(["ridge_alpha", "method"]).agg(
            mean_closure_residual_ratio=("closure_residual_ratio", "mean"),
            mean_matrix_commutator_norm=("matrix_commutator_norm", "mean"),
            mean_random_subspace_null=("random_subspace_null_mean", "mean"),
            mean_empirical_p=("empirical_p_below_random_subspace", "mean"),
            cells=("closure_residual_ratio", "count"),
        ).reset_index().to_csv(CSV_DIR / "matrix_closure_by_alpha_method.csv", index=False)

    if not jacobi.empty:
        jacobi.to_csv(CSV_DIR / "matrix_jacobi_all_models.csv", index=False)
        jacobi.groupby(["ridge_alpha", "method", "triple"]).agg(
            mean_relative_jacobi_operator_norm=("relative_jacobi_operator_norm", "mean"),
            std_relative_jacobi_operator_norm=("relative_jacobi_operator_norm", "std"),
            cells=("relative_jacobi_operator_norm", "count"),
        ).reset_index().to_csv(CSV_DIR / "matrix_jacobi_summary.csv", index=False)
        jacobi.groupby(["ridge_alpha", "method"]).agg(
            mean_relative_jacobi_operator_norm=("relative_jacobi_operator_norm", "mean"),
            mean_jacobi_coeff_norm=("jacobi_coeff_norm", "mean"),
            cells=("relative_jacobi_operator_norm", "count"),
        ).reset_index().to_csv(CSV_DIR / "matrix_jacobi_by_alpha_method.csv", index=False)


def run_model(model_name: str, df: pd.DataFrame, config: dict, alphas: list[float]) -> None:
    safe = safe_model_name(model_name)
    print(f"\n=== MODEL {model_name} ===", flush=True)
    texts = all_texts(df)
    print(f"templates={len(df)} texts={len(texts)}", flush=True)

    raw = embed_texts(model_name, texts, config["device"], config["batch_size"])
    vecs = fit_pca(raw, config["pca_dim"])
    vecs_by_text = dict(zip(texts, vecs))
    dim = vecs.shape[1]

    for alpha in alphas:
        label = alpha_label(alpha)
        marker = CKPT_DIR / f"{safe}__alpha_{label}.done.json"
        if marker.exists() and not config["force"]:
            print(f"SKIP {model_name} alpha={alpha:g}", flush=True)
            continue

        print(f"alpha={alpha:g}", flush=True)
        alpha_config = {**config, "ridge_alpha": alpha}
        fit, comp, closure, jacobi = model_language_audit(model_name, df, vecs_by_text, dim, alpha_config)

        fit = add_alpha(fit, alpha)
        comp = add_alpha(comp, alpha)
        closure = add_alpha(closure, alpha)
        jacobi = add_alpha(jacobi, alpha)

        fit.to_csv(CSV_DIR / f"operator_fit_{safe}__alpha_{label}.csv", index=False)
        comp.to_csv(CSV_DIR / f"composition_prediction_{safe}__alpha_{label}.csv", index=False)
        closure.to_csv(CSV_DIR / f"matrix_closure_{safe}__alpha_{label}.csv", index=False)
        jacobi.to_csv(CSV_DIR / f"matrix_jacobi_{safe}__alpha_{label}.csv", index=False)
        marker.write_text(
            json.dumps({"model": model_name, "ridge_alpha": alpha, "finished_at": time.ctime()}, indent=2),
            encoding="utf-8",
        )
        aggregate()
        del fit, comp, closure, jacobi
        gc.collect()

    del raw, vecs, vecs_by_text
    gc.collect()


def main() -> None:
    config = {
        "seed": int(os.getenv("GLT_MOLT_SEED", "20260624")),
        "n_templates_per_language": int(os.getenv("GLT_MOLT_TEMPLATES_PER_LANGUAGE", "160")),
        "pca_dim": int(os.getenv("GLT_MOLT_PCA_DIM", "128")),
        "batch_size": int(os.getenv("GLT_MOLT_BATCH_SIZE", "1")),
        "n_nulls": int(os.getenv("GLT_MOLT_NULLS", "300")),
        "device": os.getenv("LIE_DEVICE", "cpu"),
        "force": os.getenv("GLT_MOLT_FORCE", "0") == "1",
    }
    alphas = parse_alphas()
    models_env = os.getenv("GLT_MOLT_MODELS", "").strip()
    models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else DEFAULT_MODELS
    config_with_models = {
        **config,
        "ridge_alphas": alphas,
        "models": models,
        "languages": [spec.code for spec in LANGS],
        "operators": OPS,
    }
    print("GLT-MOLT RIDGE SWEEP")
    print(json.dumps(config_with_models, indent=2), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps(config_with_models, indent=2), encoding="utf-8")

    df = build_structure_dataset(config["n_templates_per_language"], config["seed"])
    df.to_csv(CSV_DIR / "glt_molt_dataset.csv", index=False)
    print(f"templates={len(df)} texts={len(all_texts(df))}", flush=True)

    failures = []
    for model in models:
        try:
            run_model(model, df, config, alphas)
        except Exception as exc:
            failures.append({"model": model, "error": repr(exc)})
            print(f"FAILED {model}: {exc!r}", flush=True)
    aggregate()
    status = {
        "finished_at": time.ctime(),
        "failures": failures,
        "completed_markers": sorted(path.name for path in CKPT_DIR.glob("*.done.json")),
    }
    (OUT_DIR / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("DONE")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
