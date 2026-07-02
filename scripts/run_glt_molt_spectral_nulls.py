from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_glt_molt_affine_operator_audit import all_texts, project_residual_ratio  # noqa: E402
from run_glt_molt_matched_nulls import alpha_label, mean_cosine, summarize  # noqa: E402
from run_lie_endpoint_residualization_audit import DEFAULT_MODELS, safe_model_name  # noqa: E402
from run_lie_multilingual_max_audit import LANGS, PAIR_OPS, embed_texts, fit_pca  # noqa: E402
from run_lie_structure_constants_audit import OPS, build_structure_dataset  # noqa: E402


OUT_DIR = Path(os.getenv("GLT_MOLT_SPECTRAL_OUT_DIR", "results/experiments/glt_molt_spectral_nulls_results"))
CSV_DIR = OUT_DIR / "csv"
CKPT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, CKPT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def parse_alphas() -> list[float]:
    raw = os.getenv("GLT_MOLT_SPECTRAL_ALPHAS", "100")
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def fit_map(x: np.ndarray, y: np.ndarray, alpha: float, fit_intercept: bool) -> tuple[np.ndarray, np.ndarray]:
    reg = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    reg.fit(x, y)
    intercept = reg.intercept_ if fit_intercept else np.zeros(y.shape[1])
    return reg.coef_, intercept


def random_signed_permutation(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = values[rng.permutation(values.size)].copy()
    out *= rng.choice(np.array([-1.0, 1.0]), size=values.size)
    return out


def givens_spectrum_matched_matrix(
    singular_values: np.ndarray,
    rng: np.random.Generator,
    n_givens: int,
) -> np.ndarray:
    mat = np.diag(random_signed_permutation(singular_values, rng))
    dim = mat.shape[0]

    for _ in range(n_givens):
        i, j = rng.choice(dim, size=2, replace=False)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        row_i = mat[i].copy()
        row_j = mat[j].copy()
        mat[i] = c * row_i + s * row_j
        mat[j] = -s * row_i + c * row_j

        i, j = rng.choice(dim, size=2, replace=False)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        col_i = mat[:, i].copy()
        col_j = mat[:, j].copy()
        mat[:, i] = c * col_i + s * col_j
        mat[:, j] = -s * col_i + c * col_j

    return mat


def spectral_null_ratios(
    singular_values_by_op: dict[str, np.ndarray],
    n_nulls: int,
    n_givens: int,
    rng: np.random.Generator,
) -> dict[tuple[str, str], np.ndarray]:
    ratios = {pair: [] for pair in PAIR_OPS}
    for _ in range(n_nulls):
        null_ops = {
            op: givens_spectrum_matched_matrix(singular_values, rng, n_givens)
            for op, singular_values in singular_values_by_op.items()
        }
        basis = np.vstack([null_ops[op].reshape(1, -1) for op in OPS])
        for a, b in PAIR_OPS:
            comm = null_ops[b] @ null_ops[a] - null_ops[a] @ null_ops[b]
            ratio, _ = project_residual_ratio(comm.reshape(-1), basis)
            ratios[(a, b)].append(ratio)
    return {pair: np.asarray(values) for pair, values in ratios.items()}


def model_alpha_audit(
    model_name: str,
    df: pd.DataFrame,
    vecs_by_text: dict[str, np.ndarray],
    dim: int,
    alpha: float,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows = []
    closure_rows = []
    rng = np.random.default_rng(config["seed"] + abs(hash((model_name, alpha, "spectral"))) % 1_000_000)

    source_all = np.vstack([vecs_by_text[t] for t in df["source"]])
    y_by_op = {op: np.vstack([vecs_by_text[t] for t in df[f"{op}_text"]]) for op in OPS}

    for heldout in [spec.code for spec in LANGS]:
        train_mask = ~df["language"].eq(heldout).to_numpy()
        test_mask = df["language"].eq(heldout).to_numpy()
        x_train = source_all[train_mask]
        x_test = source_all[test_mask]

        maps: dict[str, dict[str, dict[str, np.ndarray]]] = {"linear": {}, "affine": {}}
        for op in OPS:
            y_train = y_by_op[op][train_mask]
            y_test = y_by_op[op][test_mask]
            for method, fit_intercept in [("linear", False), ("affine", True)]:
                matrix, intercept = fit_map(x_train, y_train, alpha, fit_intercept)
                maps[method][op] = {"matrix": matrix, "intercept": intercept}
                pred = x_test @ matrix.T + intercept
                fit_rows.append({
                    "model": model_name,
                    "ridge_alpha": alpha,
                    "heldout_language": heldout,
                    "method": method,
                    "operator": op,
                    "target_cosine": mean_cosine(pred, y_test),
                    "mse": float(np.mean((pred - y_test) ** 2)),
                    "n_test": int(test_mask.sum()),
                })

        for method in ["linear", "affine"]:
            operators = {op: maps[method][op]["matrix"] - np.eye(dim) for op in OPS}
            basis = np.vstack([operators[op].reshape(1, -1) for op in OPS])
            singular_values_by_op = {
                op: np.linalg.svd(operators[op], compute_uv=False)
                for op in OPS
            }
            spectral_by_pair = spectral_null_ratios(
                singular_values_by_op,
                config["n_nulls"],
                config["n_givens"],
                rng,
            )

            for a, b in PAIR_OPS:
                comm = operators[b] @ operators[a] - operators[a] @ operators[b]
                vector = comm.reshape(-1)
                ratio, coeff = project_residual_ratio(vector, basis)
                spectral = spectral_by_pair[(a, b)]
                closure_rows.append({
                    "model": model_name,
                    "ridge_alpha": alpha,
                    "heldout_language": heldout,
                    "method": method,
                    "pair": f"{a}{b}_minus_{b}{a}",
                    "basis_rank": int(np.linalg.matrix_rank(basis)),
                    "matrix_commutator_norm": float(np.linalg.norm(vector)),
                    "closure_residual_ratio": ratio,
                    **summarize(spectral, ratio, "givens_spectral_matched_null"),
                    **{f"coef_{op}": float(coeff[i]) for i, op in enumerate(OPS)},
                })

    return pd.DataFrame(fit_rows), pd.DataFrame(closure_rows)


def read_many(pattern: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in CSV_DIR.glob(pattern)
        if "all_models" not in path.name and "summary" not in path.name and "by_alpha" not in path.name
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate() -> None:
    fit = read_many("operator_fit_*.csv")
    closure = read_many("spectral_closure_*.csv")

    if not fit.empty:
        fit.to_csv(CSV_DIR / "operator_fit_all_models.csv", index=False)
        fit.groupby(["ridge_alpha", "method"]).agg(
            mean_target_cosine=("target_cosine", "mean"),
            mean_mse=("mse", "mean"),
            cells=("target_cosine", "count"),
        ).reset_index().to_csv(CSV_DIR / "operator_fit_by_alpha_method.csv", index=False)

    if not closure.empty:
        closure.to_csv(CSV_DIR / "spectral_closure_all_models.csv", index=False)
        closure.groupby(["ridge_alpha", "method"]).agg(
            mean_closure_residual_ratio=("closure_residual_ratio", "mean"),
            mean_givens_spectral_matched_null=("givens_spectral_matched_null_mean", "mean"),
            mean_p_below_givens_spectral_matched=("empirical_p_below_givens_spectral_matched_null", "mean"),
            cells=("closure_residual_ratio", "count"),
        ).reset_index().to_csv(CSV_DIR / "spectral_closure_by_alpha_method.csv", index=False)
        closure.groupby(["ridge_alpha", "method", "pair"]).agg(
            mean_closure_residual_ratio=("closure_residual_ratio", "mean"),
            mean_givens_spectral_matched_null=("givens_spectral_matched_null_mean", "mean"),
            mean_p_below_givens_spectral_matched=("empirical_p_below_givens_spectral_matched_null", "mean"),
            cells=("closure_residual_ratio", "count"),
        ).reset_index().to_csv(CSV_DIR / "spectral_closure_summary.csv", index=False)


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
        fit, closure = model_alpha_audit(model_name, df, vecs_by_text, dim, alpha, config)
        fit.to_csv(CSV_DIR / f"operator_fit_{safe}__alpha_{label}.csv", index=False)
        closure.to_csv(CSV_DIR / f"spectral_closure_{safe}__alpha_{label}.csv", index=False)
        marker.write_text(
            json.dumps({"model": model_name, "ridge_alpha": alpha, "finished_at": time.ctime()}, indent=2),
            encoding="utf-8",
        )
        aggregate()
        del fit, closure
        gc.collect()

    del raw, vecs, vecs_by_text
    gc.collect()


def main() -> None:
    config = {
        "seed": int(os.getenv("GLT_MOLT_SEED", "20260624")),
        "n_templates_per_language": int(os.getenv("GLT_MOLT_TEMPLATES_PER_LANGUAGE", "160")),
        "pca_dim": int(os.getenv("GLT_MOLT_PCA_DIM", "128")),
        "batch_size": int(os.getenv("GLT_MOLT_BATCH_SIZE", "1")),
        "n_nulls": int(os.getenv("GLT_MOLT_SPECTRAL_NULLS", "300")),
        "n_givens": int(os.getenv("GLT_MOLT_SPECTRAL_GIVENS", "256")),
        "device": os.getenv("LIE_DEVICE", "cpu"),
        "force": os.getenv("GLT_MOLT_FORCE", "0") == "1",
    }
    alphas = parse_alphas()
    models_env = os.getenv("GLT_MOLT_MODELS", "").strip()
    models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else DEFAULT_MODELS
    run_config = {
        **config,
        "ridge_alphas": alphas,
        "models": models,
        "languages": [spec.code for spec in LANGS],
        "operators": OPS,
        "nulls": ["givens_spectral_matched"],
    }
    print("GLT-MOLT SPECTRAL NULLS")
    print(json.dumps(run_config, indent=2), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

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
