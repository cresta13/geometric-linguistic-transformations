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

from run_glt_molt_affine_operator_audit import all_texts, project_residual_ratio, random_subspace_ratios  # noqa: E402
from run_lie_endpoint_residualization_audit import DEFAULT_MODELS, safe_model_name  # noqa: E402
from run_lie_multilingual_max_audit import LANGS, PAIR_OPS, embed_texts, fit_pca  # noqa: E402
from run_lie_structure_constants_audit import OPS, build_structure_dataset  # noqa: E402


OUT_DIR = Path(os.getenv("GLT_MOLT_MATCHED_OUT_DIR", "results/experiments/glt_molt_matched_nulls_results"))
CSV_DIR = OUT_DIR / "csv"
CKPT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, CKPT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def parse_alphas() -> list[float]:
    raw = os.getenv("GLT_MOLT_MATCHED_ALPHAS", "10,100")
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def alpha_label(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.sum(unit_rows(a) * unit_rows(b), axis=1)))


def fit_map(x: np.ndarray, y: np.ndarray, alpha: float, fit_intercept: bool) -> tuple[np.ndarray, np.ndarray]:
    reg = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    reg.fit(x, y)
    intercept = reg.intercept_ if fit_intercept else np.zeros(y.shape[1])
    return reg.coef_, intercept


def signed_permute_matrix(mat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rows = rng.permutation(mat.shape[0])
    cols = rng.permutation(mat.shape[1])
    row_signs = rng.choice(np.array([-1.0, 1.0]), size=mat.shape[0])
    col_signs = rng.choice(np.array([-1.0, 1.0]), size=mat.shape[1])
    return row_signs[:, None] * mat[rows][:, cols] * col_signs[None, :]


def gaussian_norm_matched_ops(operators: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    for op, mat in operators.items():
        sample = rng.normal(size=mat.shape)
        sample *= np.linalg.norm(mat) / (np.linalg.norm(sample) + 1e-12)
        out[op] = sample
    return out


def signed_permutation_matched_ops(operators: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {op: signed_permute_matrix(mat, rng) for op, mat in operators.items()}


def null_closure_ratios(
    operators: dict[str, np.ndarray],
    n_nulls: int,
    rng: np.random.Generator,
    null_kind: str,
) -> dict[tuple[str, str], np.ndarray]:
    ratios = {pair: [] for pair in PAIR_OPS}
    for _ in range(n_nulls):
        if null_kind == "gaussian_norm_matched":
            null_ops = gaussian_norm_matched_ops(operators, rng)
        elif null_kind == "signed_permutation_matched":
            null_ops = signed_permutation_matched_ops(operators, rng)
        else:
            raise ValueError(f"unknown null kind: {null_kind}")
        basis = np.vstack([null_ops[op].reshape(1, -1) for op in OPS])
        for a, b in PAIR_OPS:
            comm = null_ops[b] @ null_ops[a] - null_ops[a] @ null_ops[b]
            ratio, _ = project_residual_ratio(comm.reshape(-1), basis)
            ratios[(a, b)].append(ratio)
    return {pair: np.asarray(values) for pair, values in ratios.items()}


def summarize(values: np.ndarray, observed: float, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"empirical_p_below_{prefix}": float(((values <= observed).sum() + 1) / (len(values) + 1)),
    }


def model_alpha_audit(
    model_name: str,
    df: pd.DataFrame,
    vecs_by_text: dict[str, np.ndarray],
    dim: int,
    alpha: float,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_rows = []
    closure_rows = []
    norm_rows = []
    rng = np.random.default_rng(config["seed"] + abs(hash((model_name, alpha))) % 1_000_000)

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
            rank = int(np.linalg.matrix_rank(basis))
            gaussian_matched_by_pair = null_closure_ratios(
                operators,
                config["n_nulls"],
                rng,
                "gaussian_norm_matched",
            )
            signed_perm_by_pair = null_closure_ratios(
                operators,
                config["n_nulls"],
                rng,
                "signed_permutation_matched",
            )

            for op in OPS:
                mat = operators[op]
                norm_rows.append({
                    "model": model_name,
                    "ridge_alpha": alpha,
                    "heldout_language": heldout,
                    "method": method,
                    "operator": op,
                    "operator_fro_norm": float(np.linalg.norm(mat)),
                    "operator_spectral_norm": float(np.linalg.norm(mat, ord=2)),
                    "operator_trace": float(np.trace(mat)),
                })

            for a, b in PAIR_OPS:
                comm = operators[b] @ operators[a] - operators[a] @ operators[b]
                vector = comm.reshape(-1)
                ratio, coeff = project_residual_ratio(vector, basis)
                random_subspace = random_subspace_ratios(vector, max(rank, 1), config["n_nulls"], rng)
                gaussian_matched = gaussian_matched_by_pair[(a, b)]
                signed_perm = signed_perm_by_pair[(a, b)]
                closure_rows.append({
                    "model": model_name,
                    "ridge_alpha": alpha,
                    "heldout_language": heldout,
                    "method": method,
                    "pair": f"{a}{b}_minus_{b}{a}",
                    "basis_rank": rank,
                    "matrix_commutator_norm": float(np.linalg.norm(vector)),
                    "closure_residual_ratio": ratio,
                    **summarize(random_subspace, ratio, "random_subspace_null"),
                    **summarize(gaussian_matched, ratio, "gaussian_norm_matched_null"),
                    **summarize(signed_perm, ratio, "signed_permutation_matched_null"),
                    **{f"coef_{op}": float(coeff[i]) for i, op in enumerate(OPS)},
                })

    return pd.DataFrame(fit_rows), pd.DataFrame(closure_rows), pd.DataFrame(norm_rows)


def read_many(pattern: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in CSV_DIR.glob(pattern)
        if "all_models" not in path.name and "summary" not in path.name and "by_alpha" not in path.name
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate() -> None:
    fit = read_many("operator_fit_*.csv")
    closure = read_many("matched_closure_*.csv")
    norms = read_many("operator_norms_*.csv")

    if not fit.empty:
        fit.to_csv(CSV_DIR / "operator_fit_all_models.csv", index=False)
        fit.groupby(["ridge_alpha", "method"]).agg(
            mean_target_cosine=("target_cosine", "mean"),
            mean_mse=("mse", "mean"),
            cells=("target_cosine", "count"),
        ).reset_index().to_csv(CSV_DIR / "operator_fit_by_alpha_method.csv", index=False)

    if not norms.empty:
        norms.to_csv(CSV_DIR / "operator_norms_all_models.csv", index=False)
        norms.groupby(["ridge_alpha", "method", "operator"]).agg(
            mean_fro_norm=("operator_fro_norm", "mean"),
            mean_spectral_norm=("operator_spectral_norm", "mean"),
            mean_trace=("operator_trace", "mean"),
            cells=("operator_fro_norm", "count"),
        ).reset_index().to_csv(CSV_DIR / "operator_norms_summary.csv", index=False)

    if not closure.empty:
        closure.to_csv(CSV_DIR / "matched_closure_all_models.csv", index=False)
        closure.groupby(["ridge_alpha", "method"]).agg(
            mean_closure_residual_ratio=("closure_residual_ratio", "mean"),
            mean_random_subspace_null=("random_subspace_null_mean", "mean"),
            mean_gaussian_norm_matched_null=("gaussian_norm_matched_null_mean", "mean"),
            mean_signed_permutation_matched_null=("signed_permutation_matched_null_mean", "mean"),
            mean_p_below_random_subspace=("empirical_p_below_random_subspace_null", "mean"),
            mean_p_below_gaussian_norm_matched=("empirical_p_below_gaussian_norm_matched_null", "mean"),
            mean_p_below_signed_permutation_matched=("empirical_p_below_signed_permutation_matched_null", "mean"),
            cells=("closure_residual_ratio", "count"),
        ).reset_index().to_csv(CSV_DIR / "matched_closure_by_alpha_method.csv", index=False)
        closure.groupby(["ridge_alpha", "method", "pair"]).agg(
            mean_closure_residual_ratio=("closure_residual_ratio", "mean"),
            mean_random_subspace_null=("random_subspace_null_mean", "mean"),
            mean_gaussian_norm_matched_null=("gaussian_norm_matched_null_mean", "mean"),
            mean_signed_permutation_matched_null=("signed_permutation_matched_null_mean", "mean"),
            cells=("closure_residual_ratio", "count"),
        ).reset_index().to_csv(CSV_DIR / "matched_closure_summary.csv", index=False)


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
        fit, closure, norms = model_alpha_audit(model_name, df, vecs_by_text, dim, alpha, config)
        fit.to_csv(CSV_DIR / f"operator_fit_{safe}__alpha_{label}.csv", index=False)
        closure.to_csv(CSV_DIR / f"matched_closure_{safe}__alpha_{label}.csv", index=False)
        norms.to_csv(CSV_DIR / f"operator_norms_{safe}__alpha_{label}.csv", index=False)
        marker.write_text(
            json.dumps({"model": model_name, "ridge_alpha": alpha, "finished_at": time.ctime()}, indent=2),
            encoding="utf-8",
        )
        aggregate()
        del fit, closure, norms
        gc.collect()

    del raw, vecs, vecs_by_text
    gc.collect()


def main() -> None:
    config = {
        "seed": int(os.getenv("GLT_MOLT_SEED", "20260624")),
        "n_templates_per_language": int(os.getenv("GLT_MOLT_TEMPLATES_PER_LANGUAGE", "160")),
        "pca_dim": int(os.getenv("GLT_MOLT_PCA_DIM", "128")),
        "batch_size": int(os.getenv("GLT_MOLT_BATCH_SIZE", "1")),
        "n_nulls": int(os.getenv("GLT_MOLT_MATCHED_NULLS", "1000")),
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
        "nulls": ["random_subspace", "gaussian_norm_matched", "signed_permutation_matched"],
    }
    print("GLT-MOLT MATCHED NULLS")
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
