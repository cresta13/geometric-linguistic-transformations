from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_lie_endpoint_residualization_audit import DEFAULT_MODELS, safe_model_name  # noqa: E402
from run_lie_multilingual_max_audit import LANGS, PAIR_OPS, embed_texts, fit_pca  # noqa: E402
from run_lie_structure_constants_audit import OPS, build_structure_dataset  # noqa: E402


OUT_DIR = Path("results/experiments/glt_molt_affine_operator_results")
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CKPT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, FIG_DIR, CKPT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def all_texts(df: pd.DataFrame) -> list[str]:
    texts: set[str] = set()
    for col in df.columns:
        if col == "source" or col.endswith("_text"):
            texts.update(df[col].dropna().astype(str))
    return sorted(texts)


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.sum(unit_rows(a) * unit_rows(b), axis=1)))


def fit_affine(x: np.ndarray, y: np.ndarray, alpha: float, fit_intercept: bool) -> tuple[np.ndarray, np.ndarray]:
    reg = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    reg.fit(x, y)
    matrix = reg.coef_
    intercept = reg.intercept_ if fit_intercept else np.zeros(y.shape[1])
    return matrix, intercept


def apply_operator(x: np.ndarray, op_name: str, method: str, maps: dict) -> np.ndarray:
    if method == "additive":
        return x + maps[method][op_name]["delta"]
    matrix = maps[method][op_name]["matrix"]
    intercept = maps[method][op_name]["intercept"]
    return x @ matrix.T + intercept


def random_subspace_ratios(vector: np.ndarray, rank: int, n_nulls: int, rng: np.random.Generator) -> np.ndarray:
    dim = vector.size
    norm = np.linalg.norm(vector) + 1e-12
    ratios = []
    for _ in range(n_nulls):
        mat = rng.normal(size=(dim, rank))
        q, _ = np.linalg.qr(mat)
        pred = q[:, :rank] @ (q[:, :rank].T @ vector)
        ratios.append(float(np.linalg.norm(vector - pred) / norm))
    return np.asarray(ratios)


def project_residual_ratio(vector: np.ndarray, basis: np.ndarray) -> tuple[float, np.ndarray]:
    norm = np.linalg.norm(vector) + 1e-12
    coeff, *_ = np.linalg.lstsq(basis.T, vector, rcond=None)
    pred = basis.T @ coeff
    return float(np.linalg.norm(vector - pred) / norm), coeff


def bracket_coefficients(pair_coeffs: dict[tuple[str, str], np.ndarray], a: str, b: str) -> np.ndarray:
    if a == b:
        return np.zeros(len(OPS))
    if (a, b) in pair_coeffs:
        return pair_coeffs[(a, b)]
    return -pair_coeffs[(b, a)]


def bracket_linear(pair_coeffs: dict[tuple[str, str], np.ndarray], a: str, coeff: np.ndarray) -> np.ndarray:
    out = np.zeros(len(OPS))
    for op, weight in zip(OPS, coeff):
        if abs(weight) > 1e-12:
            out += weight * bracket_coefficients(pair_coeffs, a, op)
    return out


def model_language_audit(model_name: str, df: pd.DataFrame, vecs_by_text: dict[str, np.ndarray], dim: int, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_rows = []
    comp_rows = []
    closure_rows = []
    jacobi_rows = []
    rng = np.random.default_rng(config["seed"] + abs(hash(model_name)) % 1_000_000)

    source_all = np.vstack([vecs_by_text[t] for t in df["source"]])
    y_by_op = {op: np.vstack([vecs_by_text[t] for t in df[f"{op}_text"]]) for op in OPS}

    for heldout in [spec.code for spec in LANGS]:
        train_mask = ~df["language"].eq(heldout).to_numpy()
        test_mask = df["language"].eq(heldout).to_numpy()
        x_train = source_all[train_mask]
        x_test = source_all[test_mask]

        maps = {"additive": {}, "linear": {}, "affine": {}}
        for op in OPS:
            y_train = y_by_op[op][train_mask]
            y_test = y_by_op[op][test_mask]
            delta = (y_train - x_train).mean(axis=0)
            maps["additive"][op] = {"delta": delta}
            linear_matrix, linear_intercept = fit_affine(x_train, y_train, config["ridge_alpha"], fit_intercept=False)
            affine_matrix, affine_intercept = fit_affine(x_train, y_train, config["ridge_alpha"], fit_intercept=True)
            maps["linear"][op] = {"matrix": linear_matrix, "intercept": linear_intercept}
            maps["affine"][op] = {"matrix": affine_matrix, "intercept": affine_intercept}

            for method in ["additive", "linear", "affine"]:
                pred = apply_operator(x_test, op, method, maps)
                fit_rows.append({
                    "model": model_name,
                    "heldout_language": heldout,
                    "method": method,
                    "operator": op,
                    "target_cosine": mean_cosine(pred, y_test),
                    "mse": float(np.mean((pred - y_test) ** 2)),
                    "n_test": int(test_mask.sum()),
                })

        heldout_df = df[test_mask].reset_index(drop=True)
        for a, b in PAIR_OPS:
            actual_ab = np.vstack([vecs_by_text[t] for t in heldout_df[f"{a}{b}_text"]])
            actual_ba = np.vstack([vecs_by_text[t] for t in heldout_df[f"{b}{a}_text"]])
            actual_comm = actual_ab - actual_ba
            for method in ["additive", "linear", "affine"]:
                pred_ab = apply_operator(apply_operator(x_test, a, method, maps), b, method, maps)
                pred_ba = apply_operator(apply_operator(x_test, b, method, maps), a, method, maps)
                pred_comm = pred_ab - pred_ba
                comp_rows.append({
                    "model": model_name,
                    "heldout_language": heldout,
                    "method": method,
                    "pair": f"{a}{b}_minus_{b}{a}",
                    "ab_target_cosine": mean_cosine(pred_ab, actual_ab),
                    "ba_target_cosine": mean_cosine(pred_ba, actual_ba),
                    "commutator_cosine": mean_cosine(pred_comm, actual_comm),
                    "pred_commutator_norm": float(np.linalg.norm(pred_comm.mean(axis=0))),
                    "actual_commutator_norm": float(np.linalg.norm(actual_comm.mean(axis=0))),
                    "n_test": int(test_mask.sum()),
                })

        for method in ["linear", "affine"]:
            matrices = {op: maps[method][op]["matrix"] for op in OPS}
            operators = {op: matrices[op] - np.eye(dim) for op in OPS}
            basis = np.vstack([operators[op].reshape(1, -1) for op in OPS])
            rank = int(np.linalg.matrix_rank(basis))
            pair_coeffs = {}
            for a, b in PAIR_OPS:
                comm = operators[b] @ operators[a] - operators[a] @ operators[b]
                vector = comm.reshape(-1)
                ratio, coeff = project_residual_ratio(vector, basis)
                null = random_subspace_ratios(vector, max(rank, 1), config["n_nulls"], rng)
                pair_coeffs[(a, b)] = coeff
                closure_rows.append({
                    "model": model_name,
                    "heldout_language": heldout,
                    "method": method,
                    "pair": f"{a}{b}_minus_{b}{a}",
                    "basis_rank": rank,
                    "matrix_commutator_norm": float(np.linalg.norm(vector)),
                    "closure_residual_ratio": ratio,
                    "random_subspace_null_mean": float(null.mean()),
                    "random_subspace_null_std": float(null.std(ddof=1)),
                    "empirical_p_below_random_subspace": float(((null <= ratio).sum() + 1) / (len(null) + 1)),
                    **{f"coef_{op}": float(coeff[i]) for i, op in enumerate(OPS)},
                })

            for triple in [("N", "Q", "M"), ("N", "Q", "T"), ("N", "M", "T"), ("Q", "M", "T")]:
                a, b, c = triple
                bc = bracket_coefficients(pair_coeffs, b, c)
                ca = bracket_coefficients(pair_coeffs, c, a)
                ab = bracket_coefficients(pair_coeffs, a, b)
                jac_coeff = bracket_linear(pair_coeffs, a, bc) + bracket_linear(pair_coeffs, b, ca) + bracket_linear(pair_coeffs, c, ab)
                jac_vec = basis.T @ jac_coeff
                scale = np.mean([
                    np.linalg.norm(basis.T @ bracket_coefficients(pair_coeffs, x, y))
                    for x, y in [(a, b), (b, c), (c, a)]
                ]) + 1e-12
                jacobi_rows.append({
                    "model": model_name,
                    "heldout_language": heldout,
                    "method": method,
                    "triple": "".join(triple),
                    "basis_rank": rank,
                    "relative_jacobi_operator_norm": float(np.linalg.norm(jac_vec) / scale),
                    "jacobi_coeff_norm": float(np.linalg.norm(jac_coeff)),
                    **{f"jacobi_coef_{op}": float(jac_coeff[i]) for i, op in enumerate(OPS)},
                })

    return pd.DataFrame(fit_rows), pd.DataFrame(comp_rows), pd.DataFrame(closure_rows), pd.DataFrame(jacobi_rows)


def run_model(model_name: str, df: pd.DataFrame, config: dict) -> None:
    safe = safe_model_name(model_name)
    marker = CKPT_DIR / f"{safe}.done.json"
    if marker.exists() and not config["force"]:
        print(f"SKIP {model_name}")
        return

    print(f"\n=== MODEL {model_name} ===", flush=True)
    texts = all_texts(df)
    print(f"templates={len(df)} texts={len(texts)}", flush=True)
    raw = embed_texts(model_name, texts, config["device"], config["batch_size"])
    vecs = fit_pca(raw, config["pca_dim"])
    vecs_by_text = dict(zip(texts, vecs))
    dim = vecs.shape[1]

    fit, comp, closure, jacobi = model_language_audit(model_name, df, vecs_by_text, dim, config)
    fit.to_csv(CSV_DIR / f"operator_fit_{safe}.csv", index=False)
    comp.to_csv(CSV_DIR / f"composition_prediction_{safe}.csv", index=False)
    closure.to_csv(CSV_DIR / f"matrix_closure_{safe}.csv", index=False)
    jacobi.to_csv(CSV_DIR / f"matrix_jacobi_{safe}.csv", index=False)
    marker.write_text(json.dumps({"model": model_name, "finished_at": time.ctime()}, indent=2), encoding="utf-8")
    del raw, vecs, vecs_by_text, fit, comp, closure, jacobi
    gc.collect()


def read_many(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in CSV_DIR.glob(pattern) if "all_models" not in path.name and "summary" not in path.name]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate() -> None:
    fit = read_many("operator_fit_*.csv")
    comp = read_many("composition_prediction_*.csv")
    closure = read_many("matrix_closure_*.csv")
    jacobi = read_many("matrix_jacobi_*.csv")
    if not fit.empty:
        fit.to_csv(CSV_DIR / "operator_fit_all_models.csv", index=False)
        fit.groupby(["method", "operator"]).agg(
            mean_target_cosine=("target_cosine", "mean"),
            std_target_cosine=("target_cosine", "std"),
            mean_mse=("mse", "mean"),
            cells=("target_cosine", "count"),
        ).reset_index().to_csv(CSV_DIR / "operator_fit_summary.csv", index=False)
    if not comp.empty:
        comp.to_csv(CSV_DIR / "composition_prediction_all_models.csv", index=False)
        comp.groupby(["method", "pair"]).agg(
            mean_ab_target_cosine=("ab_target_cosine", "mean"),
            mean_ba_target_cosine=("ba_target_cosine", "mean"),
            mean_commutator_cosine=("commutator_cosine", "mean"),
            cells=("ab_target_cosine", "count"),
        ).reset_index().to_csv(CSV_DIR / "composition_prediction_summary.csv", index=False)
    if not closure.empty:
        closure.to_csv(CSV_DIR / "matrix_closure_all_models.csv", index=False)
        summary = closure.groupby(["method", "pair"]).agg(
            mean_closure_residual_ratio=("closure_residual_ratio", "mean"),
            std_closure_residual_ratio=("closure_residual_ratio", "std"),
            mean_random_subspace_null=("random_subspace_null_mean", "mean"),
            mean_empirical_p=("empirical_p_below_random_subspace", "mean"),
            cells=("closure_residual_ratio", "count"),
        ).reset_index()
        summary.to_csv(CSV_DIR / "matrix_closure_summary.csv", index=False)
        for method, sub in summary.groupby("method"):
            plt.figure(figsize=(10, 5))
            x = np.arange(len(sub))
            plt.bar(x - 0.2, sub["mean_closure_residual_ratio"], 0.4, label="operator-span residual")
            plt.bar(x + 0.2, sub["mean_random_subspace_null"], 0.4, label="random subspace null")
            plt.xticks(x, sub["pair"], rotation=30, ha="right")
            plt.ylabel("Residual ratio")
            plt.title(f"GLT-MOLT matrix closure audit ({method})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"matrix_closure_{method}.png", dpi=220, bbox_inches="tight")
            plt.close()
    if not jacobi.empty:
        jacobi.to_csv(CSV_DIR / "matrix_jacobi_all_models.csv", index=False)
        jacobi.groupby(["method", "triple"]).agg(
            mean_relative_jacobi_operator_norm=("relative_jacobi_operator_norm", "mean"),
            std_relative_jacobi_operator_norm=("relative_jacobi_operator_norm", "std"),
            cells=("relative_jacobi_operator_norm", "count"),
        ).reset_index().to_csv(CSV_DIR / "matrix_jacobi_summary.csv", index=False)


def main() -> None:
    config = {
        "seed": int(os.getenv("GLT_MOLT_SEED", "20260624")),
        "n_templates_per_language": int(os.getenv("GLT_MOLT_TEMPLATES_PER_LANGUAGE", "160")),
        "pca_dim": int(os.getenv("GLT_MOLT_PCA_DIM", "128")),
        "batch_size": int(os.getenv("GLT_MOLT_BATCH_SIZE", "8")),
        "ridge_alpha": float(os.getenv("GLT_MOLT_RIDGE_ALPHA", "10.0")),
        "n_nulls": int(os.getenv("GLT_MOLT_NULLS", "200")),
        "device": os.getenv("LIE_DEVICE", "cpu"),
        "force": os.getenv("GLT_MOLT_FORCE", "0") == "1",
    }
    models_env = os.getenv("GLT_MOLT_MODELS", "").strip()
    models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else DEFAULT_MODELS
    config_with_models = {**config, "models": models, "languages": [spec.code for spec in LANGS], "operators": OPS}
    print("GLT-MOLT AFFINE OPERATOR AUDIT")
    print(json.dumps(config_with_models, indent=2), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps(config_with_models, indent=2), encoding="utf-8")

    df = build_structure_dataset(config["n_templates_per_language"], config["seed"])
    df.to_csv(CSV_DIR / "glt_molt_dataset.csv", index=False)
    print(f"templates={len(df)} texts={len(all_texts(df))}", flush=True)

    failures = []
    for model in models:
        try:
            run_model(model, df, config)
            aggregate()
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
