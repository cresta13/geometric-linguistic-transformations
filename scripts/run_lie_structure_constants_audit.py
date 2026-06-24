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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_lie_endpoint_residualization_audit import DEFAULT_MODELS, safe_model_name  # noqa: E402
from run_lie_multilingual_max_audit import (  # noqa: E402
    LANGS,
    PAIR_OPS,
    TRIPLE_OPS,
    clean,
    compose_text,
    embed_texts,
    fit_pca,
)


OUT_DIR = Path("results/experiments/lie_structure_constants_results")
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CKPT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, FIG_DIR, CKPT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

OPS = ["N", "Q", "M", "T"]
OP_INDEX = {op: i for i, op in enumerate(OPS)}


def build_structure_dataset(n_templates_per_language: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for spec in LANGS:
        combos = [(subject, action, context) for subject in spec.subjects for action in spec.actions for context in spec.contexts]
        rng.shuffle(combos)
        combos = combos[:n_templates_per_language]
        for template_id, (subject, action, context) in enumerate(combos):
            past, base, obj = action
            form = {
                "neg": rng.choice(spec.neg),
                "mod": rng.choice(spec.mod),
                "future": rng.choice(spec.future),
            }
            source = clean(spec.source.format(subject=subject, past=past, obj=obj, context=context))
            row = {
                "language": spec.code,
                "template_id": template_id,
                "source": source,
            }
            for op in OPS:
                row[f"{op}_text"] = compose_text(spec, (op,), subject, past, base, obj, context, form)
            for a, b in PAIR_OPS:
                row[f"{a}{b}_text"] = compose_text(spec, (a, b), subject, past, base, obj, context, form)
                row[f"{b}{a}_text"] = compose_text(spec, (b, a), subject, past, base, obj, context, form)
            for triple in TRIPLE_OPS:
                for name, perm in [
                    ("abc", (0, 1, 2)),
                    ("bca", (1, 2, 0)),
                    ("cab", (2, 0, 1)),
                    ("acb", (0, 2, 1)),
                    ("cba", (2, 1, 0)),
                    ("bac", (1, 0, 2)),
                ]:
                    seq = tuple(triple[i] for i in perm)
                    row[f"{''.join(triple)}_{name}_text"] = compose_text(spec, seq, subject, past, base, obj, context, form)
            rows.append(row)
    return pd.DataFrame(rows)


def all_texts(df: pd.DataFrame) -> list[str]:
    texts: set[str] = set()
    for col in df.columns:
        if col == "source" or col.endswith("_text"):
            texts.update(df[col].dropna().astype(str))
    return sorted(texts)


def project_residual_ratio(vector: np.ndarray, basis_rows: np.ndarray) -> tuple[float, float]:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12 or basis_rows.size == 0:
        return 1.0, norm
    coef, *_ = np.linalg.lstsq(basis_rows.T, vector, rcond=None)
    pred = basis_rows.T @ coef
    residual = vector - pred
    return float(np.linalg.norm(residual) / (norm + 1e-12)), norm


def random_subspace_ratios(vector: np.ndarray, dim: int, rank: int, n_nulls: int, rng: np.random.Generator) -> np.ndarray:
    ratios = []
    for _ in range(n_nulls):
        mat = rng.normal(size=(dim, rank))
        q, _ = np.linalg.qr(mat)
        basis_rows = q[:, :rank].T
        ratio, _ = project_residual_ratio(vector, basis_rows)
        ratios.append(ratio)
    return np.asarray(ratios)


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
    rng = np.random.default_rng(config["seed"] + abs(hash(model_name)) % 1_000_000)

    op_rows = []
    pair_rows = []
    jacobi_rows = []

    for lang, sub in df.groupby("language"):
        source = np.vstack([vecs_by_text[t] for t in sub["source"]])
        op_centroids = {}
        for op in OPS:
            deltas = np.vstack([vecs_by_text[t] for t in sub[f"{op}_text"]]) - source
            centroid = deltas.mean(axis=0)
            op_centroids[op] = centroid
            op_rows.append({
                "model": model_name,
                "language": lang,
                "operator": op,
                "centroid_norm": float(np.linalg.norm(centroid)),
                "n": int(len(sub)),
            })

        basis = np.vstack([op_centroids[op] for op in OPS])
        rank = int(np.linalg.matrix_rank(basis))
        pair_coeffs: dict[tuple[str, str], np.ndarray] = {}

        for a, b in PAIR_OPS:
            ab = np.vstack([vecs_by_text[t] for t in sub[f"{a}{b}_text"]]) - source
            ba = np.vstack([vecs_by_text[t] for t in sub[f"{b}{a}_text"]]) - source
            comms = ab - ba
            comm = comms.mean(axis=0)
            coeff, *_ = np.linalg.lstsq(basis.T, comm, rcond=None)
            pred = basis.T @ coeff
            residual = comm - pred
            closure_ratio = float(np.linalg.norm(residual) / (np.linalg.norm(comm) + 1e-12))
            null = random_subspace_ratios(comm, dim, max(rank, 1), config["n_nulls"], rng)
            pair_coeffs[(a, b)] = coeff
            pair_rows.append({
                "model": model_name,
                "language": lang,
                "pair": f"{a}{b}_minus_{b}{a}",
                "basis_rank": rank,
                "commutator_norm": float(np.linalg.norm(comm)),
                "closure_residual_ratio": closure_ratio,
                "random_subspace_null_mean": float(null.mean()),
                "random_subspace_null_std": float(null.std(ddof=1)),
                "empirical_p_below_random_subspace": float(((null <= closure_ratio).sum() + 1) / (len(null) + 1)),
                **{f"coef_{op}": float(coeff[i]) for i, op in enumerate(OPS)},
            })

        for triple in TRIPLE_OPS:
            a, b, c = triple
            bc = bracket_coefficients(pair_coeffs, b, c)
            ca = bracket_coefficients(pair_coeffs, c, a)
            ab = bracket_coefficients(pair_coeffs, a, b)
            jac_coeff = bracket_linear(pair_coeffs, a, bc) + bracket_linear(pair_coeffs, b, ca) + bracket_linear(pair_coeffs, c, ab)
            jac_vec = basis.T @ jac_coeff
            scale = np.mean([np.linalg.norm(basis.T @ bracket_coefficients(pair_coeffs, x, y)) for x, y in [(a, b), (b, c), (c, a)]]) + 1e-12
            jacobi_rows.append({
                "model": model_name,
                "language": lang,
                "triple": "".join(triple),
                "basis_rank": rank,
                "jacobi_coeff_norm": float(np.linalg.norm(jac_coeff)),
                "jacobi_embedding_norm": float(np.linalg.norm(jac_vec)),
                "relative_jacobi_embedding_norm": float(np.linalg.norm(jac_vec) / scale),
                **{f"jacobi_coef_{op}": float(jac_coeff[i]) for i, op in enumerate(OPS)},
            })

    pd.DataFrame(op_rows).to_csv(CSV_DIR / f"operator_centroids_{safe}.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(CSV_DIR / f"closure_structure_constants_{safe}.csv", index=False)
    pd.DataFrame(jacobi_rows).to_csv(CSV_DIR / f"jacobi_closure_{safe}.csv", index=False)
    marker.write_text(json.dumps({"model": model_name, "finished_at": time.ctime()}, indent=2), encoding="utf-8")
    del raw, vecs, vecs_by_text
    gc.collect()


def read_many(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in CSV_DIR.glob(pattern) if "all_models" not in path.name and "global" not in path.name]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate() -> None:
    pairs = read_many("closure_structure_constants_*.csv")
    jac = read_many("jacobi_closure_*.csv")
    ops = read_many("operator_centroids_*.csv")
    if not pairs.empty:
        pairs.to_csv(CSV_DIR / "closure_structure_constants_all_models.csv", index=False)
        summary = pairs.groupby("pair").agg(
            mean_closure_residual_ratio=("closure_residual_ratio", "mean"),
            std_closure_residual_ratio=("closure_residual_ratio", "std"),
            mean_random_subspace_null=("random_subspace_null_mean", "mean"),
            mean_empirical_p=("empirical_p_below_random_subspace", "mean"),
            cells=("closure_residual_ratio", "count"),
        ).reset_index()
        summary.to_csv(CSV_DIR / "closure_global_summary.csv", index=False)
        plt.figure(figsize=(10, 5))
        x = np.arange(len(summary))
        plt.bar(x - 0.2, summary["mean_closure_residual_ratio"], 0.4, label="operator-span residual")
        plt.bar(x + 0.2, summary["mean_random_subspace_null"], 0.4, label="random subspace null")
        plt.xticks(x, summary["pair"], rotation=30, ha="right")
        plt.ylabel("Residual ratio")
        plt.title("Lie-style closure audit: commutators projected onto operator span")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "01_closure_residual_vs_random_subspace.png", dpi=220, bbox_inches="tight")
        plt.close()
    if not jac.empty:
        jac.to_csv(CSV_DIR / "jacobi_closure_all_models.csv", index=False)
        jac_summary = jac.groupby("triple").agg(
            mean_relative_jacobi_embedding_norm=("relative_jacobi_embedding_norm", "mean"),
            std_relative_jacobi_embedding_norm=("relative_jacobi_embedding_norm", "std"),
            cells=("relative_jacobi_embedding_norm", "count"),
        ).reset_index()
        jac_summary.to_csv(CSV_DIR / "jacobi_closure_global_summary.csv", index=False)
    if not ops.empty:
        ops.to_csv(CSV_DIR / "operator_centroids_all_models.csv", index=False)


def main() -> None:
    config = {
        "seed": int(os.getenv("LIE_STRUCTURE_SEED", "20260624")),
        "n_templates_per_language": int(os.getenv("LIE_STRUCTURE_TEMPLATES_PER_LANGUAGE", "160")),
        "pca_dim": int(os.getenv("LIE_STRUCTURE_PCA_DIM", "128")),
        "batch_size": int(os.getenv("LIE_STRUCTURE_BATCH_SIZE", "8")),
        "n_nulls": int(os.getenv("LIE_STRUCTURE_NULLS", "1000")),
        "device": os.getenv("LIE_DEVICE", "cpu"),
        "force": os.getenv("LIE_STRUCTURE_FORCE", "0") == "1",
    }
    models_env = os.getenv("LIE_STRUCTURE_MODELS", "").strip()
    models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else DEFAULT_MODELS
    config_with_models = {**config, "models": models, "languages": [spec.code for spec in LANGS], "operators": OPS}
    print("LIE STRUCTURE CONSTANTS AUDIT")
    print(json.dumps(config_with_models, indent=2), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps(config_with_models, indent=2), encoding="utf-8")

    df = build_structure_dataset(config["n_templates_per_language"], config["seed"])
    df.to_csv(CSV_DIR / "structure_constants_dataset.csv", index=False)
    print(f"templates={len(df)} texts={len(all_texts(df))}", flush=True)

    failures = []
    for model in models:
        try:
            run_model(model, df, config)
            aggregate()
        except Exception as exc:  # keep overnight run moving
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
