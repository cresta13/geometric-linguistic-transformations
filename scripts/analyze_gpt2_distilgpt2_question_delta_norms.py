from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))

os.environ.setdefault("STEERING_CLASSES", "question,negation,modality,tense_shift")
os.environ.setdefault("STEERING_EVAL_CLASSES", "question")

from run_gpt2_activation_steering_pilot import build_dataset, last_token_hidden  # noqa: E402


MODELS = [m.strip() for m in os.getenv("QUESTION_DELTA_MODELS", "gpt2,distilgpt2").split(",") if m.strip()]
TARGET_CLASS = os.getenv("QUESTION_DELTA_CLASS", "question")
MAX_ROWS = int(os.getenv("QUESTION_DELTA_MAX_ROWS", "100"))
OUT_DIR = Path(
    os.getenv(
        "QUESTION_DELTA_OUT_DIR",
        "results/experiments/gpt2_distilgpt2_question_delta_norms_20260716_results",
    )
)
CSV_DIR = OUT_DIR / "csv"
STATUS_PATH = OUT_DIR / "run_status.json"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def write_status(**kwargs) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current = {}
    if STATUS_PATH.exists():
        try:
            current = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(kwargs)
    current["updated_at"] = now()
    STATUS_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


def pairwise_sample_cosine(vectors: np.ndarray, max_pairs: int = 5000, seed: int = 20260716) -> np.ndarray:
    if len(vectors) < 2:
        return np.array([], dtype=np.float32)
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    unit = vectors / norms
    total_pairs = len(unit) * (len(unit) - 1) // 2
    if total_pairs <= max_pairs:
        vals = []
        for i in range(len(unit)):
            vals.extend((unit[i + 1 :] @ unit[i]).tolist())
        return np.array(vals, dtype=np.float32)
    idx_a = rng.integers(0, len(unit), size=max_pairs)
    idx_b = rng.integers(0, len(unit), size=max_pairs)
    keep = idx_a != idx_b
    idx_a = idx_a[keep]
    idx_b = idx_b[keep]
    return np.sum(unit[idx_a] * unit[idx_b], axis=1).astype(np.float32)


def get_layers(model) -> list[int]:
    override = os.getenv("QUESTION_DELTA_LAYERS", "").strip()
    n_layers = len(model.transformer.h)
    if override:
        layers = []
        for raw in override.split(","):
            if raw.strip():
                idx = int(raw)
                if idx < 0:
                    idx = n_layers + idx
                if 0 <= idx < n_layers:
                    layers.append(idx)
        return sorted(set(layers))
    return list(range(n_layers))


def analyze_model(model_name: str, rows_df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    write_status(phase="loading_model", current_model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    layers = get_layers(model)
    write_status(phase="embedding", current_model=model_name, current_layers=layers)
    source_h = last_token_hidden(tokenizer, model, rows_df["source"].tolist(), layers)
    target_h = last_token_hidden(tokenizer, model, rows_df["target"].tolist(), layers)
    n_layers = len(model.transformer.h)

    summary_rows = []
    raw_rows = []
    for layer in layers:
        delta = target_h[layer] - source_h[layer]
        norms = np.linalg.norm(delta, axis=1)
        centroid = delta.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        mean_norm = float(norms.mean())
        pair_cos = pairwise_sample_cosine(delta)
        unit = delta / (norms[:, None] + 1e-12)
        resultant = np.linalg.norm(unit.mean(axis=0))
        summary_rows.append(
            {
                "model": model_name,
                "class": TARGET_CLASS,
                "layer": layer,
                "n_layers": n_layers,
                "relative_layer": layer / max(1, n_layers - 1),
                "n": int(len(delta)),
                "mean_delta_norm": mean_norm,
                "median_delta_norm": float(np.median(norms)),
                "std_delta_norm": float(norms.std()),
                "centroid_norm": centroid_norm,
                "centroid_to_mean_norm": centroid_norm / (mean_norm + 1e-12),
                "mean_pairwise_cosine": float(pair_cos.mean()) if len(pair_cos) else np.nan,
                "median_pairwise_cosine": float(np.median(pair_cos)) if len(pair_cos) else np.nan,
                "resultant_length_unit_deltas": float(resultant),
            }
        )
        for idx, norm in enumerate(norms):
            raw_rows.append(
                {
                    "model": model_name,
                    "class": TARGET_CLASS,
                    "layer": layer,
                    "relative_layer": layer / max(1, n_layers - 1),
                    "row_index": idx,
                    "source": rows_df.iloc[idx]["source"],
                    "target": rows_df.iloc[idx]["target"],
                    "delta_norm": float(norm),
                }
            )
    del model
    return summary_rows, raw_rows


def build_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rel in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        nearest = []
        for model_name, group in summary_df.groupby("model"):
            idx = (group["relative_layer"] - rel).abs().idxmin()
            nearest.append(group.loc[idx])
        if len(nearest) == 2:
            a, b = nearest
            if a["model"] == "distilgpt2":
                distil, gpt = a, b
            else:
                gpt, distil = a, b
            rows.append(
                {
                    "relative_layer_target": rel,
                    "gpt2_layer": int(gpt["layer"]),
                    "distilgpt2_layer": int(distil["layer"]),
                    "gpt2_mean_delta_norm": float(gpt["mean_delta_norm"]),
                    "distilgpt2_mean_delta_norm": float(distil["mean_delta_norm"]),
                    "distil_to_gpt2_mean_norm_ratio": float(distil["mean_delta_norm"] / (gpt["mean_delta_norm"] + 1e-12)),
                    "gpt2_centroid_norm": float(gpt["centroid_norm"]),
                    "distilgpt2_centroid_norm": float(distil["centroid_norm"]),
                    "distil_to_gpt2_centroid_norm_ratio": float(distil["centroid_norm"] / (gpt["centroid_norm"] + 1e-12)),
                    "gpt2_mean_pairwise_cosine": float(gpt["mean_pairwise_cosine"]),
                    "distilgpt2_mean_pairwise_cosine": float(distil["mean_pairwise_cosine"]),
                }
            )
    return pd.DataFrame(rows)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    write_status(status="running", started_at=now(), models=MODELS, target_class=TARGET_CLASS, phase="building_dataset")
    df = build_dataset()
    rows_df = df[df["split"].eq("test") & df["class"].eq(TARGET_CLASS)].head(MAX_ROWS).reset_index(drop=True)
    rows_df.to_csv(CSV_DIR / "question_delta_sources.csv", index=False)

    summary_rows = []
    raw_rows = []
    failures = []
    for model_name in MODELS:
        try:
            s_rows, r_rows = analyze_model(model_name, rows_df)
            summary_rows.extend(s_rows)
            raw_rows.extend(r_rows)
            pd.DataFrame(summary_rows).to_csv(CSV_DIR / "question_delta_norm_summary.csv", index=False)
            pd.DataFrame(raw_rows).to_csv(CSV_DIR / "question_delta_norm_raw.csv", index=False)
        except Exception as exc:  # pragma: no cover - run artifact should record failures.
            failures.append({"model": model_name, "error": repr(exc)})
            write_status(phase="failed_model", current_model=model_name, failures=failures)

    summary_df = pd.DataFrame(summary_rows)
    comparison_df = build_comparison(summary_df) if not summary_df.empty else pd.DataFrame()
    summary_df.to_csv(CSV_DIR / "question_delta_norm_summary.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(CSV_DIR / "question_delta_norm_raw.csv", index=False)
    comparison_df.to_csv(CSV_DIR / "gpt2_distilgpt2_relative_layer_comparison.csv", index=False)
    write_status(
        status="finished" if not failures else "finished_with_failures",
        phase="finished",
        rows=int(len(summary_df)),
        raw_rows=int(len(raw_rows)),
        failures=failures,
        finished_at=now(),
        summary_csv=str(CSV_DIR / "question_delta_norm_summary.csv"),
        comparison_csv=str(CSV_DIR / "gpt2_distilgpt2_relative_layer_comparison.csv"),
    )
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    if not comparison_df.empty:
        print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    run()
