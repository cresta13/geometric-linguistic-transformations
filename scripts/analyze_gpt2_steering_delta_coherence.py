from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))

os.environ.setdefault("STEERING_CLASSES", "question,negation,modality,tense_shift")
os.environ.setdefault("STEERING_EVAL_CLASSES", "question,negation")
os.environ.setdefault("STEERING_LAYERS", ",".join(str(i) for i in range(12)))

from run_gpt2_activation_steering_pilot import build_dataset, last_token_hidden  # noqa: E402


OUT_DIR = Path(
    os.getenv(
        "STEERING_COHERENCE_OUT_DIR",
        "results/experiments/gpt2_steering_delta_coherence_20260716_results",
    )
)
CSV_DIR = OUT_DIR / "csv"
STATUS_PATH = OUT_DIR / "run_status.json"

MODEL_NAME = os.getenv("STEERING_COHERENCE_MODEL", "gpt2")
CLASSES = [c.strip() for c in os.getenv("STEERING_COHERENCE_CLASSES", "question,negation").split(",") if c.strip()]
LAYERS = [int(x) for x in os.getenv("STEERING_COHERENCE_LAYERS", ",".join(str(i) for i in range(12))).split(",") if x.strip()]
MAX_ROWS_PER_CLASS = int(os.getenv("STEERING_COHERENCE_MAX_ROWS_PER_CLASS", "400"))
SEED = int(os.getenv("STEERING_COHERENCE_SEED", "20260716"))


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
    STATUS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def upper_triangle_values(sim: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(sim.shape[0], k=1)
    return sim[idx]


def class_coherence(vectors: np.ndarray) -> dict[str, float]:
    unit = l2_normalize(vectors)
    sim = unit @ unit.T
    vals = upper_triangle_values(sim)
    centroid = vectors.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    mean_norm = float(np.linalg.norm(vectors, axis=1).mean())
    resultant = float(np.linalg.norm(unit.mean(axis=0)))
    return {
        "n": float(len(vectors)),
        "mean_pairwise_cosine": float(vals.mean()),
        "median_pairwise_cosine": float(np.median(vals)),
        "std_pairwise_cosine": float(vals.std()),
        "p10_pairwise_cosine": float(np.quantile(vals, 0.10)),
        "p90_pairwise_cosine": float(np.quantile(vals, 0.90)),
        "centroid_norm": centroid_norm,
        "mean_delta_norm": mean_norm,
        "centroid_to_mean_norm": float(centroid_norm / (mean_norm + 1e-12)),
        "resultant_length_unit_deltas": resultant,
    }


def between_class_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    sim = l2_normalize(a) @ l2_normalize(b).T
    return {
        "between_mean_cosine": float(sim.mean()),
        "between_median_cosine": float(np.median(sim)),
        "between_std_cosine": float(sim.std()),
    }


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    write_status(status="running", started_at=now(), model=MODEL_NAME, classes=CLASSES, layers=LAYERS)

    df = build_dataset()
    train = df[df["split"].eq("train") & df["class"].isin(CLASSES)].copy().reset_index(drop=True)
    pieces = []
    for cls in CLASSES:
        sub = train[train["class"].eq(cls)].head(MAX_ROWS_PER_CLASS)
        pieces.append(sub)
    audit_df = pd.concat(pieces, ignore_index=True)
    audit_df.to_csv(CSV_DIR / "coherence_pairs.csv", index=False)

    write_status(phase="loading_model", rows=len(audit_df))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")

    write_status(phase="embedding_sources", rows=len(audit_df))
    source_h = last_token_hidden(tokenizer, model, audit_df["source"].tolist(), LAYERS)
    write_status(phase="embedding_targets", rows=len(audit_df))
    target_h = last_token_hidden(tokenizer, model, audit_df["target"].tolist(), LAYERS)

    labels = audit_df["class"].to_numpy()
    rows = []
    deltas_by_layer_class: dict[tuple[int, str], np.ndarray] = {}
    for layer in LAYERS:
        delta = target_h[layer] - source_h[layer]
        for cls in CLASSES:
            vecs = delta[labels == cls]
            deltas_by_layer_class[(layer, cls)] = vecs
            rows.append({"model": MODEL_NAME, "layer": layer, "class": cls, **class_coherence(vecs)})
    summary = pd.DataFrame(rows)
    summary.to_csv(CSV_DIR / "delta_coherence_by_layer_class.csv", index=False)

    between_rows = []
    for layer in LAYERS:
        for a, b in combinations(CLASSES, 2):
            between_rows.append(
                {
                    "model": MODEL_NAME,
                    "layer": layer,
                    "class_a": a,
                    "class_b": b,
                    **between_class_stats(deltas_by_layer_class[(layer, a)], deltas_by_layer_class[(layer, b)]),
                }
            )
    between = pd.DataFrame(between_rows)
    between.to_csv(CSV_DIR / "delta_between_class_cosine.csv", index=False)

    pivot = summary.pivot(index="layer", columns="class", values="mean_pairwise_cosine").reset_index()
    if set(CLASSES).issuperset({"question", "negation"}):
        pivot["question_minus_negation"] = pivot["question"] - pivot["negation"]
    pivot.to_csv(CSV_DIR / "question_negation_layerwise_contrast.csv", index=False)

    write_status(
        status="finished",
        finished_at=now(),
        rows=len(audit_df),
        summary_csv=str(CSV_DIR / "delta_coherence_by_layer_class.csv"),
        contrast_csv=str(CSV_DIR / "question_negation_layerwise_contrast.csv"),
        failures=[],
    )
    print(summary.to_string(index=False))
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
