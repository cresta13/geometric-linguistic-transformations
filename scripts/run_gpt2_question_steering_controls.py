from __future__ import annotations

import json
import os
import random
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
os.environ.setdefault("STEERING_LAYERS", "2,3,4,5,6")
os.environ.setdefault("STEERING_GAINS", "0.5,0.75,1.0")

from run_gpt2_activation_steering_pilot import (  # noqa: E402
    build_dataset,
    generate_one,
    get_layer_indices,
    learn_delta_centroids,
    make_prompt,
    norm_match_random,
    score_output,
)


OUT_DIR = Path(
    os.getenv(
        "QUESTION_STEERING_CONTROL_OUT_DIR",
        "results/experiments/gpt2_question_steering_controls_20260714_results",
    )
)
CSV_DIR = OUT_DIR / "csv"
STATUS_PATH = OUT_DIR / "run_status.json"

MODEL_NAME = os.getenv("QUESTION_STEERING_CONTROL_MODEL", "gpt2")
LAYERS = [int(x) for x in os.getenv("QUESTION_STEERING_CONTROL_LAYERS", "2,3,4,5,6").split(",") if x.strip()]
GAINS = [float(x) for x in os.getenv("QUESTION_STEERING_CONTROL_GAINS", "0.5,0.75,1.0").split(",") if x.strip()]
MAX_TRAIN_TEMPLATES = int(os.getenv("QUESTION_STEERING_CONTROL_TRAIN_TEMPLATES", "100"))
MAX_IN_TEMPLATE_SOURCES = int(os.getenv("QUESTION_STEERING_CONTROL_IN_TEMPLATE_SOURCES", "80"))
MAX_OUT_OF_TEMPLATE_SOURCES = int(os.getenv("QUESTION_STEERING_CONTROL_OUT_OF_TEMPLATE_SOURCES", "40"))
MAX_NEW_TOKENS = int(os.getenv("STEERING_MAX_NEW_TOKENS", "28"))
SEED = int(os.getenv("QUESTION_STEERING_CONTROL_SEED", "20260714"))


OUT_OF_TEMPLATE_SOURCES = [
    "The cat sat on the mat.",
    "The kettle whistled on the stove.",
    "A student solved the puzzle.",
    "The gardener watered the roses.",
    "The musician tuned the violin.",
    "A doctor checked the chart.",
    "The driver parked the car.",
    "The child opened the window.",
    "A researcher cleaned the dataset.",
    "The painter mixed the colors.",
    "The pilot landed the plane.",
    "A chef sliced the tomato.",
    "The librarian stamped the card.",
    "The engineer tested the bridge.",
    "A nurse folded the blanket.",
    "The baker warmed the bread.",
    "The photographer adjusted the lens.",
    "A farmer repaired the fence.",
    "The swimmer crossed the pool.",
    "The analyst reviewed the table.",
    "A teacher erased the board.",
    "The carpenter measured the shelf.",
    "The cyclist crossed the square.",
    "A lawyer signed the document.",
    "The mechanic replaced the battery.",
    "The dancer practiced the routine.",
    "A manager approved the invoice.",
    "The sailor tied the rope.",
    "The architect sketched the room.",
    "A journalist recorded the interview.",
    "The botanist labeled the sample.",
    "The courier delivered the package.",
    "A programmer fixed the bug.",
    "The pianist played the melody.",
    "The guard locked the gate.",
    "A scientist weighed the powder.",
    "The tailor adjusted the jacket.",
    "The waiter cleared the table.",
    "A climber packed the rope.",
    "The dentist polished the mirror.",
]


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


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    group_cols = ["source_set", "control", "gain"]
    return (
        raw_df.groupby(group_cols)
        .agg(
            question_mark_rate=("question_mark_hit", "mean"),
            target_marker_rate=("target_marker_hit", "mean"),
            other_marker_rate=("any_other_marker_hit", "mean"),
            mean_generated_chars=("generated_chars", "mean"),
            rows=("question_mark_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def make_question_target(source: str) -> str:
    text = source.strip()
    if text.endswith("."):
        text = text[:-1]
    return f"Could someone ask whether {text.lower()}?"


def choose_wrong_vector(centroids: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[str, np.ndarray]:
    choices = [c for c in sorted(centroids) if c != "question"]
    cls = choices[int(rng.integers(0, len(choices)))]
    return cls, centroids[cls]


def generate_rows(tokenizer, model, centroids_by_layer, source_rows: list[dict]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    total = len(source_rows) * len(LAYERS) * (1 + len(GAINS) * 4)
    done = 0

    for layer in LAYERS:
        centroids = centroids_by_layer[layer]
        question_vec = centroids["question"]
        for rec in source_rows:
            prompt = make_prompt(rec["source"])
            plan = [("none", 0.0, None, "")]
            for gain in GAINS:
                wrong_cls, wrong_vec = choose_wrong_vector(centroids, rng)
                plan.extend(
                    [
                        ("target", gain, question_vec, "question"),
                        ("wrong_class", gain, wrong_vec, wrong_cls),
                        ("random_norm", gain, norm_match_random(question_vec, rng), "random_norm"),
                        ("negative_target", gain, -question_vec, "question"),
                    ]
                )
            for control, gain, vec, vector_class in plan:
                done += 1
                write_status(phase="generating", current_layer=layer, current_control=control, progress_done=done, progress_total=total, rows=len(rows))
                generated = generate_one(tokenizer, model, prompt, layer if vec is not None else None, vec, gain)
                score = score_output(generated, "question")
                rows.append(
                    {
                        "model": MODEL_NAME,
                        "layer": layer,
                        "source_set": rec["source_set"],
                        "source": rec["source"],
                        "target": rec["target"],
                        "target_class": "question",
                        "control": control,
                        "gain": gain,
                        "vector_class": vector_class,
                        "prompt": prompt,
                        "generated": generated,
                        **score,
                    }
                )
                if len(rows) % 50 == 0:
                    raw_df = pd.DataFrame(rows)
                    raw_df.to_csv(CSV_DIR / "question_steering_controls_raw.csv", index=False)
                    summarize(raw_df).to_csv(CSV_DIR / "question_steering_controls_summary.csv", index=False)
                    print(f"generated {done}/{total}, rows={len(rows)}", flush=True)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(CSV_DIR / "question_steering_controls_raw.csv", index=False)
    summarize(raw_df).to_csv(CSV_DIR / "question_steering_controls_summary.csv", index=False)
    return raw_df


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    write_status(
        status="running",
        started_at=now(),
        model=MODEL_NAME,
        layers=LAYERS,
        gains=GAINS,
        phase="loading_model",
    )

    df = build_dataset()
    train_df = df[df["split"].eq("train")].reset_index(drop=True)
    if MAX_TRAIN_TEMPLATES:
        template_ids = list(dict.fromkeys(train_df["template_id"].tolist()))[:MAX_TRAIN_TEMPLATES] if "template_id" in train_df.columns else None
        if template_ids is not None:
            train_df = train_df[train_df["template_id"].isin(template_ids)].reset_index(drop=True)

    in_template_question = df[df["split"].eq("test") & df["class"].eq("question")].reset_index(drop=True)
    unique_sources = list(dict.fromkeys(in_template_question["source"].tolist()))[:MAX_IN_TEMPLATE_SOURCES]
    in_template_question = in_template_question[in_template_question["source"].isin(unique_sources)].reset_index(drop=True)

    source_rows = [
        {"source_set": "in_template_test", "source": row.source, "target": row.target}
        for row in in_template_question.itertuples(index=False)
    ]
    source_rows.extend(
        {"source_set": "out_of_template_freeform", "source": source, "target": make_question_target(source)}
        for source in OUT_OF_TEMPLATE_SOURCES[:MAX_OUT_OF_TEMPLATE_SOURCES]
    )
    pd.DataFrame(source_rows).to_csv(CSV_DIR / "question_steering_control_sources.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")

    available_layers = get_layer_indices(model)
    missing = sorted(set(LAYERS) - set(available_layers))
    if missing:
        raise ValueError(f"Requested unavailable layers: {missing}; available from script config: {available_layers}")

    write_status(phase="learning_centroids", train_rows=len(train_df), source_rows=len(source_rows))
    centroids = learn_delta_centroids(tokenizer, model, train_df, LAYERS)

    raw_df = generate_rows(tokenizer, model, centroids, source_rows)
    summary = summarize(raw_df)

    base_rate = (
        raw_df[raw_df["control"].eq("none")]
        .groupby("source_set")
        .agg(
            base_question_mark_rate=("question_mark_hit", "mean"),
            base_target_marker_rate=("target_marker_hit", "mean"),
            rows=("question_mark_hit", "count"),
        )
        .reset_index()
    )
    base_rate.to_csv(CSV_DIR / "question_mark_base_rate.csv", index=False)

    write_status(
        status="finished",
        finished_at=now(),
        rows=len(raw_df),
        summary_rows=len(summary),
        base_rate_csv=str(CSV_DIR / "question_mark_base_rate.csv"),
        raw_csv=str(CSV_DIR / "question_steering_controls_raw.csv"),
        summary_csv=str(CSV_DIR / "question_steering_controls_summary.csv"),
        failures=[],
    )
    print(f"Saved raw: {CSV_DIR / 'question_steering_controls_raw.csv'}", flush=True)
    print(f"Saved summary: {CSV_DIR / 'question_steering_controls_summary.csv'}", flush=True)
    print(f"Saved base rate: {CSV_DIR / 'question_mark_base_rate.csv'}", flush=True)


if __name__ == "__main__":
    main()
