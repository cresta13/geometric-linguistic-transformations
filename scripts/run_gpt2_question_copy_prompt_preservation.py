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
os.environ.setdefault("STEERING_LAYERS", "2,3")
os.environ.setdefault("STEERING_GAINS", "0.75")
os.environ.setdefault("STEERING_MAX_NEW_TOKENS", "28")

from run_gpt2_activation_steering_pilot import (  # noqa: E402
    build_dataset,
    generate_one,
    learn_delta_centroids,
    norm_match_random,
    score_output,
)
from run_gpt2_question_steering_controls import OUT_OF_TEMPLATE_SOURCES  # noqa: E402


OUT_DIR = Path(
    os.getenv(
        "QUESTION_COPY_PROMPT_OUT_DIR",
        "results/experiments/gpt2_question_copy_prompt_preservation_20260716_results",
    )
)
CSV_DIR = OUT_DIR / "csv"
STATUS_PATH = OUT_DIR / "run_status.json"

MODEL_NAME = os.getenv("QUESTION_COPY_PROMPT_MODEL", "gpt2")
LAYERS = [int(x) for x in os.getenv("QUESTION_COPY_PROMPT_LAYERS", "2,3").split(",") if x.strip()]
GAIN = float(os.getenv("QUESTION_COPY_PROMPT_GAIN", "0.75"))
MAX_IN_TEMPLATE_SOURCES = int(os.getenv("QUESTION_COPY_PROMPT_IN_TEMPLATE_SOURCES", "40"))
MAX_OUT_OF_TEMPLATE_SOURCES = int(os.getenv("QUESTION_COPY_PROMPT_OUT_OF_TEMPLATE_SOURCES", "40"))
SEED = int(os.getenv("QUESTION_COPY_PROMPT_SEED", "20260716"))


PROMPT_STYLES = {
    "repeat_sentence": lambda source: f"Repeat this sentence:\n{source}\nRepeated:",
    "copy_sentence": lambda source: f"Copy this sentence:\n{source}\nCopy:",
    "same_sentence": lambda source: f"Same sentence:\n{source}\nSame:",
    "quoted_echo": lambda source: f"\"{source}\"",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


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


def normalize_token(token: str) -> str:
    token = "".join(ch for ch in token.lower() if ch.isalnum())
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_tokens(text: str) -> set[str]:
    tokens = set()
    for raw in str(text).replace(".", " ").replace("?", " ").split():
        tok = normalize_token(raw)
        if tok and tok not in STOPWORDS:
            tokens.add(tok)
    return tokens


def preservation_score(source: str, generated: str) -> float:
    src = content_tokens(source)
    if not src:
        return 0.0
    gen = content_tokens(generated)
    return len(src & gen) / len(src)


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    group_cols = ["source_set", "prompt_style", "control"]
    return (
        raw_df.groupby(group_cols)
        .agg(
            question_mark_rate=("question_mark_hit", "mean"),
            mean_content_preservation=("content_preservation", "mean"),
            question_and_preserved_rate=("question_and_preserved", "mean"),
            target_marker_rate=("target_marker_hit", "mean"),
            other_marker_rate=("any_other_marker_hit", "mean"),
            rows=("question_mark_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def choose_wrong_vector(centroids: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[str, np.ndarray]:
    choices = [c for c in sorted(centroids) if c != "question"]
    cls = choices[int(rng.integers(0, len(choices)))]
    return cls, centroids[cls]


def build_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_dataset()
    question_test = df[df["split"].eq("test") & df["class"].eq("question")].reset_index(drop=True)
    unique_sources = list(dict.fromkeys(question_test["source"].tolist()))[:MAX_IN_TEMPLATE_SOURCES]
    question_test = question_test[question_test["source"].isin(unique_sources)].reset_index(drop=True)
    rows = [
        {"source_set": "in_template_test", "source": row.source, "target": row.target}
        for row in question_test.itertuples(index=False)
    ]
    rows.extend(
        {"source_set": "out_of_template_freeform", "source": source, "target": ""}
        for source in OUT_OF_TEMPLATE_SOURCES[:MAX_OUT_OF_TEMPLATE_SOURCES]
    )
    train_df = df[df["split"].eq("train")].reset_index(drop=True)
    return pd.DataFrame(rows), train_df


def flush(rows: list[dict]) -> None:
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(CSV_DIR / "question_copy_prompt_raw.csv", index=False)
    summarize(raw_df).to_csv(CSV_DIR / "question_copy_prompt_summary.csv", index=False)


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
        gain=GAIN,
        prompt_styles=list(PROMPT_STYLES),
        phase="building_sources",
    )
    sources, train_df = build_sources()
    sources.to_csv(CSV_DIR / "question_copy_prompt_sources.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")

    write_status(phase="learning_centroids", train_rows=len(train_df), source_rows=len(sources))
    centroids_by_layer = learn_delta_centroids(tokenizer, model, train_df, LAYERS)

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    total = len(sources) * len(PROMPT_STYLES) * len(LAYERS) * 5
    done = 0

    for layer in LAYERS:
        centroids = centroids_by_layer[layer]
        question_vec = centroids["question"]
        for source_rec in sources.to_dict("records"):
            for prompt_style, prompt_fn in PROMPT_STYLES.items():
                prompt = prompt_fn(source_rec["source"])
                wrong_cls, wrong_vec = choose_wrong_vector(centroids, rng)
                plan = [
                    ("none", 0.0, None, ""),
                    ("target", GAIN, question_vec, "question"),
                    ("random_norm", GAIN, norm_match_random(question_vec, rng), "random_norm"),
                    ("wrong_class", GAIN, wrong_vec, wrong_cls),
                    ("negative_target", GAIN, -question_vec, "question"),
                ]
                for control, gain, vec, vector_class in plan:
                    done += 1
                    write_status(
                        phase="generating",
                        current_layer=layer,
                        current_prompt_style=prompt_style,
                        current_control=control,
                        progress_done=done,
                        progress_total=total,
                        rows=len(rows),
                    )
                    generated = generate_one(tokenizer, model, prompt, layer if vec is not None else None, vec, gain)
                    score = score_output(generated, "question")
                    content_preservation = preservation_score(source_rec["source"], generated)
                    rows.append(
                        {
                            "model": MODEL_NAME,
                            "layer": layer,
                            "source_set": source_rec["source_set"],
                            "prompt_style": prompt_style,
                            "source": source_rec["source"],
                            "target": source_rec["target"],
                            "target_class": "question",
                            "control": control,
                            "gain": gain,
                            "vector_class": vector_class,
                            "prompt": prompt,
                            "generated": generated,
                            "content_preservation": content_preservation,
                            "question_and_preserved": float(score["question_mark_hit"] == 1.0 and content_preservation >= 0.67),
                            **score,
                        }
                    )
                    if len(rows) % 100 == 0:
                        flush(rows)
                        print(f"generated {done}/{total}, rows={len(rows)}", flush=True)

    flush(rows)
    raw_df = pd.DataFrame(rows)
    summary = summarize(raw_df)
    write_status(
        status="finished",
        finished_at=now(),
        rows=len(raw_df),
        summary_rows=len(summary),
        raw_csv=str(CSV_DIR / "question_copy_prompt_raw.csv"),
        summary_csv=str(CSV_DIR / "question_copy_prompt_summary.csv"),
        failures=[],
    )
    print(f"Saved raw: {CSV_DIR / 'question_copy_prompt_raw.csv'}", flush=True)
    print(f"Saved summary: {CSV_DIR / 'question_copy_prompt_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
