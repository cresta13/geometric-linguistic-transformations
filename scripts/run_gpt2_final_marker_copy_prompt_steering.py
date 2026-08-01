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

os.environ.setdefault("STEERING_MAX_NEW_TOKENS", os.getenv("FINAL_MARKER_MAX_NEW_TOKENS", "28"))

from run_gpt2_activation_steering_pilot import generate_one, last_token_hidden, norm_match_random  # noqa: E402
from run_gpt2_exclamation_copy_prompt_steering import (  # noqa: E402
    ADVERBIALS,
    HARD_SOURCES,
    PROMPT_STYLES,
    STOPWORDS,
    SUBJECTS,
    VERBS,
    normalize_statement,
)


MODEL_NAME = os.getenv("FINAL_MARKER_MODEL", "gpt2")
LAYERS = [int(x) for x in os.getenv("FINAL_MARKER_LAYERS", "2,3").split(",") if x.strip()]
GAIN = float(os.getenv("FINAL_MARKER_GAIN", "0.75"))
TRAIN_ROWS = int(os.getenv("FINAL_MARKER_TRAIN_ROWS", "120"))
TEST_ROWS = int(os.getenv("FINAL_MARKER_TEST_ROWS", "40"))
OUT_OF_TEMPLATE_ROWS = int(os.getenv("FINAL_MARKER_OUT_OF_TEMPLATE_ROWS", "40"))
SEED = int(os.getenv("FINAL_MARKER_SEED", "20260801"))
TARGET_NAME = os.getenv("FINAL_MARKER_TARGET_NAME", "ellipsis")
TARGET_SUFFIX = os.getenv("FINAL_MARKER_TARGET_SUFFIX", "...")
CONTRAST_NAME = os.getenv("FINAL_MARKER_CONTRAST_NAME", "question_mark")
CONTRAST_SUFFIX = os.getenv("FINAL_MARKER_CONTRAST_SUFFIX", "?")
OUT_DIR = Path(
    os.getenv(
        "FINAL_MARKER_OUT_DIR",
        f"results/experiments/{MODEL_NAME}_{TARGET_NAME}_copy_prompt_steering_20260801_results",
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


def make_sources() -> list[str]:
    sources = []
    for subject in SUBJECTS:
        for verb in VERBS:
            for adverbial in ADVERBIALS:
                sources.append(normalize_statement(f"{subject} {verb} {adverbial}."))
    return sources


def with_suffix(source: str, suffix: str) -> str:
    stem = source.strip().rstrip(".!?")
    return stem + suffix


def build_pairs() -> pd.DataFrame:
    rng = random.Random(SEED)
    sources = make_sources()
    rng.shuffle(sources)
    rows = []
    for split, selected in [
        ("train", sources[:TRAIN_ROWS]),
        ("test", sources[TRAIN_ROWS : TRAIN_ROWS + TEST_ROWS]),
    ]:
        for source in selected:
            rows.append({"split": split, "class": TARGET_NAME, "source": source, "target": with_suffix(source, TARGET_SUFFIX)})
            rows.append(
                {
                    "split": split,
                    "class": CONTRAST_NAME,
                    "source": source,
                    "target": with_suffix(source, CONTRAST_SUFFIX),
                }
            )
    return pd.DataFrame(rows)


def learn_centroids(tokenizer, model, train_df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    sources = train_df["source"].tolist()
    targets = train_df["target"].tolist()
    labels = train_df["class"].to_numpy()
    source_h = last_token_hidden(tokenizer, model, sources, LAYERS)
    target_h = last_token_hidden(tokenizer, model, targets, LAYERS)
    centroids = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        delta = target_h[layer] - source_h[layer]
        for cls in [TARGET_NAME, CONTRAST_NAME]:
            mask = labels == cls
            centroids[layer][cls] = delta[mask].mean(axis=0).astype(np.float32)
    return centroids


def content_preserved(source: str, generated: str) -> float:
    generated_low = generated.lower()
    words = [
        w.strip(".,!?;:\"'()[]").lower()
        for w in source.split()
        if w.strip(".,!?;:\"'()[]").lower() and w.strip(".,!?;:\"'()[]").lower() not in STOPWORDS
    ]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in generated_low)
    return float(hits / len(words) >= 0.6)


def marker_hit(text: str, suffix: str) -> float:
    if suffix == "...":
        return float("..." in text or "\u2026" in text)
    return float(suffix in text)


def score(source: str, generated: str) -> dict[str, float]:
    target = marker_hit(generated, TARGET_SUFFIX)
    contrast = marker_hit(generated, CONTRAST_SUFFIX)
    preserved = content_preserved(source, generated)
    return {
        "target_marker_hit": target,
        "contrast_marker_hit": contrast,
        "content_preserved": preserved,
        "target_and_preserved": float(target and preserved),
        "contrast_and_preserved": float(contrast and preserved),
        "generated_chars": float(len(generated)),
    }


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "target_name", "source_set", "prompt_style", "layer", "control"]
    return (
        raw_df.groupby(group_cols)
        .agg(
            target_marker_rate=("target_marker_hit", "mean"),
            contrast_marker_rate=("contrast_marker_hit", "mean"),
            content_preserved_rate=("content_preserved", "mean"),
            target_and_preserved_rate=("target_and_preserved", "mean"),
            contrast_and_preserved_rate=("contrast_and_preserved", "mean"),
            mean_generated_chars=("generated_chars", "mean"),
            rows=("target_marker_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def flush(rows: list[dict]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(CSV_DIR / "final_marker_copy_prompt_raw.csv", index=False)
    summarize(raw_df).to_csv(CSV_DIR / "final_marker_copy_prompt_summary.csv", index=False)


def run() -> None:
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    write_status(
        status="running",
        started_at=now(),
        model=MODEL_NAME,
        target_name=TARGET_NAME,
        target_suffix=TARGET_SUFFIX,
        contrast_name=CONTRAST_NAME,
        contrast_suffix=CONTRAST_SUFFIX,
        layers=LAYERS,
        gain=GAIN,
        phase="loading_model",
    )
    pairs = build_pairs()
    pairs.to_csv(CSV_DIR / "final_marker_training_pairs.csv", index=False)
    test_sources = pairs[pairs["split"].eq("test") & pairs["class"].eq(TARGET_NAME)]["source"].head(TEST_ROWS).tolist()
    out_sources = [normalize_statement(x) for x in HARD_SOURCES[:OUT_OF_TEMPLATE_ROWS]]
    source_records = (
        [{"source_set": "in_template", "source": x} for x in test_sources]
        + [{"source_set": "hard_out_of_template", "source": x} for x in out_sources]
    )
    pd.DataFrame(source_records).to_csv(CSV_DIR / "final_marker_copy_prompt_sources.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")

    train = pairs[pairs["split"].eq("train")].reset_index(drop=True)
    write_status(phase="learning_centroids", train_rows=len(train), source_rows=len(source_records))
    centroids = learn_centroids(tokenizer, model, train)

    total = len(source_records) * len(PROMPT_STYLES) * len(LAYERS) * 5
    rows = []
    done = 0
    write_status(phase="generating", progress_done=0, progress_total=total, rows=0)
    for layer in LAYERS:
        target_vec = centroids[layer][TARGET_NAME]
        contrast_vec = centroids[layer][CONTRAST_NAME]
        for source_record in source_records:
            source = source_record["source"]
            for prompt_style, prompt_fn in PROMPT_STYLES.items():
                prompt = prompt_fn(source)
                plan = [
                    ("none", 0.0, None),
                    ("target_marker", GAIN, target_vec),
                    ("wrong_contrast", GAIN, contrast_vec),
                    ("random_norm", GAIN, norm_match_random(target_vec, rng)),
                    ("negative_target", GAIN, -target_vec),
                ]
                for control, gain, vec in plan:
                    generated = generate_one(tokenizer, model, prompt, layer if vec is not None else None, vec, gain)
                    rows.append(
                        {
                            "model": MODEL_NAME,
                            "target_name": TARGET_NAME,
                            "target_suffix": TARGET_SUFFIX,
                            "contrast_name": CONTRAST_NAME,
                            "contrast_suffix": CONTRAST_SUFFIX,
                            "source_set": source_record["source_set"],
                            "source": source,
                            "prompt_style": prompt_style,
                            "layer": layer,
                            "control": control,
                            "gain": gain,
                            "generated": generated,
                            **score(source, generated),
                        }
                    )
                    done += 1
                    if done % 25 == 0:
                        flush(rows)
                        write_status(
                            phase="generating",
                            current_layer=layer,
                            current_prompt_style=prompt_style,
                            current_control=control,
                            progress_done=done,
                            progress_total=total,
                            rows=len(rows),
                        )
    flush(rows)
    write_status(
        status="finished",
        phase="finished",
        progress_done=done,
        progress_total=total,
        rows=len(rows),
        finished_at=now(),
        summary_csv=str(CSV_DIR / "final_marker_copy_prompt_summary.csv"),
        raw_csv=str(CSV_DIR / "final_marker_copy_prompt_raw.csv"),
        failures=[],
    )
    print(summarize(pd.DataFrame(rows)).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
