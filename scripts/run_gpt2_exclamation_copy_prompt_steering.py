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

from run_gpt2_activation_steering_pilot import generate_one, last_token_hidden, norm_match_random  # noqa: E402


MODEL_NAME = os.getenv("EXCLAMATION_STEERING_MODEL", "gpt2")
LAYERS = [int(x) for x in os.getenv("EXCLAMATION_STEERING_LAYERS", "2,3").split(",") if x.strip()]
GAIN = float(os.getenv("EXCLAMATION_STEERING_GAIN", "0.75"))
TRAIN_ROWS = int(os.getenv("EXCLAMATION_STEERING_TRAIN_ROWS", "120"))
TEST_ROWS = int(os.getenv("EXCLAMATION_STEERING_TEST_ROWS", "40"))
OUT_OF_TEMPLATE_ROWS = int(os.getenv("EXCLAMATION_STEERING_OUT_OF_TEMPLATE_ROWS", "40"))
SEED = int(os.getenv("EXCLAMATION_STEERING_SEED", "20260716"))
OUT_DIR = Path(
    os.getenv(
        "EXCLAMATION_STEERING_OUT_DIR",
        "results/experiments/gpt2_exclamation_copy_prompt_steering_20260716_results",
    )
)
CSV_DIR = OUT_DIR / "csv"
STATUS_PATH = OUT_DIR / "run_status.json"


PROMPT_STYLES = {
    "repeat_sentence": lambda source: f"Repeat this sentence:\n{source}\nRepeated:",
    "copy_sentence": lambda source: f"Copy this sentence:\n{source}\nCopy:",
    "same_sentence": lambda source: f"Same sentence:\n{source}\nSame:",
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


SUBJECTS = [
    "The analyst",
    "The engineer",
    "The teacher",
    "The artist",
    "The researcher",
    "The student",
    "The pilot",
    "The doctor",
    "The writer",
    "The gardener",
    "The manager",
    "The musician",
]

VERBS = [
    "completed the task",
    "opened the archive",
    "checked the result",
    "designed the prototype",
    "cleaned the dataset",
    "published the note",
    "reviewed the plan",
    "tested the system",
    "updated the report",
    "fixed the problem",
    "measured the signal",
    "explained the pattern",
]

ADVERBIALS = [
    "today",
    "before lunch",
    "after the meeting",
    "in the lab",
    "during the session",
    "with care",
    "on Monday",
    "for the team",
    "near the station",
    "without delay",
]

HARD_SOURCES = [
    "The sample was collected by Maria in Paris.",
    "Although the sensor failed twice, the archive remained intact.",
    "Dr. Ivanov reviewed 17 documents before sunrise.",
    "The bridge was inspected after the storm ended.",
    "When the lights went out, Elena saved the last checkpoint.",
    "The experiment ran for 14 days in a sealed chamber.",
    "A backup copy was created before the server rebooted.",
    "Because the battery was low, the robot returned to the dock.",
    "Professor Kim compared three models on Tuesday.",
    "The package was delivered to Building 12 at noon.",
    "After the committee voted, the proposal moved forward.",
    "Nina measured 42 samples during the morning shift.",
    "The old manuscript was restored by a small museum team.",
    "If the pressure rises, the valve closes automatically.",
    "The telescope captured Jupiter at 03:15.",
    "Before Anna left Berlin, she archived the recordings.",
    "The report was signed by two reviewers yesterday.",
    "While the model trained, the laptop stayed plugged in.",
    "The clinic processed 86 forms before closing.",
    "Because the route changed, Alex updated the map.",
    "The signal was detected by the second antenna.",
    "When the camera focused, the image became sharp.",
    "The library stored 300 boxes in the basement.",
    "After the rehearsal ended, Sofia tuned the piano.",
    "The database was migrated during the night.",
    "Although the river froze, the sensors kept running.",
    "The document was translated by a volunteer in Madrid.",
    "Before the timer expired, Pavel saved the file.",
    "The company shipped 25 devices to Lisbon.",
    "When the wind slowed, the drone landed safely.",
    "The glass panel was replaced after the inspection.",
    "Because the password changed, the script requested a token.",
    "The survey included 512 participants in June.",
    "After the alarm sounded, the system locked the door.",
    "The painting was moved to Gallery 4 on Friday.",
    "Although the network lagged, the call continued.",
    "The rover crossed 9 meters of loose sand.",
    "When the freezer opened, the samples stayed labeled.",
    "The treaty was approved by the council in 2025.",
    "Before the lecture began, Olesya checked the projector.",
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


def normalize_statement(text: str) -> str:
    return text.strip().rstrip(".!?") + "."


def make_sources() -> list[str]:
    sources = []
    for subject in SUBJECTS:
        for verb in VERBS:
            for adverbial in ADVERBIALS:
                sources.append(normalize_statement(f"{subject} {verb} {adverbial}."))
    return sources


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
            stem = source.rstrip(".")
            rows.append({"split": split, "class": "exclamation", "source": source, "target": stem + "!"})
            rows.append({"split": split, "class": "question_mark", "source": source, "target": stem + "?"})
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
        for cls in ["exclamation", "question_mark"]:
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


def score(source: str, generated: str) -> dict[str, float]:
    exclamation = float("!" in generated)
    question = float("?" in generated)
    preserved = content_preserved(source, generated)
    return {
        "exclamation_mark_hit": exclamation,
        "question_mark_hit": question,
        "content_preserved": preserved,
        "exclamation_and_preserved": float(exclamation and preserved),
        "generated_chars": float(len(generated)),
    }


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "source_set", "prompt_style", "layer", "control"]
    return (
        raw_df.groupby(group_cols)
        .agg(
            exclamation_mark_rate=("exclamation_mark_hit", "mean"),
            question_mark_rate=("question_mark_hit", "mean"),
            content_preserved_rate=("content_preserved", "mean"),
            exclamation_and_preserved_rate=("exclamation_and_preserved", "mean"),
            mean_generated_chars=("generated_chars", "mean"),
            rows=("exclamation_mark_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def flush(rows: list[dict]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(CSV_DIR / "exclamation_copy_prompt_raw.csv", index=False)
    summarize(raw_df).to_csv(CSV_DIR / "exclamation_copy_prompt_summary.csv", index=False)


def run() -> None:
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    write_status(
        status="running",
        started_at=now(),
        model=MODEL_NAME,
        layers=LAYERS,
        gain=GAIN,
        phase="loading_model",
    )
    pairs = build_pairs()
    pairs.to_csv(CSV_DIR / "exclamation_training_pairs.csv", index=False)
    test_sources = (
        pairs[pairs["split"].eq("test") & pairs["class"].eq("exclamation")]["source"].head(TEST_ROWS).tolist()
    )
    out_sources = [normalize_statement(x) for x in HARD_SOURCES[:OUT_OF_TEMPLATE_ROWS]]
    pd.DataFrame(
        [{"source_set": "in_template", "source": x} for x in test_sources]
        + [{"source_set": "hard_out_of_template", "source": x} for x in out_sources]
    ).to_csv(CSV_DIR / "exclamation_copy_prompt_sources.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")

    write_status(phase="learning_centroids", train_rows=int(len(pairs[pairs["split"].eq("train")])))
    centroids = learn_centroids(tokenizer, model, pairs[pairs["split"].eq("train")].reset_index(drop=True))
    source_records = (
        [{"source_set": "in_template", "source": x} for x in test_sources]
        + [{"source_set": "hard_out_of_template", "source": x} for x in out_sources]
    )
    total = len(source_records) * len(PROMPT_STYLES) * len(LAYERS) * 5
    rows = []
    done = 0
    write_status(phase="generating", source_rows=len(source_records), progress_done=0, progress_total=total)
    for layer in LAYERS:
        target_vec = centroids[layer]["exclamation"]
        question_vec = centroids[layer]["question_mark"]
        for source_record in source_records:
            source = source_record["source"]
            for prompt_style, prompt_fn in PROMPT_STYLES.items():
                prompt = prompt_fn(source)
                plan = [
                    ("none", 0.0, None),
                    ("target_exclamation", GAIN, target_vec),
                    ("wrong_question", GAIN, question_vec),
                    ("random_norm", GAIN, norm_match_random(target_vec, rng)),
                    ("negative_target", GAIN, -target_vec),
                ]
                for control, gain, vec in plan:
                    generated = generate_one(tokenizer, model, prompt, layer if vec is not None else None, vec, gain)
                    metrics = score(source, generated)
                    rows.append(
                        {
                            "model": MODEL_NAME,
                            "source_set": source_record["source_set"],
                            "source": source,
                            "prompt_style": prompt_style,
                            "layer": layer,
                            "control": control,
                            "gain": gain,
                            "generated": generated,
                            **metrics,
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
        summary_csv=str(CSV_DIR / "exclamation_copy_prompt_summary.csv"),
        raw_csv=str(CSV_DIR / "exclamation_copy_prompt_raw.csv"),
        failures=[],
    )
    print(summarize(pd.DataFrame(rows)).to_string(index=False))


if __name__ == "__main__":
    run()
