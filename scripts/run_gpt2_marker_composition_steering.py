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

os.environ.setdefault("STEERING_MAX_NEW_TOKENS", os.getenv("MARKER_COMPOSITION_MAX_NEW_TOKENS", "28"))

from run_gpt2_activation_steering_pilot import attach_hook, decode_new_text, last_token_hidden, norm_match_random  # noqa: E402
from run_gpt2_exclamation_copy_prompt_steering import (  # noqa: E402
    ADVERBIALS,
    HARD_SOURCES,
    PROMPT_STYLES,
    STOPWORDS,
    SUBJECTS,
    VERBS,
    normalize_statement,
)


MODEL_NAME = os.getenv("MARKER_COMPOSITION_MODEL", "gpt2")
EARLY_LAYER = int(os.getenv("MARKER_COMPOSITION_EARLY_LAYER", "2"))
LATE_LAYER = int(os.getenv("MARKER_COMPOSITION_LATE_LAYER", "3"))
GAIN = float(os.getenv("MARKER_COMPOSITION_GAIN", "0.75"))
TRAIN_ROWS = int(os.getenv("MARKER_COMPOSITION_TRAIN_ROWS", "120"))
TEST_ROWS = int(os.getenv("MARKER_COMPOSITION_TEST_ROWS", "40"))
OUT_OF_TEMPLATE_ROWS = int(os.getenv("MARKER_COMPOSITION_OUT_OF_TEMPLATE_ROWS", "40"))
SEED = int(os.getenv("MARKER_COMPOSITION_SEED", "20260801"))
OP_A_NAME = os.getenv("MARKER_COMPOSITION_A_NAME", "question_mark")
OP_A_SUFFIX = os.getenv("MARKER_COMPOSITION_A_SUFFIX", "?")
OP_B_NAME = os.getenv("MARKER_COMPOSITION_B_NAME", "exclamation")
OP_B_SUFFIX = os.getenv("MARKER_COMPOSITION_B_SUFFIX", "!")
OUT_DIR = Path(
    os.getenv(
        "MARKER_COMPOSITION_OUT_DIR",
        f"results/experiments/{MODEL_NAME}_question_exclamation_marker_composition_20260801_results",
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
            rows.append({"split": split, "class": OP_A_NAME, "source": source, "target": with_suffix(source, OP_A_SUFFIX)})
            rows.append({"split": split, "class": OP_B_NAME, "source": source, "target": with_suffix(source, OP_B_SUFFIX)})
    return pd.DataFrame(rows)


def learn_centroids(tokenizer, model, train_df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    layers = [EARLY_LAYER, LATE_LAYER]
    sources = train_df["source"].tolist()
    targets = train_df["target"].tolist()
    labels = train_df["class"].to_numpy()
    source_h = last_token_hidden(tokenizer, model, sources, layers)
    target_h = last_token_hidden(tokenizer, model, targets, layers)
    centroids = {layer: {} for layer in layers}
    for layer in layers:
        delta = target_h[layer] - source_h[layer]
        for cls in [OP_A_NAME, OP_B_NAME]:
            mask = labels == cls
            centroids[layer][cls] = delta[mask].mean(axis=0).astype(np.float32)
    return centroids


def generate_with_hooks(tokenizer, model, prompt: str, hook_specs: list[tuple[int, np.ndarray, float]]) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    handles = []
    try:
        for layer, vector, gain in hook_specs:
            if abs(gain) > 0:
                handles.append(attach_hook(model, layer, vector, gain))
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=int(os.getenv("MARKER_COMPOSITION_MAX_NEW_TOKENS", "28")),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return decode_new_text(tokenizer, out[0].detach().cpu(), enc["input_ids"].shape[1])
    finally:
        for handle in handles:
            handle.remove()


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
    a_hit = marker_hit(generated, OP_A_SUFFIX)
    b_hit = marker_hit(generated, OP_B_SUFFIX)
    preserved = content_preserved(source, generated)
    return {
        "a_marker_hit": a_hit,
        "b_marker_hit": b_hit,
        "both_markers_hit": float(a_hit and b_hit),
        "any_marker_hit": float(a_hit or b_hit),
        "content_preserved": preserved,
        "a_and_preserved": float(a_hit and preserved),
        "b_and_preserved": float(b_hit and preserved),
        "both_and_preserved": float(a_hit and b_hit and preserved),
        "generated_chars": float(len(generated)),
    }


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "source_set", "prompt_style", "control"]
    return (
        raw_df.groupby(group_cols)
        .agg(
            a_marker_rate=("a_marker_hit", "mean"),
            b_marker_rate=("b_marker_hit", "mean"),
            both_markers_rate=("both_markers_hit", "mean"),
            any_marker_rate=("any_marker_hit", "mean"),
            content_preserved_rate=("content_preserved", "mean"),
            a_and_preserved_rate=("a_and_preserved", "mean"),
            b_and_preserved_rate=("b_and_preserved", "mean"),
            both_and_preserved_rate=("both_and_preserved", "mean"),
            mean_generated_chars=("generated_chars", "mean"),
            rows=("a_marker_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def order_contrast(raw_df: pd.DataFrame) -> pd.DataFrame:
    pivot_cols = ["source_set", "source", "prompt_style"]
    small = raw_df[raw_df["control"].isin(["a_then_b_layers", "b_then_a_layers", "a_plus_b_late"])]
    if small.empty:
        return pd.DataFrame()
    records = []
    grouped = {key: group for key, group in small.groupby(pivot_cols)}
    for key, group in grouped.items():
        by_control = {row.control: row for row in group.itertuples(index=False)}
        if "a_then_b_layers" not in by_control or "b_then_a_layers" not in by_control:
            continue
        ab = by_control["a_then_b_layers"]
        ba = by_control["b_then_a_layers"]
        plus = by_control.get("a_plus_b_late")
        records.append(
            {
                "source_set": key[0],
                "source": key[1],
                "prompt_style": key[2],
                "ab_equals_ba": float(ab.generated == ba.generated),
                "ab_ba_marker_profile_equal": float(
                    ab.a_marker_hit == ba.a_marker_hit and ab.b_marker_hit == ba.b_marker_hit
                ),
                "ab_equals_plus": float(plus is not None and ab.generated == plus.generated),
                "ba_equals_plus": float(plus is not None and ba.generated == plus.generated),
                "ab_a_marker_hit": ab.a_marker_hit,
                "ab_b_marker_hit": ab.b_marker_hit,
                "ba_a_marker_hit": ba.a_marker_hit,
                "ba_b_marker_hit": ba.b_marker_hit,
                "plus_a_marker_hit": float(plus.a_marker_hit) if plus is not None else np.nan,
                "plus_b_marker_hit": float(plus.b_marker_hit) if plus is not None else np.nan,
            }
        )
    if not records:
        return pd.DataFrame()
    contrast = pd.DataFrame(records)
    summary = (
        contrast.groupby(["source_set", "prompt_style"])
        .agg(
            ab_equals_ba_rate=("ab_equals_ba", "mean"),
            ab_ba_marker_profile_equal_rate=("ab_ba_marker_profile_equal", "mean"),
            ab_equals_plus_rate=("ab_equals_plus", "mean"),
            ba_equals_plus_rate=("ba_equals_plus", "mean"),
            ab_a_marker_rate=("ab_a_marker_hit", "mean"),
            ab_b_marker_rate=("ab_b_marker_hit", "mean"),
            ba_a_marker_rate=("ba_a_marker_hit", "mean"),
            ba_b_marker_rate=("ba_b_marker_hit", "mean"),
            plus_a_marker_rate=("plus_a_marker_hit", "mean"),
            plus_b_marker_rate=("plus_b_marker_hit", "mean"),
            rows=("ab_equals_ba", "count"),
        )
        .reset_index()
    )
    contrast.to_csv(CSV_DIR / "marker_composition_order_contrast_raw.csv", index=False)
    return summary


def flush(rows: list[dict]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(CSV_DIR / "marker_composition_steering_raw.csv", index=False)
    summarize(raw_df).to_csv(CSV_DIR / "marker_composition_steering_summary.csv", index=False)
    order_contrast(raw_df).to_csv(CSV_DIR / "marker_composition_order_contrast_summary.csv", index=False)


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
        early_layer=EARLY_LAYER,
        late_layer=LATE_LAYER,
        gain=GAIN,
        op_a_name=OP_A_NAME,
        op_a_suffix=OP_A_SUFFIX,
        op_b_name=OP_B_NAME,
        op_b_suffix=OP_B_SUFFIX,
        phase="loading_model",
    )

    pairs = build_pairs()
    pairs.to_csv(CSV_DIR / "marker_composition_training_pairs.csv", index=False)
    test_sources = pairs[pairs["split"].eq("test") & pairs["class"].eq(OP_A_NAME)]["source"].head(TEST_ROWS).tolist()
    out_sources = [normalize_statement(x) for x in HARD_SOURCES[:OUT_OF_TEMPLATE_ROWS]]
    source_records = (
        [{"source_set": "in_template", "source": x} for x in test_sources]
        + [{"source_set": "hard_out_of_template", "source": x} for x in out_sources]
    )
    pd.DataFrame(source_records).to_csv(CSV_DIR / "marker_composition_sources.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")

    train = pairs[pairs["split"].eq("train")].reset_index(drop=True)
    write_status(phase="learning_centroids", train_rows=len(train), source_rows=len(source_records))
    centroids = learn_centroids(tokenizer, model, train)
    a_early = centroids[EARLY_LAYER][OP_A_NAME]
    b_early = centroids[EARLY_LAYER][OP_B_NAME]
    a_late = centroids[LATE_LAYER][OP_A_NAME]
    b_late = centroids[LATE_LAYER][OP_B_NAME]
    sum_late = a_late + b_late
    random_sum = norm_match_random(sum_late, rng)

    controls = {
        "none": [],
        "a_only_late": [(LATE_LAYER, a_late, GAIN)],
        "b_only_late": [(LATE_LAYER, b_late, GAIN)],
        "a_plus_b_late": [(LATE_LAYER, sum_late, GAIN)],
        "a_then_b_layers": [(EARLY_LAYER, a_early, GAIN), (LATE_LAYER, b_late, GAIN)],
        "b_then_a_layers": [(EARLY_LAYER, b_early, GAIN), (LATE_LAYER, a_late, GAIN)],
        "random_sum_late": [(LATE_LAYER, random_sum, GAIN)],
    }

    total = len(source_records) * len(PROMPT_STYLES) * len(controls)
    rows = []
    done = 0
    write_status(phase="generating", progress_done=0, progress_total=total, rows=0)
    for source_record in source_records:
        source = source_record["source"]
        for prompt_style, prompt_fn in PROMPT_STYLES.items():
            prompt = prompt_fn(source)
            for control, hooks in controls.items():
                generated = generate_with_hooks(tokenizer, model, prompt, hooks)
                rows.append(
                    {
                        "model": MODEL_NAME,
                        "early_layer": EARLY_LAYER,
                        "late_layer": LATE_LAYER,
                        "gain": GAIN,
                        "op_a_name": OP_A_NAME,
                        "op_a_suffix": OP_A_SUFFIX,
                        "op_b_name": OP_B_NAME,
                        "op_b_suffix": OP_B_SUFFIX,
                        "source_set": source_record["source_set"],
                        "source": source,
                        "prompt_style": prompt_style,
                        "control": control,
                        "generated": generated,
                        **score(source, generated),
                    }
                )
                done += 1
                if done % 25 == 0:
                    flush(rows)
                    write_status(
                        phase="generating",
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
        summary_csv=str(CSV_DIR / "marker_composition_steering_summary.csv"),
        raw_csv=str(CSV_DIR / "marker_composition_steering_raw.csv"),
        order_contrast_csv=str(CSV_DIR / "marker_composition_order_contrast_summary.csv"),
        failures=[],
    )
    print(summarize(pd.DataFrame(rows)).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
