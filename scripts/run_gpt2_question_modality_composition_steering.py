from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("STEERING_CLASSES", "question,modality,negation,tense_shift")
os.environ.setdefault("STEERING_EVAL_CLASSES", "question,modality")

MODEL_NAME = os.getenv("QM_COMPOSITION_MODEL", "gpt2")
EARLY_LAYER = int(os.getenv("QM_COMPOSITION_EARLY_LAYER", "2"))
LATE_LAYER = int(os.getenv("QM_COMPOSITION_LATE_LAYER", "3"))
GAIN = float(os.getenv("QM_COMPOSITION_GAIN", "0.75"))
OUT_OF_TEMPLATE_ROWS = int(os.getenv("QM_COMPOSITION_OUT_OF_TEMPLATE_ROWS", "40"))
SEED = int(os.getenv("QM_COMPOSITION_SEED", "20260808"))
MAX_NEW_TOKENS = int(os.getenv("QM_COMPOSITION_MAX_NEW_TOKENS", "32"))
OUT_DIR = Path(
    os.getenv(
        "QM_COMPOSITION_OUT_DIR",
        f"results/experiments/{MODEL_NAME}_question_modality_composition_layer2_3_20260808_results",
    )
)
CSV_DIR = OUT_DIR / "csv"
STATUS_PATH = OUT_DIR / "run_status.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
STATUS_PATH.write_text(
    json.dumps(
        {
            "status": "starting",
            "phase": "importing_dependencies",
            "model": MODEL_NAME,
            "early_layer": EARLY_LAYER,
            "late_layer": LATE_LAYER,
            "gain": GAIN,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))

from run_gpt2_activation_steering_pilot import (  # noqa: E402
    attach_hook,
    build_dataset,
    decode_new_text,
    learn_delta_centroids,
    norm_match_random,
)
from run_gpt2_exclamation_copy_prompt_steering import (  # noqa: E402
    HARD_SOURCES,
    PROMPT_STYLES,
    STOPWORDS,
    normalize_statement,
)

MODALITY_MARKERS = [
    "apparently",
    "reportedly",
    "seemingly",
    "allegedly",
    "supposedly",
    "rumored",
    "appears",
    "according to",
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
                max_new_tokens=MAX_NEW_TOKENS,
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


def modality_hit(generated: str) -> float:
    low = " " + generated.lower().strip() + " "
    return float(any(marker in low for marker in MODALITY_MARKERS))


def score(source: str, generated: str) -> dict[str, float]:
    q_hit = float("?" in generated)
    m_hit = modality_hit(generated)
    preserved = content_preserved(source, generated)
    return {
        "question_marker_hit": q_hit,
        "modality_marker_hit": m_hit,
        "both_markers_hit": float(q_hit and m_hit),
        "any_marker_hit": float(q_hit or m_hit),
        "content_preserved": preserved,
        "question_and_preserved": float(q_hit and preserved),
        "modality_and_preserved": float(m_hit and preserved),
        "both_and_preserved": float(q_hit and m_hit and preserved),
        "generated_chars": float(len(generated)),
    }


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "source_set", "prompt_style", "control"]
    return (
        raw_df.groupby(group_cols)
        .agg(
            question_rate=("question_marker_hit", "mean"),
            modality_rate=("modality_marker_hit", "mean"),
            both_markers_rate=("both_markers_hit", "mean"),
            any_marker_rate=("any_marker_hit", "mean"),
            content_preserved_rate=("content_preserved", "mean"),
            question_and_preserved_rate=("question_and_preserved", "mean"),
            modality_and_preserved_rate=("modality_and_preserved", "mean"),
            both_and_preserved_rate=("both_and_preserved", "mean"),
            mean_generated_chars=("generated_chars", "mean"),
            rows=("question_marker_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def order_contrast(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["source_set", "source", "prompt_style"]
    small = raw_df[raw_df["control"].isin(["question_then_modality_layers", "modality_then_question_layers", "question_plus_modality_late"])]
    records = []
    for key, group in small.groupby(group_cols):
        by_control = {row.control: row for row in group.itertuples(index=False)}
        if "question_then_modality_layers" not in by_control or "modality_then_question_layers" not in by_control:
            continue
        qm = by_control["question_then_modality_layers"]
        mq = by_control["modality_then_question_layers"]
        plus = by_control.get("question_plus_modality_late")
        records.append(
            {
                "source_set": key[0],
                "source": key[1],
                "prompt_style": key[2],
                "qm_equals_mq": float(qm.generated == mq.generated),
                "qm_mq_marker_profile_equal": float(
                    qm.question_marker_hit == mq.question_marker_hit
                    and qm.modality_marker_hit == mq.modality_marker_hit
                ),
                "qm_equals_plus": float(plus is not None and qm.generated == plus.generated),
                "mq_equals_plus": float(plus is not None and mq.generated == plus.generated),
                "qm_question_hit": qm.question_marker_hit,
                "qm_modality_hit": qm.modality_marker_hit,
                "mq_question_hit": mq.question_marker_hit,
                "mq_modality_hit": mq.modality_marker_hit,
                "plus_question_hit": float(plus.question_marker_hit) if plus is not None else np.nan,
                "plus_modality_hit": float(plus.modality_marker_hit) if plus is not None else np.nan,
            }
        )
    if not records:
        return pd.DataFrame()
    contrast = pd.DataFrame(records)
    contrast.to_csv(CSV_DIR / "question_modality_order_contrast_raw.csv", index=False)
    return (
        contrast.groupby(["source_set", "prompt_style"])
        .agg(
            qm_equals_mq_rate=("qm_equals_mq", "mean"),
            qm_mq_marker_profile_equal_rate=("qm_mq_marker_profile_equal", "mean"),
            qm_equals_plus_rate=("qm_equals_plus", "mean"),
            mq_equals_plus_rate=("mq_equals_plus", "mean"),
            qm_question_rate=("qm_question_hit", "mean"),
            qm_modality_rate=("qm_modality_hit", "mean"),
            mq_question_rate=("mq_question_hit", "mean"),
            mq_modality_rate=("mq_modality_hit", "mean"),
            plus_question_rate=("plus_question_hit", "mean"),
            plus_modality_rate=("plus_modality_hit", "mean"),
            rows=("qm_equals_mq", "count"),
        )
        .reset_index()
    )


def flush(rows: list[dict]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(CSV_DIR / "question_modality_composition_raw.csv", index=False)
    summarize(raw_df).to_csv(CSV_DIR / "question_modality_composition_summary.csv", index=False)
    order_contrast(raw_df).to_csv(CSV_DIR / "question_modality_order_contrast_summary.csv", index=False)


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
        op_a_name="question",
        op_b_name="modality",
        phase="loading_model",
    )

    train_df = build_dataset()
    train_df = train_df[train_df["split"].eq("train")].reset_index(drop=True)
    train_df.to_csv(CSV_DIR / "question_modality_training_pairs.csv", index=False)

    source_records = [
        {"source_set": "hard_out_of_template", "source": normalize_statement(source)}
        for source in HARD_SOURCES[:OUT_OF_TEMPLATE_ROWS]
    ]
    pd.DataFrame(source_records).to_csv(CSV_DIR / "question_modality_sources.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")

    layers = [EARLY_LAYER, LATE_LAYER]
    write_status(phase="learning_centroids", train_rows=len(train_df), source_rows=len(source_records))
    centroids = learn_delta_centroids(tokenizer, model, train_df, layers)
    q_early = centroids[EARLY_LAYER]["question"]
    m_early = centroids[EARLY_LAYER]["modality"]
    q_late = centroids[LATE_LAYER]["question"]
    m_late = centroids[LATE_LAYER]["modality"]
    sum_late = q_late + m_late
    random_sum = norm_match_random(sum_late, rng)

    controls = {
        "none": [],
        "question_only_late": [(LATE_LAYER, q_late, GAIN)],
        "modality_only_late": [(LATE_LAYER, m_late, GAIN)],
        "question_plus_modality_late": [(LATE_LAYER, sum_late, GAIN)],
        "question_then_modality_layers": [(EARLY_LAYER, q_early, GAIN), (LATE_LAYER, m_late, GAIN)],
        "modality_then_question_layers": [(EARLY_LAYER, m_early, GAIN), (LATE_LAYER, q_late, GAIN)],
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
        raw_csv=str(CSV_DIR / "question_modality_composition_raw.csv"),
        summary_csv=str(CSV_DIR / "question_modality_composition_summary.csv"),
        order_contrast_csv=str(CSV_DIR / "question_modality_order_contrast_summary.csv"),
        failures=[],
    )
    print(summarize(pd.DataFrame(rows)).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
