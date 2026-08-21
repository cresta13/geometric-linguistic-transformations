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

from run_gpt2_activation_steering_pilot import attach_hook, last_token_hidden, norm_match_random  # noqa: E402
from run_gpt2_exclamation_copy_prompt_steering import (  # noqa: E402
    ADVERBIALS,
    HARD_SOURCES,
    PROMPT_STYLES,
    SUBJECTS,
    VERBS,
    normalize_statement,
)


MODEL_NAME = os.getenv("FINAL_MARKER_LOGIT_MODEL", "gpt2")
LAYERS = [int(x) for x in os.getenv("FINAL_MARKER_LOGIT_LAYERS", "2,3").split(",") if x.strip()]
GAIN = float(os.getenv("FINAL_MARKER_LOGIT_GAIN", "0.75"))
TRAIN_ROWS = int(os.getenv("FINAL_MARKER_LOGIT_TRAIN_ROWS", "120"))
TEST_ROWS = int(os.getenv("FINAL_MARKER_LOGIT_TEST_ROWS", "12"))
HARD_ROWS = int(os.getenv("FINAL_MARKER_LOGIT_HARD_ROWS", "12"))
MAX_NEW_TOKENS = int(os.getenv("FINAL_MARKER_LOGIT_MAX_NEW_TOKENS", "24"))
SEED = int(os.getenv("FINAL_MARKER_LOGIT_SEED", "20260808"))
OUT_DIR = Path(
    os.getenv(
        "FINAL_MARKER_LOGIT_OUT_DIR",
        "results/experiments/gpt2_final_marker_logit_audit_layer2_3_20260808_results",
    )
)
CSV_DIR = OUT_DIR / "csv"
STATUS_PATH = OUT_DIR / "run_status.json"
RAW_PATH = CSV_DIR / "final_marker_logit_sequence_raw.csv"
STEP_PATH = CSV_DIR / "final_marker_logit_steps.csv"
SUMMARY_PATH = CSV_DIR / "final_marker_logit_summary.csv"

MARKERS = {
    "question": {
        "suffix": "?",
        "token_variants": ["?", " ?"],
        "hit": lambda text: "?" in text,
    },
    "exclamation": {
        "suffix": "!",
        "token_variants": ["!", " !"],
        "hit": lambda text: "!" in text,
    },
    "ellipsis": {
        "suffix": "...",
        "token_variants": ["...", " ...", "\u2026", " \u2026"],
        "hit": lambda text: "..." in text or "\u2026" in text,
    },
}
TARGET_CLASSES = [x.strip() for x in os.getenv("FINAL_MARKER_LOGIT_TARGETS", "question,exclamation,ellipsis").split(",") if x.strip()]
PROMPT_STYLE_NAMES = [
    x.strip()
    for x in os.getenv("FINAL_MARKER_LOGIT_PROMPTS", "copy_sentence,same_sentence").split(",")
    if x.strip()
]
CONTROLS = ["none", "target", "wrong_marker", "random_norm", "negative_target"]


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
    return source.strip().rstrip(".!?") + suffix


def build_pairs() -> pd.DataFrame:
    rng = random.Random(SEED)
    sources = make_sources()
    rng.shuffle(sources)
    rows = []
    split_specs = [
        ("train", sources[:TRAIN_ROWS]),
        ("test", sources[TRAIN_ROWS : TRAIN_ROWS + TEST_ROWS]),
    ]
    for split, selected in split_specs:
        for source in selected:
            for cls in TARGET_CLASSES:
                rows.append(
                    {
                        "split": split,
                        "class": cls,
                        "source": source,
                        "target": with_suffix(source, MARKERS[cls]["suffix"]),
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
        for cls in TARGET_CLASSES:
            mask = labels == cls
            centroids[layer][cls] = delta[mask].mean(axis=0).astype(np.float32)
    return centroids


def marker_token_ids(tokenizer) -> dict[str, list[int]]:
    by_marker: dict[str, list[int]] = {}
    for cls, cfg in MARKERS.items():
        ids = []
        for variant in cfg["token_variants"]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(int(encoded[0]))
        by_marker[cls] = sorted(set(ids))
    return by_marker


def token_stats(logits: torch.Tensor, token_ids: list[int]) -> dict[str, float | int]:
    logits_f = logits.detach().float()
    probs = torch.softmax(logits_f, dim=-1)
    if not token_ids:
        return {
            "best_token_id": -1,
            "best_logit": float("nan"),
            "best_prob": float("nan"),
            "best_rank": -1,
            "top1": 0.0,
            "top5": 0.0,
            "top10": 0.0,
        }
    ids = torch.tensor(token_ids, device=logits_f.device, dtype=torch.long)
    marker_logits = logits_f[ids]
    marker_probs = probs[ids]
    local_idx = int(torch.argmax(marker_logits).item())
    best_id = int(ids[local_idx].item())
    best_logit = float(marker_logits[local_idx].item())
    best_prob = float(marker_probs[local_idx].item())
    rank = int(torch.sum(logits_f > logits_f[best_id]).item()) + 1
    return {
        "best_token_id": best_id,
        "best_logit": best_logit,
        "best_prob": best_prob,
        "best_rank": rank,
        "top1": float(rank == 1),
        "top5": float(rank <= 5),
        "top10": float(rank <= 10),
    }


def decode_new_text(tokenizer, full_ids: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True).strip()


def wrong_class_for(target_class: str) -> str:
    choices = [cls for cls in TARGET_CLASSES if cls != target_class]
    return choices[0]


def build_sources(pairs: pd.DataFrame) -> pd.DataFrame:
    test_sources = (
        pairs[pairs["split"].eq("test") & pairs["class"].eq(TARGET_CLASSES[0])]["source"]
        .head(TEST_ROWS)
        .tolist()
    )
    hard_sources = [normalize_statement(x) for x in HARD_SOURCES[:HARD_ROWS]]
    return pd.DataFrame(
        [{"source_set": "in_template", "source": x} for x in test_sources]
        + [{"source_set": "hard_out_of_template", "source": x} for x in hard_sources]
    )


def generate_with_logit_trace(
    tokenizer,
    model,
    prompt: str,
    layer: int | None,
    vector: np.ndarray | None,
    gain: float,
    token_ids_by_marker: dict[str, list[int]],
) -> tuple[str, list[dict]]:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    handle = None
    try:
        if layer is not None and vector is not None and abs(gain) > 0:
            handle = attach_hook(model, layer, vector, gain)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
    finally:
        if handle is not None:
            handle.remove()

    sequence = out.sequences[0].detach().cpu()
    generated_ids = sequence[prompt_len:]
    steps = []
    for step, logits_batch in enumerate(out.scores):
        logits = logits_batch[0]
        next_id = int(generated_ids[step].item()) if step < len(generated_ids) else int(torch.argmax(logits).item())
        decoded_next = tokenizer.decode([next_id], skip_special_tokens=True)
        row = {
            "step": step,
            "next_token_id": next_id,
            "next_token": decoded_next,
        }
        for marker_cls, token_ids in token_ids_by_marker.items():
            stats = token_stats(logits, token_ids)
            for key, value in stats.items():
                row[f"{marker_cls}_{key}"] = value
        steps.append(row)
    return decode_new_text(tokenizer, sequence, prompt_len), steps


def summarize_sequences(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "target_class", "source_set", "prompt_style", "layer", "control"]
    return (
        raw_df.groupby(group_cols)
        .agg(
            target_marker_rate=("target_marker_hit", "mean"),
            mean_target_step0_prob=("target_step0_prob", "mean"),
            median_target_step0_rank=("target_step0_rank", "median"),
            mean_target_best_prob=("target_best_prob", "mean"),
            median_target_best_rank=("target_best_rank", "median"),
            target_top1_any_rate=("target_top1_any", "mean"),
            target_top5_any_rate=("target_top5_any", "mean"),
            target_top10_any_rate=("target_top10_any", "mean"),
            mean_generated_chars=("generated_chars", "mean"),
            rows=("target_marker_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def flush(sequence_rows: list[dict], step_rows: list[dict]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(sequence_rows)
    step_df = pd.DataFrame(step_rows)
    raw_df.to_csv(RAW_PATH, index=False)
    step_df.to_csv(STEP_PATH, index=False)
    if not raw_df.empty:
        summarize_sequences(raw_df).to_csv(SUMMARY_PATH, index=False)


def run() -> None:
    rng = np.random.default_rng(SEED)
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
        targets=TARGET_CLASSES,
        prompt_styles=PROMPT_STYLE_NAMES,
        phase="loading_model",
    )

    pairs = build_pairs()
    pairs.to_csv(CSV_DIR / "final_marker_logit_training_pairs.csv", index=False)
    sources = build_sources(pairs)
    sources.to_csv(CSV_DIR / "final_marker_logit_sources.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")
    token_ids_by_marker = marker_token_ids(tokenizer)
    (CSV_DIR / "final_marker_logit_token_ids.json").write_text(
        json.dumps(token_ids_by_marker, indent=2),
        encoding="utf-8",
    )

    train = pairs[pairs["split"].eq("train")].reset_index(drop=True)
    write_status(phase="learning_centroids", train_rows=len(train), source_rows=len(sources))
    centroids = learn_centroids(tokenizer, model, train)

    prompt_styles = {name: PROMPT_STYLES[name] for name in PROMPT_STYLE_NAMES}
    total = len(TARGET_CLASSES) * len(LAYERS) * len(sources) * len(prompt_styles) * len(CONTROLS)
    sequence_rows: list[dict] = []
    step_rows: list[dict] = []
    done = 0
    write_status(phase="generating", progress_done=0, progress_total=total, rows=0, step_rows=0)

    for target_class in TARGET_CLASSES:
        for layer in LAYERS:
            target_vec = centroids[layer][target_class]
            wrong_cls = wrong_class_for(target_class)
            wrong_vec = centroids[layer][wrong_cls]
            for source_rec in sources.to_dict("records"):
                for prompt_style, prompt_fn in prompt_styles.items():
                    prompt = prompt_fn(source_rec["source"])
                    plan = [
                        ("none", 0.0, None, ""),
                        ("target", GAIN, target_vec, target_class),
                        ("wrong_marker", GAIN, wrong_vec, wrong_cls),
                        ("random_norm", GAIN, norm_match_random(target_vec, rng), "random_norm"),
                        ("negative_target", GAIN, -target_vec, target_class),
                    ]
                    for control, gain, vec, vector_class in plan:
                        generated, trace = generate_with_logit_trace(
                            tokenizer,
                            model,
                            prompt,
                            layer if vec is not None else None,
                            vec,
                            gain,
                            token_ids_by_marker,
                        )
                        for step in trace:
                            step_rows.append(
                                {
                                    "model": MODEL_NAME,
                                    "target_class": target_class,
                                    "source_set": source_rec["source_set"],
                                    "source": source_rec["source"],
                                    "prompt_style": prompt_style,
                                    "layer": layer,
                                    "control": control,
                                    "gain": gain,
                                    "vector_class": vector_class,
                                    **step,
                                }
                            )
                        target_hit = float(MARKERS[target_class]["hit"](generated))
                        target_prefix = target_class
                        step0 = trace[0] if trace else {}
                        best_rank = min(float(step.get(f"{target_prefix}_best_rank", np.inf)) for step in trace)
                        best_prob = max(float(step.get(f"{target_prefix}_best_prob", np.nan)) for step in trace)
                        sequence_rows.append(
                            {
                                "model": MODEL_NAME,
                                "target_class": target_class,
                                "source_set": source_rec["source_set"],
                                "source": source_rec["source"],
                                "prompt_style": prompt_style,
                                "layer": layer,
                                "control": control,
                                "gain": gain,
                                "vector_class": vector_class,
                                "prompt": prompt,
                                "generated": generated,
                                "generated_chars": float(len(generated)),
                                "target_marker_hit": target_hit,
                                "target_step0_prob": float(step0.get(f"{target_prefix}_best_prob", np.nan)),
                                "target_step0_rank": float(step0.get(f"{target_prefix}_best_rank", np.nan)),
                                "target_best_prob": best_prob,
                                "target_best_rank": best_rank,
                                "target_top1_any": float(any(step.get(f"{target_prefix}_top1", 0.0) for step in trace)),
                                "target_top5_any": float(any(step.get(f"{target_prefix}_top5", 0.0) for step in trace)),
                                "target_top10_any": float(any(step.get(f"{target_prefix}_top10", 0.0) for step in trace)),
                            }
                        )
                        done += 1
                        if done % 25 == 0:
                            flush(sequence_rows, step_rows)
                            write_status(
                                phase="generating",
                                current_target=target_class,
                                current_layer=layer,
                                current_prompt_style=prompt_style,
                                current_control=control,
                                progress_done=done,
                                progress_total=total,
                                rows=len(sequence_rows),
                                step_rows=len(step_rows),
                            )

    flush(sequence_rows, step_rows)
    summary = summarize_sequences(pd.DataFrame(sequence_rows))
    write_status(
        status="finished",
        phase="finished",
        progress_done=done,
        progress_total=total,
        rows=len(sequence_rows),
        step_rows=len(step_rows),
        summary_rows=len(summary),
        finished_at=now(),
        raw_csv=str(RAW_PATH),
        step_csv=str(STEP_PATH),
        summary_csv=str(SUMMARY_PATH),
        failures=[],
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
