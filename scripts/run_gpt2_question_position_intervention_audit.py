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
    decode_new_text,
    learn_delta_centroids,
    norm_match_random,
    score_output,
)
from run_gpt2_exclamation_copy_prompt_steering import HARD_SOURCES, normalize_statement  # noqa: E402
from run_gpt2_question_copy_prompt_preservation import PROMPT_STYLES, preservation_score  # noqa: E402


MODEL_NAME = os.getenv("QUESTION_POSITION_MODEL", "gpt2")
LAYERS = [int(x) for x in os.getenv("QUESTION_POSITION_LAYERS", "2,3").split(",") if x.strip()]
GAIN = float(os.getenv("QUESTION_POSITION_GAIN", "0.75"))
IN_TEMPLATE_SOURCES = int(os.getenv("QUESTION_POSITION_IN_TEMPLATE_SOURCES", "40"))
HARD_SOURCES_N = int(os.getenv("QUESTION_POSITION_HARD_SOURCES", "40"))
MAX_NEW_TOKENS = int(os.getenv("QUESTION_POSITION_MAX_NEW_TOKENS", "28"))
SEED = int(os.getenv("QUESTION_POSITION_SEED", "20260821"))
PROMPT_STYLE_NAMES = [
    x.strip()
    for x in os.getenv("QUESTION_POSITION_PROMPTS", "repeat_sentence,copy_sentence,same_sentence").split(",")
    if x.strip()
]
OUT_DIR = Path(
    os.getenv(
        "QUESTION_POSITION_OUT_DIR",
        "results/experiments/gpt2_question_position_intervention_audit_layer2_3_20260821_results",
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


def attach_position_hook(
    model,
    layer: int,
    vector: np.ndarray,
    gain: float,
    position_mode: str,
    prompt_len: int,
):
    tensor = torch.tensor(vector, dtype=torch.float32, device=model.device)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        seq_len = int(hidden.shape[1])
        positions: list[int] = []
        if position_mode == "last_each_step":
            positions = [seq_len - 1]
        elif seq_len == prompt_len:
            if position_mode == "prompt_first":
                positions = [0]
            elif position_mode == "prompt_middle":
                positions = [max(0, prompt_len // 2)]
            elif position_mode == "prompt_last":
                positions = [prompt_len - 1]
            elif position_mode == "prompt_all":
                positions = list(range(prompt_len))
            else:
                raise ValueError(f"Unknown position_mode: {position_mode}")

        if not positions:
            return output

        edited = hidden.clone()
        edited[:, positions, :] = edited[:, positions, :] + gain * tensor.to(edited.dtype)
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    return model.transformer.h[layer].register_forward_hook(hook)


def generate_one_position(
    tokenizer,
    model,
    prompt: str,
    layer: int | None,
    vector: np.ndarray | None,
    gain: float,
    position_mode: str,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    handle = None
    try:
        if layer is not None and vector is not None and abs(gain) > 0:
            handle = attach_position_hook(
                model,
                layer=layer,
                vector=vector,
                gain=gain,
                position_mode=position_mode,
                prompt_len=int(enc["input_ids"].shape[1]),
            )
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return decode_new_text(tokenizer, out[0].detach().cpu(), int(enc["input_ids"].shape[1]))
    finally:
        if handle is not None:
            handle.remove()


def build_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_dataset()
    question_test = df[df["split"].eq("test") & df["class"].eq("question")].reset_index(drop=True)
    unique_sources = list(dict.fromkeys(question_test["source"].tolist()))[:IN_TEMPLATE_SOURCES]
    rows = [{"source_set": "in_template_test", "source": source} for source in unique_sources]
    rows.extend(
        {"source_set": "hard_out_of_template", "source": normalize_statement(source)}
        for source in HARD_SOURCES[:HARD_SOURCES_N]
    )
    return pd.DataFrame(rows), df[df["split"].eq("train")].reset_index(drop=True)


def choose_wrong_vector(centroids: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[str, np.ndarray]:
    choices = [c for c in sorted(centroids) if c != "question"]
    cls = choices[int(rng.integers(0, len(choices)))]
    return cls, centroids[cls]


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "source_set", "prompt_style", "layer", "control", "position_mode"]
    return (
        raw_df.groupby(group_cols)
        .agg(
            question_mark_rate=("question_mark_hit", "mean"),
            target_marker_rate=("target_marker_hit", "mean"),
            other_marker_rate=("any_other_marker_hit", "mean"),
            mean_content_preservation=("content_preservation", "mean"),
            question_and_preserved_rate=("question_and_preserved", "mean"),
            mean_generated_chars=("generated_chars", "mean"),
            rows=("question_mark_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def flush(rows: list[dict]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(CSV_DIR / "question_position_intervention_raw.csv", index=False)
    summarize(raw_df).to_csv(CSV_DIR / "question_position_intervention_summary.csv", index=False)


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
        prompt_styles=PROMPT_STYLE_NAMES,
        phase="loading_model",
    )

    sources, train_df = build_sources()
    sources.to_csv(CSV_DIR / "question_position_sources.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")

    write_status(phase="learning_centroids", train_rows=len(train_df), source_rows=len(sources))
    centroids_by_layer = learn_delta_centroids(tokenizer, model, train_df, LAYERS)

    prompt_styles = {name: PROMPT_STYLES[name] for name in PROMPT_STYLE_NAMES}
    rows: list[dict] = []
    conditions = [
        ("none", "none", None, 0.0, ""),
        ("target_prompt_first", "prompt_first", "question", GAIN, "question"),
        ("target_prompt_middle", "prompt_middle", "question", GAIN, "question"),
        ("target_prompt_last_once", "prompt_last", "question", GAIN, "question"),
        ("target_prompt_all_once", "prompt_all", "question", GAIN, "question"),
        ("target_last_each_step", "last_each_step", "question", GAIN, "question"),
        ("wrong_last_each_step", "last_each_step", "wrong", GAIN, "wrong"),
        ("random_last_each_step", "last_each_step", "random", GAIN, "random_norm"),
        ("negative_last_each_step", "last_each_step", "negative", GAIN, "question"),
    ]
    total = len(LAYERS) * len(sources) * len(prompt_styles) * len(conditions)
    done = 0
    write_status(phase="generating", progress_done=0, progress_total=total, rows=0)

    for layer in LAYERS:
        centroids = centroids_by_layer[layer]
        question_vec = centroids["question"]
        wrong_cls, wrong_vec = choose_wrong_vector(centroids, rng)
        for source_rec in sources.to_dict("records"):
            for prompt_style, prompt_fn in prompt_styles.items():
                prompt = prompt_fn(source_rec["source"])
                for control, position_mode, vec_kind, gain, vector_class in conditions:
                    if vec_kind is None:
                        vec = None
                    elif vec_kind == "question":
                        vec = question_vec
                    elif vec_kind == "wrong":
                        vec = wrong_vec
                        vector_class = wrong_cls
                    elif vec_kind == "random":
                        vec = norm_match_random(question_vec, rng)
                    elif vec_kind == "negative":
                        vec = -question_vec
                    else:
                        raise ValueError(f"Unknown vec_kind: {vec_kind}")

                    generated = generate_one_position(
                        tokenizer,
                        model,
                        prompt=prompt,
                        layer=layer if vec is not None else None,
                        vector=vec,
                        gain=gain,
                        position_mode=position_mode,
                    )
                    score = score_output(generated, "question")
                    content_preservation = preservation_score(source_rec["source"], generated)
                    rows.append(
                        {
                            "model": MODEL_NAME,
                            "layer": layer,
                            "source_set": source_rec["source_set"],
                            "source": source_rec["source"],
                            "prompt_style": prompt_style,
                            "control": control,
                            "position_mode": position_mode,
                            "gain": gain,
                            "vector_class": vector_class,
                            "prompt": prompt,
                            "generated": generated,
                            "content_preservation": content_preservation,
                            "question_and_preserved": float(
                                score["question_mark_hit"] == 1.0 and content_preservation >= 0.67
                            ),
                            **score,
                        }
                    )
                    done += 1
                    if done % 100 == 0:
                        flush(rows)
                        write_status(
                            phase="generating",
                            current_layer=layer,
                            current_prompt_style=prompt_style,
                            current_control=control,
                            current_position_mode=position_mode,
                            progress_done=done,
                            progress_total=total,
                            rows=len(rows),
                        )

    flush(rows)
    summary = summarize(pd.DataFrame(rows))
    write_status(
        status="finished",
        phase="finished",
        progress_done=done,
        progress_total=total,
        rows=len(rows),
        summary_rows=len(summary),
        finished_at=now(),
        raw_csv=str(CSV_DIR / "question_position_intervention_raw.csv"),
        summary_csv=str(CSV_DIR / "question_position_intervention_summary.csv"),
        failures=[],
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
