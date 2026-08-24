from __future__ import annotations

import gc
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
    PROMPT_STYLES,
    STOPWORDS,
    SUBJECTS,
    VERBS,
    normalize_statement,
)


SEED = int(os.getenv("GLT_STEER_CONFIRM_SEED", "20260825"))
TRAIN_ROWS = int(os.getenv("GLT_STEER_CONFIRM_TRAIN_ROWS", "120"))
HELDOUT_ROWS = int(os.getenv("GLT_STEER_CONFIRM_HELDOUT_ROWS", "48"))
MAX_NEW_TOKENS = int(os.getenv("GLT_STEER_CONFIRM_MAX_NEW_TOKENS", "24"))
OUT_DIR = Path(
    os.getenv(
        "GLT_STEER_CONFIRM_OUT_DIR",
        "results/experiments/glt_steer_confirmatory_fixed_params_20260825_results",
    )
)
CSV_DIR = OUT_DIR / "csv"
STATUS_PATH = OUT_DIR / "run_status.json"
RAW_PATH = CSV_DIR / "glt_steer_confirmatory_raw.csv"
SUMMARY_PATH = CSV_DIR / "glt_steer_confirmatory_summary.csv"
SOURCE_PATH = CSV_DIR / "glt_steer_confirmatory_sources.csv"
TRAINING_PATH = CSV_DIR / "glt_steer_confirmatory_training_pairs.csv"
TOKEN_IDS_PATH = CSV_DIR / "glt_steer_confirmatory_marker_token_ids.json"

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
TARGET_CLASSES = ["question", "exclamation", "ellipsis"]
PROMPT_STYLE_NAMES = ["repeat_sentence", "copy_sentence", "same_sentence"]
CONTROLS = ["none", "target", "wrong_marker", "random_norm", "negative_target"]
MODEL_CONFIGS = [
    {
        "model": "gpt2",
        "layers": [2, 3],
        "gain": 0.75,
        "setting_source": "fixed from prior GPT-2 question/final-marker audits; no tuning in this run",
    },
    {
        "model": "distilgpt2",
        "layers": [2],
        "gain": 1.0,
        "setting_source": "fixed from prior DistilGPT-2 layer/gain sweep; no tuning in this run",
    },
]

CONFIRMATORY_HELDOUT_SOURCES = [
    "The grant application was revised after the committee meeting.",
    "Although the prototype overheated, the backup controller stayed online.",
    "Marina counted 73 labeled samples before the freezer alarm sounded.",
    "The dataset was anonymized by the hospital team in Krakow.",
    "When the parser failed, the notebook saved a diagnostic snapshot.",
    "A second telescope observed Saturn during the early morning window.",
    "Because the elevator stopped, the courier used the emergency stairs.",
    "The calibration report was signed by Victor at 18:40.",
    "After the river level dropped, the engineers inspected the dam.",
    "The microscope recorded 512 frames before the lamp dimmed.",
    "If the validator rejects the file, the pipeline opens a review ticket.",
    "The old turbine was replaced during the overnight maintenance shift.",
    "Professor Novak compared four baselines on the university cluster.",
    "While the train waited outside Lyon, the server finished indexing.",
    "The survey invitation was translated into Portuguese on Friday.",
    "Because the route was blocked, Alina updated the delivery schedule.",
    "The archive stored 24 encrypted drives in a locked cabinet.",
    "After the model converged, the researcher exported the confusion matrix.",
    "The chemical sample was sealed by two technicians in Room 7.",
    "When the satellite passed overhead, the antenna logged a weak signal.",
    "The migration script copied 19 tables before the database restarted.",
    "Although the lecture began late, the recording captured every slide.",
    "The legal memo was reviewed by Nadia before noon.",
    "Because the sensor drifted, the robot recalibrated its camera.",
    "The container was inspected at Gate 6 after midnight.",
    "When the timer expired, the incubator locked the control panel.",
    "The museum catalog listed 38 restored photographs from 1921.",
    "After the password reset, the dashboard requested a fresh token.",
    "The clinical team processed 64 consent forms during the audit.",
    "Although the storm damaged the pier, the lighthouse remained powered.",
    "The finance report was delivered to Building 4 by courier.",
    "When the compressor restarted, the pressure gauge returned to baseline.",
    "The research assistant labeled nine folders before lunch.",
    "Because the network queue filled, the batch job paused automatically.",
    "The manuscript was scanned by a volunteer in Tallinn.",
    "After the rehearsal, Sofia checked the metronome settings.",
    "The warehouse shipped 31 replacement batteries to Lisbon.",
    "When the freezer door opened, the sample tags stayed readable.",
    "The council approved the transport plan in July 2026.",
    "Although the browser crashed, the form preserved the draft.",
    "The medical device logged 27 alerts during the night shift.",
    "Before the demo started, Olesya checked the microphone cable.",
    "The bridge inspection was completed by a local engineering crew.",
    "Because the invoice changed, the script regenerated the summary.",
    "The rover crossed 12 meters of loose gravel near the ridge.",
    "When the camera refocused, the barcode became readable.",
    "The committee archived five proposals after the public hearing.",
    "Although the meeting moved online, the interpreter joined on time.",
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


def make_training_sources() -> list[str]:
    sources = []
    for subject in SUBJECTS:
        for verb in VERBS:
            for adverbial in ADVERBIALS:
                sources.append(normalize_statement(f"{subject} {verb} {adverbial}."))
    return sources


def with_suffix(source: str, suffix: str) -> str:
    return source.strip().rstrip(".!?") + suffix


def build_training_pairs() -> pd.DataFrame:
    rng = random.Random(SEED)
    sources = make_training_sources()
    rng.shuffle(sources)
    selected = sources[:TRAIN_ROWS]
    rows = []
    for source in selected:
        for cls, cfg in MARKERS.items():
            rows.append(
                {
                    "split": "train",
                    "class": cls,
                    "source": source,
                    "target": with_suffix(source, cfg["suffix"]),
                }
            )
    return pd.DataFrame(rows)


def build_heldout_sources() -> pd.DataFrame:
    rows = CONFIRMATORY_HELDOUT_SOURCES[:HELDOUT_ROWS]
    return pd.DataFrame(
        {
            "source_set": "confirmatory_hard_heldout",
            "source_id": list(range(len(rows))),
            "source": [normalize_statement(x) for x in rows],
        }
    )


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


def learn_centroids(tokenizer, model, train_df: pd.DataFrame, layers: list[int]) -> dict[int, dict[str, np.ndarray]]:
    source_h = last_token_hidden(tokenizer, model, train_df["source"].tolist(), layers)
    target_h = last_token_hidden(tokenizer, model, train_df["target"].tolist(), layers)
    labels = train_df["class"].to_numpy()
    centroids = {layer: {} for layer in layers}
    for layer in layers:
        delta = target_h[layer] - source_h[layer]
        for cls in TARGET_CLASSES:
            centroids[layer][cls] = delta[labels == cls].mean(axis=0).astype(np.float32)
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
    hits = sum(1 for word in words if word in generated_low)
    return float(hits / len(words) >= 0.6)


def generated_text(tokenizer, sequence: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True).strip()


def token_stats(logits: torch.Tensor, token_ids: list[int]) -> dict[str, float | int]:
    logits_f = logits.detach().float()
    probs = torch.softmax(logits_f, dim=-1)
    if not token_ids:
        return {"best_prob": float("nan"), "best_rank": -1, "top1": 0.0, "top5": 0.0}
    ids = torch.tensor(token_ids, device=logits_f.device, dtype=torch.long)
    marker_logits = logits_f[ids]
    local_idx = int(torch.argmax(marker_logits).item())
    best_id = int(ids[local_idx].item())
    rank = int(torch.sum(logits_f > logits_f[best_id]).item()) + 1
    return {
        "best_prob": float(probs[best_id].item()),
        "best_rank": rank,
        "top1": float(rank == 1),
        "top5": float(rank <= 5),
    }


def generate_with_trace(
    tokenizer,
    model,
    prompt: str,
    layer: int | None,
    vector: np.ndarray | None,
    gain: float,
    target_token_ids: list[int],
) -> tuple[str, dict[str, float]]:
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

    per_step = [token_stats(scores[0], target_token_ids) for scores in out.scores]
    if per_step:
        best_rank = min(float(step["best_rank"]) for step in per_step)
        best_prob = max(float(step["best_prob"]) for step in per_step)
        top1_any = float(any(step["top1"] for step in per_step))
        top5_any = float(any(step["top5"] for step in per_step))
        step0 = per_step[0]
    else:
        best_rank = float("nan")
        best_prob = float("nan")
        top1_any = 0.0
        top5_any = 0.0
        step0 = {"best_prob": float("nan"), "best_rank": float("nan")}
    return generated_text(tokenizer, out.sequences[0].detach().cpu(), prompt_len), {
        "target_step0_prob": float(step0["best_prob"]),
        "target_step0_rank": float(step0["best_rank"]),
        "target_best_prob": float(best_prob),
        "target_best_rank": float(best_rank),
        "target_top1_any": top1_any,
        "target_top5_any": top5_any,
    }


def wrong_class(target_class: str) -> str:
    return next(cls for cls in TARGET_CLASSES if cls != target_class)


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "target_class", "source_set", "prompt_style", "layer", "control", "gain"]
    return (
        raw_df.groupby(group_cols)
        .agg(
            target_marker_rate=("target_marker_hit", "mean"),
            content_preserved_rate=("content_preserved", "mean"),
            target_and_preserved_rate=("target_and_preserved", "mean"),
            mean_target_step0_prob=("target_step0_prob", "mean"),
            median_target_step0_rank=("target_step0_rank", "median"),
            mean_target_best_prob=("target_best_prob", "mean"),
            median_target_best_rank=("target_best_rank", "median"),
            target_top1_any_rate=("target_top1_any", "mean"),
            target_top5_any_rate=("target_top5_any", "mean"),
            rows=("target_marker_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def flush(rows: list[dict]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(RAW_PATH, index=False)
    if not raw_df.empty:
        summarize(raw_df).to_csv(SUMMARY_PATH, index=False)


def run_model(config: dict, train_df: pd.DataFrame, sources: pd.DataFrame, rows: list[dict], start_done: int, total: int) -> int:
    model_name = config["model"]
    layers = list(config["layers"])
    gain = float(config["gain"])
    rng = np.random.default_rng(SEED + len(rows) + len(model_name))
    write_status(
        status="running",
        phase="loading_model",
        current_model=model_name,
        progress_done=start_done,
        progress_total=total,
        rows=len(rows),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to("cpu")

    token_ids = marker_token_ids(tokenizer)
    token_id_payload = {}
    if TOKEN_IDS_PATH.exists():
        try:
            token_id_payload = json.loads(TOKEN_IDS_PATH.read_text(encoding="utf-8"))
        except Exception:
            token_id_payload = {}
    token_id_payload[model_name] = token_ids
    TOKEN_IDS_PATH.write_text(json.dumps(token_id_payload, indent=2), encoding="utf-8")

    write_status(
        phase="learning_centroids",
        current_model=model_name,
        layers=layers,
        gain=gain,
        train_rows=len(train_df),
        heldout_rows=len(sources),
        progress_done=start_done,
        progress_total=total,
        rows=len(rows),
    )
    centroids = learn_centroids(tokenizer, model, train_df, layers)
    done = start_done
    prompt_styles = {name: PROMPT_STYLES[name] for name in PROMPT_STYLE_NAMES}

    for target_class in TARGET_CLASSES:
        for layer in layers:
            target_vec = centroids[layer][target_class]
            wrong_cls = wrong_class(target_class)
            wrong_vec = centroids[layer][wrong_cls]
            for source_rec in sources.to_dict("records"):
                for prompt_style, prompt_fn in prompt_styles.items():
                    prompt = prompt_fn(source_rec["source"])
                    plan = [
                        ("none", 0.0, None, ""),
                        ("target", gain, target_vec, target_class),
                        ("wrong_marker", gain, wrong_vec, wrong_cls),
                        ("random_norm", gain, norm_match_random(target_vec, rng), "random_norm"),
                        ("negative_target", gain, -target_vec, target_class),
                    ]
                    for control, row_gain, vec, vector_class in plan:
                        generated, logit_metrics = generate_with_trace(
                            tokenizer,
                            model,
                            prompt,
                            layer if vec is not None else None,
                            vec,
                            row_gain,
                            token_ids[target_class],
                        )
                        target_marker_hit = float(MARKERS[target_class]["hit"](generated))
                        preserved = content_preserved(source_rec["source"], generated)
                        rows.append(
                            {
                                "model": model_name,
                                "target_class": target_class,
                                "source_set": source_rec["source_set"],
                                "source_id": source_rec["source_id"],
                                "source": source_rec["source"],
                                "prompt_style": prompt_style,
                                "layer": layer,
                                "control": control,
                                "gain": row_gain,
                                "fixed_setting_source": config["setting_source"],
                                "vector_class": vector_class,
                                "prompt": prompt,
                                "generated": generated,
                                "generated_chars": float(len(generated)),
                                "target_marker_hit": target_marker_hit,
                                "content_preserved": preserved,
                                "target_and_preserved": float(target_marker_hit and preserved),
                                **logit_metrics,
                            }
                        )
                        done += 1
                        if done % 50 == 0:
                            flush(rows)
                            write_status(
                                phase="generating",
                                current_model=model_name,
                                current_target=target_class,
                                current_layer=layer,
                                current_prompt_style=prompt_style,
                                current_control=control,
                                progress_done=done,
                                progress_total=total,
                                rows=len(rows),
                            )

    del model
    del tokenizer
    gc.collect()
    return done


def run() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    train_df = build_training_pairs()
    sources = build_heldout_sources()
    train_df.to_csv(TRAINING_PATH, index=False)
    sources.to_csv(SOURCE_PATH, index=False)
    total = sum(
        len(TARGET_CLASSES) * len(config["layers"]) * len(sources) * len(PROMPT_STYLE_NAMES) * len(CONTROLS)
        for config in MODEL_CONFIGS
    )

    write_status(
        status="running",
        started_at=now(),
        phase="initialized",
        seed=SEED,
        train_rows=len(train_df),
        heldout_rows=len(sources),
        target_classes=TARGET_CLASSES,
        prompt_styles=PROMPT_STYLE_NAMES,
        controls=CONTROLS,
        model_configs=MODEL_CONFIGS,
        progress_done=0,
        progress_total=total,
        rows=0,
    )

    rows: list[dict] = []
    done = 0
    failures = []
    for config in MODEL_CONFIGS:
        try:
            done = run_model(config, train_df, sources, rows, done, total)
            flush(rows)
        except Exception as exc:  # noqa: BLE001
            failures.append({"model": config["model"], "error": repr(exc)})
            flush(rows)
            write_status(
                status="failed",
                phase="failed",
                current_model=config["model"],
                progress_done=done,
                progress_total=total,
                rows=len(rows),
                failures=failures,
            )
            raise

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
        raw_csv=str(RAW_PATH),
        summary_csv=str(SUMMARY_PATH),
        source_csv=str(SOURCE_PATH),
        training_csv=str(TRAINING_PATH),
        token_ids_json=str(TOKEN_IDS_PATH),
        failures=failures,
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
