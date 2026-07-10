from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))
from upat_dataset import UPATDataset, UPATDatasetConfig  # noqa: E402


OUT_DIR = Path(os.getenv("STEERING_OUT_DIR", "results/experiments/gpt2_activation_steering_pilot_results"))
CSV_DIR = OUT_DIR / "csv"
RAW_PATH = CSV_DIR / "activation_steering_raw.csv"
SUMMARY_PATH = CSV_DIR / "activation_steering_summary.csv"
STATUS_PATH = OUT_DIR / "run_status.json"

MODELS = [m.strip() for m in os.getenv("STEERING_MODELS", "distilgpt2,gpt2").split(",") if m.strip()]
CLASSES = [c.strip() for c in os.getenv("STEERING_CLASSES", "question,negation,modality,tense_shift").split(",") if c.strip()]
GAINS = [float(x) for x in os.getenv("STEERING_GAINS", "0.75,1.5,3.0").split(",") if x.strip()]
CONTROLS = [c.strip() for c in os.getenv("STEERING_CONTROLS", "none,target,wrong_class,random_norm,negative_target").split(",") if c.strip()]
MAX_TEST_SOURCES = int(os.getenv("STEERING_MAX_TEST_SOURCES", "24"))
TRAIN_TEMPLATES = int(os.getenv("STEERING_TRAIN_TEMPLATES", "100"))
TEST_TEMPLATES = int(os.getenv("STEERING_TEST_TEMPLATES", "80"))
MAX_NEW_TOKENS = int(os.getenv("STEERING_MAX_NEW_TOKENS", "28"))
SEED = int(os.getenv("STEERING_SEED", "20260711"))
FLUSH_EVERY = int(os.getenv("STEERING_FLUSH_EVERY", "25"))


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


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_").replace(".", "_")


def get_layer_indices(model) -> list[int]:
    n_layers = len(model.transformer.h)
    override = os.getenv("STEERING_LAYERS")
    if override:
        vals = []
        for raw in override.split(","):
            if raw.strip():
                idx = int(raw)
                if idx < 0:
                    idx = n_layers + idx
                if 0 <= idx < n_layers:
                    vals.append(idx)
        return sorted(set(vals))
    candidates = [max(0, n_layers // 3), max(0, (2 * n_layers) // 3), n_layers - 1]
    return sorted(set(candidates))


def build_dataset() -> pd.DataFrame:
    cfg = UPATDatasetConfig(train_templates=TRAIN_TEMPLATES, test_templates=TEST_TEMPLATES)
    df = UPATDataset(cfg).build()
    return df[df["class"].isin(CLASSES)].reset_index(drop=True)


def last_token_hidden(tokenizer, model, texts: list[str], layers: list[int], batch_size: int = 16) -> dict[int, np.ndarray]:
    by_layer = {layer: [] for layer in layers}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)
            attn = enc["attention_mask"]
            last_idx = attn.sum(dim=1) - 1
            row_idx = torch.arange(attn.shape[0], device=model.device)
            for layer in layers:
                hidden = out.hidden_states[layer + 1]
                vals = hidden[row_idx, last_idx, :].detach().float().cpu().numpy()
                by_layer[layer].append(vals)
    return {layer: np.vstack(parts) for layer, parts in by_layer.items()}


def learn_delta_centroids(tokenizer, model, train_df: pd.DataFrame, layers: list[int]) -> dict[int, dict[str, np.ndarray]]:
    sources = train_df["source"].tolist()
    targets = train_df["target"].tolist()
    labels = train_df["class"].tolist()
    source_h = last_token_hidden(tokenizer, model, sources, layers)
    target_h = last_token_hidden(tokenizer, model, targets, layers)
    centroids: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in layers}
    for layer in layers:
        delta = target_h[layer] - source_h[layer]
        for cls in CLASSES:
            mask = np.array(labels) == cls
            centroids[layer][cls] = delta[mask].mean(axis=0).astype(np.float32)
    return centroids


def make_prompt(source: str) -> str:
    return f"Input: {source}\nOutput:"


def decode_new_text(tokenizer, full_ids: torch.Tensor, prompt_len: int) -> str:
    new_ids = full_ids[prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def choose_wrong_class(target_class: str) -> str:
    choices = [c for c in CLASSES if c != target_class]
    return random.choice(choices)


def norm_match_random(vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rnd = rng.normal(size=vec.shape).astype(np.float32)
    rnd_norm = np.linalg.norm(rnd) + 1e-12
    return rnd / rnd_norm * (np.linalg.norm(vec) + 1e-12)


def class_markers() -> dict[str, list[str]]:
    return {
        "question": ["?", "whether", "could", "did", "was", "were", "is it", "someone asked", "wonder"],
        "negation": ["not", "n't", "never", "failed", "refused", "declined", "avoided", "no "],
        "modality": ["apparently", "reportedly", "seemingly", "allegedly", "supposedly", "rumored", "appears", "according to"],
        "tense_shift": ["will", "going to", "tomorrow", "later", "next week", "previously", "earlier", "used to", "once"],
        "passive": ["was", "were", "became", "received", "came under", "subject to", "marked by"],
    }


def score_output(text: str, target_class: str) -> dict[str, float | str]:
    low = " " + text.lower().strip() + " "
    markers = class_markers()
    scores = {}
    for cls, pats in markers.items():
        scores[f"{cls}_marker"] = float(any(p in low for p in pats))
    target_hit = scores.get(f"{target_class}_marker", 0.0)
    other_hits = sum(v for k, v in scores.items() if k.endswith("_marker") and k != f"{target_class}_marker")
    scores["target_marker_hit"] = float(target_hit)
    scores["any_other_marker_hit"] = float(other_hits > 0)
    scores["generated_chars"] = float(len(text))
    return scores


def attach_hook(model, layer: int, vector: np.ndarray, gain: float):
    tensor = torch.tensor(vector, dtype=torch.float32, device=model.device)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + gain * tensor.to(hidden.dtype)
            return (hidden,) + output[1:]
        hidden = output.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + gain * tensor.to(hidden.dtype)
        return hidden

    return model.transformer.h[layer].register_forward_hook(hook)


def generate_one(tokenizer, model, prompt: str, layer: int | None, vector: np.ndarray | None, gain: float) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
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
            )
        return decode_new_text(tokenizer, out[0].detach().cpu(), enc["input_ids"].shape[1])
    finally:
        if handle is not None:
            handle.remove()


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    group_cols = ["model", "layer", "target_class", "control", "gain"]
    return (
        raw_df.groupby(group_cols)
        .agg(
            target_marker_rate=("target_marker_hit", "mean"),
            other_marker_rate=("any_other_marker_hit", "mean"),
            mean_generated_chars=("generated_chars", "mean"),
            rows=("target_marker_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def flush(rows: list[dict]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(RAW_PATH, index=False)
    summarize(raw_df).to_csv(SUMMARY_PATH, index=False)


def run_model(model_name: str, df: pd.DataFrame, rows: list[dict]) -> None:
    print(f"\n=== Loading {model_name} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to("cpu")

    layers = get_layer_indices(model)
    write_status(current_model=model_name, current_layers=layers, phase="learning_centroids")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    centroids = learn_delta_centroids(tokenizer, model, train_df, layers)

    unique_sources = list(dict.fromkeys(test_df["source"].tolist()))[:MAX_TEST_SOURCES]
    eval_df = test_df[test_df["source"].isin(unique_sources)].copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED + len(rows))
    total = len(eval_df) * len(layers) * (1 + (len(CONTROLS) - int("none" in CONTROLS)) * len(GAINS))
    done = 0

    for layer in layers:
        for rec in eval_df.to_dict("records"):
            prompt = make_prompt(rec["source"])
            target_class = rec["class"]
            target_vec = centroids[layer][target_class]
            wrong_class = choose_wrong_class(target_class)
            control_plan = []
            if "none" in CONTROLS:
                control_plan.append(("none", 0.0, None, ""))
            for gain in GAINS:
                if "target" in CONTROLS:
                    control_plan.append(("target", gain, target_vec, target_class))
                if "wrong_class" in CONTROLS:
                    control_plan.append(("wrong_class", gain, centroids[layer][wrong_class], wrong_class))
                if "random_norm" in CONTROLS:
                    control_plan.append(("random_norm", gain, norm_match_random(target_vec, rng), "random_norm"))
                if "negative_target" in CONTROLS:
                    control_plan.append(("negative_target", gain, -target_vec, target_class))

            for control, gain, vec, vector_class in control_plan:
                done += 1
                write_status(
                    phase="generating",
                    current_model=model_name,
                    current_layer=layer,
                    current_class=target_class,
                    current_control=control,
                    progress_done=done,
                    progress_total=total,
                    rows=len(rows),
                )
                try:
                    generated = generate_one(tokenizer, model, prompt, layer if vec is not None else None, vec, gain)
                    score = score_output(generated, target_class)
                    rows.append(
                        {
                            "model": model_name,
                            "layer": layer,
                            "source": rec["source"],
                            "target": rec["target"],
                            "target_class": target_class,
                            "control": control,
                            "gain": gain,
                            "vector_class": vector_class,
                            "prompt": prompt,
                            "generated": generated,
                            **score,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "model": model_name,
                            "layer": layer,
                            "source": rec["source"],
                            "target": rec["target"],
                            "target_class": target_class,
                            "control": control,
                            "gain": gain,
                            "vector_class": vector_class,
                            "prompt": prompt,
                            "generated": "",
                            "error": repr(exc),
                            "target_marker_hit": math.nan,
                            "any_other_marker_hit": math.nan,
                            "generated_chars": 0.0,
                        }
                    )
                if len(rows) % FLUSH_EVERY == 0:
                    flush(rows)
                    print(f"{model_name}: generated {done}/{total}, total rows={len(rows)}", flush=True)

    flush(rows)
    del model
    del tokenizer


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    write_status(
        status="running",
        started_at=now(),
        models=MODELS,
        classes=CLASSES,
        controls=CONTROLS,
        gains=GAINS,
        max_test_sources=MAX_TEST_SOURCES,
        max_new_tokens=MAX_NEW_TOKENS,
        phase="building_dataset",
    )
    df = build_dataset()
    df.to_csv(CSV_DIR / "activation_steering_dataset.csv", index=False)
    rows: list[dict] = []

    failures = []
    for model_name in MODELS:
        try:
            run_model(model_name, df, rows)
        except Exception as exc:
            failures.append({"model": model_name, "error": repr(exc)})
            write_status(status="running", failures=failures, phase="model_failed", current_model=model_name)
            flush(rows)

    flush(rows)
    final = pd.DataFrame(rows)
    summary = summarize(final)
    write_status(
        status="finished" if not failures else "finished_with_failures",
        finished_at=now(),
        failures=failures,
        rows=len(final),
        summary_rows=len(summary),
        raw_csv=str(RAW_PATH),
        summary_csv=str(SUMMARY_PATH),
    )
    print(f"\nSaved raw: {RAW_PATH}", flush=True)
    print(f"Saved summary: {SUMMARY_PATH}", flush=True)
    print(f"Failures: {failures}", flush=True)


if __name__ == "__main__":
    main()
