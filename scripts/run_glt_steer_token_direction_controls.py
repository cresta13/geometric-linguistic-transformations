from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("STEERING_MAX_NEW_TOKENS", "24")

from run_gpt2_activation_steering_pilot import last_token_hidden, norm_match_random  # noqa: E402
from run_glt_steer_confirmatory_fixed_params import (  # noqa: E402
    CONFIRMATORY_HELDOUT_SOURCES,
    MARKERS,
    PROMPT_STYLE_NAMES,
    TARGET_CLASSES,
    content_preserved,
    generate_with_trace,
    learn_centroids,
    make_training_sources,
    marker_token_ids,
    with_suffix,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "experiments" / "glt_steer_token_direction_controls_20261009_results"
CSV_DIR = OUT_DIR / "csv"
RAW_PATH = CSV_DIR / "glt_steer_token_direction_controls_raw.csv"
SUMMARY_PATH = CSV_DIR / "glt_steer_token_direction_controls_summary.csv"
SUMMARY_MD = OUT_DIR / "SUMMARY.md"
STATUS_PATH = OUT_DIR / "run_status.json"
SEED = 20261009
LAYERS = [2]
GAIN = 0.75
TRAIN_ROWS = 120
HELDOUT_ROWS = 40
AUDIT_CLASSES = ["question"]
AUDIT_PROMPT_STYLES = ["same_sentence"]


def now() -> str:
    return pd.Timestamp.now(tz="Europe/Moscow").isoformat()


def write_status(**updates: object) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current = {}
    if STATUS_PATH.exists():
        current = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    current.update(updates)
    current["updated_at"] = now()
    STATUS_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


def single_token_id(tokenizer, variants: list[str]) -> int | None:
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def token_embedding(model, token_id: int) -> np.ndarray:
    return model.transformer.wte.weight[token_id].detach().float().cpu().numpy().astype(np.float32)


def shuffled_pair_centroids(tokenizer, model, train_df: pd.DataFrame, layers: list[int]) -> dict[int, dict[str, np.ndarray]]:
    sources = train_df["source"].tolist()
    targets = train_df["target"].tolist()
    source_h = last_token_hidden(tokenizer, model, sources, layers)
    target_h = last_token_hidden(tokenizer, model, targets, layers)
    permutation = np.random.default_rng(SEED).permutation(len(train_df))
    labels = train_df["class"].to_numpy()
    result: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in layers}
    for layer in layers:
        delta = target_h[layer][permutation] - source_h[layer]
        for cls in TARGET_CLASSES:
            result[layer][cls] = delta[labels == cls].mean(axis=0).astype(np.float32)
    return result


def normalize_source(source: str) -> str:
    return source.strip()


def make_prompt(source: str, style: str) -> str:
    if style == "repeat_sentence":
        return f"Repeat the following sentence exactly, but make it a question:\n{source}\n"
    if style == "copy_sentence":
        return f"Copy this sentence and add a question mark:\n{source}\n"
    return f"{source}\n"


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["target_class", "control", "layer"], dropna=False)
        .agg(
            target_marker_rate=("target_marker_hit", "mean"),
            target_and_preserved_rate=("target_and_preserved", "mean"),
            content_preserved_rate=("content_preserved", "mean"),
            rows=("target_marker_hit", "count"),
        )
        .reset_index()
        .sort_values(["target_class", "control", "layer"])
    )


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    write_status(status="running", phase="loading_model", model="gpt2", layers=LAYERS, gain=GAIN)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    model.to("cpu")

    train_sources = make_training_sources()[:TRAIN_ROWS]
    train_rows = []
    for source in train_sources:
        for cls, config in MARKERS.items():
            train_rows.append({"source": source, "target": with_suffix(source, config["suffix"]), "class": cls})
    train_df = pd.DataFrame(train_rows)
    sources = [normalize_source(x) for x in CONFIRMATORY_HELDOUT_SOURCES[:HELDOUT_ROWS]]
    prompt_styles = AUDIT_PROMPT_STYLES
    target_ids = marker_token_ids(tokenizer)

    write_status(phase="learning_centroids", train_rows=len(train_df), heldout_sources=len(sources))
    centroids = learn_centroids(tokenizer, model, train_df, LAYERS)
    shuffled = shuffled_pair_centroids(tokenizer, model, train_df, LAYERS)
    token_variants = {
        "question": ["?", " ?"],
        "exclamation": ["!", " !"],
        "ellipsis": ["...", " ...", "\u2026", " \u2026"],
        "semicolon": [";", " ;"],
        "comma": [",", " ,"],
        "colon": [":", " :"],
        "random_word": [" the", "The"],
    }
    token_ids = {name: single_token_id(tokenizer, variants) for name, variants in token_variants.items()}
    if any(value is None for value in token_ids.values()):
        raise RuntimeError(f"Could not find single-token controls: {token_ids}")
    (CSV_DIR / "token_control_ids.json").write_text(json.dumps(token_ids, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    controls = [
        "none",
        "target_delta",
        "shuffled_pair_delta",
        "target_token_embedding_raw",
        "target_token_embedding_normmatched",
        "other_punctuation_normmatched",
        "random_word_normmatched",
        "random_norm",
        "negative_target_delta",
    ]
    total = len(AUDIT_CLASSES) * len(LAYERS) * len(sources) * len(prompt_styles) * len(controls)
    done = 0
    rng = np.random.default_rng(SEED)
    for target_class in AUDIT_CLASSES:
        for layer in LAYERS:
            target_vec = centroids[layer][target_class]
            target_token_vec = token_embedding(model, int(target_ids[target_class][0]))
            other_token_name = "semicolon" if target_class != "question" else "colon"
            other_token_vec = token_embedding(model, int(token_ids[other_token_name]))
            word_vec = token_embedding(model, int(token_ids["random_word"]))
            vectors = {
                "none": None,
                "target_delta": target_vec,
                "shuffled_pair_delta": shuffled[layer][target_class],
                "target_token_embedding_raw": target_token_vec,
                "target_token_embedding_normmatched": target_token_vec,
                "other_punctuation_normmatched": other_token_vec,
                "random_word_normmatched": word_vec,
                "random_norm": norm_match_random(target_vec, rng),
                "negative_target_delta": -target_vec,
            }
            target_norm = np.linalg.norm(target_vec) + 1e-12
            for key in ["target_token_embedding_normmatched", "other_punctuation_normmatched", "random_word_normmatched"]:
                vectors[key] = vectors[key] / (np.linalg.norm(vectors[key]) + 1e-12) * target_norm
            for source_id, source in enumerate(sources):
                for prompt_style in prompt_styles:
                    prompt = make_prompt(source, prompt_style)
                    for control in controls:
                        done += 1
                        write_status(phase="generating", target_class=target_class, layer=layer, control=control, progress_done=done, progress_total=total, rows=len(rows))
                        vec = vectors[control]
                        generated, trace = generate_with_trace(tokenizer, model, prompt, layer if vec is not None else None, vec, GAIN, target_ids[target_class])
                        marker_hit = float(MARKERS[target_class]["hit"](generated))
                        preserved = float(content_preserved(source, generated))
                        rows.append({
                            "model": "gpt2",
                            "source_id": source_id,
                            "source": source,
                            "prompt_style": prompt_style,
                            "target_class": target_class,
                            "layer": layer,
                            "control": control,
                            "gain": GAIN,
                            "token_control": token_ids.get(control),
                            "generated": generated,
                            "target_marker_hit": marker_hit,
                            "content_preserved": preserved,
                            "target_and_preserved": float(marker_hit and preserved),
                            **trace,
                        })
                        if len(rows) % 100 == 0:
                            raw = pd.DataFrame(rows)
                            raw.to_csv(RAW_PATH, index=False)
                            summarize(raw).to_csv(SUMMARY_PATH, index=False)
                            print(f"generated {done}/{total}", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(RAW_PATH, index=False)
    summary = summarize(raw)
    summary.to_csv(SUMMARY_PATH, index=False)
    question = summary[summary["target_class"].eq("question")].copy()
    lines = [
        "# GLT-STEER Token-Direction Controls",
        "",
        "This GPT-2 audit tests whether final-marker steering is specific to the learned sentence-pair delta direction or can be reproduced by unrelated token and matched-norm directions.",
        "",
        "- Model: `gpt2`",
        "- Target: question marker `?`",
        "- Layer: `2`",
        "- Gain: `0.75`",
        "- Sources: `40` structurally varied held-out sentences",
        "- Prompt: `same_sentence`",
        "- Controls: learned target delta, shuffled-pair delta, target-token embedding, other punctuation, random word, random norm, and negative target delta.",
        "",
        "## Results",
        "",
        "| control | question marker rate | marker+preserved rate | rows |",
        "|---|---:|---:|---:|",
    ]
    for row in question.itertuples(index=False):
        lines.append(f"| `{row.control}` | {row.target_marker_rate:.4f} | {row.target_and_preserved_rate:.4f} | {int(row.rows)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The target delta is the prespecified positive direction. The token and shuffled-pair controls are tests of the alternative explanation that any direction with a related norm, or a direction associated with a punctuation token, is sufficient. This audit is limited to one model, one layer, one marker, and one prompt protocol; it is a specificity control, not a general semantic-editing test.",
            "",
            "The raw generations and aggregate table are retained so that the control can be inspected without rerunning inference.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_status(status="finished", finished_at=now(), rows=len(raw), summary_rows=len(summary), raw_csv=str(RAW_PATH), summary_csv=str(SUMMARY_PATH), summary_md=str(SUMMARY_MD), failures=[])
    print(f"Saved {RAW_PATH}")


if __name__ == "__main__":
    main()
