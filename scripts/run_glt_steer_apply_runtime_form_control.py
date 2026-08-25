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

from run_glt_steer_confirmatory_fixed_params import (  # noqa: E402
    CONFIRMATORY_HELDOUT_SOURCES,
    MARKERS,
    MODEL_CONFIGS,
    TARGET_CLASSES,
    build_training_pairs,
    marker_token_ids,
    normalize_statement,
)
from run_gpt2_activation_steering_pilot import (  # noqa: E402
    attach_hook,
    last_token_hidden,
    norm_match_random,
)


SEED = int(os.getenv("GLT_STEER_APPLY_SEED", "20260825"))
SOURCE_ROWS = int(os.getenv("GLT_STEER_APPLY_SOURCE_ROWS", "80"))
TRAIN_ROWS = int(os.getenv("GLT_STEER_APPLY_TRAIN_ROWS", "120"))
MAX_NEW_TOKENS = int(os.getenv("GLT_STEER_APPLY_MAX_NEW_TOKENS", "24"))
OUT_DIR = Path(
    os.getenv(
        "GLT_STEER_APPLY_OUT_DIR",
        "results/experiments/glt_steer_apply_runtime_form_control_20260825_results",
    )
)
CSV_DIR = OUT_DIR / "csv"
STATUS_PATH = OUT_DIR / "run_status.json"
RAW_PATH = CSV_DIR / "glt_steer_apply_raw.csv"
SUMMARY_PATH = CSV_DIR / "glt_steer_apply_summary.csv"
GLOBAL_PATH = CSV_DIR / "glt_steer_apply_global_summary.csv"
SOURCE_PATH = CSV_DIR / "glt_steer_apply_sources.csv"
TRAINING_PATH = CSV_DIR / "glt_steer_apply_training_pairs.csv"
TOKEN_IDS_PATH = CSV_DIR / "glt_steer_apply_marker_token_ids.json"

APPLY_MODEL_CONFIGS = [
    {
        **cfg,
        "layers": [
            int(x)
            for x in os.getenv(
                f"GLT_STEER_APPLY_{cfg['model'].upper().replace('-', '_')}_LAYERS",
                ",".join(str(v) for v in cfg["layers"]),
            ).split(",")
            if x.strip()
        ],
    }
    for cfg in MODEL_CONFIGS
    if cfg["model"] in {m.strip() for m in os.getenv("GLT_STEER_APPLY_MODELS", "gpt2,distilgpt2").split(",")}
]

PROMPT_STYLES = {
    "neutral_restate": lambda source: f"Restate the following sentence exactly.\nSentence: {source}\nRestatement:",
    "neutral_same": lambda source: f"Same meaning, same wording:\n{source}\nOutput:",
}

STRONG_PROMPTS = {
    "question": lambda source: (
        "Restate the following sentence exactly, but end it with a question mark.\n"
        f"Sentence: {source}\nRestatement:"
    ),
    "exclamation": lambda source: (
        "Restate the following sentence exactly, but end it with an exclamation mark.\n"
        f"Sentence: {source}\nRestatement:"
    ),
    "ellipsis": lambda source: (
        "Restate the following sentence exactly, but end it with an ellipsis.\n"
        f"Sentence: {source}\nRestatement:"
    ),
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
    "into",
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

APPLICATION_SOURCES = [
    "The contract was signed in Berlin.",
    "Maria submitted the report after midnight.",
    "The backup server failed during migration.",
    "The committee approved the proposal.",
    "A pediatric clinic opened near the railway station.",
    "The research team archived the calibration logs.",
    "Anton replaced the battery before the field test.",
    "The invoice matched the purchase order.",
    "Several volunteers cleaned the lab benches.",
    "The solar array produced enough power for the rover.",
    "The judge postponed the hearing until Monday.",
    "Nadia checked the microscope before the lecture.",
    "The file checksum changed after compression.",
    "The museum restored three photographs from 1912.",
    "A courier delivered the tablets to Room 406.",
    "The telescope recorded a faint signal near Jupiter.",
    "The hospital updated the consent form.",
    "Mila copied the dataset to an encrypted drive.",
    "The safety valve opened during the pressure test.",
    "The archive listed twelve missing pages.",
    "A data analyst reviewed the outlier report.",
    "The cargo ship reached Lisbon before sunrise.",
    "The browser saved the form draft automatically.",
    "A violin student practiced near the old theater.",
    "The compiler emitted a warning during packaging.",
    "The bridge inspection finished ahead of schedule.",
    "A climate sensor drifted after the storm.",
    "The robot mapped the corridor on its second pass.",
    "The translation service returned a malformed token.",
    "Sofia labeled the audio clips before lunch.",
    "The drone photographed the flooded road.",
    "The legal memo cited the revised statute.",
    "A warehouse scanner missed the damaged barcode.",
    "The classroom projector overheated during rehearsal.",
    "The finance team reconciled the quarterly accounts.",
    "A backup script copied the database snapshots.",
    "The library catalog indexed the new manuscripts.",
    "The satellite crossed the equator at 03:14.",
    "A pharmacist verified the batch number.",
    "The research grant covered the conference fee.",
    "The elevator stopped between the ninth and tenth floors.",
    "A field engineer replaced the damaged antenna.",
    "The model checkpoint loaded without errors.",
    "The clinic scheduled forty follow-up calls.",
    "A weather balloon carried the sensor package.",
    "The editor accepted the revised appendix.",
    "The pipeline rejected three duplicate rows.",
    "A ceramic filter removed the visible sediment.",
    "The orchestra rehearsed the final movement.",
    "The meeting transcript mentioned the budget delay.",
    "A customs officer inspected the sealed container.",
    "The freezer alarm sounded during the night shift.",
    "The prototype passed the vibration test.",
    "A maintenance crew repaired the tunnel lights.",
    "The survey reached participants in five cities.",
    "The spreadsheet formula rounded the tax estimate.",
    "A court clerk scanned the signed affidavit.",
    "The image classifier flagged the blurry frame.",
    "The train departed from platform seven.",
    "A graduate student plotted the residual errors.",
    "The medical device logged twenty-seven alerts.",
    "The lighthouse remained powered after the storm.",
    "A bookstore ordered the translated edition.",
    "The conference badge expired after checkout.",
    "The cache cleanup freed several gigabytes.",
    "A security review found no critical findings.",
    "The parser recovered from the missing bracket.",
    "The rover crossed twelve meters of loose gravel.",
    "A food inspector documented the storage temperature.",
    "The notebook exported the confusion matrix.",
    "The committee archived five public comments.",
    "A radio operator confirmed the emergency channel.",
    "The server restarted after the kernel update.",
    "The research assistant measured the sample twice.",
    "The invoice changed after the exchange-rate update.",
    "A moderator closed the discussion thread.",
    "The bicycle courier avoided the flooded avenue.",
    "The database migration paused at table nineteen.",
    "A therapist reviewed the intake questionnaire.",
    "The antenna logged a weak signal.",
    "The validator opened a review ticket.",
    "A technician sealed the chemical sample.",
    "The report included a confidence interval.",
    "The registration desk printed a replacement badge.",
    "A local crew inspected the bridge supports.",
    "The search index refreshed before the demo.",
    "The package arrived with a torn label.",
    "A laboratory notebook recorded the reagent lot.",
    "The microphone cable failed during setup.",
    "The route planner avoided the closed bridge.",
    "A patient portal displayed the test result.",
    "The dashboard requested a fresh token.",
    "The backup controller stayed online.",
    "A translation reviewer corrected the subtitle timing.",
    "The camera refocused on the barcode.",
    "The batch job paused automatically.",
    "A public hearing delayed the transport plan.",
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


def build_sources() -> pd.DataFrame:
    sources = list(dict.fromkeys(CONFIRMATORY_HELDOUT_SOURCES + APPLICATION_SOURCES))[:SOURCE_ROWS]
    return pd.DataFrame(
        {
            "source_set": "application_naturalish_heldout",
            "source_id": list(range(len(sources))),
            "source": [normalize_statement(x) for x in sources],
        }
    )


def content_tokens(text: str) -> list[str]:
    tokens = []
    for raw in str(text).replace("\n", " ").split():
        tok = "".join(ch for ch in raw.lower() if ch.isalnum())
        if tok and tok not in STOPWORDS:
            tokens.append(tok)
    return tokens


def content_preserved(source: str, generated: str) -> float:
    src = set(content_tokens(source))
    if not src:
        return 0.0
    gen = set(content_tokens(generated))
    return float(len(src & gen) / len(src) >= 0.6)


def malformed_or_repetitive(source: str, generated: str) -> float:
    text = str(generated).strip()
    if not text:
        return 1.0
    if len(text) > max(180, int(2.8 * len(source))):
        return 1.0
    if text.count("\n") >= 3:
        return 1.0
    tokens = content_tokens(text)
    if tokens:
        counts = {tok: tokens.count(tok) for tok in set(tokens)}
        if max(counts.values()) >= 4:
            return 1.0
    return 0.0


def append_source(source: str, target_class: str) -> str:
    return source.strip().rstrip(".!?") + MARKERS[target_class]["suffix"]


def generated_text(tokenizer, sequence: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True).strip()


def token_stats(logits: torch.Tensor, token_ids: list[int]) -> dict[str, float]:
    logits_f = logits.detach().float()
    probs = torch.softmax(logits_f, dim=-1)
    if not token_ids:
        return {"target_best_prob": float("nan"), "target_best_rank": float("nan"), "target_top1_any": 0.0}
    ids = torch.tensor(token_ids, device=logits_f.device, dtype=torch.long)
    marker_logits = logits_f[ids]
    best_id = int(ids[int(torch.argmax(marker_logits).item())].item())
    rank = int(torch.sum(logits_f > logits_f[best_id]).item()) + 1
    return {
        "target_best_prob": float(probs[best_id].item()),
        "target_best_rank": float(rank),
        "target_top1_any": float(rank == 1),
    }


def generate_with_trace(
    tokenizer,
    model,
    prompt: str,
    target_token_ids: list[int],
    layer: int | None = None,
    vector: np.ndarray | None = None,
    gain: float = 0.0,
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
    steps = [token_stats(scores[0], target_token_ids) for scores in out.scores]
    if not steps:
        return generated_text(tokenizer, out.sequences[0].detach().cpu(), prompt_len), {
            "target_best_prob": float("nan"),
            "target_best_rank": float("nan"),
            "target_top1_any": 0.0,
        }
    best_rank = min(step["target_best_rank"] for step in steps)
    best_prob = max(step["target_best_prob"] for step in steps)
    top1_any = float(any(step["target_top1_any"] for step in steps))
    return generated_text(tokenizer, out.sequences[0].detach().cpu(), prompt_len), {
        "target_best_prob": best_prob,
        "target_best_rank": best_rank,
        "target_top1_any": top1_any,
    }


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


def wrong_class(target_class: str) -> str:
    return next(cls for cls in TARGET_CLASSES if cls != target_class)


def score_row(source: str, generated: str, target_class: str) -> dict[str, float]:
    marker_hit = float(MARKERS[target_class]["hit"](generated))
    preserved = content_preserved(source, generated)
    malformed = malformed_or_repetitive(source, generated)
    return {
        "target_marker_hit": marker_hit,
        "content_preserved": preserved,
        "target_and_preserved": float(marker_hit and preserved),
        "malformed_or_repetitive": malformed,
        "generated_chars": float(len(str(generated))),
    }


def summarize(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["model", "target_class", "prompt_style", "layer", "control", "gain"]
    summary = (
        raw_df.groupby(group_cols)
        .agg(
            target_marker_rate=("target_marker_hit", "mean"),
            content_preserved_rate=("content_preserved", "mean"),
            target_and_preserved_rate=("target_and_preserved", "mean"),
            malformed_or_repetitive_rate=("malformed_or_repetitive", "mean"),
            mean_target_best_prob=("target_best_prob", "mean"),
            median_target_best_rank=("target_best_rank", "median"),
            target_top1_any_rate=("target_top1_any", "mean"),
            rows=("target_marker_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    global_summary = (
        raw_df.groupby(["model", "target_class", "control"])
        .agg(
            target_marker_rate=("target_marker_hit", "mean"),
            content_preserved_rate=("content_preserved", "mean"),
            target_and_preserved_rate=("target_and_preserved", "mean"),
            malformed_or_repetitive_rate=("malformed_or_repetitive", "mean"),
            rows=("target_marker_hit", "count"),
        )
        .reset_index()
        .sort_values(["model", "target_class", "control"])
    )
    return summary, global_summary


def flush(rows: list[dict]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(RAW_PATH, index=False)
    if not raw_df.empty:
        summary, global_summary = summarize(raw_df)
        summary.to_csv(SUMMARY_PATH, index=False)
        global_summary.to_csv(GLOBAL_PATH, index=False)


def run_model(config: dict, train_df: pd.DataFrame, sources: pd.DataFrame, rows: list[dict], done: int, total: int) -> int:
    model_name = config["model"]
    layers = list(config["layers"])
    gain = float(config["gain"])
    rng = np.random.default_rng(SEED + len(model_name))
    write_status(
        status="running",
        phase="loading_model",
        current_model=model_name,
        progress_done=done,
        progress_total=total,
        rows=len(rows),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to("cpu")

    token_ids = marker_token_ids(tokenizer)
    token_payload = {}
    if TOKEN_IDS_PATH.exists():
        try:
            token_payload = json.loads(TOKEN_IDS_PATH.read_text(encoding="utf-8"))
        except Exception:
            token_payload = {}
    token_payload[model_name] = token_ids
    TOKEN_IDS_PATH.write_text(json.dumps(token_payload, indent=2), encoding="utf-8")

    write_status(
        phase="learning_centroids",
        current_model=model_name,
        train_rows=len(train_df),
        source_rows=len(sources),
        layers=layers,
        gain=gain,
        progress_done=done,
        progress_total=total,
        rows=len(rows),
    )
    centroids = learn_centroids(tokenizer, model, train_df, layers)
    vector_controls = ["target", "wrong_marker", "random_norm", "negative_target"]
    no_vector_controls = ["none", "strong_prompt", "string_append_source"]

    for target_class in TARGET_CLASSES:
        for source_rec in sources.to_dict("records"):
            for prompt_style, prompt_fn in PROMPT_STYLES.items():
                neutral_prompt = prompt_fn(source_rec["source"])
                for control in no_vector_controls:
                    if control == "string_append_source":
                        generated = append_source(source_rec["source"], target_class)
                        logits = {
                            "target_best_prob": float("nan"),
                            "target_best_rank": float("nan"),
                            "target_top1_any": float("nan"),
                        }
                        prompt = "<postprocess: append target marker to source>"
                    elif control == "strong_prompt":
                        prompt = STRONG_PROMPTS[target_class](source_rec["source"])
                        generated, logits = generate_with_trace(
                            tokenizer, model, prompt, token_ids[target_class]
                        )
                    else:
                        prompt = neutral_prompt
                        generated, logits = generate_with_trace(
                            tokenizer, model, prompt, token_ids[target_class]
                        )
                    rows.append(
                        {
                            "model": model_name,
                            "target_class": target_class,
                            "source_set": source_rec["source_set"],
                            "source_id": source_rec["source_id"],
                            "source": source_rec["source"],
                            "prompt_style": prompt_style,
                            "layer": -1,
                            "control": control,
                            "gain": 0.0,
                            "vector_class": "",
                            "prompt": prompt,
                            "generated": generated,
                            **score_row(source_rec["source"], generated, target_class),
                            **logits,
                        }
                    )
                    done += 1
                    if done % 100 == 0:
                        flush(rows)
                        write_status(
                            phase="generating",
                            current_model=model_name,
                            current_target=target_class,
                            current_prompt_style=prompt_style,
                            current_control=control,
                            progress_done=done,
                            progress_total=total,
                            rows=len(rows),
                        )

                for layer in layers:
                    target_vec = centroids[layer][target_class]
                    wrong_cls = wrong_class(target_class)
                    wrong_vec = centroids[layer][wrong_cls]
                    plan = [
                        ("target", target_vec, target_class),
                        ("wrong_marker", wrong_vec, wrong_cls),
                        ("random_norm", norm_match_random(target_vec, rng), "random_norm"),
                        ("negative_target", -target_vec, target_class),
                    ]
                    for control, vec, vector_class in plan:
                        generated, logits = generate_with_trace(
                            tokenizer,
                            model,
                            neutral_prompt,
                            token_ids[target_class],
                            layer=layer,
                            vector=vec,
                            gain=gain,
                        )
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
                                "gain": gain,
                                "vector_class": vector_class,
                                "prompt": neutral_prompt,
                                "generated": generated,
                                **score_row(source_rec["source"], generated, target_class),
                                **logits,
                            }
                        )
                        done += 1
                        if done % 100 == 0:
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
    flush(rows)
    return done


def total_rows(sources: pd.DataFrame) -> int:
    total = 0
    no_vector = 3
    vector = 4
    for config in APPLY_MODEL_CONFIGS:
        total += len(TARGET_CLASSES) * len(sources) * len(PROMPT_STYLES) * (
            no_vector + vector * len(config["layers"])
        )
    return total


def run() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    train_df = build_training_pairs()
    if TRAIN_ROWS > 0:
        # Keep the same number of source templates per target class as the confirmatory run.
        train_df = train_df.groupby("class", group_keys=False).head(TRAIN_ROWS)
    sources = build_sources()
    train_df.to_csv(TRAINING_PATH, index=False)
    sources.to_csv(SOURCE_PATH, index=False)
    total = total_rows(sources)
    write_status(
        status="running",
        started_at=now(),
        phase="initialized",
        seed=SEED,
        source_rows=len(sources),
        train_rows=len(train_df),
        target_classes=TARGET_CLASSES,
        prompt_styles=list(PROMPT_STYLES),
        controls=[
            "none",
            "strong_prompt",
            "string_append_source",
            "target",
            "wrong_marker",
            "random_norm",
            "negative_target",
        ],
        model_configs=APPLY_MODEL_CONFIGS,
        progress_done=0,
        progress_total=total,
        rows=0,
        note=(
            "Applicability audit for runtime final-marker form control. Includes strong_prompt "
            "and string_append_source baselines; practical claims must be interpreted against them."
        ),
    )
    rows: list[dict] = []
    done = 0
    failures = []
    for config in APPLY_MODEL_CONFIGS:
        try:
            done = run_model(config, train_df, sources, rows, done, total)
        except Exception as exc:  # noqa: BLE001
            failures.append({"model": config["model"], "error": repr(exc)})
            flush(rows)
            write_status(
                status="failed",
                phase="failed",
                progress_done=done,
                progress_total=total,
                rows=len(rows),
                failures=failures,
            )
            raise

    flush(rows)
    summary, global_summary = summarize(pd.DataFrame(rows))
    write_status(
        status="finished",
        phase="finished",
        progress_done=done,
        progress_total=total,
        rows=len(rows),
        summary_rows=len(summary),
        global_summary_rows=len(global_summary),
        finished_at=now(),
        raw_csv=str(RAW_PATH),
        summary_csv=str(SUMMARY_PATH),
        global_csv=str(GLOBAL_PATH),
        source_csv=str(SOURCE_PATH),
        training_csv=str(TRAINING_PATH),
        failures=failures,
    )
    print(global_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
