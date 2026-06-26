from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_lie_endpoint_residualization_audit import DEFAULT_MODELS, safe_model_name  # noqa: E402
from run_lie_multilingual_max_audit import fit_pca, mean_pool  # noqa: E402


OUT_DIR = Path("results/experiments/glt_affect_marker_pool_control_results")
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CKPT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, FIG_DIR, CKPT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MarkerSpec:
    code: str
    subjects: list[str]
    objects: list[str]
    contexts: list[str]
    templates: dict[int, str]
    markers: dict[int, str]


LANGS = [
    MarkerSpec(
        code="en",
        subjects=["I", "she", "the critic", "the user", "the teacher", "the artist", "the researcher", "the neighbor"],
        objects=["you", "this film", "this city", "this idea", "this system", "the proposal", "the song", "the design"],
        contexts=["today", "after the meeting", "in private", "during the review", "at the end"],
        markers={-2: "hates", -1: "dislikes", 0: "indifferent", 1: "likes", 2: "loves"},
        templates={
            -2: "{subject} hates {object} {context}.",
            -1: "{subject} dislikes {object} {context}.",
            0: "{subject} feels indifferent toward {object} {context}.",
            1: "{subject} likes {object} {context}.",
            2: "{subject} loves {object} {context}.",
        },
    ),
    MarkerSpec(
        code="ru",
        subjects=["ya", "ona", "kritik", "polzovatel", "uchitel", "hudozhnik", "issledovatel", "sosed"],
        objects=["tebya", "etot film", "etot gorod", "etu ideyu", "etu sistemu", "eto predlozhenie", "etu pesnyu", "etot dizayn"],
        contexts=["segodnya", "posle vstrechi", "naedine", "vo vremya obzora", "v kontse"],
        markers={-2: "nenavidit", -1: "nedolyublivaet", 0: "ravnodushen", 1: "simpatiziruet", 2: "lyubit"},
        templates={
            -2: "{subject} nenavidit {object} {context}.",
            -1: "{subject} nedolyublivaet {object} {context}.",
            0: "{subject} ravnodushen k {object} {context}.",
            1: "{subject} simpatiziruet {object} {context}.",
            2: "{subject} lyubit {object} {context}.",
        },
    ),
    MarkerSpec(
        code="zh",
        subjects=["wo", "ta", "pinglun zhe", "yonghu", "laoshi", "yishujia", "yanjiu zhe", "linju"],
        objects=["ni", "zhe bu dianying", "zhe zuo chengshi", "zhe ge xiangfa", "zhe ge xitong", "zhe ge jianyi", "zhe shou ge", "zhe ge sheji"],
        contexts=["jintian", "huiyi hou", "sixiadi", "shencha shi", "zuihou"],
        markers={-2: "taoyan", -1: "bu tai xihuan", 0: "wu gan", 1: "xihuan", 2: "reai"},
        templates={
            -2: "{subject} taoyan {object} {context}.",
            -1: "{subject} bu tai xihuan {object} {context}.",
            0: "{subject} dui {object} wu gan {context}.",
            1: "{subject} xihuan {object} {context}.",
            2: "{subject} reai {object} {context}.",
        },
    ),
]

LEVELS = [-2, -1, 0, 1, 2]
ADJACENT = [(-2, -1), (-1, 0), (0, 1), (1, 2)]


def clean(text: str) -> str:
    return " ".join(text.split())


def build_dataset(n_templates_per_language: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for spec in LANGS:
        combos = [(subject, obj, context) for subject in spec.subjects for obj in spec.objects for context in spec.contexts]
        rng.shuffle(combos)
        combos = combos[:n_templates_per_language]
        for template_id, (subject, obj, context) in enumerate(combos):
            for level in LEVELS:
                rows.append({
                    "language": spec.code,
                    "template_id": template_id,
                    "level": level,
                    "text": clean(spec.templates[level].format(subject=subject, object=obj, context=context)),
                    "marker": spec.markers[level],
                })
    return pd.DataFrame(rows)


def find_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int] | None:
    if not subsequence:
        return None
    for start in range(0, len(sequence) - len(subsequence) + 1):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start, start + len(subsequence)
    return None


@torch.no_grad()
def embed_mean_and_marker(model_name: str, df: pd.DataFrame, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    prefix = "query: " if "e5" in model_name.lower() else ""

    mean_vectors = []
    marker_vectors = []
    audit_rows = []
    records = df.to_dict("records")
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        texts = [prefix + r["text"] for r in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out = model(**enc)
        hidden = out.last_hidden_state.detach().cpu().numpy()
        mean_vec = mean_pool(out.last_hidden_state, enc["attention_mask"]).detach().cpu().numpy()
        ids = enc["input_ids"].detach().cpu().tolist()
        masks = enc["attention_mask"].detach().cpu().numpy()

        for i, row in enumerate(batch):
            marker_ids = tokenizer(prefix + row["marker"], add_special_tokens=False)["input_ids"]
            full_ids = ids[i][: int(masks[i].sum())]
            span = find_subsequence(full_ids, marker_ids)
            if span is None and prefix:
                marker_ids = tokenizer(row["marker"], add_special_tokens=False)["input_ids"]
                span = find_subsequence(full_ids, marker_ids)
            if span is None:
                # Fallback: use the first non-special token. The audit CSV makes misses visible.
                special = set(tokenizer.all_special_ids)
                usable = [j for j, token_id in enumerate(full_ids) if token_id not in special]
                span = (usable[0], usable[0] + 1) if usable else (0, 1)
                found = False
            else:
                found = True
            marker_vec = hidden[i, span[0]:span[1]].mean(axis=0)
            mean_vectors.append(mean_vec[i])
            marker_vectors.append(marker_vec)
            audit_rows.append({**row, "marker_found": found, "marker_span_start": span[0], "marker_span_end": span[1], "marker_token_count": span[1] - span[0]})

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(mean_vectors), np.vstack(marker_vectors), pd.DataFrame(audit_rows)


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / ((np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def summarize_geometry(model_name: str, audit: pd.DataFrame, vectors: np.ndarray, representation: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local = audit.copy().reset_index(drop=True)
    local["vec_idx"] = np.arange(len(local))
    step_rows = []
    opposition_rows = []
    curvature_rows = []
    for lang, sub_lang in local.groupby("language"):
        for template_id, sub in sub_lang.groupby("template_id"):
            by_level = {int(row.level): vectors[int(row.vec_idx)] for row in sub.itertuples()}
            if any(level not in by_level for level in LEVELS):
                continue
            for a, b in ADJACENT + [(-2, 0), (-1, 1), (0, 2), (-2, 2)]:
                delta = by_level[b] - by_level[a]
                step_rows.append({
                    "model": model_name,
                    "language": lang,
                    "template_id": int(template_id),
                    "representation": representation,
                    "from_level": a,
                    "to_level": b,
                    "step_size": b - a,
                    "delta_norm": float(np.linalg.norm(delta)),
                })
            pos = by_level[2] - by_level[0]
            neg = by_level[-2] - by_level[0]
            opposition_rows.append({
                "model": model_name,
                "language": lang,
                "template_id": int(template_id),
                "representation": representation,
                "row_cosine": float(row_cosines(pos.reshape(1, -1), neg.reshape(1, -1))[0]),
                "norm_ratio_pos_over_neg": float(np.linalg.norm(pos) / (np.linalg.norm(neg) + 1e-12)),
            })
            adjacent = [by_level[b] - by_level[a] for a, b in ADJACENT]
            norms = np.asarray([np.linalg.norm(v) for v in adjacent])
            curvature_rows.append({
                "model": model_name,
                "language": lang,
                "template_id": int(template_id),
                "representation": representation,
                "cv_adjacent_norm": float(norms.std(ddof=1) / (norms.mean() + 1e-12)),
                "cos_step_12": cosine(adjacent[0], adjacent[1]),
                "cos_step_23": cosine(adjacent[1], adjacent[2]),
                "cos_step_34": cosine(adjacent[2], adjacent[3]),
            })
    return pd.DataFrame(step_rows), pd.DataFrame(opposition_rows), pd.DataFrame(curvature_rows)


def run_model(model_name: str, df: pd.DataFrame, config: dict) -> None:
    safe = safe_model_name(model_name)
    marker = CKPT_DIR / f"{safe}.done.json"
    if marker.exists() and not config["force"]:
        print(f"SKIP {model_name}")
        return
    print(f"\n=== MODEL {model_name} ===", flush=True)
    mean_raw, marker_raw, audit = embed_mean_and_marker(model_name, df, config["device"], config["batch_size"])
    mean_vecs = fit_pca(mean_raw, config["pca_dim"])
    marker_vecs = fit_pca(marker_raw, config["pca_dim"])
    parts = []
    opp_parts = []
    curv_parts = []
    for representation, vectors in [("mean_pool", mean_vecs), ("marker_pool", marker_vecs)]:
        steps, opp, curv = summarize_geometry(model_name, audit, vectors, representation)
        parts.append(steps)
        opp_parts.append(opp)
        curv_parts.append(curv)
    pd.concat(parts, ignore_index=True).to_csv(CSV_DIR / f"pooling_steps_{safe}.csv", index=False)
    pd.concat(opp_parts, ignore_index=True).to_csv(CSV_DIR / f"pooling_opposition_{safe}.csv", index=False)
    pd.concat(curv_parts, ignore_index=True).to_csv(CSV_DIR / f"pooling_curvature_{safe}.csv", index=False)
    audit.to_csv(CSV_DIR / f"marker_span_audit_{safe}.csv", index=False)
    marker.write_text(json.dumps({"model": model_name, "finished_at": time.ctime()}, indent=2), encoding="utf-8")


def read_many(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in CSV_DIR.glob(pattern) if "all_models" not in path.name and "summary" not in path.name]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate() -> None:
    steps = read_many("pooling_steps_*.csv")
    opp = read_many("pooling_opposition_*.csv")
    curv = read_many("pooling_curvature_*.csv")
    spans = read_many("marker_span_audit_*.csv")
    if not steps.empty:
        steps.to_csv(CSV_DIR / "pooling_steps_all_models.csv", index=False)
        summary = steps.groupby(["representation", "from_level", "to_level", "step_size"]).agg(
            mean_delta_norm=("delta_norm", "mean"),
            std_delta_norm=("delta_norm", "std"),
            cells=("delta_norm", "count"),
        ).reset_index()
        summary.to_csv(CSV_DIR / "pooling_steps_summary.csv", index=False)
        adj = summary[summary["step_size"].eq(1)].copy()
        for representation, sub in adj.groupby("representation"):
            plt.figure(figsize=(8, 5))
            labels = [f"{int(r.from_level)}->{int(r.to_level)}" for r in sub.itertuples()]
            plt.bar(labels, sub["mean_delta_norm"])
            plt.ylabel("Mean adjacent delta norm")
            plt.title(f"GLT-AFFECT step norms ({representation})")
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"step_norms_{representation}.png", dpi=220, bbox_inches="tight")
            plt.close()
    if not opp.empty:
        opp.to_csv(CSV_DIR / "pooling_opposition_all_models.csv", index=False)
        opp.groupby("representation").agg(
            mean_row_cosine=("row_cosine", "mean"),
            std_row_cosine=("row_cosine", "std"),
            mean_norm_ratio_pos_over_neg=("norm_ratio_pos_over_neg", "mean"),
            cells=("row_cosine", "count"),
        ).reset_index().to_csv(CSV_DIR / "pooling_opposition_summary.csv", index=False)
    if not curv.empty:
        curv.to_csv(CSV_DIR / "pooling_curvature_all_models.csv", index=False)
        curv.groupby("representation").agg(
            mean_cv_adjacent_norm=("cv_adjacent_norm", "mean"),
            mean_cos_step_12=("cos_step_12", "mean"),
            mean_cos_step_23=("cos_step_23", "mean"),
            mean_cos_step_34=("cos_step_34", "mean"),
            cells=("cv_adjacent_norm", "count"),
        ).reset_index().to_csv(CSV_DIR / "pooling_curvature_summary.csv", index=False)
    if not spans.empty:
        spans.to_csv(CSV_DIR / "marker_span_audit_all_models.csv", index=False)
        spans.groupby(["language", "level"]).agg(
            marker_found_rate=("marker_found", "mean"),
            mean_marker_token_count=("marker_token_count", "mean"),
            cells=("marker_found", "count"),
        ).reset_index().to_csv(CSV_DIR / "marker_span_summary.csv", index=False)


def main() -> None:
    config = {
        "seed": int(os.getenv("GLT_AFFECT_POOL_SEED", "20260626")),
        "n_templates_per_language": int(os.getenv("GLT_AFFECT_POOL_TEMPLATES_PER_LANGUAGE", "160")),
        "pca_dim": int(os.getenv("GLT_AFFECT_POOL_PCA_DIM", "128")),
        "batch_size": int(os.getenv("GLT_AFFECT_POOL_BATCH_SIZE", "8")),
        "device": os.getenv("LIE_DEVICE", "cpu"),
        "force": os.getenv("GLT_AFFECT_POOL_FORCE", "0") == "1",
    }
    models_env = os.getenv("GLT_AFFECT_POOL_MODELS", "").strip()
    models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else DEFAULT_MODELS
    run_config = {**config, "models": models, "languages": [spec.code for spec in LANGS], "levels": LEVELS}
    print("GLT-AFFECT MARKER POOL CONTROL")
    print(json.dumps(run_config, indent=2), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    df = build_dataset(config["n_templates_per_language"], config["seed"])
    df.to_csv(CSV_DIR / "marker_pool_control_dataset.csv", index=False)
    failures = []
    for model_name in models:
        try:
            run_model(model_name, df, config)
            aggregate()
        except Exception as exc:
            failures.append({"model": model_name, "error": repr(exc)})
            print(f"FAILED {model_name}: {exc!r}", flush=True)
    aggregate()
    status = {
        "finished_at": time.ctime(),
        "failures": failures,
        "completed_markers": sorted(path.name for path in CKPT_DIR.glob("*.done.json")),
    }
    (OUT_DIR / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("DONE")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
