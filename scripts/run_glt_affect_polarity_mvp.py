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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_lie_endpoint_residualization_audit import DEFAULT_MODELS, safe_model_name  # noqa: E402
from run_lie_multilingual_max_audit import embed_texts, fit_pca  # noqa: E402


OUT_DIR = Path("results/experiments/glt_affect_polarity_mvp_results")
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CKPT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, FIG_DIR, CKPT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AffectLanguage:
    code: str
    subjects: list[str]
    objects: list[str]
    contexts: list[str]
    templates: dict[int, str]


LANGS = [
    AffectLanguage(
        code="en",
        subjects=["I", "she", "the critic", "the user", "the teacher", "the artist", "the researcher", "the neighbor"],
        objects=["you", "this film", "this city", "this idea", "this system", "the proposal", "the song", "the design"],
        contexts=["today", "after the meeting", "in private", "during the review", "at the end"],
        templates={
            -2: "{subject} hates {object} {context}.",
            -1: "{subject} dislikes {object} {context}.",
            0: "{subject} feels indifferent toward {object} {context}.",
            1: "{subject} likes {object} {context}.",
            2: "{subject} loves {object} {context}.",
        },
    ),
    AffectLanguage(
        code="es",
        subjects=["yo", "ella", "el critico", "el usuario", "la profesora", "el artista", "la investigadora", "el vecino"],
        objects=["a ti", "esta pelicula", "esta ciudad", "esta idea", "este sistema", "la propuesta", "la cancion", "el diseno"],
        contexts=["hoy", "despues de la reunion", "en privado", "durante la revision", "al final"],
        templates={
            -2: "{subject} odia {object} {context}.",
            -1: "{subject} no aprecia {object} {context}.",
            0: "{subject} es indiferente hacia {object} {context}.",
            1: "{subject} aprecia {object} {context}.",
            2: "{subject} ama {object} {context}.",
        },
    ),
    AffectLanguage(
        code="fr",
        subjects=["je", "elle", "le critique", "l utilisateur", "la professeure", "l artiste", "la chercheuse", "le voisin"],
        objects=["toi", "ce film", "cette ville", "cette idee", "ce systeme", "la proposition", "la chanson", "le design"],
        contexts=["aujourd hui", "apres la reunion", "en prive", "pendant l evaluation", "a la fin"],
        templates={
            -2: "{subject} deteste {object} {context}.",
            -1: "{subject} n aime pas beaucoup {object} {context}.",
            0: "{subject} est indifferent envers {object} {context}.",
            1: "{subject} aime bien {object} {context}.",
            2: "{subject} adore {object} {context}.",
        },
    ),
    AffectLanguage(
        code="de",
        subjects=["ich", "sie", "der Kritiker", "der Nutzer", "die Lehrerin", "der Kunstler", "die Forscherin", "der Nachbar"],
        objects=["dich", "diesen Film", "diese Stadt", "diese Idee", "dieses System", "den Vorschlag", "das Lied", "das Design"],
        contexts=["heute", "nach dem Treffen", "privat", "wahrend der Prufung", "am Ende"],
        templates={
            -2: "{subject} hasst {object} {context}.",
            -1: "{subject} mag {object} kaum {context}.",
            0: "{subject} ist {object} gegenuber gleichgultig {context}.",
            1: "{subject} mag {object} {context}.",
            2: "{subject} liebt {object} {context}.",
        },
    ),
    AffectLanguage(
        code="ru",
        subjects=["ya", "ona", "kritik", "polzovatel", "uchitel", "hudozhnik", "issledovatel", "sosed"],
        objects=["tebya", "etot film", "etot gorod", "etu ideyu", "etu sistemu", "eto predlozhenie", "etu pesnyu", "etot dizayn"],
        contexts=["segodnya", "posle vstrechi", "naedine", "vo vremya obzora", "v kontse"],
        templates={
            -2: "{subject} nenavidit {object} {context}.",
            -1: "{subject} nedolyublivaet {object} {context}.",
            0: "{subject} ravnodushen k {object} {context}.",
            1: "{subject} simpatiziruet {object} {context}.",
            2: "{subject} lyubit {object} {context}.",
        },
    ),
    AffectLanguage(
        code="zh",
        subjects=["wo", "ta", "pinglun zhe", "yonghu", "laoshi", "yishujia", "yanjiu zhe", "linju"],
        objects=["ni", "zhe bu dianying", "zhe zuo chengshi", "zhe ge xiangfa", "zhe ge xitong", "zhe ge jianyi", "zhe shou ge", "zhe ge sheji"],
        contexts=["jintian", "huiyi hou", "sixiadi", "shencha shi", "zuihou"],
        templates={
            -2: "{subject} taoyan {object} {context}.",
            -1: "{subject} bu tai xihuan {object} {context}.",
            0: "{subject} dui {object} wu gan {context}.",
            1: "{subject} xihuan {object} {context}.",
            2: "{subject} reai {object} {context}.",
        },
    ),
    AffectLanguage(
        code="ar",
        subjects=["ana", "hiya", "al-naqid", "al-mustakhdim", "al-muallim", "al-fannan", "al-bahith", "al-jar"],
        objects=["anta", "hatha al-film", "hathihi al-madina", "hathihi al-fikra", "hatha al-nizam", "al-iqtirah", "al-ughniya", "al-tasmim"],
        contexts=["alyawm", "baad al-ijtima", "sirran", "athna al-murajaa", "fi al-nihaya"],
        templates={
            -2: "{subject} yakrah {object} {context}.",
            -1: "{subject} la yastahsin {object} {context}.",
            0: "{subject} ghayr mubal bi {object} {context}.",
            1: "{subject} yastahsin {object} {context}.",
            2: "{subject} yuhib {object} {context}.",
        },
    ),
]

LEVELS = [-2, -1, 0, 1, 2]
ADJACENT = [(-2, -1), (-1, 0), (0, 1), (1, 2)]
LONG_STEPS = [(-2, 0), (-1, 1), (0, 2), (-2, 2)]


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
            row = {"language": spec.code, "template_id": template_id, "subject": subject, "object": obj, "context": context}
            for level in LEVELS:
                row[f"level_{level}_text"] = clean(spec.templates[level].format(subject=subject, object=obj, context=context))
            rows.append(row)
    return pd.DataFrame(rows)


def all_texts(df: pd.DataFrame) -> list[str]:
    texts = set()
    for col in df.columns:
        if col.endswith("_text"):
            texts.update(df[col].dropna().astype(str))
    return sorted(texts)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / ((np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12)


def run_model(model_name: str, df: pd.DataFrame, config: dict) -> None:
    safe = safe_model_name(model_name)
    marker = CKPT_DIR / f"{safe}.done.json"
    if marker.exists() and not config["force"]:
        print(f"SKIP {model_name}")
        return

    print(f"\n=== MODEL {model_name} ===", flush=True)
    texts = all_texts(df)
    print(f"rows={len(df)} texts={len(texts)}", flush=True)
    raw = embed_texts(model_name, texts, config["device"], config["batch_size"])
    vecs = fit_pca(raw, config["pca_dim"])
    vecs_by_text = dict(zip(texts, vecs))

    step_rows = []
    linearity_rows = []
    opposition_rows = []
    curvature_rows = []

    for lang, sub in df.groupby("language"):
        level_vecs = {level: np.vstack([vecs_by_text[t] for t in sub[f"level_{level}_text"]]) for level in LEVELS}
        adjacent_deltas = {(a, b): level_vecs[b] - level_vecs[a] for a, b in ADJACENT}

        for a, b in ADJACENT + LONG_STEPS:
            delta = level_vecs[b] - level_vecs[a]
            centroid = delta.mean(axis=0)
            step_rows.append({
                "model": model_name,
                "language": lang,
                "from_level": a,
                "to_level": b,
                "step_size": b - a,
                "mean_delta_norm": float(np.linalg.norm(delta, axis=1).mean()),
                "centroid_norm": float(np.linalg.norm(centroid)),
                "n": int(len(sub)),
            })

        for a, m, b in [(-2, -1, 0), (-1, 0, 1), (0, 1, 2), (-2, 0, 2)]:
            composed = (level_vecs[m] - level_vecs[a]) + (level_vecs[b] - level_vecs[m])
            direct = level_vecs[b] - level_vecs[a]
            residual = composed - direct
            # This should be numerically zero; kept as a bug-catching sanity check only.
            linearity_rows.append({
                "model": model_name,
                "language": lang,
                "path": f"{a}->{m}->{b}",
                "direct": f"{a}->{b}",
                "mean_residual_norm": float(np.linalg.norm(residual, axis=1).mean()),
                "mean_direct_norm": float(np.linalg.norm(direct, axis=1).mean()),
                "relative_residual_norm": float(np.linalg.norm(residual, axis=1).mean() / (np.linalg.norm(direct, axis=1).mean() + 1e-12)),
                "mean_cosine_composed_direct": float(row_cosines(composed, direct).mean()),
                "note": "sanity_identity_for_same_endpoint_path",
            })

        pos = level_vecs[2] - level_vecs[0]
        neg = level_vecs[-2] - level_vecs[0]
        opposition_rows.append({
            "model": model_name,
            "language": lang,
            "comparison": "neutral_to_love_vs_neutral_to_hate",
            "mean_cosine": float(row_cosines(pos, neg).mean()),
            "centroid_cosine": cosine(pos.mean(axis=0), neg.mean(axis=0)),
            "mean_norm_pos": float(np.linalg.norm(pos, axis=1).mean()),
            "mean_norm_neg": float(np.linalg.norm(neg, axis=1).mean()),
            "norm_ratio_pos_over_neg": float(np.linalg.norm(pos, axis=1).mean() / (np.linalg.norm(neg, axis=1).mean() + 1e-12)),
        })

        adjacent_norms = np.column_stack([np.linalg.norm(adjacent_deltas[pair], axis=1) for pair in ADJACENT])
        adjacent_centroids = [adjacent_deltas[pair].mean(axis=0) for pair in ADJACENT]
        curvature_rows.append({
            "model": model_name,
            "language": lang,
            "mean_adjacent_norm": float(adjacent_norms.mean()),
            "std_adjacent_norm_across_steps": float(adjacent_norms.mean(axis=0).std(ddof=1)),
            "cv_adjacent_norm_across_steps": float(adjacent_norms.mean(axis=0).std(ddof=1) / (adjacent_norms.mean() + 1e-12)),
            "cos_hate_dislike_to_dislike_neutral": cosine(adjacent_centroids[0], adjacent_centroids[1]),
            "cos_dislike_neutral_to_neutral_like": cosine(adjacent_centroids[1], adjacent_centroids[2]),
            "cos_neutral_like_to_like_love": cosine(adjacent_centroids[2], adjacent_centroids[3]),
            "cos_hate_dislike_to_neutral_like": cosine(adjacent_centroids[0], adjacent_centroids[2]),
            "cos_dislike_neutral_to_like_love": cosine(adjacent_centroids[1], adjacent_centroids[3]),
        })

    pd.DataFrame(step_rows).to_csv(CSV_DIR / f"affect_steps_{safe}.csv", index=False)
    pd.DataFrame(linearity_rows).to_csv(CSV_DIR / f"affect_linearity_sanity_{safe}.csv", index=False)
    pd.DataFrame(opposition_rows).to_csv(CSV_DIR / f"affect_opposition_{safe}.csv", index=False)
    pd.DataFrame(curvature_rows).to_csv(CSV_DIR / f"affect_curvature_{safe}.csv", index=False)
    marker.write_text(json.dumps({"model": model_name, "finished_at": time.ctime()}, indent=2), encoding="utf-8")
    del raw, vecs, vecs_by_text
    gc.collect()


def read_many(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in CSV_DIR.glob(pattern) if "all_models" not in path.name and "summary" not in path.name]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate() -> None:
    steps = read_many("affect_steps_*.csv")
    sanity = read_many("affect_linearity_sanity_*.csv")
    opposition = read_many("affect_opposition_*.csv")
    curvature = read_many("affect_curvature_*.csv")
    if not steps.empty:
        steps.to_csv(CSV_DIR / "affect_steps_all_models.csv", index=False)
        summary = steps.groupby(["from_level", "to_level", "step_size"]).agg(
            mean_delta_norm=("mean_delta_norm", "mean"),
            std_delta_norm=("mean_delta_norm", "std"),
            mean_centroid_norm=("centroid_norm", "mean"),
            cells=("mean_delta_norm", "count"),
        ).reset_index()
        summary.to_csv(CSV_DIR / "affect_steps_summary.csv", index=False)
        adj = summary[summary["step_size"].eq(1)].copy()
        if not adj.empty:
            plt.figure(figsize=(8, 5))
            labels = [f"{int(r.from_level)}->{int(r.to_level)}" for r in adj.itertuples()]
            plt.bar(labels, adj["mean_delta_norm"])
            plt.ylabel("Mean adjacent delta norm")
            plt.title("GLT-AFFECT adjacent emotional polarity step norms")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "01_adjacent_step_norms.png", dpi=220, bbox_inches="tight")
            plt.close()
    if not sanity.empty:
        sanity.to_csv(CSV_DIR / "affect_linearity_sanity_all_models.csv", index=False)
        sanity.groupby(["path", "direct"]).agg(
            mean_relative_residual_norm=("relative_residual_norm", "mean"),
            max_relative_residual_norm=("relative_residual_norm", "max"),
            mean_cosine_composed_direct=("mean_cosine_composed_direct", "mean"),
            cells=("relative_residual_norm", "count"),
        ).reset_index().to_csv(CSV_DIR / "affect_linearity_sanity_summary.csv", index=False)
    if not opposition.empty:
        opposition.to_csv(CSV_DIR / "affect_opposition_all_models.csv", index=False)
        opposition.groupby("comparison").agg(
            mean_row_cosine=("mean_cosine", "mean"),
            mean_centroid_cosine=("centroid_cosine", "mean"),
            mean_norm_ratio_pos_over_neg=("norm_ratio_pos_over_neg", "mean"),
            cells=("mean_cosine", "count"),
        ).reset_index().to_csv(CSV_DIR / "affect_opposition_summary.csv", index=False)
    if not curvature.empty:
        curvature.to_csv(CSV_DIR / "affect_curvature_all_models.csv", index=False)
        curvature.agg({
            "mean_adjacent_norm": "mean",
            "cv_adjacent_norm_across_steps": "mean",
            "cos_hate_dislike_to_dislike_neutral": "mean",
            "cos_dislike_neutral_to_neutral_like": "mean",
            "cos_neutral_like_to_like_love": "mean",
            "cos_hate_dislike_to_neutral_like": "mean",
            "cos_dislike_neutral_to_like_love": "mean",
        }).to_frame("value").reset_index(names="metric").to_csv(CSV_DIR / "affect_curvature_summary.csv", index=False)


def main() -> None:
    config = {
        "seed": int(os.getenv("GLT_AFFECT_SEED", "20260625")),
        "n_templates_per_language": int(os.getenv("GLT_AFFECT_TEMPLATES_PER_LANGUAGE", "160")),
        "pca_dim": int(os.getenv("GLT_AFFECT_PCA_DIM", "128")),
        "batch_size": int(os.getenv("GLT_AFFECT_BATCH_SIZE", "8")),
        "device": os.getenv("LIE_DEVICE", "cpu"),
        "force": os.getenv("GLT_AFFECT_FORCE", "0") == "1",
    }
    models_env = os.getenv("GLT_AFFECT_MODELS", "").strip()
    models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else DEFAULT_MODELS
    config_with_models = {**config, "models": models, "languages": [spec.code for spec in LANGS], "levels": LEVELS}
    print("GLT-AFFECT POLARITY MVP")
    print(json.dumps(config_with_models, indent=2), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps(config_with_models, indent=2), encoding="utf-8")

    df = build_dataset(config["n_templates_per_language"], config["seed"])
    df.to_csv(CSV_DIR / "glt_affect_polarity_dataset.csv", index=False)
    print(f"rows={len(df)} texts={len(all_texts(df))}", flush=True)

    failures = []
    for model in models:
        try:
            run_model(model, df, config)
            aggregate()
        except Exception as exc:
            failures.append({"model": model, "error": repr(exc)})
            print(f"FAILED {model}: {exc!r}", flush=True)
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
