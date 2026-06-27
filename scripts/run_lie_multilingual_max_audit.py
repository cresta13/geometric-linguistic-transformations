import gc
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import percentileofscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer


OUT_DIR = Path(os.getenv("LIE_MULTI_OUT_DIR", "results/experiments/lie_multilingual_max_results"))
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
for directory in [CSV_DIR, FIG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


PAIR_OPS = [("N", "Q"), ("N", "M"), ("N", "T"), ("Q", "M"), ("Q", "T"), ("M", "T")]
TRIPLE_OPS = [("N", "Q", "M"), ("N", "Q", "T"), ("N", "M", "T"), ("Q", "M", "T")]
CYCLIC = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
ANTICYCLIC = [(0, 2, 1), (2, 1, 0), (1, 0, 2)]


MODELS = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "bert-base-multilingual-cased",
]


@dataclass
class LanguageSpec:
    code: str
    subjects: list[str]
    actions: list[tuple[str, str, str]]
    contexts: list[str]
    source: str
    neg: list[str]
    mod: list[str]
    future: list[str]
    statement: str
    question: str
    false_question: str
    modal_question: str
    future_truth_question: str
    future_question: str
    did_fail_question: str
    modal_did_question: str


LANGS = [
    LanguageSpec(
        code="en",
        subjects=["scientist", "engineer", "teacher", "doctor", "programmer", "researcher", "analyst", "manager"],
        actions=[
            ("accepted", "accept", "the explanation"),
            ("completed", "complete", "the repair"),
            ("confirmed", "confirm", "the answer"),
            ("approved", "approve", "the treatment"),
            ("fixed", "fix", "the bug"),
            ("supported", "support", "the theory"),
        ],
        contexts=["during the meeting", "after the review", "before the deadline", "in the morning"],
        source="The {subject} {past} {obj} {context}.",
        neg=["failed to {base}", "did not {base}", "refused to {base}"],
        mod=["allegedly", "reportedly", "apparently"],
        future=["will {base}", "is going to {base}", "plans to {base}"],
        statement="The {subject} {vp} {obj} {context}.",
        question="Did the {subject} {base} {obj} {context}?",
        false_question="Is it false that the {subject} {vp} {obj} {context}?",
        modal_question="Is it {marker} true that the {subject} {vp} {obj} {context}?",
        future_truth_question="Will it be true that the {subject} {vp} {obj} {context}?",
        future_question="Will the {subject} {vp} {obj} {context}?",
        did_fail_question="Did the {subject} fail to {base} {obj} {context}?",
        modal_did_question="Did the {subject} {marker} {base} {obj} {context}?",
    ),
    LanguageSpec(
        code="es",
        subjects=["cientifico", "ingeniero", "profesor", "medico", "programador", "investigador", "analista", "gerente"],
        actions=[
            ("acepto", "aceptar", "la explicacion"),
            ("completo", "completar", "la reparacion"),
            ("confirmo", "confirmar", "la respuesta"),
            ("aprobo", "aprobar", "el tratamiento"),
            ("arreglo", "arreglar", "el error"),
            ("apoyo", "apoyar", "la teoria"),
        ],
        contexts=["durante la reunion", "despues de la revision", "antes del plazo", "por la manana"],
        source="El {subject} {past} {obj} {context}.",
        neg=["no logro {base}", "no quiso {base}", "se nego a {base}"],
        mod=["supuestamente", "al parecer", "segun se informa"],
        future=["va a {base}", "planea {base}", "intentara {base}"],
        statement="El {subject} {vp} {obj} {context}.",
        question="El {subject} {base} {obj} {context}?",
        false_question="Es falso que el {subject} {vp} {obj} {context}?",
        modal_question="Es {marker} cierto que el {subject} {vp} {obj} {context}?",
        future_truth_question="Sera cierto que el {subject} {vp} {obj} {context}?",
        future_question="El {subject} va a {vp} {obj} {context}?",
        did_fail_question="El {subject} no logro {base} {obj} {context}?",
        modal_did_question="El {subject} {marker} {base} {obj} {context}?",
    ),
    LanguageSpec(
        code="fr",
        subjects=["scientifique", "ingenieur", "enseignant", "medecin", "programmeur", "chercheur", "analyste", "directeur"],
        actions=[
            ("a accepte", "accepter", "l'explication"),
            ("a termine", "terminer", "la reparation"),
            ("a confirme", "confirmer", "la reponse"),
            ("a approuve", "approuver", "le traitement"),
            ("a corrige", "corriger", "le bogue"),
            ("a soutenu", "soutenir", "la theorie"),
        ],
        contexts=["pendant la reunion", "apres l'examen", "avant la date limite", "le matin"],
        source="Le {subject} {past} {obj} {context}.",
        neg=["n'a pas reussi a {base}", "n'a pas voulu {base}", "a refuse de {base}"],
        mod=["pretendument", "apparemment", "selon le rapport"],
        future=["va {base}", "prevoit de {base}", "tentera de {base}"],
        statement="Le {subject} {vp} {obj} {context}.",
        question="Le {subject} peut-il {base} {obj} {context}?",
        false_question="Est-il faux que le {subject} {vp} {obj} {context}?",
        modal_question="Est-il {marker} vrai que le {subject} {vp} {obj} {context}?",
        future_truth_question="Sera-t-il vrai que le {subject} {vp} {obj} {context}?",
        future_question="Le {subject} va-t-il {vp} {obj} {context}?",
        did_fail_question="Le {subject} n'a-t-il pas reussi a {base} {obj} {context}?",
        modal_did_question="Le {subject} a-t-il {marker} {base} {obj} {context}?",
    ),
    LanguageSpec(
        code="de",
        subjects=["Wissenschaftler", "Ingenieur", "Lehrer", "Arzt", "Programmierer", "Forscher", "Analyst", "Manager"],
        actions=[
            ("akzeptierte", "akzeptieren", "die Erklaerung"),
            ("beendete", "beenden", "die Reparatur"),
            ("bestaetigte", "bestaetigen", "die Antwort"),
            ("genehmigte", "genehmigen", "die Behandlung"),
            ("reparierte", "reparieren", "den Fehler"),
            ("unterstuetzte", "unterstuetzen", "die Theorie"),
        ],
        contexts=["waehrend der Besprechung", "nach der Pruefung", "vor der Frist", "am Morgen"],
        source="Der {subject} {past} {obj} {context}.",
        neg=["schaffte es nicht zu {base}", "weigerte sich zu {base}", "konnte nicht {base}"],
        mod=["angeblich", "offenbar", "berichten zufolge"],
        future=["wird {base}", "plant zu {base}", "versucht zu {base}"],
        statement="Der {subject} {vp} {obj} {context}.",
        question="Kann der {subject} {base} {obj} {context}?",
        false_question="Ist es falsch, dass der {subject} {vp} {obj} {context}?",
        modal_question="Ist es {marker} wahr, dass der {subject} {vp} {obj} {context}?",
        future_truth_question="Wird es wahr sein, dass der {subject} {vp} {obj} {context}?",
        future_question="Wird der {subject} {vp} {obj} {context}?",
        did_fail_question="Schaffte es der {subject} nicht zu {base} {obj} {context}?",
        modal_did_question="Hat der {subject} {marker} {base} {obj} {context}?",
    ),
    LanguageSpec(
        code="ru",
        subjects=["uchenyj", "inzhener", "uchitel", "vrach", "programmist", "issledovatel", "analitik", "menedzher"],
        actions=[
            ("prinyal", "prinyat", "obyasnenie"),
            ("zavershil", "zavershit", "remont"),
            ("podtverdil", "podtverdit", "otvet"),
            ("odobril", "odobrit", "lechenie"),
            ("ispravil", "ispravit", "oshibku"),
            ("podderzhal", "podderzhat", "teoriyu"),
        ],
        contexts=["vo vremya vstrechi", "posle proverki", "do sroka", "utrom"],
        source="{subject} {past} {obj} {context}.",
        neg=["ne smog {base}", "otkazalsya {base}", "ne stal {base}"],
        mod=["predpolozhitelno", "po soobshcheniyam", "po-vidimomu"],
        future=["budet {base}", "planiruet {base}", "poprobuet {base}"],
        statement="{subject} {vp} {obj} {context}.",
        question="{subject} mozhet {base} {obj} {context}?",
        false_question="Nepravda li, chto {subject} {vp} {obj} {context}?",
        modal_question="Pravda li, chto {subject} {marker} {vp} {obj} {context}?",
        future_truth_question="Budet li pravdoy, chto {subject} {vp} {obj} {context}?",
        future_question="Budet li {subject} {vp} {obj} {context}?",
        did_fail_question="{subject} ne smog {base} {obj} {context}?",
        modal_did_question="{subject} {marker} {base} {obj} {context}?",
    ),
    LanguageSpec(
        code="zh",
        subjects=["科学家", "工程师", "老师", "医生", "程序员", "研究员", "分析师", "经理"],
        actions=[
            ("接受了", "接受", "解释"),
            ("完成了", "完成", "修理"),
            ("确认了", "确认", "答案"),
            ("批准了", "批准", "治疗"),
            ("修复了", "修复", "错误"),
            ("支持了", "支持", "理论"),
        ],
        contexts=["在会议期间", "在审查之后", "在截止日期之前", "早上"],
        source="{subject}{context}{past}{obj}。",
        neg=["未能{base}", "没有{base}", "拒绝{base}"],
        mod=["据称", "据报道", "显然"],
        future=["将{base}", "计划{base}", "打算{base}"],
        statement="{subject}{context}{vp}{obj}。",
        question="{subject}{context}{base}{obj}吗？",
        false_question="{subject}{context}{vp}{obj}是假的么？",
        modal_question="{subject}{context}{marker}{vp}{obj}是真的吗？",
        future_truth_question="{subject}{context}{vp}{obj}将是真的吗？",
        future_question="{subject}{context}会{vp}{obj}吗？",
        did_fail_question="{subject}{context}未能{base}{obj}吗？",
        modal_did_question="{subject}{context}{marker}{base}{obj}吗？",
    ),
    LanguageSpec(
        code="ar",
        subjects=["al-bahith", "al-muhandis", "al-muallim", "al-tabib", "al-mubarmij", "al-muhallil", "al-mudir", "al-muraji"],
        actions=[
            ("qabila", "yaqbal", "al-tafsir"),
            ("akmala", "yukmil", "al-islah"),
            ("akkada", "yuakkid", "al-ijaba"),
            ("wafaqa ala", "yuwafiq ala", "al-ilaj"),
            ("aslaha", "yuslih", "al-khata"),
            ("daama", "yadaam", "al-nazariya"),
        ],
        contexts=["athna al-ijtimaa", "baad al-murajaa", "qabla al-mawed", "sabahan"],
        source="{subject} {past} {obj} {context}.",
        neg=["lam {base}", "fashila an {base}", "rafada an {base}"],
        mod=["mazuman", "hasab al-taqrir", "ala ma yabdu"],
        future=["sawfa {base}", "yukhatit an {base}", "sayuhawil an {base}"],
        statement="{subject} {vp} {obj} {context}.",
        question="hal {subject} {base} {obj} {context}?",
        false_question="hal min al-khata anna {subject} {vp} {obj} {context}?",
        modal_question="hal min al-sahih {marker} anna {subject} {vp} {obj} {context}?",
        future_truth_question="hal sayakun sahihan anna {subject} {vp} {obj} {context}?",
        future_question="hal sawfa {subject} {vp} {obj} {context}?",
        did_fail_question="hal {subject} fashila an {base} {obj} {context}?",
        modal_did_question="hal {subject} {marker} {base} {obj} {context}?",
    ),
]


def clean(text: str) -> str:
    return " ".join(text.split())


def format_vp(spec: LanguageSpec, ops: tuple[str, ...], past: str, base: str, form: dict) -> str:
    negated = "N" in ops
    modal = "M" in ops
    future = "T" in ops

    if future:
        vp = form["future"].format(base=base)
    elif negated:
        vp = form["neg"].format(base=base)
    else:
        vp = past

    if negated and future:
        vp = form["future"].format(base=base)
        if ops.index("N") < ops.index("T"):
            vp = spec.neg[0].format(base=vp)
        else:
            vp = spec.neg[1].format(base=base)

    if modal:
        marker = form["mod"]
        if spec.code == "zh":
            vp = marker + vp
        elif ops.index("M") == 0:
            vp = f"{marker} {vp}"
        else:
            vp = f"{vp} {marker}"

    return clean(vp)


def compose_text(spec: LanguageSpec, ops: tuple[str, ...], subject: str, past: str, base: str, obj: str, context: str, form: dict) -> str:
    vp = format_vp(spec, ops, past, base, form)
    question = "Q" in ops
    negated = "N" in ops
    modal = "M" in ops
    future = "T" in ops

    if not question:
        return clean(spec.statement.format(subject=subject, vp=vp, obj=obj, context=context))

    if negated and ops.index("Q") < ops.index("N"):
        positive_vp = format_vp(spec, tuple(op for op in ops if op != "N"), past, base, form)
        return clean(spec.false_question.format(subject=subject, vp=positive_vp, obj=obj, context=context))

    if modal and ops.index("Q") < ops.index("M"):
        positive_vp = format_vp(spec, tuple(op for op in ops if op != "M"), past, base, form)
        return clean(spec.modal_question.format(subject=subject, marker=form["mod"], vp=positive_vp, obj=obj, context=context))

    if future and ops.index("Q") < ops.index("T"):
        positive_vp = format_vp(spec, tuple(op for op in ops if op != "T"), past, base, form)
        return clean(spec.future_truth_question.format(subject=subject, vp=positive_vp, obj=obj, context=context))

    if future:
        q_vp = format_vp(spec, tuple(op for op in ops if op != "T"), past, base, form)
        return clean(spec.future_question.format(subject=subject, vp=q_vp, obj=obj, context=context))

    if negated:
        return clean(spec.did_fail_question.format(subject=subject, base=base, obj=obj, context=context))

    if modal:
        return clean(spec.modal_did_question.format(subject=subject, marker=form["mod"], base=base, obj=obj, context=context))

    return clean(spec.question.format(subject=subject, base=base, obj=obj, context=context))


def build_dataset(n_templates_per_language: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for spec in LANGS:
        combos = [(subject, action, context) for subject in spec.subjects for action in spec.actions for context in spec.contexts]
        rng.shuffle(combos)
        combos = combos[:n_templates_per_language]
        for template_id, (subject, action, context) in enumerate(combos):
            past, base, obj = action
            form = {
                "neg": rng.choice(spec.neg),
                "mod": rng.choice(spec.mod),
                "future": rng.choice(spec.future),
            }
            source = clean(spec.source.format(subject=subject, past=past, obj=obj, context=context))
            for a, b in PAIR_OPS:
                ab = compose_text(spec, (a, b), subject, past, base, obj, context, form)
                ba = compose_text(spec, (b, a), subject, past, base, obj, context, form)
                rows.append({
                    "language": spec.code,
                    "template_id": template_id,
                    "kind": "pair",
                    "label": f"{a}{b}_vs_{b}{a}",
                    "source": source,
                    "op_a": a,
                    "op_b": b,
                    "op_c": "",
                    "ab_text": ab,
                    "ba_text": ba,
                })
            for triple in TRIPLE_OPS:
                row = {
                    "language": spec.code,
                    "template_id": template_id,
                    "kind": "triple",
                    "label": "".join(triple),
                    "source": source,
                    "op_a": triple[0],
                    "op_b": triple[1],
                    "op_c": triple[2],
                }
                for name, perm in [
                    ("abc", (0, 1, 2)),
                    ("bca", (1, 2, 0)),
                    ("cab", (2, 0, 1)),
                    ("acb", (0, 2, 1)),
                    ("cba", (2, 1, 0)),
                    ("bac", (1, 0, 2)),
                ]:
                    seq = tuple(triple[i] for i in perm)
                    row[f"{name}_text"] = compose_text(spec, seq, subject, past, base, obj, context, form)
                rows.append(row)
    return pd.DataFrame(rows)


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


@torch.no_grad()
def embed_texts(model_name: str, texts: list[str], device: str, batch_size: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    prefix = "query: " if "e5" in model_name.lower() else ""
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start:start + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        embeddings.append(pooled.detach().cpu().numpy())
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(embeddings)


def relative_norm(vec, scale_vecs):
    scale = np.mean([np.linalg.norm(v) for v in scale_vecs]) + 1e-12
    return float(np.linalg.norm(vec) / scale)


def empirical_p(null_values, observed, smaller_is_better=True):
    null_values = np.asarray(null_values)
    if smaller_is_better:
        return float(((null_values <= observed).sum() + 1) / (len(null_values) + 1))
    return float(((null_values >= observed).sum() + 1) / (len(null_values) + 1))


def fit_pca(raw: np.ndarray, pca_dim: int) -> np.ndarray:
    dim = min(pca_dim, raw.shape[0] - 1, raw.shape[1])
    return PCA(n_components=dim, random_state=42).fit_transform(raw)


def classifier_controls(df, vecs_by_text, label_col, endpoint_cols, train_langs, test_langs):
    rows = []
    sub = df[df["kind"].eq("pair")].copy()
    labels = sorted(sub[label_col].unique())
    y_map = {label: i for i, label in enumerate(labels)}
    y = sub[label_col].map(y_map).to_numpy()
    train_mask = sub["language"].isin(train_langs).to_numpy()
    test_mask = sub["language"].isin(test_langs).to_numpy()
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return pd.DataFrame()

    feature_sets = {
        "source_only": np.vstack([vecs_by_text[t] for t in sub["source"]]),
        "ab_endpoint_only": np.vstack([vecs_by_text[t] for t in sub["ab_text"]]),
        "ba_endpoint_only": np.vstack([vecs_by_text[t] for t in sub["ba_text"]]),
        "ab_delta_only": np.vstack([vecs_by_text[row.ab_text] - vecs_by_text[row.source] for row in sub.itertuples()]),
        "ba_delta_only": np.vstack([vecs_by_text[row.ba_text] - vecs_by_text[row.source] for row in sub.itertuples()]),
        "commutator_delta": np.vstack([
            (vecs_by_text[row.ab_text] - vecs_by_text[row.source]) - (vecs_by_text[row.ba_text] - vecs_by_text[row.source])
            for row in sub.itertuples()
        ]),
    }
    for feature, x in feature_sets.items():
        clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
        clf.fit(normalize(x[train_mask]), y[train_mask])
        pred = clf.predict(normalize(x[test_mask]))
        rows.append({
            "feature": feature,
            "train_languages": ",".join(train_langs),
            "test_languages": ",".join(test_langs),
            "accuracy": accuracy_score(y[test_mask], pred),
            "macro_f1": f1_score(y[test_mask], pred, average="macro"),
            "chance": 1.0 / len(labels),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        })
    return pd.DataFrame(rows)


def analyze_model(model_name: str, df: pd.DataFrame, args: dict):
    safe = model_name.replace("/", "__").replace("-", "_")
    done_marker = CHECKPOINT_DIR / f"{safe}.done.json"
    if done_marker.exists() and not args["force"]:
        print(f"[SKIP] {model_name} already done", flush=True)
        return

    print(f"\n=== MODEL {model_name} ===", flush=True)
    text_cols = ["source", "ab_text", "ba_text", "abc_text", "bca_text", "cab_text", "acb_text", "cba_text", "bac_text"]
    texts = sorted({str(v) for col in text_cols if col in df.columns for v in df[col].dropna().tolist()})
    print(f"texts={len(texts)}", flush=True)

    raw = embed_texts(model_name, texts, args["device"], args["batch_size"])
    vecs = fit_pca(raw, args["pca_dim"])
    vecs_by_text = {text: vecs[i] for i, text in enumerate(texts)}

    pair_rows = []
    for row in df[df["kind"].eq("pair")].itertuples(index=False):
        x = vecs_by_text[row.source]
        dab = vecs_by_text[row.ab_text] - x
        dba = vecs_by_text[row.ba_text] - x
        comm = dab - dba
        rel = relative_norm(comm, [dab, dba])
        cos = float(np.sum(normalize(dab.reshape(1, -1)) * normalize(dba.reshape(1, -1))))
        pair_rows.append({
            "model": model_name,
            "language": row.language,
            "template_id": row.template_id,
            "pair": row.label,
            "relative_commutator_norm": rel,
            "commutator_norm": float(np.linalg.norm(comm)),
            "cosine_ab_ba": cos,
            "noncommutativity": 1.0 - cos,
        })
    pair_raw = pd.DataFrame(pair_rows)

    triple_rows = []
    rng = np.random.default_rng(args["seed"])
    for row in df[df["kind"].eq("triple")].itertuples(index=False):
        x = vecs_by_text[row.source]
        endpoint_names = ["abc_text", "bca_text", "cab_text", "acb_text", "cba_text", "bac_text"]
        deltas = [vecs_by_text[getattr(row, name)] - x for name in endpoint_names]
        signed = deltas[0] + deltas[1] + deltas[2] - deltas[3] - deltas[4] - deltas[5]
        scale = np.mean([np.linalg.norm(delta) for delta in deltas]) + 1e-12
        rel = float(np.linalg.norm(signed) / scale)
        nulls = []
        for _ in range(args["n_null"]):
            pos = set(rng.choice(6, size=3, replace=False).tolist())
            s = np.zeros_like(signed)
            for i, delta in enumerate(deltas):
                s += delta if i in pos else -delta
            nulls.append(float(np.linalg.norm(s) / scale))
        nulls = np.asarray(nulls)
        triple_rows.append({
            "model": model_name,
            "language": row.language,
            "template_id": row.template_id,
            "triple": row.label,
            "relative_signed_permutation_norm": rel,
            "null_mean": float(nulls.mean()),
            "null_std": float(nulls.std(ddof=1)),
            "ratio_to_null_mean": float(rel / (nulls.mean() + 1e-12)),
            "null_percentile_smaller_is_better": float(percentileofscore(nulls, rel, kind="mean")),
            "p_smaller_or_equal_null": empirical_p(nulls, rel, smaller_is_better=True),
        })
    triple_raw = pd.DataFrame(triple_rows)

    pair_summary = pair_raw.groupby(["model", "language", "pair"]).agg(
        mean_relative_commutator_norm=("relative_commutator_norm", "mean"),
        std_relative_commutator_norm=("relative_commutator_norm", "std"),
        mean_noncommutativity=("noncommutativity", "mean"),
        n=("pair", "count"),
    ).reset_index()

    triple_summary = triple_raw.groupby(["model", "language", "triple"]).agg(
        mean_relative_signed_permutation_norm=("relative_signed_permutation_norm", "mean"),
        mean_ratio_to_null_mean=("ratio_to_null_mean", "mean"),
        mean_null_percentile=("null_percentile_smaller_is_better", "mean"),
        max_p_smaller_or_equal_null=("p_smaller_or_equal_null", "max"),
        n=("triple", "count"),
    ).reset_index()

    cent_rows = []
    for kind, raw_df, label_col, value_col in [
        ("pair_commutator", pair_raw, "pair", "relative_commutator_norm"),
        ("triple_signed", triple_raw, "triple", "relative_signed_permutation_norm"),
    ]:
        labels = sorted(raw_df[label_col].unique())
        langs = sorted(raw_df["language"].unique())
        for label in labels:
            lang_vecs = {}
            for lang in langs:
                sub_df = df[(df["language"].eq(lang)) & (df["kind"].eq("pair" if kind == "pair_commutator" else "triple"))]
                if kind == "pair_commutator":
                    sub_df = sub_df[sub_df["label"].eq(label)]
                    vec_list = []
                    for r in sub_df.itertuples(index=False):
                        x = vecs_by_text[r.source]
                        vec_list.append((vecs_by_text[r.ab_text] - x) - (vecs_by_text[r.ba_text] - x))
                else:
                    sub_df = sub_df[sub_df["label"].eq(label)]
                    vec_list = []
                    for r in sub_df.itertuples(index=False):
                        x = vecs_by_text[r.source]
                        d = [vecs_by_text[getattr(r, name)] - x for name in ["abc_text", "bca_text", "cab_text", "acb_text", "cba_text", "bac_text"]]
                        vec_list.append(d[0] + d[1] + d[2] - d[3] - d[4] - d[5])
                lang_vecs[lang] = np.mean(vec_list, axis=0)
            for i, lang_a in enumerate(langs):
                for lang_b in langs[i + 1:]:
                    va = normalize(lang_vecs[lang_a].reshape(1, -1))
                    vb = normalize(lang_vecs[lang_b].reshape(1, -1))
                    cent_rows.append({
                        "model": model_name,
                        "kind": kind,
                        "label": label,
                        "language_a": lang_a,
                        "language_b": lang_b,
                        "centroid_cosine": float(np.sum(va * vb)),
                    })

    centroid_consistency = pd.DataFrame(cent_rows)
    control_parts = []
    all_langs = [spec.code for spec in LANGS]
    for heldout in all_langs:
        train_langs = [lang for lang in all_langs if lang != heldout]
        controls = classifier_controls(df, vecs_by_text, "label", [], train_langs=train_langs, test_langs=[heldout])
        if not controls.empty:
            controls.insert(0, "model", model_name)
            control_parts.append(controls)
    controls = pd.concat(control_parts, ignore_index=True)

    pair_raw.to_csv(CSV_DIR / f"pair_raw_{safe}.csv", index=False)
    triple_raw.to_csv(CSV_DIR / f"triple_raw_{safe}.csv", index=False)
    pair_summary.to_csv(CSV_DIR / f"pair_summary_{safe}.csv", index=False)
    triple_summary.to_csv(CSV_DIR / f"triple_summary_{safe}.csv", index=False)
    centroid_consistency.to_csv(CSV_DIR / f"cross_language_centroid_consistency_{safe}.csv", index=False)
    controls.to_csv(CSV_DIR / f"endpoint_controls_{safe}.csv", index=False)
    done_marker.write_text(json.dumps({"model": model_name, "finished_at": time.ctime()}, indent=2), encoding="utf-8")


def aggregate_outputs(models: list[str]):
    def read_many(pattern):
        paths = [
            path for path in sorted(CSV_DIR.glob(pattern))
            if "all_models" not in path.name and path.name != "endpoint_controls_summary.csv"
        ]
        if not paths:
            return pd.DataFrame()
        return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    outputs = {
        "pair_raw_all_models.csv": read_many("pair_raw_*.csv"),
        "triple_raw_all_models.csv": read_many("triple_raw_*.csv"),
        "pair_summary_all_models.csv": read_many("pair_summary_*.csv"),
        "triple_summary_all_models.csv": read_many("triple_summary_*.csv"),
        "cross_language_centroid_consistency_all_models.csv": read_many("cross_language_centroid_consistency_*.csv"),
        "endpoint_controls_all_models.csv": read_many("endpoint_controls_*.csv"),
    }
    for filename, df in outputs.items():
        if not df.empty:
            df.to_csv(CSV_DIR / filename, index=False)

    if not outputs["triple_summary_all_models.csv"].empty:
        triples = outputs["triple_summary_all_models.csv"]
        triple_global = (
            triples.groupby("triple")
            .agg(
                mean_ratio_to_null_mean=("mean_ratio_to_null_mean", "mean"),
                std_ratio_to_null_mean=("mean_ratio_to_null_mean", "std"),
                min_ratio_to_null_mean=("mean_ratio_to_null_mean", "min"),
                max_ratio_to_null_mean=("mean_ratio_to_null_mean", "max"),
                below_null_cells=("mean_ratio_to_null_mean", lambda s: int((s < 1.0).sum())),
                n_cells=("mean_ratio_to_null_mean", "count"),
            )
            .reset_index()
        )
        triple_global["frac_below_null_cells"] = triple_global["below_null_cells"] / triple_global["n_cells"]
        triple_global.to_csv(CSV_DIR / "triple_global_summary.csv", index=False)

        triple_by_model = (
            triples.groupby(["model", "triple"])
            .agg(
                mean_ratio_to_null_mean=("mean_ratio_to_null_mean", "mean"),
                std_ratio_to_null_mean=("mean_ratio_to_null_mean", "std"),
                min_ratio_to_null_mean=("mean_ratio_to_null_mean", "min"),
                max_ratio_to_null_mean=("mean_ratio_to_null_mean", "max"),
                below_null_languages=("mean_ratio_to_null_mean", lambda s: int((s < 1.0).sum())),
                n_languages=("mean_ratio_to_null_mean", "count"),
            )
            .reset_index()
        )
        triple_by_model.to_csv(CSV_DIR / "triple_by_model_summary.csv", index=False)

        triple_by_language = (
            triples.groupby(["language", "triple"])
            .agg(
                mean_ratio_to_null_mean=("mean_ratio_to_null_mean", "mean"),
                std_ratio_to_null_mean=("mean_ratio_to_null_mean", "std"),
                min_ratio_to_null_mean=("mean_ratio_to_null_mean", "min"),
                max_ratio_to_null_mean=("mean_ratio_to_null_mean", "max"),
                below_null_models=("mean_ratio_to_null_mean", lambda s: int((s < 1.0).sum())),
                n_models=("mean_ratio_to_null_mean", "count"),
            )
            .reset_index()
        )
        triple_by_language.to_csv(CSV_DIR / "triple_by_language_summary.csv", index=False)

        pivot = triples.pivot_table(index=["model", "language"], columns="triple", values="mean_ratio_to_null_mean")
        plt.figure(figsize=(10, max(4, len(pivot) * 0.25)))
        plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=max(1.5, np.nanmax(pivot.values)))
        plt.colorbar(label="Mean ratio to signed-null mean")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
        plt.yticks(np.arange(len(pivot.index)), [f"{m}\n{l}" for m, l in pivot.index], fontsize=6)
        plt.title("Multilingual third-order signed permutation ratios")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "01_multilingual_signed_permutation_ratios.png", dpi=220, bbox_inches="tight")
        plt.close()

        ordered = triple_global.sort_values("mean_ratio_to_null_mean")
        plt.figure(figsize=(7, 4))
        plt.bar(ordered["triple"], ordered["mean_ratio_to_null_mean"], yerr=ordered["std_ratio_to_null_mean"], capsize=4)
        plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
        plt.ylabel("Mean ratio to signed-null mean")
        plt.title("Global multilingual signed-permutation ratio by triple")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "02_triple_global_ratio_summary.png", dpi=220, bbox_inches="tight")
        plt.close()

    if not outputs["pair_summary_all_models.csv"].empty:
        pairs = outputs["pair_summary_all_models.csv"]
        pair_global = (
            pairs.groupby("pair")
            .agg(
                mean_relative_commutator_norm=("mean_relative_commutator_norm", "mean"),
                std_relative_commutator_norm=("mean_relative_commutator_norm", "std"),
                min_relative_commutator_norm=("mean_relative_commutator_norm", "min"),
                max_relative_commutator_norm=("mean_relative_commutator_norm", "max"),
                n_cells=("mean_relative_commutator_norm", "count"),
            )
            .reset_index()
        )
        pair_global.to_csv(CSV_DIR / "pair_global_summary.csv", index=False)

    if not outputs["endpoint_controls_all_models.csv"].empty:
        controls = outputs["endpoint_controls_all_models.csv"]
        summary = controls.groupby(["model", "feature"]).agg(mean_macro_f1=("macro_f1", "mean"), min_macro_f1=("macro_f1", "min"), n=("macro_f1", "count")).reset_index()
        summary.to_csv(CSV_DIR / "endpoint_controls_summary.csv", index=False)

        control_pivot = summary.pivot_table(index="model", columns="feature", values="mean_macro_f1")
        plt.figure(figsize=(10, 4.8))
        plt.imshow(control_pivot.values, aspect="auto", vmin=0, vmax=1)
        plt.colorbar(label="Mean macro F1 across held-out languages")
        plt.xticks(np.arange(len(control_pivot.columns)), control_pivot.columns, rotation=35, ha="right")
        plt.yticks(np.arange(len(control_pivot.index)), control_pivot.index, fontsize=7)
        plt.title("Endpoint/control feature transfer across held-out languages")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "03_endpoint_control_macro_f1.png", dpi=220, bbox_inches="tight")
        plt.close()

    if not outputs["cross_language_centroid_consistency_all_models.csv"].empty:
        consistency = outputs["cross_language_centroid_consistency_all_models.csv"]
        consistency_summary = (
            consistency.groupby("kind")
            .agg(
                mean_centroid_cosine=("centroid_cosine", "mean"),
                std_centroid_cosine=("centroid_cosine", "std"),
                min_centroid_cosine=("centroid_cosine", "min"),
                max_centroid_cosine=("centroid_cosine", "max"),
                n=("centroid_cosine", "count"),
            )
            .reset_index()
        )
        consistency_summary.to_csv(CSV_DIR / "cross_language_centroid_summary.csv", index=False)

        labels = list(consistency_summary["kind"])
        data = [consistency.loc[consistency["kind"] == label, "centroid_cosine"].values for label in labels]
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, tick_labels=labels, showfliers=False)
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.ylabel("Cross-language centroid cosine")
        plt.title("Cross-language consistency of composition centroids")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "04_cross_language_centroid_consistency.png", dpi=220, bbox_inches="tight")
        plt.close()


def main():
    args = {
        "seed": int(os.getenv("LIE_MULTI_SEED", "20260622")),
        "n_templates_per_language": int(os.getenv("LIE_MULTI_TEMPLATES_PER_LANGUAGE", "48")),
        "n_null": int(os.getenv("LIE_MULTI_N_NULL", "2000")),
        "pca_dim": int(os.getenv("LIE_MULTI_PCA_DIM", "96")),
        "batch_size": int(os.getenv("LIE_MULTI_BATCH_SIZE", "8")),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "force": os.getenv("LIE_MULTI_FORCE", "0") == "1",
    }
    model_env = os.getenv("LIE_MULTI_MODELS", "")
    models = [m.strip() for m in model_env.split(",") if m.strip()] or MODELS

    print("LIE MULTILINGUAL MAX AUDIT")
    print(json.dumps({**args, "models": models, "languages": [spec.code for spec in LANGS]}, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps({**args, "models": models, "languages": [spec.code for spec in LANGS]}, indent=2), encoding="utf-8")

    df = build_dataset(args["n_templates_per_language"], args["seed"])
    df.to_csv(CSV_DIR / "multilingual_composition_dataset.csv", index=False)
    print("dataset rows", len(df), "texts", len({v for col in df.columns if col.endswith("_text") or col == "source" for v in df[col].dropna()}), flush=True)

    failures = []
    for model in models:
        try:
            analyze_model(model, df, args)
            aggregate_outputs(models)
        except Exception as exc:
            print(f"[FAIL] {model}: {exc}", flush=True)
            traceback.print_exc()
            failures.append({"model": model, "error": repr(exc), "traceback": traceback.format_exc()})
            (OUT_DIR / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
            continue

    aggregate_outputs(models)
    status = {
        "finished_at": time.ctime(),
        "failures": failures,
        "completed_markers": [p.name for p in CHECKPOINT_DIR.glob("*.done.json")],
    }
    (OUT_DIR / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("DONE")
    print(json.dumps(status, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
