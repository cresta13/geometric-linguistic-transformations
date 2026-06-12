import os
import gc
import time
import random
import itertools
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

load_dotenv()

OUT_DIR = "lie_llm_syntax_results"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [m.strip() for m in os.getenv(
    "LIE_MODELS",
    "bert-base-uncased,roberta-base,distilroberta-base,gpt2,distilgpt2"
).split(",") if m.strip()]

N_BASE = int(os.getenv("LIE_N_BASE", "1000"))
BATCH_SIZE = int(os.getenv("LIE_BATCH_SIZE", "32"))
SEED = int(os.getenv("LIE_SEED", "42"))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SUBJECTS = [
    "the doctor", "the teacher", "the scientist", "the engineer", "the artist",
    "the student", "the child", "the manager", "the programmer", "the writer",
    "the musician", "the researcher", "the nurse", "the lawyer", "the farmer",
    "the pilot", "the designer", "the analyst", "the journalist", "the chef",
]

VERBS = [
    ("examined", "examine"), ("explained", "explain"), ("discovered", "discover"),
    ("designed", "design"), ("created", "create"), ("solved", "solve"),
    ("found", "find"), ("approved", "approve"), ("tested", "test"),
    ("wrote", "write"), ("studied", "study"), ("helped", "help"),
    ("defended", "defend"), ("improved", "improve"), ("analyzed", "analyze"),
    ("reported", "report"), ("prepared", "prepare"), ("built", "build"),
    ("measured", "measure"), ("observed", "observe"),
]

OBJECTS = [
    "the patient", "the rule", "a new principle", "a complex machine",
    "a beautiful painting", "the problem", "a black cat", "the decision",
    "the system", "a short story", "the question", "the family",
    "the client", "the garden", "the interface", "the data",
    "the event", "the document", "the experiment", "the signal",
]

CONTEXTS = [
    "during the meeting",
    "after the discussion",
    "before the deadline",
    "in the laboratory",
    "at the school",
    "during the experiment",
    "after the interview",
    "before the presentation",
]


def cap(text):
    return text[0].upper() + text[1:]


def mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def make_base_rows(n_base):
    triples = list(itertools.product(SUBJECTS, VERBS, OBJECTS, CONTEXTS))
    random.shuffle(triples)
    triples = triples[:n_base]

    rows = []

    for i, (subject, (past, base), obj, ctx) in enumerate(triples):
        rows.append({
            "base_id": i,
            "subject": subject,
            "S": cap(subject),
            "past": past,
            "base": base,
            "object": obj,
            "O": cap(obj),
            "context": ctx,
        })

    return pd.DataFrame(rows)


def source_sentence(r, syntax_family):
    S, s = r["S"], r["subject"]
    past, obj, ctx = r["past"], r["object"], r["context"]

    if syntax_family == "simple_svo":
        return f"{S} {past} {obj}."

    if syntax_family == "with_context":
        return f"{S} {past} {obj} {ctx}."

    if syntax_family == "relative_clause":
        return f"The person who {past} {obj} was {s}."

    if syntax_family == "temporal_clause":
        return f"After {s} {past} {obj}, the situation changed."

    if syntax_family == "reported_statement":
        return f"The report said that {s} {past} {obj}."

    if syntax_family == "conditional":
        return f"If {s} {past} {obj}, the result mattered."

    raise ValueError(syntax_family)


def transform_sentence(r, syntax_family, transform_class):
    S, s = r["S"], r["subject"]
    past, base, obj, O, ctx = r["past"], r["base"], r["object"], r["O"], r["context"]

    if syntax_family in ["simple_svo", "with_context"]:
        suffix = f" {ctx}" if syntax_family == "with_context" else ""

        if transform_class == "negation":
            return f"{S} did not {base} {obj}{suffix}."
        if transform_class == "question":
            return f"Did {s} {base} {obj}{suffix}?"
        if transform_class == "future":
            return f"{S} will {base} {obj}{suffix}."
        if transform_class == "uncertainty":
            return f"{S} may have {past} {obj}{suffix}."
        if transform_class == "passive":
            return f"{O} was {past} by {s}{suffix}."
        if transform_class == "formalization":
            return f"It is true that {s} {past} {obj}{suffix}."

    if syntax_family == "relative_clause":
        if transform_class == "negation":
            return f"The person who did not {base} {obj} was {s}."
        if transform_class == "question":
            return f"Was the person who {past} {obj} {s}?"
        if transform_class == "future":
            return f"The person who will {base} {obj} is {s}."
        if transform_class == "uncertainty":
            return f"The person who may have {past} {obj} was {s}."
        if transform_class == "passive":
            return f"The person by whom {obj} was {past} was {s}."
        if transform_class == "formalization":
            return f"It is true that the person who {past} {obj} was {s}."

    if syntax_family == "temporal_clause":
        if transform_class == "negation":
            return f"After {s} did not {base} {obj}, the situation changed."
        if transform_class == "question":
            return f"Did the situation change after {s} {past} {obj}?"
        if transform_class == "future":
            return f"After {s} will {base} {obj}, the situation will change."
        if transform_class == "uncertainty":
            return f"After {s} may have {past} {obj}, the situation changed."
        if transform_class == "passive":
            return f"After {obj} was {past} by {s}, the situation changed."
        if transform_class == "formalization":
            return f"It is true that after {s} {past} {obj}, the situation changed."

    if syntax_family == "reported_statement":
        if transform_class == "negation":
            return f"The report said that {s} did not {base} {obj}."
        if transform_class == "question":
            return f"Did the report say that {s} {past} {obj}?"
        if transform_class == "future":
            return f"The report said that {s} will {base} {obj}."
        if transform_class == "uncertainty":
            return f"The report said that {s} may have {past} {obj}."
        if transform_class == "passive":
            return f"The report said that {obj} was {past} by {s}."
        if transform_class == "formalization":
            return f"It is true that the report said that {s} {past} {obj}."

    if syntax_family == "conditional":
        if transform_class == "negation":
            return f"If {s} did not {base} {obj}, the result mattered."
        if transform_class == "question":
            return f"Did the result matter if {s} {past} {obj}?"
        if transform_class == "future":
            return f"If {s} will {base} {obj}, the result will matter."
        if transform_class == "uncertainty":
            return f"If {s} may have {past} {obj}, the result mattered."
        if transform_class == "passive":
            return f"If {obj} was {past} by {s}, the result mattered."
        if transform_class == "formalization":
            return f"It is true that if {s} {past} {obj}, the result mattered."

    raise ValueError((syntax_family, transform_class))


def build_pairs(base_df):
    transform_classes = ["negation", "question", "future", "uncertainty", "passive", "formalization"]
    syntax_families = [
        "simple_svo",
        "with_context",
        "relative_clause",
        "temporal_clause",
        "reported_statement",
        "conditional",
    ]

    rows = []

    for _, r in base_df.iterrows():
        for syntax_family in syntax_families:
            source = source_sentence(r, syntax_family)

            for cls in transform_classes:
                target = transform_sentence(r, syntax_family, cls)

                rows.append({
                    "base_id": int(r["base_id"]),
                    "syntax_family": syntax_family,
                    "class": cls,
                    "source": source,
                    "target": target,
                })

    return pd.DataFrame(rows)


def index_prompts(pair_df):
    prompts = []
    seen = {}

    for text in list(pair_df["source"]) + list(pair_df["target"]):
        if text not in seen:
            seen[text] = len(prompts)
            prompts.append(text)

    pair_df = pair_df.copy()
    pair_df["source_idx"] = pair_df["source"].map(seen)
    pair_df["target_idx"] = pair_df["target"].map(seen)

    return prompts, pair_df


def get_vectors(model_name, prompts):
    safe = model_name.replace("/", "__")
    cache_path = os.path.join(OUT_DIR, f"vectors_{safe}.npy")

    if os.path.exists(cache_path):
        print("Loading cached:", cache_path)
        return np.load(cache_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    vectors = []
    started = time.time()

    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=160,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)

        pooled = mean_pool(outputs.hidden_states[-1], inputs["attention_mask"])
        vectors.append(pooled.cpu().numpy())

        done = start + len(batch)
        print(f"  {model_name}: {done}/{len(prompts)}, {time.time() - started:.1f}s")

    vectors = np.vstack(vectors)
    np.save(cache_path, vectors)

    del model
    gc.collect()

    return vectors


def build_deltas(model_name, vectors, pair_df):
    rows = []
    deltas = []

    for _, r in pair_df.iterrows():
        src = vectors[int(r["source_idx"])]
        tgt = vectors[int(r["target_idx"])]
        delta = tgt - src
        deltas.append(delta)

        rows.append({
            "model": model_name,
            "base_id": int(r["base_id"]),
            "syntax_family": r["syntax_family"],
            "class": r["class"],
            "source": r["source"],
            "target": r["target"],
            "delta_norm": float(np.linalg.norm(delta)),
        })

    return pd.DataFrame(rows), np.vstack(deltas)


def run_syntax_holdout(model_name, delta_df, D):
    labels = delta_df["class"].astype(str).to_numpy()
    syntax = delta_df["syntax_family"].astype(str).to_numpy()

    train_mask = np.isin(syntax, ["simple_svo", "with_context"])
    test_mask = np.isin(syntax, ["relative_clause", "temporal_clause", "reported_statement", "conditional"])

    X_train = D[train_mask]
    y_train = labels[train_mask]
    X_test = D[test_mask]
    y_test = labels[test_mask]

    classes = sorted(set(labels))
    random_baseline = 1 / len(classes)

    classifiers = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=-1),
        ),
        "linear_svc": make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", max_iter=20000),
        ),
    }

    rows = []

    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)

        rows.append({
            "model": model_name,
            "classifier": clf_name,
            "accuracy": float(acc),
            "random_baseline": float(random_baseline),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        })

        safe = model_name.replace("/", "__")

        pd.DataFrame(
            classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
        ).transpose().to_csv(os.path.join(OUT_DIR, f"report_{safe}_{clf_name}.csv"))

        pd.DataFrame(
            confusion_matrix(y_test, pred, labels=classes),
            index=classes,
            columns=classes,
        ).to_csv(os.path.join(OUT_DIR, f"confusion_{safe}_{clf_name}.csv"))

    return pd.DataFrame(rows)


def main():
    base_df = make_base_rows(N_BASE)
    pair_df = build_pairs(base_df)
    prompts, pair_df = index_prompts(pair_df)

    base_df.to_csv(os.path.join(OUT_DIR, "base_sentences.csv"), index=False)
    pair_df.to_csv(os.path.join(OUT_DIR, "transformation_pairs.csv"), index=False)
    pd.DataFrame({"prompt_id": range(len(prompts)), "prompt": prompts}).to_csv(
        os.path.join(OUT_DIR, "prompts.csv"),
        index=False,
    )

    print("Models:", MODELS)
    print("Base:", len(base_df))
    print("Pairs:", len(pair_df))
    print("Prompts:", len(prompts))

    all_results = []

    for model_name in MODELS:
        print(f"\n=== MODEL: {model_name} ===")

        vectors = get_vectors(model_name, prompts)
        delta_df, D = build_deltas(model_name, vectors, pair_df)

        safe = model_name.replace("/", "__")
        delta_df.to_csv(os.path.join(OUT_DIR, f"delta_vectors_metrics_{safe}.csv"), index=False)
        np.save(os.path.join(OUT_DIR, f"deltas_{safe}.npy"), D)

        result = run_syntax_holdout(model_name, delta_df, D)
        print(result)

        all_results.append(result)

    summary = pd.concat(all_results, ignore_index=True)
    summary.to_csv(os.path.join(OUT_DIR, "syntax_holdout_summary.csv"), index=False)

    print("\n=== SUMMARY ===")
    print(summary)
    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()