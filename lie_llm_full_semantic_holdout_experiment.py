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

OUT_DIR = "lie_llm_full_semantic_holdout_results"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [m.strip() for m in os.getenv(
    "LIE_MODELS",
    "bert-base-uncased,roberta-base,distilroberta-base,gpt2,distilgpt2"
).split(",") if m.strip()]

N_BASE = int(os.getenv("LIE_N_BASE", "300"))
BATCH_SIZE = int(os.getenv("LIE_BATCH_SIZE", "32"))
SEED = int(os.getenv("LIE_SEED", "42"))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_SUBJECTS = [
    "the doctor", "the teacher", "the scientist", "the engineer", "the artist",
    "the student", "the child", "the manager", "the programmer", "the writer",
]

TEST_SUBJECTS = [
    "the musician", "the researcher", "the nurse", "the lawyer", "the farmer",
    "the pilot", "the designer", "the analyst", "the journalist", "the chef",
]

TRAIN_OBJECTS = [
    "the patient", "the rule", "a new principle", "a complex machine",
    "a beautiful painting", "the problem", "a black cat", "the decision",
    "the system", "a short story",
]

TEST_OBJECTS = [
    "the question", "the family", "the client", "the garden",
    "the interface", "the data", "the event", "the document",
    "the experiment", "the signal",
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

TRAIN_SYNTAX = ["simple_svo", "with_context"]
TEST_SYNTAX = ["relative_clause", "temporal_clause", "reported_statement", "conditional"]

CLASSES = ["negation", "question", "future", "uncertainty", "passive", "formalization"]


def cap(text):
    return text[0].upper() + text[1:]


def mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def source_sentence(r, syntax_family):
    S = cap(r["subject"])
    s = r["subject"]
    past = r["past"]
    obj = r["object"]
    ctx = r["context"]

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


def train_transform(r, syntax_family, cls):
    S = cap(r["subject"])
    s = r["subject"]
    past = r["past"]
    base = r["base"]
    obj = r["object"]
    O = cap(obj)
    ctx = r["context"]

    suffix = f" {ctx}" if syntax_family == "with_context" else ""

    if cls == "negation":
        return random.choice([
            f"{S} did not {base} {obj}{suffix}.",
            f"{S} never {past} {obj}{suffix}.",
        ])

    if cls == "question":
        return random.choice([
            f"Did {s} {base} {obj}{suffix}?",
            f"Could {s} have {past} {obj}{suffix}?",
        ])

    if cls == "future":
        return random.choice([
            f"{S} will {base} {obj}{suffix}.",
            f"{S} is going to {base} {obj}{suffix}.",
        ])

    if cls == "uncertainty":
        return random.choice([
            f"{S} may have {past} {obj}{suffix}.",
            f"{S} might have {past} {obj}{suffix}.",
        ])

    if cls == "passive":
        return random.choice([
            f"{O} was {past} by {s}{suffix}.",
            f"{O} had been {past} by {s}{suffix}.",
        ])

    if cls == "formalization":
        return random.choice([
            f"It is true that {s} {past} {obj}{suffix}.",
            f"The fact is that {s} {past} {obj}{suffix}.",
        ])

    raise ValueError(cls)


def test_transform(r, syntax_family, cls):
    s = r["subject"]
    S = cap(s)
    past = r["past"]
    base = r["base"]
    obj = r["object"]
    O = cap(obj)

    if syntax_family == "relative_clause":
        if cls == "negation":
            return random.choice([
                f"The person who failed to {base} {obj} was {s}.",
                f"The person who was unable to {base} {obj} was {s}.",
                f"The person who could not {base} {obj} was {s}.",
            ])
        if cls == "question":
            return random.choice([
                f"Is there evidence that the person who {past} {obj} was {s}?",
                f"Can we confirm that the person who {past} {obj} was {s}?",
                f"Was it true that the person who {past} {obj} was {s}?",
            ])
        if cls == "future":
            return random.choice([
                f"The person who is expected to {base} {obj} is {s}.",
                f"The person who is likely to {base} {obj} is {s}.",
                f"The person who plans to {base} {obj} is {s}.",
            ])
        if cls == "uncertainty":
            return random.choice([
                f"It is possible that the person who {past} {obj} was {s}.",
                f"There is a chance that the person who {past} {obj} was {s}.",
                f"The person who probably {past} {obj} was {s}.",
            ])
        if cls == "passive":
            return random.choice([
                f"The person by whom {obj} got {past} was {s}.",
                f"The person by whom {obj} was successfully {past} was {s}.",
                f"The person by whom {obj} was eventually {past} was {s}.",
            ])
        if cls == "formalization":
            return random.choice([
                f"One can state that the person who {past} {obj} was {s}.",
                f"It can be formally said that the person who {past} {obj} was {s}.",
                f"The statement that the person who {past} {obj} was {s} is true.",
            ])

    if syntax_family == "temporal_clause":
        if cls == "negation":
            return random.choice([
                f"After {s} failed to {base} {obj}, the situation changed.",
                f"After {s} was unable to {base} {obj}, the situation changed.",
                f"After {s} could not {base} {obj}, the situation changed.",
            ])
        if cls == "question":
            return random.choice([
                f"Is there evidence that the situation changed after {s} {past} {obj}?",
                f"Can we confirm that the situation changed after {s} {past} {obj}?",
                f"Was it true that the situation changed after {s} {past} {obj}?",
            ])
        if cls == "future":
            return random.choice([
                f"After {s} is expected to {base} {obj}, the situation will change.",
                f"After {s} is likely to {base} {obj}, the situation will change.",
                f"After {s} plans to {base} {obj}, the situation will change.",
            ])
        if cls == "uncertainty":
            return random.choice([
                f"It is possible that after {s} {past} {obj}, the situation changed.",
                f"There is a chance that after {s} {past} {obj}, the situation changed.",
                f"After {s} probably {past} {obj}, the situation changed.",
            ])
        if cls == "passive":
            return random.choice([
                f"After {obj} got {past} by {s}, the situation changed.",
                f"After {obj} was successfully {past} by {s}, the situation changed.",
                f"After {obj} was eventually {past} by {s}, the situation changed.",
            ])
        if cls == "formalization":
            return random.choice([
                f"One can state that after {s} {past} {obj}, the situation changed.",
                f"It can be formally said that after {s} {past} {obj}, the situation changed.",
                f"The statement that after {s} {past} {obj}, the situation changed is true.",
            ])

    if syntax_family == "reported_statement":
        if cls == "negation":
            return random.choice([
                f"The report said that {s} failed to {base} {obj}.",
                f"The report said that {s} was unable to {base} {obj}.",
                f"The report said that {s} could not {base} {obj}.",
            ])
        if cls == "question":
            return random.choice([
                f"Is there evidence that the report said {s} {past} {obj}?",
                f"Can we confirm that the report said {s} {past} {obj}?",
                f"Was it true that the report said {s} {past} {obj}?",
            ])
        if cls == "future":
            return random.choice([
                f"The report said that {s} is expected to {base} {obj}.",
                f"The report said that {s} is likely to {base} {obj}.",
                f"The report said that {s} plans to {base} {obj}.",
            ])
        if cls == "uncertainty":
            return random.choice([
                f"The report said it is possible that {s} {past} {obj}.",
                f"The report said there is a chance that {s} {past} {obj}.",
                f"The report said that {s} probably {past} {obj}.",
            ])
        if cls == "passive":
            return random.choice([
                f"The report said that {obj} got {past} by {s}.",
                f"The report said that {obj} was successfully {past} by {s}.",
                f"The report said that {obj} was eventually {past} by {s}.",
            ])
        if cls == "formalization":
            return random.choice([
                f"One can state that the report said {s} {past} {obj}.",
                f"It can be formally said that the report said {s} {past} {obj}.",
                f"The statement that the report said {s} {past} {obj} is true.",
            ])

    if syntax_family == "conditional":
        if cls == "negation":
            return random.choice([
                f"If {s} failed to {base} {obj}, the result mattered.",
                f"If {s} was unable to {base} {obj}, the result mattered.",
                f"If {s} could not {base} {obj}, the result mattered.",
            ])
        if cls == "question":
            return random.choice([
                f"Is there evidence that the result mattered if {s} {past} {obj}?",
                f"Can we confirm that the result mattered if {s} {past} {obj}?",
                f"Was it true that the result mattered if {s} {past} {obj}?",
            ])
        if cls == "future":
            return random.choice([
                f"If {s} is expected to {base} {obj}, the result will matter.",
                f"If {s} is likely to {base} {obj}, the result will matter.",
                f"If {s} plans to {base} {obj}, the result will matter.",
            ])
        if cls == "uncertainty":
            return random.choice([
                f"If it is possible that {s} {past} {obj}, the result mattered.",
                f"If there is a chance that {s} {past} {obj}, the result mattered.",
                f"If {s} probably {past} {obj}, the result mattered.",
            ])
        if cls == "passive":
            return random.choice([
                f"If {obj} got {past} by {s}, the result mattered.",
                f"If {obj} was successfully {past} by {s}, the result mattered.",
                f"If {obj} was eventually {past} by {s}, the result mattered.",
            ])
        if cls == "formalization":
            return random.choice([
                f"One can state that if {s} {past} {obj}, the result mattered.",
                f"It can be formally said that if {s} {past} {obj}, the result mattered.",
                f"The statement that if {s} {past} {obj}, the result mattered is true.",
            ])

    raise ValueError((syntax_family, cls))


def make_rows(subjects, objects, syntax_families, split_name, n_base):
    triples = list(itertools.product(subjects, VERBS, objects, CONTEXTS))
    random.shuffle(triples)
    triples = triples[:n_base]

    rows = []
    for i, (subject, (past, base), obj, ctx) in enumerate(triples):
        r = {
            "base_id": f"{split_name}_{i}",
            "subject": subject,
            "past": past,
            "base": base,
            "object": obj,
            "context": ctx,
        }

        for syntax_family in syntax_families:
            source = source_sentence(r, syntax_family)

            for cls in CLASSES:
                if split_name == "train":
                    target = train_transform(r, syntax_family, cls)
                else:
                    target = test_transform(r, syntax_family, cls)

                rows.append({
                    "split": split_name,
                    "base_id": r["base_id"],
                    "syntax_family": syntax_family,
                    "class": cls,
                    "source": source,
                    "target": target,
                })

    return rows


def build_dataset():
    train_rows = make_rows(TRAIN_SUBJECTS, TRAIN_OBJECTS, TRAIN_SYNTAX, "train", N_BASE)
    test_rows = make_rows(TEST_SUBJECTS, TEST_OBJECTS, TEST_SYNTAX, "test", N_BASE)
    return pd.DataFrame(train_rows + test_rows)


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
            max_length=180,
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
            "split": r["split"],
            "base_id": r["base_id"],
            "syntax_family": r["syntax_family"],
            "class": r["class"],
            "source": r["source"],
            "target": r["target"],
            "delta_norm": float(np.linalg.norm(delta)),
        })

    return pd.DataFrame(rows), np.vstack(deltas)


def run_classifier(model_name, delta_df, D):
    labels = delta_df["class"].astype(str).to_numpy()
    split = delta_df["split"].astype(str).to_numpy()

    train_mask = split == "train"
    test_mask = split == "test"

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
    safe = model_name.replace("/", "__")

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

        pd.DataFrame(
            classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
        ).transpose().to_csv(
            os.path.join(OUT_DIR, f"report_{safe}_{clf_name}.csv")
        )

        pd.DataFrame(
            confusion_matrix(y_test, pred, labels=classes),
            index=classes,
            columns=classes,
        ).to_csv(
            os.path.join(OUT_DIR, f"confusion_{safe}_{clf_name}.csv")
        )

    return pd.DataFrame(rows)


def main():
    pair_df = build_dataset()
    prompts, pair_df = index_prompts(pair_df)

    pair_df.to_csv(os.path.join(OUT_DIR, "transformation_pairs.csv"), index=False)
    pd.DataFrame({"prompt_id": range(len(prompts)), "prompt": prompts}).to_csv(
        os.path.join(OUT_DIR, "prompts.csv"), index=False
    )

    print("Models:", MODELS)
    print("Rows:", len(pair_df))
    print("Prompts:", len(prompts))
    print("Train rows:", len(pair_df[pair_df["split"] == "train"]))
    print("Test rows:", len(pair_df[pair_df["split"] == "test"]))

    all_results = []

    for model_name in MODELS:
        print(f"\n=== MODEL: {model_name} ===")

        vectors = get_vectors(model_name, prompts)
        delta_df, D = build_deltas(model_name, vectors, pair_df)

        safe = model_name.replace("/", "__")
        delta_df.to_csv(os.path.join(OUT_DIR, f"delta_vectors_metrics_{safe}.csv"), index=False)
        np.save(os.path.join(OUT_DIR, f"deltas_{safe}.npy"), D)

        result = run_classifier(model_name, delta_df, D)
        print(result)

        all_results.append(result)

    summary = pd.concat(all_results, ignore_index=True)
    summary.to_csv(os.path.join(OUT_DIR, "full_semantic_holdout_summary.csv"), index=False)

    print("\n=== SUMMARY ===")
    print(summary)
    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()