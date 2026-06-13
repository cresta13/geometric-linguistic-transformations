import os
import gc
import time
import random
import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

load_dotenv()

MODELS = [m.strip() for m in os.getenv(
    "LIE_MODELS",
    "roberta-base,distilroberta-base,bert-base-uncased,gpt2,distilgpt2"
).split(",") if m.strip()]

OUT_DIR = "results/experiments/lie_llm_diverse_results"
os.makedirs(OUT_DIR, exist_ok=True)

N_BASE = int(os.getenv("LIE_N_BASE", "1000"))
BATCH_SIZE = int(os.getenv("LIE_BATCH_SIZE", "32"))
PAIR_SAMPLE = int(os.getenv("LIE_PAIR_SAMPLE", "300000"))
N_PERMUTATIONS = int(os.getenv("LIE_N_PERMUTATIONS", "100"))
SEED = int(os.getenv("LIE_SEED", "42"))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SUBJECTS = [
    "the doctor", "the teacher", "the scientist", "the engineer", "the artist",
    "the student", "the child", "the manager", "the programmer", "the writer",
    "the musician", "the researcher", "the nurse", "the lawyer", "the farmer",
    "the pilot", "the designer", "the analyst", "the journalist", "the chef",
    "the historian", "the biologist", "the economist", "the developer", "the editor",
]

VERBS = [
    ("examined", "examine"), ("explained", "explain"), ("discovered", "discover"),
    ("designed", "design"), ("created", "create"), ("solved", "solve"),
    ("found", "find"), ("approved", "approve"), ("tested", "test"),
    ("wrote", "write"), ("studied", "study"), ("helped", "help"),
    ("defended", "defend"), ("improved", "improve"), ("analyzed", "analyze"),
    ("reported", "report"), ("prepared", "prepare"), ("built", "build"),
    ("measured", "measure"), ("observed", "observe"), ("confirmed", "confirm"),
    ("rejected", "reject"), ("accepted", "accept"), ("predicted", "predict"),
    ("described", "describe"),
]

OBJECTS = [
    "the patient", "the rule", "a new principle", "a complex machine",
    "a beautiful painting", "the problem", "a black cat", "the decision",
    "the system", "a short story", "the question", "the family",
    "the client", "the garden", "the interface", "the data",
    "the event", "the document", "the experiment", "the signal",
    "the theory", "the project", "the report", "the lesson", "the model",
]


def cap(text):
    return text[0].upper() + text[1:]


def mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def make_base_sentences(n_base):
    triples = list(itertools.product(SUBJECTS, VERBS, OBJECTS))
    random.shuffle(triples)
    triples = triples[:n_base]

    rows = []
    for i, (subject, (past, base), obj) in enumerate(triples):
        rows.append({
            "base_id": i,
            "subject": subject,
            "subject_cap": cap(subject),
            "verb_past": past,
            "verb_base": base,
            "object": obj,
            "object_cap": cap(obj),
            "source": f"{cap(subject)} {past} {obj}.",
        })
    return pd.DataFrame(rows)


def transformation_variants(r):
    s = r["subject"]
    S = r["subject_cap"]
    past = r["verb_past"]
    base = r["verb_base"]
    obj = r["object"]
    O = r["object_cap"]

    return {
        "negation": [
            f"{S} did not {base} {obj}.",
            f"{S} never {past} {obj}.",
            f"{S} failed to {base} {obj}.",
            f"{S} was unable to {base} {obj}.",
            f"{S} could not {base} {obj}.",
        ],

        "question": [
            f"Did {s} {base} {obj}?",
            f"Was it true that {s} {past} {obj}?",
            f"Could {s} have {past} {obj}?",
            f"Is there evidence that {s} {past} {obj}?",
            f"Can we confirm that {s} {past} {obj}?",
        ],

        "past_to_future": [
            f"{S} will {base} {obj}.",
            f"{S} is going to {base} {obj}.",
            f"{S} is expected to {base} {obj}.",
            f"{S} is likely to {base} {obj}.",
            f"{S} plans to {base} {obj}.",
        ],

        "certainty_to_uncertainty": [
            f"{S} may have {past} {obj}.",
            f"{S} might have {past} {obj}.",
            f"It is possible that {s} {past} {obj}.",
            f"There is a chance that {s} {past} {obj}.",
            f"{S} probably {past} {obj}.",
        ],

        "active_to_passive": [
            f"{O} was {past} by {s}.",
            f"{O} had been {past} by {s}.",
            f"{O} got {past} by {s}.",
            f"{O} was successfully {past} by {s}.",
            f"{O} was eventually {past} by {s}.",
        ],

        "formalization": [
            f"It is true that {s} {past} {obj}.",
            f"The statement that {s} {past} {obj} is true.",
            f"One can state that {s} {past} {obj}.",
            f"It can be formally said that {s} {past} {obj}.",
            f"The fact is that {s} {past} {obj}.",
        ],
    }


def build_pairs(base_df):
    rows = []

    for _, r in base_df.iterrows():
        source = r["source"]
        variants = transformation_variants(r)

        for class_name, targets in variants.items():
            for variant_id, target in enumerate(targets):
                rows.append({
                    "base_id": int(r["base_id"]),
                    "class": class_name,
                    "variant_id": variant_id,
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
        print(f"Loading cached vectors: {cache_path}")
        return np.load(cache_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
    )
    model.eval()

    vectors = []
    started = time.time()

    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)

        final_hidden = outputs.hidden_states[-1]
        pooled = mean_pool(final_hidden, inputs["attention_mask"])
        vectors.append(pooled.cpu().numpy())

        done = start + len(batch)
        print(f"  {model_name}: {done}/{len(prompts)} prompts, {time.time() - started:.1f}s")

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
            "class": r["class"],
            "variant_id": int(r["variant_id"]),
            "source": r["source"],
            "target": r["target"],
            "delta_norm": float(np.linalg.norm(delta)),
        })

    return pd.DataFrame(rows), np.vstack(deltas)


def class_stability(model_name, delta_df, D):
    Dn = normalize(D)
    rows = []

    for cls in sorted(delta_df["class"].unique()):
        idx = np.where(delta_df["class"].to_numpy() == cls)[0]
        centroid = normalize(D[idx].mean(axis=0, keepdims=True))
        sims = Dn[idx] @ centroid.T

        rows.append({
            "model": model_name,
            "class": cls,
            "n": len(idx),
            "mean_cosine_to_centroid": float(sims.mean()),
            "std_cosine_to_centroid": float(sims.std()),
        })

    return pd.DataFrame(rows)


def sampled_pair_indices(n, max_pairs):
    total = n * (n - 1) // 2
    if total <= max_pairs:
        return np.array(list(itertools.combinations(range(n), 2)), dtype=np.int32)

    pairs = set()
    while len(pairs) < max_pairs:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))

    return np.array(list(pairs), dtype=np.int32)


def separability(model_name, delta_df, D):
    labels = delta_df["class"].astype(str).to_numpy()
    Dn = normalize(D)

    pairs = sampled_pair_indices(len(labels), PAIR_SAMPLE)
    i = pairs[:, 0]
    j = pairs[:, 1]

    cosines = np.sum(Dn[i] * Dn[j], axis=1)
    same = labels[i] == labels[j]

    within = cosines[same]
    between = cosines[~same]
    observed = float(within.mean() - between.mean())

    perm_deltas = []
    for _ in range(N_PERMUTATIONS):
        shuffled = np.random.permutation(labels)
        same_perm = shuffled[i] == shuffled[j]
        w = cosines[same_perm].mean()
        b = cosines[~same_perm].mean()
        perm_deltas.append(float(w - b))

    perm_deltas = np.array(perm_deltas)

    return pd.DataFrame([{
        "model": model_name,
        "n_transformations": len(labels),
        "sampled_pairs": len(pairs),
        "within_class_mean_cosine": float(within.mean()),
        "between_class_mean_cosine": float(between.mean()),
        "separation_delta": observed,
        "permutation_p_value": float((np.sum(perm_deltas >= observed) + 1) / (len(perm_deltas) + 1)),
        "permutation_mean_delta": float(perm_deltas.mean()),
        "permutation_std_delta": float(perm_deltas.std()),
    }])


def leave_one_out_centroid_classifier(model_name, delta_df, D):
    labels = delta_df["class"].astype(str).to_numpy()
    classes = sorted(set(labels))
    Dn = normalize(D)

    correct = 0
    predictions = []

    for idx in range(len(labels)):
        scores = {}

        for cls in classes:
            train_idx = np.where(labels == cls)[0]
            train_idx = train_idx[train_idx != idx]

            centroid = normalize(D[train_idx].mean(axis=0, keepdims=True))[0]
            scores[cls] = float(Dn[idx] @ centroid)

        pred = max(scores, key=scores.get)
        correct += int(pred == labels[idx])

        predictions.append({
            "model": model_name,
            "true_class": labels[idx],
            "predicted_class": pred,
            "correct": pred == labels[idx],
            **{f"score_{cls}": scores[cls] for cls in classes}
        })

    pred_df = pd.DataFrame(predictions)

    return pd.DataFrame([{
        "model": model_name,
        "accuracy": float(correct / len(labels)),
        "random_baseline": float(1 / len(classes)),
        "n_samples": int(len(labels)),
        "n_classes": int(len(classes)),
    }]), pred_df


def main():
    base_df = make_base_sentences(N_BASE)
    pair_df = build_pairs(base_df)
    prompts, pair_df = index_prompts(pair_df)

    base_df.to_csv(os.path.join(OUT_DIR, "base_sentences.csv"), index=False)
    pair_df.to_csv(os.path.join(OUT_DIR, "transformation_pairs.csv"), index=False)
    pd.DataFrame({"prompt_id": range(len(prompts)), "prompt": prompts}).to_csv(
        os.path.join(OUT_DIR, "prompts.csv"),
        index=False,
    )

    print("Models:", MODELS)
    print("Base sentences:", len(base_df))
    print("Pairs:", len(pair_df))
    print("Unique prompts:", len(prompts))

    all_delta = []
    all_stability = []
    all_sep = []
    all_clf = []

    for model_name in MODELS:
        print(f"\n=== MODEL: {model_name} ===")

        vectors = get_vectors(model_name, prompts)
        delta_df, D = build_deltas(model_name, vectors, pair_df)

        stability_df = class_stability(model_name, delta_df, D)
        sep_df = separability(model_name, delta_df, D)
        clf_df, pred_df = leave_one_out_centroid_classifier(model_name, delta_df, D)

        safe = model_name.replace("/", "__")

        delta_df.to_csv(os.path.join(OUT_DIR, f"delta_vectors_metrics_{safe}.csv"), index=False)
        np.save(os.path.join(OUT_DIR, f"deltas_{safe}.npy"), D)
        stability_df.to_csv(os.path.join(OUT_DIR, f"class_stability_{safe}.csv"), index=False)
        sep_df.to_csv(os.path.join(OUT_DIR, f"separability_{safe}.csv"), index=False)
        clf_df.to_csv(os.path.join(OUT_DIR, f"centroid_classifier_{safe}.csv"), index=False)
        pred_df.to_csv(os.path.join(OUT_DIR, f"predictions_{safe}.csv"), index=False)

        all_delta.append(delta_df)
        all_stability.append(stability_df)
        all_sep.append(sep_df)
        all_clf.append(clf_df)

        print(sep_df)
        print(clf_df)

    delta_all = pd.concat(all_delta, ignore_index=True)
    stability_all = pd.concat(all_stability, ignore_index=True)
    sep_all = pd.concat(all_sep, ignore_index=True)
    clf_all = pd.concat(all_clf, ignore_index=True)

    delta_all.to_csv(os.path.join(OUT_DIR, "ALL_delta_vectors_metrics.csv"), index=False)
    stability_all.to_csv(os.path.join(OUT_DIR, "ALL_class_stability.csv"), index=False)
    sep_all.to_csv(os.path.join(OUT_DIR, "ALL_separability.csv"), index=False)
    clf_all.to_csv(os.path.join(OUT_DIR, "ALL_centroid_classifier.csv"), index=False)

    plt.figure(figsize=(11, 5))
    plt.bar(sep_all["model"], sep_all["separation_delta"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Within-class cosine minus between-class cosine")
    plt.title("Diverse transformation separability")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ALL_separability.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(clf_all["model"], clf_all["accuracy"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Leave-one-out centroid accuracy")
    plt.title("Diverse transformation class prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ALL_centroid_classifier.png"), dpi=180)
    plt.close()

    pivot = stability_all.pivot(
        index="class",
        columns="model",
        values="mean_cosine_to_centroid",
    )

    pivot.plot(kind="bar", figsize=(14, 7))
    plt.ylabel("Mean cosine to class centroid")
    plt.title("Diverse class stability by model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ALL_class_stability.png"), dpi=180)
    plt.close()

    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
