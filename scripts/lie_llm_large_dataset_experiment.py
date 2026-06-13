import os
import gc
import itertools
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

load_dotenv()

MODELS = os.getenv(
    "LIE_MODELS",
    "bert-base-uncased,distilgpt2,gpt2,roberta-base,distilroberta-base"
).split(",")

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
OUT_DIR = "results/experiments/lie_llm_large_results"
os.makedirs(OUT_DIR, exist_ok=True)

N_BASE = int(os.getenv("LIE_N_BASE", "200"))
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
    ("examined", "examine"),
    ("explained", "explain"),
    ("discovered", "discover"),
    ("designed", "design"),
    ("created", "create"),
    ("solved", "solve"),
    ("found", "find"),
    ("approved", "approve"),
    ("tested", "test"),
    ("wrote", "write"),
    ("played", "play"),
    ("studied", "study"),
    ("helped", "help"),
    ("defended", "defend"),
    ("grew", "grow"),
    ("controlled", "control"),
    ("improved", "improve"),
    ("analyzed", "analyze"),
    ("reported", "report"),
    ("prepared", "prepare"),
]

OBJECTS = [
    "the patient", "the rule", "a new principle", "a complex machine",
    "a beautiful painting", "the problem", "a black cat", "the decision",
    "the system", "a short story", "a sad melody", "the question",
    "the family", "the client", "the garden", "the aircraft",
    "the interface", "the data", "the event", "the meal",
]


def cap(text: str) -> str:
    return text[0].upper() + text[1:]


def make_base_sentences(n_base):
    triples = list(itertools.product(SUBJECTS, VERBS, OBJECTS))
    random.shuffle(triples)
    triples = triples[:n_base]

    rows = []

    for i, (subject, (past, base), obj) in enumerate(triples):
        base_sentence = f"{cap(subject)} {past} {obj}."

        rows.append({
            "base_id": i,
            "subject": subject,
            "verb_past": past,
            "verb_base": base,
            "object": obj,
            "base_sentence": base_sentence,
        })

    return pd.DataFrame(rows)


def build_transformation_pairs(base_df):
    rows = []

    for _, r in base_df.iterrows():
        subject = r["subject"]
        subject_cap = cap(subject)
        past = r["verb_past"]
        base = r["verb_base"]
        obj = r["object"]
        obj_cap = cap(obj)

        source = f"{subject_cap} {past} {obj}."

        transformations = {
            "negation": f"{subject_cap} did not {base} {obj}.",
            "question": f"Did {subject} {base} {obj}?",
            "past_to_future": f"{subject_cap} will {base} {obj}.",
            "certainty_to_uncertainty": f"{subject_cap} may have {past} {obj}.",
            "active_to_passive": f"{obj_cap} was {past} by {subject}.",
            "formalization": f"It is true that {subject} {past} {obj}.",
        }

        for class_name, target in transformations.items():
            rows.append({
                "base_id": r["base_id"],
                "class": class_name,
                "source": source,
                "target": target,
            })

    return pd.DataFrame(rows)


def build_prompt_index(pair_df):
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


def mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def get_final_vectors(model_name, prompts, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        token=HF_TOKEN,
    )
    model.eval()

    vectors = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]

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

        print(f"  batch {start + len(batch)}/{len(prompts)}")

    vectors = np.vstack(vectors)

    del model
    gc.collect()

    return vectors


def analyze_model(model_name, prompts, pair_df):
    print(f"\n=== MODEL: {model_name} ===")

    vectors = get_final_vectors(model_name, prompts)

    rows = []

    for _, r in pair_df.iterrows():
        src = vectors[int(r["source_idx"])]
        tgt = vectors[int(r["target_idx"])]
        delta = tgt - src

        rows.append({
            "model": model_name,
            "base_id": r["base_id"],
            "class": r["class"],
            "source": r["source"],
            "target": r["target"],
            "delta_norm": float(np.linalg.norm(delta)),
            "delta": delta,
        })

    delta_df = pd.DataFrame(rows)

    similarity_rows = []
    classes = sorted(delta_df["class"].unique())

    for class_name in classes:
        sub = delta_df[delta_df["class"] == class_name]
        deltas = list(sub["delta"])

        centroid = np.mean(deltas, axis=0)
        sims = [cosine(v, centroid) for v in deltas]

        similarity_rows.append({
            "model": model_name,
            "class": class_name,
            "mean_cosine_to_centroid": float(np.mean(sims)),
            "std_cosine_to_centroid": float(np.std(sims)),
            "mean_delta_norm": float(sub["delta_norm"].mean()),
            "n": len(sub),
        })

    pairwise_rows = []

    sample_df = delta_df.copy()

    for i, j in itertools.combinations(range(len(sample_df)), 2):
        a = sample_df.iloc[i]
        b = sample_df.iloc[j]

        sim = cosine(a["delta"], b["delta"])

        pairwise_rows.append({
            "model": model_name,
            "class_a": a["class"],
            "class_b": b["class"],
            "same_class": a["class"] == b["class"],
            "cosine": sim,
        })

    pairwise_df = pd.DataFrame(pairwise_rows)

    within = pairwise_df[pairwise_df["same_class"]]["cosine"]
    between = pairwise_df[~pairwise_df["same_class"]]["cosine"]

    sep_row = {
        "model": model_name,
        "within_class_mean_cosine": float(within.mean()),
        "between_class_mean_cosine": float(between.mean()),
        "separation_delta": float(within.mean() - between.mean()),
        "within_n": int(len(within)),
        "between_n": int(len(between)),
    }

    # permutation test
    observed = sep_row["separation_delta"]
    perm_deltas = []

    labels = delta_df["class"].astype(str).to_numpy(copy=True)
    deltas = list(delta_df["delta"])

    # 50 вместо 300 для скорости
    for _ in range(50):

        shuffled_labels = np.random.permutation(labels)

        same_mask = []
        cos_values = []

        for i, j in itertools.combinations(range(len(deltas)), 2):
            same_mask.append(
                shuffled_labels[i] == shuffled_labels[j]
            )

            cos_values.append(
                cosine(deltas[i], deltas[j])
            )

        temp_df = pd.DataFrame({
            "same": same_mask,
            "cos": cos_values,
        })

        w = temp_df.loc[temp_df["same"], "cos"].mean()
        b = temp_df.loc[~temp_df["same"], "cos"].mean()

        perm_deltas.append(w - b)

    p_value = (
                      np.sum(np.array(perm_deltas) >= observed) + 1
              ) / (len(perm_deltas) + 1)

    sep_row["permutation_p_value"] = float(p_value)
    sep_row["permutation_mean_delta"] = float(np.mean(perm_deltas))
    sep_row["permutation_std_delta"] = float(np.std(perm_deltas))

    clean_delta_df = delta_df.drop(columns=["delta"])

    return clean_delta_df, pd.DataFrame(similarity_rows), pairwise_df, pd.DataFrame([sep_row])


def main():
    base_df = make_base_sentences(N_BASE)
    pair_df = build_transformation_pairs(base_df)
    prompts, pair_df = build_prompt_index(pair_df)

    base_df.to_csv(f"{OUT_DIR}/base_sentences.csv", index=False)
    pair_df.to_csv(f"{OUT_DIR}/transformation_pairs.csv", index=False)
    pd.DataFrame({"prompt_id": range(len(prompts)), "prompt": prompts}).to_csv(
        f"{OUT_DIR}/prompts.csv",
        index=False,
    )

    print(f"Base sentences: {len(base_df)}")
    print(f"Transformation pairs: {len(pair_df)}")
    print(f"Unique prompts: {len(prompts)}")

    all_delta = []
    all_stability = []
    all_pairwise = []
    all_separability = []

    for model_name in MODELS:
        model_name = model_name.strip()

        try:
            delta_df, stability_df, pairwise_df, sep_df = analyze_model(
                model_name,
                prompts,
                pair_df,
            )

            all_delta.append(delta_df)
            all_stability.append(stability_df)
            all_pairwise.append(pairwise_df)
            all_separability.append(sep_df)

        except Exception as e:
            print(f"FAILED: {model_name}")
            print(e)

    delta_all = pd.concat(all_delta, ignore_index=True)
    stability_all = pd.concat(all_stability, ignore_index=True)
    pairwise_all = pd.concat(all_pairwise, ignore_index=True)
    separability_all = pd.concat(all_separability, ignore_index=True)

    delta_all.to_csv(f"{OUT_DIR}/delta_vectors_metrics.csv", index=False)
    stability_all.to_csv(f"{OUT_DIR}/class_centroid_stability.csv", index=False)
    pairwise_all.to_csv(f"{OUT_DIR}/pairwise_delta_similarity.csv", index=False)
    separability_all.to_csv(f"{OUT_DIR}/class_separability_with_permutation.csv", index=False)

    print("\n=== SEPARABILITY WITH PERMUTATION TEST ===")
    print(separability_all)

    print("\n=== CLASS STABILITY ===")
    print(stability_all)

    plt.figure(figsize=(10, 5))
    plt.bar(separability_all["model"], separability_all["separation_delta"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Within-class cosine minus between-class cosine")
    plt.title("Large dataset transformation-class separability")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/large_separability_by_model.png", dpi=180)
    plt.close()

    pivot = stability_all.pivot(
        index="class",
        columns="model",
        values="mean_cosine_to_centroid",
    )

    pivot.plot(kind="bar", figsize=(13, 6))
    plt.ylabel("Mean cosine to class centroid")
    plt.title("Large dataset class stability by model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/large_class_stability.png", dpi=180)
    plt.close()

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
