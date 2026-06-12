import os
import gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

load_dotenv()

MODELS = os.getenv(
    "LIE_MODELS",
    "bert-base-uncased,distilgpt2,gpt2"
).split(",")

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
OUT_DIR = "lie_llm_compare_results"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = [
    "The scientist discovered a new principle of symmetry.",
    "The mathematician discovered a new principle of symmetry.",
    "The scientist did not discover a new principle of symmetry.",
    "A child found a small black cat in the garden.",
    "A child did not find a small black cat in the garden.",
    "The government announced a new economic policy.",
    "The poet wrote a beautiful poem about memory.",
    "The engineer designed a machine that predicts weather.",
    "If the theory is correct, the experiment will confirm it.",
    "If the theory is wrong, the experiment will disprove it.",
]

PAIRS = [
    (0, 1, "scientist → mathematician"),
    (0, 2, "affirmative → negative"),
    (3, 4, "found → did not find"),
    (8, 9, "correct → wrong"),
]


def mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def analyze_model(model_name):
    model_name = model_name.strip()
    print(f"\n=== MODEL: {model_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        token=HF_TOKEN,
    )
    model.eval()

    inputs = tokenizer(
        PROMPTS,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states

    layer_vectors = []
    for h in hidden_states:
        pooled = mean_pool(h, inputs["attention_mask"])
        layer_vectors.append(pooled.cpu().numpy())

    traj = np.stack(layer_vectors, axis=1)

    trajectory_rows = []

    for i, prompt in enumerate(PROMPTS):
        vectors = traj[i]
        transitions = vectors[1:] - vectors[:-1]
        norms = np.linalg.norm(transitions, axis=1)

        adjacent_cosines = [
            cosine(transitions[j], transitions[j + 1])
            for j in range(len(transitions) - 1)
        ]

        path_length = float(norms.sum())
        displacement = float(np.linalg.norm(vectors[-1] - vectors[0]))
        efficiency = displacement / (path_length + 1e-12)

        trajectory_rows.append({
            "model": model_name,
            "prompt_id": i,
            "prompt": prompt,
            "num_layers": len(hidden_states),
            "hidden_dim": traj.shape[-1],
            "path_length": path_length,
            "displacement": displacement,
            "trajectory_efficiency": efficiency,
            "mean_transition_norm": float(norms.mean()),
            "mean_adjacent_cosine": float(np.mean(adjacent_cosines)),
            "curvature_proxy_1_minus_cos": float(np.mean([1 - c for c in adjacent_cosines])),
        })

    pair_rows = []

    for a, b, label in PAIRS:
        layer_distances = np.linalg.norm(traj[a] - traj[b], axis=1)

        trans_a = traj[a, 1:] - traj[a, :-1]
        trans_b = traj[b, 1:] - traj[b, :-1]

        transition_cosines = [
            cosine(trans_a[j], trans_b[j])
            for j in range(trans_a.shape[0])
        ]

        pair_rows.append({
            "model": model_name,
            "pair": label,
            "mean_layer_distance": float(layer_distances.mean()),
            "final_state_distance": float(layer_distances[-1]),
            "mean_transition_cosine": float(np.mean(transition_cosines)),
        })

    del model
    gc.collect()

    return pd.DataFrame(trajectory_rows), pd.DataFrame(pair_rows)


def main():
    all_traj = []
    all_pairs = []

    for model_name in MODELS:
        try:
            traj_df, pair_df = analyze_model(model_name)
            all_traj.append(traj_df)
            all_pairs.append(pair_df)
        except Exception as e:
            print(f"FAILED: {model_name}")
            print(e)

    trajectory_df = pd.concat(all_traj, ignore_index=True)
    pair_df = pd.concat(all_pairs, ignore_index=True)

    trajectory_df.to_csv(f"{OUT_DIR}/all_trajectory_metrics.csv", index=False)
    pair_df.to_csv(f"{OUT_DIR}/all_pair_metrics.csv", index=False)

    summary = trajectory_df.groupby("model").agg({
        "num_layers": "first",
        "hidden_dim": "first",
        "path_length": "mean",
        "displacement": "mean",
        "trajectory_efficiency": "mean",
        "mean_adjacent_cosine": "mean",
        "curvature_proxy_1_minus_cos": "mean",
    }).reset_index()

    summary.to_csv(f"{OUT_DIR}/model_summary.csv", index=False)

    print("\n=== MODEL SUMMARY ===")
    print(summary)

    print("\n=== PAIR METRICS ===")
    print(pair_df)

    plt.figure(figsize=(10, 5))
    plt.bar(summary["model"], summary["curvature_proxy_1_minus_cos"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean curvature proxy")
    plt.title("Trajectory curvature proxy by model")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_curvature_comparison.png", dpi=180)
    plt.close()

    pivot = pair_df.pivot(
        index="pair",
        columns="model",
        values="mean_transition_cosine",
    )

    pivot.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Mean transition cosine")
    plt.title("Prompt-pair transition similarity by model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pair_similarity_by_model.png", dpi=180)
    plt.close()

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()