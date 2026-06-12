import os
import gc
import itertools
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
OUT_DIR = "lie_llm_class_results"
os.makedirs(OUT_DIR, exist_ok=True)


TRANSFORMATION_CLASSES = {
    "negation": [
        ("The scientist discovered a new principle.", "The scientist did not discover a new principle."),
        ("A child found a black cat.", "A child did not find a black cat."),
        ("The engineer solved the problem.", "The engineer did not solve the problem."),
        ("The teacher explained the rule.", "The teacher did not explain the rule."),
        ("The system detected an error.", "The system did not detect an error."),
        ("The artist created a painting.", "The artist did not create a painting."),
        ("The student understood the lesson.", "The student did not understand the lesson."),
        ("The company released a product.", "The company did not release a product."),
    ],

    "question": [
        ("The scientist discovered a new principle.", "Did the scientist discover a new principle?"),
        ("A child found a black cat.", "Did a child find a black cat?"),
        ("The engineer solved the problem.", "Did the engineer solve the problem?"),
        ("The teacher explained the rule.", "Did the teacher explain the rule?"),
        ("The system detected an error.", "Did the system detect an error?"),
        ("The artist created a painting.", "Did the artist create a painting?"),
        ("The student understood the lesson.", "Did the student understand the lesson?"),
        ("The company released a product.", "Did the company release a product?"),
    ],

    "past_to_future": [
        ("The scientist discovered a new principle.", "The scientist will discover a new principle."),
        ("A child found a black cat.", "A child will find a black cat."),
        ("The engineer solved the problem.", "The engineer will solve the problem."),
        ("The teacher explained the rule.", "The teacher will explain the rule."),
        ("The system detected an error.", "The system will detect an error."),
        ("The artist created a painting.", "The artist will create a painting."),
        ("The student understood the lesson.", "The student will understand the lesson."),
        ("The company released a product.", "The company will release a product."),
    ],

    "certainty_to_uncertainty": [
        ("The theory is correct.", "The theory may be correct."),
        ("The answer is obvious.", "The answer may be obvious."),
        ("The result is reliable.", "The result may be reliable."),
        ("The experiment is successful.", "The experiment may be successful."),
        ("The solution is complete.", "The solution may be complete."),
        ("The prediction is accurate.", "The prediction may be accurate."),
        ("The explanation is valid.", "The explanation may be valid."),
        ("The signal is meaningful.", "The signal may be meaningful."),
    ],

    "active_to_passive": [
        ("The scientist discovered a new principle.", "A new principle was discovered by the scientist."),
        ("A child found a black cat.", "A black cat was found by a child."),
        ("The engineer solved the problem.", "The problem was solved by the engineer."),
        ("The teacher explained the rule.", "The rule was explained by the teacher."),
        ("The system detected an error.", "An error was detected by the system."),
        ("The artist created a painting.", "A painting was created by the artist."),
        ("The student understood the lesson.", "The lesson was understood by the student."),
        ("The company released a product.", "A product was released by the company."),
    ],

    "formalization": [
        ("I don't know the answer.", "I do not know the answer."),
        ("We can't solve this problem.", "We cannot solve this problem."),
        ("They won't accept the result.", "They will not accept the result."),
        ("She isn't ready for the exam.", "She is not ready for the exam."),
        ("He doesn't understand the rule.", "He does not understand the rule."),
        ("It wasn't a simple decision.", "It was not a simple decision."),
        ("You shouldn't ignore the signal.", "You should not ignore the signal."),
        ("The system couldn't process the request.", "The system could not process the request."),
    ],
}


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def build_dataset():
    rows = []
    seen = {}
    prompts = []

    for class_name, pairs in TRANSFORMATION_CLASSES.items():
        for pair_id, (source, target) in enumerate(pairs):
            for role, text in [("source", source), ("target", target)]:
                if text not in seen:
                    seen[text] = len(prompts)
                    prompts.append(text)

            rows.append({
                "class": class_name,
                "pair_id": f"{class_name}_{pair_id}",
                "source": source,
                "target": target,
                "source_idx": seen[source],
                "target_idx": seen[target],
            })

    return prompts, pd.DataFrame(rows)


def get_trajectories(model_name, prompts):
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
        prompts,
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

    # prompts × layers × hidden_dim
    traj = np.stack(layer_vectors, axis=1)

    del model
    gc.collect()

    return traj


def transformation_vector(traj, source_idx, target_idx, mode="final"):
    if mode == "final":
        return traj[target_idx, -1] - traj[source_idx, -1]

    if mode == "all_layers_flat":
        return (traj[target_idx] - traj[source_idx]).reshape(-1)

    if mode == "transition_dynamics_flat":
        source_transitions = traj[source_idx, 1:] - traj[source_idx, :-1]
        target_transitions = traj[target_idx, 1:] - traj[target_idx, :-1]
        return (target_transitions - source_transitions).reshape(-1)

    raise ValueError(f"Unknown mode: {mode}")


def analyze_transformation_classes(model_name, prompts, pair_table):
    print(f"\n=== MODEL: {model_name} ===")

    traj = get_trajectories(model_name, prompts)

    vector_modes = [
        "final",
        "all_layers_flat",
        "transition_dynamics_flat",
    ]

    all_rows = []
    class_summary_rows = []
    separability_rows = []

    for mode in vector_modes:
        vectors = []

        for _, row in pair_table.iterrows():
            v = transformation_vector(
                traj,
                int(row["source_idx"]),
                int(row["target_idx"]),
                mode=mode,
            )

            vectors.append({
                "model": model_name,
                "mode": mode,
                "class": row["class"],
                "pair_id": row["pair_id"],
                "source": row["source"],
                "target": row["target"],
                "vector": v,
                "norm": float(np.linalg.norm(v)),
            })

        # pairwise similarities between transformation vectors
        for a, b in itertools.combinations(range(len(vectors)), 2):
            va = vectors[a]
            vb = vectors[b]

            sim = cosine(va["vector"], vb["vector"])
            same_class = va["class"] == vb["class"]

            all_rows.append({
                "model": model_name,
                "mode": mode,
                "class_a": va["class"],
                "class_b": vb["class"],
                "pair_a": va["pair_id"],
                "pair_b": vb["pair_id"],
                "same_class": same_class,
                "cosine": sim,
            })

        # class centroid stability
        for class_name in TRANSFORMATION_CLASSES.keys():
            class_vectors = [
                item["vector"]
                for item in vectors
                if item["class"] == class_name
            ]

            centroid = np.mean(class_vectors, axis=0)
            sims_to_centroid = [cosine(v, centroid) for v in class_vectors]
            norms = [np.linalg.norm(v) for v in class_vectors]

            class_summary_rows.append({
                "model": model_name,
                "mode": mode,
                "class": class_name,
                "mean_cosine_to_class_centroid": float(np.mean(sims_to_centroid)),
                "std_cosine_to_class_centroid": float(np.std(sims_to_centroid)),
                "mean_vector_norm": float(np.mean(norms)),
                "std_vector_norm": float(np.std(norms)),
            })

        sim_df = pd.DataFrame([r for r in all_rows if r["mode"] == mode and r["model"] == model_name])

        within = sim_df[sim_df["same_class"]]["cosine"]
        between = sim_df[~sim_df["same_class"]]["cosine"]

        separability_rows.append({
            "model": model_name,
            "mode": mode,
            "within_class_mean_cosine": float(within.mean()),
            "between_class_mean_cosine": float(between.mean()),
            "separation_delta": float(within.mean() - between.mean()),
        })

    return (
        pd.DataFrame(all_rows),
        pd.DataFrame(class_summary_rows),
        pd.DataFrame(separability_rows),
    )


def plot_results(separability_df, class_summary_df):
    plt.figure(figsize=(12, 6))

    pivot = separability_df.pivot(
        index="mode",
        columns="model",
        values="separation_delta",
    )

    pivot.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Within-class cosine minus between-class cosine")
    plt.title("Transformation-class separability by model and vector mode")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/separability_by_model_and_mode.png", dpi=180)
    plt.close()

    for mode in class_summary_df["mode"].unique():
        sub = class_summary_df[class_summary_df["mode"] == mode]

        pivot2 = sub.pivot(
            index="class",
            columns="model",
            values="mean_cosine_to_class_centroid",
        )

        pivot2.plot(kind="bar", figsize=(13, 6))
        plt.ylabel("Mean cosine to class centroid")
        plt.title(f"Class stability by model: {mode}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        safe_mode = mode.replace("/", "_")
        plt.savefig(f"{OUT_DIR}/class_stability_{safe_mode}.png", dpi=180)
        plt.close()


def main():
    prompts, pair_table = build_dataset()

    pd.DataFrame({"prompt_id": range(len(prompts)), "prompt": prompts}).to_csv(
        f"{OUT_DIR}/prompts.csv",
        index=False,
    )

    pair_table.to_csv(f"{OUT_DIR}/transformation_pairs.csv", index=False)

    all_similarity = []
    all_class_summary = []
    all_separability = []

    for model_name in MODELS:
        model_name = model_name.strip()

        try:
            sim_df, class_df, sep_df = analyze_transformation_classes(
                model_name,
                prompts,
                pair_table,
            )

            all_similarity.append(sim_df)
            all_class_summary.append(class_df)
            all_separability.append(sep_df)

        except Exception as e:
            print(f"\nFAILED: {model_name}")
            print(e)

    similarity_df = pd.concat(all_similarity, ignore_index=True)
    class_summary_df = pd.concat(all_class_summary, ignore_index=True)
    separability_df = pd.concat(all_separability, ignore_index=True)

    similarity_df.to_csv(f"{OUT_DIR}/pairwise_transformation_similarity.csv", index=False)
    class_summary_df.to_csv(f"{OUT_DIR}/class_centroid_stability.csv", index=False)
    separability_df.to_csv(f"{OUT_DIR}/class_separability_summary.csv", index=False)

    plot_results(separability_df, class_summary_df)

    print("\n=== CLASS SEPARABILITY ===")
    print(separability_df)

    print("\n=== CLASS CENTROID STABILITY ===")
    print(class_summary_df)

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()