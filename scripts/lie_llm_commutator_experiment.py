# pip install torch transformers pandas numpy matplotlib scikit-learn python-dotenv

import os
import gc
import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

load_dotenv()

MODELS = os.getenv(
    "LIE_MODELS",
    "bert-base-uncased,distilgpt2,gpt2,roberta-base,distilroberta-base"
).split(",")

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
OUT_DIR = "results/experiments/lie_llm_commutator_results"
os.makedirs(OUT_DIR, exist_ok=True)

PCA_DIM = int(os.getenv("LIE_PCA_DIM", "64"))
RIDGE_ALPHA = float(os.getenv("LIE_RIDGE_ALPHA", "1e-2"))


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


def build_dataset():
    prompts = []
    seen = {}
    rows = []

    for class_name, pairs in TRANSFORMATION_CLASSES.items():
        for i, (src, tgt) in enumerate(pairs):
            for text in [src, tgt]:
                if text not in seen:
                    seen[text] = len(prompts)
                    prompts.append(text)

            rows.append({
                "class": class_name,
                "pair_id": f"{class_name}_{i}",
                "source": src,
                "target": tgt,
                "source_idx": seen[src],
                "target_idx": seen[tgt],
            })

    return prompts, pd.DataFrame(rows)


def mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def get_final_vectors(model_name, prompts):
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

    final_hidden = outputs.hidden_states[-1]
    vectors = mean_pool(final_hidden, inputs["attention_mask"]).cpu().numpy()

    del model
    gc.collect()

    return vectors


def fit_affine_delta_operator(X_src, X_tgt, alpha=1e-2):
    """
    Учим оператор класса:
    delta = target - source ≈ source @ W.T + b

    Тогда трансформация:
    T(x) = x + W x + b
    """
    Y = X_tgt - X_src

    n, d = X_src.shape
    X_aug = np.concatenate([X_src, np.ones((n, 1))], axis=1)

    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0

    # B: (d+1) × d
    B = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y)

    W = B[:-1, :].T
    b = B[-1, :]

    return W, b


def affine_lie_bracket(Wa, ba, Wb, bb):
    """
    Для affine vector fields:
    A(x)=Wa x + ba
    B(x)=Wb x + bb

    [A,B](x) = Db*A - Da*B
             = Wb(Wa x + ba) - Wa(Wb x + bb)

    Matrix part: Wb Wa - Wa Wb
    Bias part:   Wb ba - Wa bb
    """
    M = Wb @ Wa - Wa @ Wb
    c = Wb @ ba - Wa @ bb
    return M, c


def apply_transform(x, W, b):
    return x + W @ x + b


def main():
    prompts, pair_table = build_dataset()

    pd.DataFrame({"prompt_id": range(len(prompts)), "prompt": prompts}).to_csv(
        f"{OUT_DIR}/prompts.csv",
        index=False,
    )
    pair_table.to_csv(f"{OUT_DIR}/transformation_pairs.csv", index=False)

    all_commutators = []
    all_operator_norms = []
    all_composition_tests = []

    for model_name in MODELS:
        model_name = model_name.strip()
        print(f"\n=== MODEL: {model_name} ===")

        try:
            raw_vectors = get_final_vectors(model_name, prompts)

            dim = min(PCA_DIM, raw_vectors.shape[0] - 1, raw_vectors.shape[1])
            pca = PCA(n_components=dim, random_state=42)
            vectors = pca.fit_transform(raw_vectors)

            operators = {}

            for class_name in TRANSFORMATION_CLASSES.keys():
                sub = pair_table[pair_table["class"] == class_name]

                src = vectors[sub["source_idx"].values]
                tgt = vectors[sub["target_idx"].values]

                W, b = fit_affine_delta_operator(src, tgt, alpha=RIDGE_ALPHA)
                operators[class_name] = (W, b)

                all_operator_norms.append({
                    "model": model_name,
                    "class": class_name,
                    "pca_dim": dim,
                    "operator_matrix_norm": float(np.linalg.norm(W, ord="fro")),
                    "operator_bias_norm": float(np.linalg.norm(b)),
                    "mean_delta_norm": float(np.mean(np.linalg.norm(tgt - src, axis=1))),
                })

            for class_a, class_b in itertools.combinations(operators.keys(), 2):
                Wa, ba = operators[class_a]
                Wb, bb = operators[class_b]

                M, c = affine_lie_bracket(Wa, ba, Wb, bb)

                norm_a = np.linalg.norm(Wa, ord="fro") + np.linalg.norm(ba)
                norm_b = np.linalg.norm(Wb, ord="fro") + np.linalg.norm(bb)

                bracket_norm = np.linalg.norm(M, ord="fro") + np.linalg.norm(c)
                normalized_bracket = bracket_norm / ((norm_a * norm_b) + 1e-12)

                empirical = []
                composition_gaps = []

                for x in vectors:
                    bracket_x = M @ x + c
                    empirical.append(np.linalg.norm(bracket_x))

                    ab = apply_transform(apply_transform(x, Wa, ba), Wb, bb)
                    ba_comp = apply_transform(apply_transform(x, Wb, bb), Wa, ba)
                    composition_gaps.append(np.linalg.norm(ab - ba_comp))

                all_commutators.append({
                    "model": model_name,
                    "class_a": class_a,
                    "class_b": class_b,
                    "pca_dim": dim,
                    "bracket_matrix_fro_norm": float(np.linalg.norm(M, ord="fro")),
                    "bracket_bias_norm": float(np.linalg.norm(c)),
                    "bracket_total_norm": float(bracket_norm),
                    "normalized_bracket_norm": float(normalized_bracket),
                    "mean_empirical_bracket_norm": float(np.mean(empirical)),
                    "mean_composition_gap": float(np.mean(composition_gaps)),
                })

                all_composition_tests.append({
                    "model": model_name,
                    "class_a": class_a,
                    "class_b": class_b,
                    "mean_TbTa_minus_TaTb_norm": float(np.mean(composition_gaps)),
                })

        except Exception as e:
            print(f"FAILED: {model_name}")
            print(e)

    comm_df = pd.DataFrame(all_commutators)
    op_df = pd.DataFrame(all_operator_norms)
    comp_df = pd.DataFrame(all_composition_tests)

    comm_df.to_csv(f"{OUT_DIR}/commutator_summary.csv", index=False)
    op_df.to_csv(f"{OUT_DIR}/operator_norms.csv", index=False)
    comp_df.to_csv(f"{OUT_DIR}/composition_gaps.csv", index=False)

    print("\n=== COMMUTATOR SUMMARY ===")
    print(comm_df.sort_values(["model", "normalized_bracket_norm"], ascending=[True, False]))

    print("\n=== OPERATOR NORMS ===")
    print(op_df)

    # График среднего normalized bracket по модели
    if not comm_df.empty:
        model_summary = comm_df.groupby("model").agg({
            "normalized_bracket_norm": "mean",
            "mean_composition_gap": "mean",
            "mean_empirical_bracket_norm": "mean",
        }).reset_index()

        model_summary.to_csv(f"{OUT_DIR}/model_commutator_summary.csv", index=False)

        plt.figure(figsize=(10, 5))
        plt.bar(model_summary["model"], model_summary["normalized_bracket_norm"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Mean normalized bracket norm")
        plt.title("Average non-commutativity by model")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/model_average_commutator.png", dpi=180)
        plt.close()

        for model_name in comm_df["model"].unique():
            sub = comm_df[comm_df["model"] == model_name]

            classes = sorted(set(sub["class_a"]).union(set(sub["class_b"])))
            matrix = pd.DataFrame(0.0, index=classes, columns=classes)

            for _, row in sub.iterrows():
                a = row["class_a"]
                b = row["class_b"]
                val = row["normalized_bracket_norm"]
                matrix.loc[a, b] = val
                matrix.loc[b, a] = val

            plt.figure(figsize=(8, 7))
            plt.imshow(matrix.values)
            plt.colorbar(label="Normalized bracket norm")
            plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
            plt.yticks(range(len(classes)), classes)
            plt.title(f"Commutator heatmap: {model_name}")
            plt.tight_layout()

            safe_name = model_name.replace("/", "_")
            plt.savefig(f"{OUT_DIR}/commutator_heatmap_{safe_name}.png", dpi=180)
            plt.close()

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
