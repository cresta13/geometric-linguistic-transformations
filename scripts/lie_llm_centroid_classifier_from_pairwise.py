import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = os.getenv("LIE_RESULTS_DIR", "results/experiments/lie_llm_large_results")
OUT_DIR = os.path.join(RESULTS_DIR, "centroid_classifier_from_pairwise")
os.makedirs(OUT_DIR, exist_ok=True)

DELTA_CSV = os.path.join(RESULTS_DIR, "delta_vectors_metrics.csv")
PAIRWISE_CSV = os.path.join(RESULTS_DIR, "pairwise_delta_similarity.csv")


def main():
    print("Loading:")
    print(DELTA_CSV)
    print(PAIRWISE_CSV)

    delta_df = pd.read_csv(DELTA_CSV)
    pairwise_df = pd.read_csv(PAIRWISE_CSV)

    models = sorted(delta_df["model"].unique())

    all_summary = []

    for model in models:
        print(f"\n=== MODEL: {model} ===")

        model_delta = delta_df[delta_df["model"] == model].reset_index(drop=True)
        model_pairwise = pairwise_df[pairwise_df["model"] == model].reset_index(drop=True)

        labels = model_delta["class"].astype(str).to_numpy()
        classes = sorted(set(labels))
        n = len(labels)

        # Восстанавливаем матрицу cosine similarity
        sim_matrix = np.eye(n, dtype=float)

        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim_matrix[i, j] = model_pairwise.iloc[idx]["cosine"]
                sim_matrix[j, i] = model_pairwise.iloc[idx]["cosine"]
                idx += 1

        correct = 0
        predictions = []

        for i in range(n):
            class_scores = {}

            for cls in classes:
                cls_idx = np.where(labels == cls)[0]
                cls_idx = cls_idx[cls_idx != i]

                if len(cls_idx) == 0:
                    class_scores[cls] = -999
                else:
                    class_scores[cls] = float(sim_matrix[i, cls_idx].mean())

            pred = max(class_scores, key=class_scores.get)

            predictions.append({
                "model": model,
                "index": i,
                "true_class": labels[i],
                "predicted_class": pred,
                "correct": pred == labels[i],
                **{f"score_{cls}": class_scores[cls] for cls in classes},
            })

            if pred == labels[i]:
                correct += 1

        pred_df = pd.DataFrame(predictions)
        acc = correct / n

        random_baseline = 1 / len(classes)

        summary_row = {
            "model": model,
            "accuracy": acc,
            "random_baseline": random_baseline,
            "n_samples": n,
            "n_classes": len(classes),
        }

        all_summary.append(summary_row)

        print(summary_row)

        pred_df.to_csv(
            os.path.join(
                OUT_DIR,
                f"predictions_{model.replace('/', '__')}.csv"
            ),
            index=False,
        )

        confusion = pd.crosstab(
            pred_df["true_class"],
            pred_df["predicted_class"],
            rownames=["true"],
            colnames=["predicted"],
            normalize="index",
        )

        confusion.to_csv(
            os.path.join(
                OUT_DIR,
                f"confusion_{model.replace('/', '__')}.csv"
            )
        )

        print(confusion)

    summary = pd.DataFrame(all_summary)
    summary.to_csv(os.path.join(OUT_DIR, "classifier_summary.csv"), index=False)

    print("\n=== SUMMARY ===")
    print(summary)
    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
