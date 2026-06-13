import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

load_dotenv()

RESULTS_DIR = os.getenv("LIE_RESULTS_DIR", "results/experiments/lie_llm_large_results")
MODEL_NAME = os.getenv("LIE_CLASSIFIER_MODEL", "roberta-base")
PCA_DIM = int(os.getenv("LIE_CLASSIFIER_PCA_DIM", "128"))
N_SPLITS = int(os.getenv("LIE_CLASSIFIER_CV", "5"))

safe_model = MODEL_NAME.replace("/", "__")

delta_csv = os.path.join(RESULTS_DIR, "delta_vectors_metrics.csv")
vectors_npy = os.path.join(RESULTS_DIR, f"vectors_{safe_model}.npy")
pairs_csv = os.path.join(RESULTS_DIR, "transformation_pairs.csv")

out_dir = os.path.join(RESULTS_DIR, "delta_classifier")
os.makedirs(out_dir, exist_ok=True)


def main():
    print("Loading:")
    print(delta_csv)
    print(vectors_npy)
    print(pairs_csv)

    delta_df = pd.read_csv(delta_csv)
    pair_df = pd.read_csv(pairs_csv)
    vectors = np.load(vectors_npy)

    if "model" in delta_df.columns:
        delta_df = delta_df[delta_df["model"] == MODEL_NAME].reset_index(drop=True)

    X = []
    y = []

    for _, row in pair_df.iterrows():
        src = vectors[int(row["source_idx"])]
        tgt = vectors[int(row["target_idx"])]
        X.append(tgt - src)
        y.append(row["class"])

    X = np.vstack(X)
    y = np.array(y)

    print("X shape:", X.shape)
    print("Classes:", sorted(set(y)))
    print("Random baseline:", round(1 / len(set(y)), 4))

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42,
    )

    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            PCA(n_components=min(PCA_DIM, X.shape[1], X.shape[0] - 1)),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        "linear_svc": make_pipeline(
            StandardScaler(),
            PCA(n_components=min(PCA_DIM, X.shape[1], X.shape[0] - 1)),
            LinearSVC(
                class_weight="balanced",
                max_iter=10000,
            ),
        ),
    }

    summary_rows = []

    for name, clf in models.items():
        scores = cross_val_score(
            clf,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        summary_rows.append({
            "model_name": MODEL_NAME,
            "classifier": name,
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            "n_samples": int(len(y)),
            "n_classes": int(len(set(y))),
            "random_baseline": float(1 / len(set(y))),
            "pca_dim": int(min(PCA_DIM, X.shape[1], X.shape[0] - 1)),
        })

        print(f"\n{name}")
        print("scores:", scores)
        print("mean:", scores.mean(), "std:", scores.std())

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "classifier_summary.csv"), index=False)

    # Финальный подробный отчёт на одном split
    train_idx, test_idx = next(cv.split(X, y))

    best_clf = models["logreg"]
    best_clf.fit(X[train_idx], y[train_idx])
    pred = best_clf.predict(X[test_idx])

    labels = sorted(set(y))

    report = classification_report(
        y[test_idx],
        pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(out_dir, "classification_report_logreg.csv"))

    cm = confusion_matrix(
        y[test_idx],
        pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix_logreg.csv"))

    print("\n=== SUMMARY ===")
    print(summary)

    print("\n=== CLASSIFICATION REPORT LOGREG ===")
    print(pd.DataFrame(report).transpose())

    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
