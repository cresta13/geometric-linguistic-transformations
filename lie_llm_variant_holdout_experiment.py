import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

load_dotenv()

RESULTS_DIR = os.getenv("LIE_RESULTS_DIR", "lie_llm_diverse_results")
OUT_DIR = os.path.join(RESULTS_DIR, "variant_holdout")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_VARIANTS = [0, 1]
TEST_VARIANTS = [2, 3, 4]

MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "distilroberta-base",
    "distilgpt2",
    "gpt2",
]


def safe_name(model_name):
    return model_name.replace("/", "__")


def load_model_data(model_name):
    safe = safe_name(model_name)

    delta_csv = os.path.join(
        RESULTS_DIR,
        f"delta_vectors_metrics_{safe}.csv",
    )

    deltas_npy = os.path.join(
        RESULTS_DIR,
        f"deltas_{safe}.npy",
    )

    if not os.path.exists(delta_csv) or not os.path.exists(deltas_npy):
        return None, None

    delta_df = pd.read_csv(delta_csv)
    D = np.load(deltas_npy)

    return delta_df, D


def run_variant_holdout(model_name, delta_df, D):
    labels = delta_df["class"].astype(str).to_numpy()
    variants = delta_df["variant_id"].astype(int).to_numpy()

    train_mask = np.isin(variants, TRAIN_VARIANTS)
    test_mask = np.isin(variants, TEST_VARIANTS)

    X_train = D[train_mask]
    y_train = labels[train_mask]

    X_test = D[test_mask]
    y_test = labels[test_mask]

    classes = sorted(set(labels))
    random_baseline = 1 / len(classes)

    classifiers = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        "linear_svc": make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                max_iter=20000,
            ),
        ),
    }

    summary_rows = []

    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)

        summary_rows.append({
            "model": model_name,
            "classifier": clf_name,
            "accuracy": float(acc),
            "random_baseline": float(random_baseline),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_classes": int(len(classes)),
            "train_variants": ",".join(map(str, TRAIN_VARIANTS)),
            "test_variants": ",".join(map(str, TEST_VARIANTS)),
        })

        report = classification_report(
            y_test,
            pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(
            os.path.join(
                OUT_DIR,
                f"report_{safe_name(model_name)}_{clf_name}.csv",
            )
        )

        cm = confusion_matrix(
            y_test,
            pred,
            labels=classes,
        )

        cm_df = pd.DataFrame(
            cm,
            index=classes,
            columns=classes,
        )

        cm_df.to_csv(
            os.path.join(
                OUT_DIR,
                f"confusion_{safe_name(model_name)}_{clf_name}.csv",
            )
        )

    return pd.DataFrame(summary_rows)


def main():
    all_summary = []

    for model_name in MODELS:
        print(f"\n=== MODEL: {model_name} ===")

        delta_df, D = load_model_data(model_name)

        if delta_df is None:
            print("Skipped: no files found")
            continue

        result = run_variant_holdout(model_name, delta_df, D)

        print(result)

        all_summary.append(result)

    if not all_summary:
        print("No results found.")
        return

    summary = pd.concat(all_summary, ignore_index=True)

    summary.to_csv(
        os.path.join(OUT_DIR, "variant_holdout_summary.csv"),
        index=False,
    )

    print("\n=== SUMMARY ===")
    print(summary)

    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()