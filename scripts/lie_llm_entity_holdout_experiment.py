import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

load_dotenv()

RESULTS_DIR = os.getenv("LIE_RESULTS_DIR", "results/experiments/lie_llm_diverse_results")
OUT_DIR = os.path.join(RESULTS_DIR, "entity_holdout")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "distilroberta-base",
    "distilgpt2",
    "gpt2",
]

# Жёсткий режим:
# train на base_id с одной половиной предложений
# test на другой половине предложений
# дополнительно можно включить variant holdout
USE_VARIANT_HOLDOUT = os.getenv("LIE_ENTITY_USE_VARIANT_HOLDOUT", "true").lower() == "true"

TRAIN_VARIANTS = [0, 1]
TEST_VARIANTS = [2, 3, 4]


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
        print("Missing:", delta_csv)
        print("Missing:", deltas_npy)
        return None, None

    return pd.read_csv(delta_csv), np.load(deltas_npy)


def make_entity_split(delta_df):
    """
    Делим по base_id, чтобы одно и то же базовое предложение
    не попадало одновременно в train и test.
    """
    base_ids = sorted(delta_df["base_id"].unique())
    midpoint = len(base_ids) // 2

    train_base = set(base_ids[:midpoint])
    test_base = set(base_ids[midpoint:])

    return train_base, test_base


def run_experiment(model_name, delta_df, D):
    labels = delta_df["class"].astype(str).to_numpy()
    base_ids = delta_df["base_id"].to_numpy()
    variants = delta_df["variant_id"].astype(int).to_numpy()

    train_base, test_base = make_entity_split(delta_df)

    train_mask = np.array([b in train_base for b in base_ids])
    test_mask = np.array([b in test_base for b in base_ids])

    if USE_VARIANT_HOLDOUT:
        train_mask = train_mask & np.isin(variants, TRAIN_VARIANTS)
        test_mask = test_mask & np.isin(variants, TEST_VARIANTS)

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

    rows = []

    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)

        row = {
            "model": model_name,
            "classifier": clf_name,
            "accuracy": float(acc),
            "random_baseline": float(random_baseline),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_classes": int(len(classes)),
            "use_variant_holdout": USE_VARIANT_HOLDOUT,
            "train_variants": ",".join(map(str, TRAIN_VARIANTS)) if USE_VARIANT_HOLDOUT else "all",
            "test_variants": ",".join(map(str, TEST_VARIANTS)) if USE_VARIANT_HOLDOUT else "all",
        }

        rows.append(row)

        report = classification_report(
            y_test,
            pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )

        pd.DataFrame(report).transpose().to_csv(
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

        pd.DataFrame(
            cm,
            index=classes,
            columns=classes,
        ).to_csv(
            os.path.join(
                OUT_DIR,
                f"confusion_{safe_name(model_name)}_{clf_name}.csv",
            )
        )

    return pd.DataFrame(rows)


def main():
    all_rows = []

    print("RESULTS_DIR:", RESULTS_DIR)
    print("USE_VARIANT_HOLDOUT:", USE_VARIANT_HOLDOUT)

    for model_name in MODELS:
        print(f"\n=== MODEL: {model_name} ===")

        delta_df, D = load_model_data(model_name)

        if delta_df is None:
            print("Skipped.")
            continue

        result = run_experiment(model_name, delta_df, D)
        print(result)

        all_rows.append(result)

    if not all_rows:
        print("No results.")
        return

    summary = pd.concat(all_rows, ignore_index=True)

    summary.to_csv(
        os.path.join(OUT_DIR, "entity_holdout_summary.csv"),
        index=False,
    )

    print("\n=== SUMMARY ===")
    print(summary)
    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
