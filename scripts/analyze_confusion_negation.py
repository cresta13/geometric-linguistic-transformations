from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def parse_name(path):
    stem = path.stem.replace("confusion_", "")
    if stem.endswith("_linear_svc"):
        return stem[:-11].replace("__", "/"), "linear_svc"
    if stem.endswith("_logreg"):
        return stem[:-7].replace("__", "/"), "logreg"
    raise ValueError(path.name)


def load_confusion(path):
    model, classifier = parse_name(path)
    cm = pd.read_csv(path, index_col=0)
    labels = list(cm.index)

    rows = []
    errors = []

    for label in labels:
        total = float(cm.loc[label].sum())
        correct = float(cm.loc[label, label])
        recall = correct / total if total else 0.0
        rows.append({
            "source": "full_semantic",
            "model": model,
            "classifier": classifier,
            "class": label,
            "support": int(total),
            "correct": int(correct),
            "recall": recall,
            "error_rate": 1.0 - recall,
            "is_negation": label == "negation",
        })

        for pred in labels:
            if pred == label:
                continue
            count = int(cm.loc[label, pred])
            if count:
                errors.append({
                    "source": "full_semantic",
                    "model": model,
                    "classifier": classifier,
                    "true_class": label,
                    "predicted_class": pred,
                    "count": count,
                    "true_support": int(total),
                    "share_of_true": count / total if total else 0.0,
                    "involves_negation": label == "negation" or pred == "negation",
                })

    return rows, errors


def main():
    paths = sorted((EXP / "lie_llm_full_semantic_holdout_results").glob("confusion_*.csv"))
    all_rows = []
    all_errors = []

    for path in paths:
        rows, errors = load_confusion(path)
        all_rows.extend(rows)
        all_errors.extend(errors)

    class_df = pd.DataFrame(all_rows)
    error_df = pd.DataFrame(all_errors)

    class_df.to_csv(OUT / "confusion_class_recall.csv", index=False)
    error_df.sort_values(["count", "share_of_true"], ascending=False).to_csv(
        OUT / "confusion_top_errors.csv",
        index=False,
    )

    neg_summary = (
        class_df
        .assign(group=lambda d: d["is_negation"].map({True: "negation", False: "non_negation"}))
        .groupby(["source", "model", "classifier", "group"])
        .agg(
            mean_recall=("recall", "mean"),
            mean_error_rate=("error_rate", "mean"),
            n_classes=("class", "count"),
        )
        .reset_index()
    )
    neg_summary.to_csv(OUT / "confusion_negation_summary.csv", index=False)

    rank_df = class_df.copy()
    rank_df["recall_rank_low_is_hard"] = rank_df.groupby(["model", "classifier"])["recall"].rank(method="min")
    rank_df.sort_values(["model", "classifier", "recall"]).to_csv(
        OUT / "confusion_class_recall_ranked.csv",
        index=False,
    )

    print("Saved:")
    for name in [
        "confusion_class_recall.csv",
        "confusion_top_errors.csv",
        "confusion_negation_summary.csv",
        "confusion_class_recall_ranked.csv",
    ]:
        print(OUT / name)


if __name__ == "__main__":
    main()
