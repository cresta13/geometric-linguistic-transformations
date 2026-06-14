import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, normalize

from run_upat_large import UPATAudit
from run_upat_rise_aware_comparison import (
    CSV_DIR,
    FIG_DIR,
    learn_mdv_raw,
    learn_mdv_unit,
    learn_rise_style,
    make_full_anchor_alignment,
    predict_mdv_raw,
    predict_mdv_unit,
    predict_rise_style,
    rows_from_vectors,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run non-leaky Hybrid RISE-Procrustes Transformation Transfer on UPAT."
    )
    parser.add_argument("--seed", type=int, default=4242)
    return parser.parse_args()


def cosine_rows(a, b):
    return np.sum(normalize(a) * normalize(b), axis=1)


def fit_predict_feature(audit, train_features, test_features, seed):
    pred = audit.fit_predict(
        train_features[audit.train_mask],
        audit.y[audit.train_mask],
        test_features[audit.test_mask],
        seed=seed,
    )
    return {
        "acc": accuracy_score(audit.y[audit.test_mask], pred),
        "f1": f1_score(audit.y[audit.test_mask], pred, average="macro"),
    }


def learn_prototypes(method, x_train, y_train, labels_train, class_ids):
    if method == "mdv_raw":
        return learn_mdv_raw(x_train, y_train, labels_train, class_ids)
    if method == "mdv_unit":
        return learn_mdv_unit(x_train, y_train, labels_train, class_ids)
    if method == "rise_style":
        return learn_rise_style(x_train, y_train, labels_train, class_ids)
    raise ValueError(f"Unknown prototype method: {method}")


def predict_all_classes(method, x_all, prototypes, class_ids):
    predictions = []
    for cls in class_ids:
        labels = np.full(x_all.shape[0], cls)
        if method == "mdv_raw":
            pred = predict_mdv_raw(x_all, prototypes, labels)
        elif method == "mdv_unit":
            pred = predict_mdv_unit(x_all, prototypes, labels)
        elif method == "rise_style":
            pred = predict_rise_style(x_all, prototypes, labels)
        else:
            raise ValueError(f"Unknown prototype method: {method}")
        predictions.append(pred)
    return np.stack(predictions, axis=1)


def prototype_score_features(method, x_train, y_train, x_eval, y_eval, labels_train, class_ids):
    """Score each pair against every class prototype without using its true label."""
    prototypes = learn_prototypes(
        method=method,
        x_train=x_train,
        y_train=y_train,
        labels_train=labels_train,
        class_ids=class_ids,
    )
    pred_all = predict_all_classes(method, x_eval, prototypes, class_ids)

    true_delta = y_eval - x_eval
    pred_delta = pred_all - x_eval[:, None, :]

    target_cos = np.column_stack([
        cosine_rows(pred_all[:, i, :], y_eval)
        for i in range(len(class_ids))
    ])
    delta_cos = np.column_stack([
        cosine_rows(pred_delta[:, i, :], true_delta)
        for i in range(len(class_ids))
    ])
    residual_norm = np.column_stack([
        np.linalg.norm(y_eval - pred_all[:, i, :], axis=1)
        for i in range(len(class_ids))
    ])

    return np.hstack([target_cos, delta_cos, residual_norm])


def standardize_from_train(audit, train_features, test_features):
    scaler = StandardScaler()
    scaler.fit(train_features[audit.train_mask])
    return scaler.transform(train_features), scaler.transform(test_features)


def build_feature_sets(audit, method, train_x, train_y, test_x, test_y):
    train_idx = np.where(audit.train_mask)[0]
    class_ids = np.unique(audit.y)

    delta_train = normalize(train_y - train_x)
    delta_test = normalize(test_y - test_x)

    score_train = prototype_score_features(
        method=method,
        x_train=train_x[train_idx],
        y_train=train_y[train_idx],
        x_eval=train_x,
        y_eval=train_y,
        labels_train=audit.y[train_idx],
        class_ids=class_ids,
    )
    score_test = prototype_score_features(
        method=method,
        x_train=train_x[train_idx],
        y_train=train_y[train_idx],
        x_eval=test_x,
        y_eval=test_y,
        labels_train=audit.y[train_idx],
        class_ids=class_ids,
    )
    score_train, score_test = standardize_from_train(audit, score_train, score_test)

    return {
        "delta_only": (delta_train, delta_test),
        f"{method}_prototype_scores": (score_train, score_test),
        f"{method}_hybrid_delta_scores": (
            normalize(np.hstack([delta_train, score_train])),
            normalize(np.hstack([delta_test, score_test])),
        ),
    }


def evaluate_feature_sets(audit, comparison, train_model, test_model, method, train_x, train_y, test_x, test_y, seed, observed=None):
    rows = []
    for feature, (train_features, test_features) in build_feature_sets(
        audit, method, train_x, train_y, test_x, test_y
    ).items():
        metrics = fit_predict_feature(audit, train_features, test_features, seed)
        row = {
            "comparison": comparison,
            "train_model": train_model,
            "test_model": test_model,
            "prototype_method": method,
            "feature": feature,
            "acc": metrics["acc"],
            "f1": metrics["f1"],
        }
        if observed is not None:
            row.update({
                "observed_raw_f1": observed.raw_f1,
                "observed_aligned_f1": observed.aligned_f1,
                "gain_vs_observed_aligned_f1": metrics["f1"] - observed.aligned_f1,
            })
        rows.append(row)
    return rows


def run_within_model(audit, seed):
    rows = []
    for model_name in audit.procrustes_models:
        print(f"\nHybrid within-model: {model_name}", flush=True)
        sp = audit.spaces[model_name]
        for method in ["mdv_raw", "mdv_unit", "rise_style"]:
            rows.extend(evaluate_feature_sets(
                audit=audit,
                comparison="within_model",
                train_model=model_name,
                test_model=model_name,
                method=method,
                train_x=sp["x_raw"],
                train_y=sp["y_raw"],
                test_x=sp["x_raw"],
                test_y=sp["y_raw"],
                seed=seed,
            ))
    return rows


def run_cross_model(audit, seed):
    observed = pd.read_csv(CSV_DIR / "cross_model_procrustes.csv")
    rows = []

    for obs in observed.itertuples(index=False):
        train_model = obs.train_model
        test_model = obs.test_model
        if train_model == test_model:
            continue

        print(f"\nHybrid cross-model: {train_model} -> {test_model}", flush=True)
        source_aligned, target_aligned = make_full_anchor_alignment(audit, train_model, test_model)
        train_x, train_y = rows_from_vectors(audit, train_model, source_aligned)
        test_x, test_y = rows_from_vectors(audit, test_model, target_aligned)

        for method in ["mdv_raw", "mdv_unit", "rise_style"]:
            rows.extend(evaluate_feature_sets(
                audit=audit,
                comparison="cross_model_full_anchor",
                train_model=train_model,
                test_model=test_model,
                method=method,
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                seed=seed,
                observed=obs,
            ))
    return rows


def summarize(raw_df):
    return (
        raw_df
        .groupby(["comparison", "prototype_method", "feature"])
        .agg(
            mean_acc=("acc", "mean"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_gain_vs_observed_aligned_f1=("gain_vs_observed_aligned_f1", "mean"),
            n=("f1", "count"),
        )
        .reset_index()
        .sort_values(["comparison", "mean_f1"], ascending=[True, False])
    )


def plot_summary(summary_df):
    cross = summary_df[summary_df["comparison"] == "cross_model_full_anchor"].copy()
    cross["label"] = cross["prototype_method"] + "\n" + cross["feature"].str.replace("_", " ")
    cross = cross.sort_values("mean_f1", ascending=False)

    plt.figure(figsize=(max(12, len(cross) * 0.55), 6))
    x = np.arange(len(cross))
    plt.bar(x, cross["mean_f1"])
    delta_mean = cross[cross["feature"] == "delta_only"]["mean_f1"].iloc[0]
    plt.axhline(delta_mean, linestyle="--", label="delta_only")
    plt.xticks(x, cross["label"], rotation=70, ha="right", fontsize=7)
    plt.ylabel("Cross-model transformation label macro F1")
    plt.title("Hybrid RISE-Procrustes Transformation Transfer")
    plt.grid(axis="y", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_hybrid_rise_procrustes_f1.png", dpi=220, bbox_inches="tight")
    plt.close()

    heat = cross.pivot_table(index="prototype_method", columns="feature", values="mean_f1")
    plt.figure(figsize=(10, 4))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="Macro F1")
    plt.xticks(np.arange(len(heat.columns)), heat.columns, rotation=35, ha="right", fontsize=8)
    plt.yticks(np.arange(len(heat.index)), heat.index)
    plt.title("Hybrid feature comparison")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            plt.text(j, i, f"{heat.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_hybrid_rise_procrustes_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    audit = UPATAudit()
    audit.build_dataset()
    audit.extract_all_spaces()

    cross_path = CSV_DIR / "cross_model_procrustes.csv"
    if not cross_path.exists():
        audit.run_cross_model_procrustes()

    rows = []
    rows.extend(run_within_model(audit, seed=args.seed))
    rows.extend(run_cross_model(audit, seed=args.seed))

    raw_df = pd.DataFrame(rows)
    summary_df = summarize(raw_df)
    raw_df.to_csv(CSV_DIR / "hybrid_rise_procrustes_raw.csv", index=False)
    summary_df.to_csv(CSV_DIR / "hybrid_rise_procrustes_summary.csv", index=False)
    plot_summary(summary_df)

    print(f"\nSaved: {CSV_DIR / 'hybrid_rise_procrustes_raw.csv'}")
    print(f"Saved: {CSV_DIR / 'hybrid_rise_procrustes_summary.csv'}")


if __name__ == "__main__":
    sys.exit(main())
