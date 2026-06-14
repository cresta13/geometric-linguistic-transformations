import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from run_upat_large import UPATAudit
from run_upat_rise_aware_comparison import (
    CSV_DIR,
    FIG_DIR,
    cosine_rows,
    exp_map_sphere,
    learn_rise_style,
    make_full_anchor_alignment,
    nearest_target_metrics,
    predict_rise_style,
    rows_from_vectors,
    unit,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run movement-level spherical delta steering tests on UPAT."
    )
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def tangent_project(x_unit, direction):
    return direction - np.sum(direction * x_unit, axis=1, keepdims=True) * x_unit


def learn_unit_delta_centroids(x_train, y_train, labels_train, class_ids):
    deltas = unit(y_train) - unit(x_train)
    return {
        cls: deltas[labels_train == cls].mean(axis=0)
        for cls in class_ids
    }


def predict_linear_delta(x_eval, centroids, labels_eval):
    x_u = unit(x_eval)
    delta = np.vstack([centroids[label] for label in labels_eval])
    return unit(x_u + delta)


def predict_spherical_delta(x_eval, centroids, labels_eval):
    x_u = unit(x_eval)
    delta = np.vstack([centroids[label] for label in labels_eval])
    tangent = tangent_project(x_u, delta)
    return exp_map_sphere(x_u, tangent)


def learn_residual_centroids(base_pred_train, y_train, labels_train, class_ids):
    base_u = unit(base_pred_train)
    residual = unit(y_train) - base_u
    return {
        cls: residual[labels_train == cls].mean(axis=0)
        for cls in class_ids
    }


def apply_residual_spherical(base_pred, residual_centroids, labels_eval):
    base_u = unit(base_pred)
    residual = np.vstack([residual_centroids[label] for label in labels_eval])
    tangent = tangent_project(base_u, residual)
    return exp_map_sphere(base_u, tangent)


def evaluate_prediction(method, comparison, train_model, test_model, y_pred, y_true, labels_true):
    cos = cosine_rows(y_pred, y_true)
    retrieval = nearest_target_metrics(y_pred, y_true, labels_true)
    return {
        "comparison": comparison,
        "train_model": train_model,
        "test_model": test_model,
        "method": method,
        "mean_target_cosine": float(cos.mean()),
        "std_target_cosine": float(cos.std(ddof=1)),
        "median_target_cosine": float(np.median(cos)),
        **retrieval,
    }


def evaluate_direction(audit, comparison, train_model, test_model, train_x, train_y, test_x, test_y):
    train_idx = np.where(audit.train_mask)[0]
    test_idx = np.where(audit.test_mask)[0]
    class_ids = np.unique(audit.y)

    labels_train = audit.y[train_idx]
    labels_test = audit.y[test_idx]

    train_x_fit = train_x[train_idx]
    train_y_fit = train_y[train_idx]
    test_x_eval = test_x[test_idx]
    test_y_eval = test_y[test_idx]

    delta_centroids = learn_unit_delta_centroids(train_x_fit, train_y_fit, labels_train, class_ids)
    rise_prototypes = learn_rise_style(train_x_fit, train_y_fit, labels_train, class_ids)

    train_linear = predict_linear_delta(train_x_fit, delta_centroids, labels_train)
    train_spherical = predict_spherical_delta(train_x_fit, delta_centroids, labels_train)
    train_rise = predict_rise_style(train_x_fit, rise_prototypes, labels_train)

    linear_pred = predict_linear_delta(test_x_eval, delta_centroids, labels_test)
    spherical_pred = predict_spherical_delta(test_x_eval, delta_centroids, labels_test)
    rise_pred = predict_rise_style(test_x_eval, rise_prototypes, labels_test)

    rise_residual = learn_residual_centroids(train_rise, train_y_fit, labels_train, class_ids)
    linear_residual = learn_residual_centroids(train_linear, train_y_fit, labels_train, class_ids)
    spherical_residual = learn_residual_centroids(train_spherical, train_y_fit, labels_train, class_ids)

    rise_then_ours = apply_residual_spherical(rise_pred, rise_residual, labels_test)
    linear_then_rise_residual = apply_residual_spherical(linear_pred, linear_residual, labels_test)
    spherical_then_rise_residual = apply_residual_spherical(spherical_pred, spherical_residual, labels_test)
    hybrid_average = unit((spherical_pred + rise_pred) / 2.0)

    predictions = {
        "linear_delta": linear_pred,
        "spherical_delta": spherical_pred,
        "rise_only": rise_pred,
        "rise_then_ours_residual": rise_then_ours,
        "linear_then_residual": linear_then_rise_residual,
        "spherical_then_residual": spherical_then_rise_residual,
        "hybrid_average": hybrid_average,
    }

    return [
        evaluate_prediction(
            method=method,
            comparison=comparison,
            train_model=train_model,
            test_model=test_model,
            y_pred=pred,
            y_true=test_y_eval,
            labels_true=labels_test,
        )
        for method, pred in predictions.items()
    ]


def run_within_model(audit):
    rows = []
    for model_name in audit.procrustes_models:
        print(f"\nSpherical steering within-model: {model_name}", flush=True)
        sp = audit.spaces[model_name]
        rows.extend(evaluate_direction(
            audit=audit,
            comparison="within_model",
            train_model=model_name,
            test_model=model_name,
            train_x=sp["x_raw"],
            train_y=sp["y_raw"],
            test_x=sp["x_raw"],
            test_y=sp["y_raw"],
        ))
    return rows


def run_cross_model(audit):
    observed = pd.read_csv(CSV_DIR / "cross_model_procrustes.csv")
    rows = []

    for obs in observed.itertuples(index=False):
        train_model = obs.train_model
        test_model = obs.test_model
        if train_model == test_model:
            continue

        print(f"\nSpherical steering cross-model: {train_model} -> {test_model}", flush=True)
        source_aligned, target_aligned = make_full_anchor_alignment(audit, train_model, test_model)
        train_x, train_y = rows_from_vectors(audit, train_model, source_aligned)
        test_x, test_y = rows_from_vectors(audit, test_model, target_aligned)
        direction_rows = evaluate_direction(
            audit=audit,
            comparison="cross_model_full_anchor",
            train_model=train_model,
            test_model=test_model,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
        )
        for row in direction_rows:
            row.update({
                "observed_raw_f1": obs.raw_f1,
                "observed_aligned_f1": obs.aligned_f1,
            })
        rows.extend(direction_rows)
    return rows


def summarize(raw_df):
    return (
        raw_df
        .groupby(["comparison", "method"])
        .agg(
            mean_target_cosine=("mean_target_cosine", "mean"),
            std_target_cosine=("mean_target_cosine", "std"),
            mean_retrieval_top1_acc=("retrieval_top1_acc", "mean"),
            mean_retrieval_label_acc=("retrieval_label_acc", "mean"),
            mean_retrieval_label_f1=("retrieval_label_f1", "mean"),
            n=("mean_target_cosine", "count"),
        )
        .reset_index()
        .sort_values(["comparison", "mean_target_cosine"], ascending=[True, False])
    )


def plot_summary(summary_df):
    for metric, ylabel, filename in [
        ("mean_target_cosine", "Mean target cosine", "14_spherical_delta_target_cosine.png"),
        ("mean_retrieval_top1_acc", "Nearest-target top-1 accuracy", "14_spherical_delta_retrieval_top1.png"),
        ("mean_retrieval_label_f1", "Nearest-target label macro F1", "14_spherical_delta_retrieval_label_f1.png"),
    ]:
        pivot = summary_df.pivot(index="method", columns="comparison", values=metric)
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel(ylabel)
        ax.set_title("UPAT Spherical Delta Steering")
        ax.grid(axis="y", alpha=0.35)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=220, bbox_inches="tight")
        plt.close()


def main():
    parse_args()
    audit = UPATAudit()
    audit.build_dataset()
    audit.extract_all_spaces()

    cross_path = CSV_DIR / "cross_model_procrustes.csv"
    if not cross_path.exists():
        audit.run_cross_model_procrustes()

    rows = []
    rows.extend(run_within_model(audit))
    rows.extend(run_cross_model(audit))

    raw_df = pd.DataFrame(rows)
    summary_df = summarize(raw_df)

    raw_df.to_csv(CSV_DIR / "spherical_delta_steering_raw.csv", index=False)
    summary_df.to_csv(CSV_DIR / "spherical_delta_steering_summary.csv", index=False)
    plot_summary(summary_df)

    print(f"\nSaved: {CSV_DIR / 'spherical_delta_steering_raw.csv'}")
    print(f"Saved: {CSV_DIR / 'spherical_delta_steering_summary.csv'}")


if __name__ == "__main__":
    sys.exit(main())
