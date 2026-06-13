import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

from run_upat_large import UPATAudit


OUT_DIR = Path("results/experiments/upat_large_results")
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare delta, MDV, and RISE-style prototype baselines on UPAT."
    )
    parser.add_argument("--seed", type=int, default=2718)
    return parser.parse_args()


def unit(x):
    return normalize(x)


def cosine_rows(a, b):
    return np.sum(unit(a) * unit(b), axis=1)


def log_map_sphere(x, y, eps=1e-7):
    dot = np.clip(np.sum(x * y, axis=1, keepdims=True), -1 + eps, 1 - eps)
    theta = np.arccos(dot)
    tangent = y - dot * x
    sin_theta = np.sin(theta)
    return tangent * (theta / np.maximum(sin_theta, eps))


def exp_map_sphere(x, tangent, eps=1e-7):
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    direction = tangent / np.maximum(norm, eps)
    out = np.cos(norm) * x + np.sin(norm) * direction
    small = norm[:, 0] < eps
    if np.any(small):
        out[small] = x[small]
    return unit(out)


def householder_to_e1_apply(x, z, eps=1e-7):
    e1 = np.zeros_like(x)
    e1[:, 0] = 1.0
    u = x - e1
    denom = np.sum(u * u, axis=1, keepdims=True)
    out = z.copy()

    regular = denom[:, 0] > eps
    if np.any(regular):
        u_reg = u[regular]
        z_reg = z[regular]
        denom_reg = denom[regular]
        out[regular] = z_reg - 2.0 * u_reg * (np.sum(u_reg * z_reg, axis=1, keepdims=True) / denom_reg)

    antipodal = (~regular) & (x[:, 0] < 0)
    if np.any(antipodal):
        out[antipodal, 0] *= -1.0

    return out


def fit_procrustes(source_anchor, target_anchor, source_all, target_all):
    source_mean = source_anchor.mean(axis=0, keepdims=True)
    target_mean = target_anchor.mean(axis=0, keepdims=True)

    source_anchor_c = source_anchor - source_mean
    target_anchor_c = target_anchor - target_mean
    source_all_c = source_all - source_mean
    target_all_c = target_all - target_mean

    min_dim = min(source_anchor_c.shape[1], target_anchor_c.shape[1], source_all_c.shape[1], target_all_c.shape[1])
    source_anchor_c = source_anchor_c[:, :min_dim]
    target_anchor_c = target_anchor_c[:, :min_dim]
    source_all_c = source_all_c[:, :min_dim]
    target_all_c = target_all_c[:, :min_dim]

    u, _, vt = svd(source_anchor_c.T @ target_anchor_c, full_matrices=False, check_finite=False)
    r = u @ vt
    return source_all_c @ r, target_all_c


def make_full_anchor_alignment(audit, train_model, test_model):
    source_sp = audit.spaces[train_model]
    target_sp = audit.spaces[test_model]
    common_texts = sorted(set(source_sp["texts"]) & set(target_sp["texts"]))
    source_idx = np.array([source_sp["idx"][text] for text in common_texts])
    target_idx = np.array([target_sp["idx"][text] for text in common_texts])
    return fit_procrustes(
        source_anchor=source_sp["raw"][source_idx],
        target_anchor=target_sp["raw"][target_idx],
        source_all=source_sp["raw"],
        target_all=target_sp["raw"],
    )


def rows_from_vectors(audit, model_name, vectors):
    sp = audit.spaces[model_name]
    idx = sp["idx"]
    x = np.array([vectors[idx[row.source]] for row in audit.df.itertuples(index=False)])
    y = np.array([vectors[idx[row.target]] for row in audit.df.itertuples(index=False)])
    return x, y


def learn_mdv_raw(x_train, y_train, labels_train, class_ids):
    deltas = y_train - x_train
    return {
        cls: deltas[labels_train == cls].mean(axis=0)
        for cls in class_ids
    }


def predict_mdv_raw(x_test, prototypes, labels_test):
    return np.vstack([x + prototypes[label] for x, label in zip(x_test, labels_test)])


def learn_mdv_unit(x_train, y_train, labels_train, class_ids):
    x_u = unit(x_train)
    y_u = unit(y_train)
    deltas = y_u - x_u
    return {
        cls: deltas[labels_train == cls].mean(axis=0)
        for cls in class_ids
    }


def predict_mdv_unit(x_test, prototypes, labels_test):
    x_u = unit(x_test)
    return unit(np.vstack([x + prototypes[label] for x, label in zip(x_u, labels_test)]))


def learn_rise_style(x_train, y_train, labels_train, class_ids):
    x_u = unit(x_train)
    y_u = unit(y_train)
    tangent = log_map_sphere(x_u, y_u)
    canonical = householder_to_e1_apply(x_u, tangent)
    prototypes = {}
    for cls in class_ids:
        proto = canonical[labels_train == cls].mean(axis=0)
        proto[0] = 0.0
        prototypes[cls] = proto
    return prototypes


def predict_rise_style(x_test, prototypes, labels_test):
    x_u = unit(x_test)
    proto = np.vstack([prototypes[label] for label in labels_test])
    tangent = householder_to_e1_apply(x_u, proto)
    return exp_map_sphere(x_u, tangent)


def nearest_target_metrics(y_pred, y_true, labels_true):
    y_pred_u = unit(y_pred)
    y_true_u = unit(y_true)
    sim = y_pred_u @ y_true_u.T
    nearest = sim.argmax(axis=1)
    pred_labels = labels_true[nearest]
    return {
        "retrieval_top1_acc": float(np.mean(nearest == np.arange(len(nearest)))),
        "retrieval_label_acc": accuracy_score(labels_true, pred_labels),
        "retrieval_label_f1": f1_score(labels_true, pred_labels, average="macro"),
    }


def classifier_transfer_metrics(audit, x_train, y_train, x_test, y_test, seed):
    train_delta = normalize(y_train - x_train)
    test_delta = normalize(y_test - x_test)
    pred = audit.fit_predict(
        train_delta[audit.train_mask],
        audit.y[audit.train_mask],
        test_delta[audit.test_mask],
        seed=seed,
    )
    return {
        "delta_classifier_acc": accuracy_score(audit.y[audit.test_mask], pred),
        "delta_classifier_f1": f1_score(audit.y[audit.test_mask], pred, average="macro"),
    }


def evaluate_prototype_method(audit, method, train_model, test_model, train_x, train_y, test_x, test_y, seed):
    train_idx = np.where(audit.train_mask)[0]
    test_idx = np.where(audit.test_mask)[0]
    class_ids = np.unique(audit.y)

    if method == "mdv_raw":
        prototypes = learn_mdv_raw(train_x[train_idx], train_y[train_idx], audit.y[train_idx], class_ids)
        pred_y = predict_mdv_raw(test_x[test_idx], prototypes, audit.y[test_idx])
    elif method == "mdv_unit":
        prototypes = learn_mdv_unit(train_x[train_idx], train_y[train_idx], audit.y[train_idx], class_ids)
        pred_y = predict_mdv_unit(test_x[test_idx], prototypes, audit.y[test_idx])
    elif method == "rise_style":
        prototypes = learn_rise_style(train_x[train_idx], train_y[train_idx], audit.y[train_idx], class_ids)
        pred_y = predict_rise_style(test_x[test_idx], prototypes, audit.y[test_idx])
    else:
        raise ValueError(f"Unknown method: {method}")

    y_true = test_y[test_idx]
    cos = cosine_rows(pred_y, y_true)
    retrieval = nearest_target_metrics(pred_y, y_true, audit.y[test_idx])

    return {
        "train_model": train_model,
        "test_model": test_model,
        "method": method,
        "mean_target_cosine": cos.mean(),
        "std_target_cosine": cos.std(ddof=1),
        "median_target_cosine": np.median(cos),
        **retrieval,
    }


def run_within_model(audit, seed):
    rows = []
    for model_name in audit.procrustes_models:
        print(f"\nWithin-model prototype comparison: {model_name}", flush=True)
        sp = audit.spaces[model_name]
        train_x = sp["x_raw"]
        train_y = sp["y_raw"]
        test_x = sp["x_raw"]
        test_y = sp["y_raw"]
        clf = classifier_transfer_metrics(audit, train_x, train_y, test_x, test_y, seed=seed)
        for method in ["mdv_raw", "mdv_unit", "rise_style"]:
            row = evaluate_prototype_method(
                audit, method, model_name, model_name, train_x, train_y, test_x, test_y, seed
            )
            row.update({"comparison": "within_model", **clf})
            rows.append(row)
    return rows


def run_cross_model(audit, seed):
    observed = pd.read_csv(CSV_DIR / "cross_model_procrustes.csv")
    rows = []

    for obs in observed.itertuples(index=False):
        train_model = obs.train_model
        test_model = obs.test_model
        if train_model == test_model:
            continue

        print(f"\nCross-model prototype comparison: {train_model} -> {test_model}", flush=True)
        source_aligned, target_aligned = make_full_anchor_alignment(audit, train_model, test_model)
        train_x, train_y = rows_from_vectors(audit, train_model, source_aligned)
        test_x, test_y = rows_from_vectors(audit, test_model, target_aligned)
        clf = classifier_transfer_metrics(audit, train_x, train_y, test_x, test_y, seed=seed)

        for method in ["mdv_raw", "mdv_unit", "rise_style"]:
            row = evaluate_prototype_method(
                audit, method, train_model, test_model, train_x, train_y, test_x, test_y, seed
            )
            row.update({
                "comparison": "cross_model_full_anchor",
                "observed_raw_f1": obs.raw_f1,
                "observed_aligned_f1": obs.aligned_f1,
                **clf,
            })
            rows.append(row)
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
            mean_delta_classifier_f1=("delta_classifier_f1", "mean"),
            n=("mean_target_cosine", "count"),
        )
        .reset_index()
    )


def plot_summary(summary_df):
    for metric, ylabel, filename in [
        ("mean_target_cosine", "Mean target cosine", "12_rise_aware_target_cosine.png"),
        ("mean_retrieval_label_f1", "Nearest-target label macro F1", "12_rise_aware_retrieval_f1.png"),
    ]:
        pivot = summary_df.pivot(index="method", columns="comparison", values=metric)
        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_ylabel(ylabel)
        ax.set_title("UPAT RISE-aware prototype comparison")
        ax.grid(axis="y", alpha=0.35)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=220, bbox_inches="tight")
        plt.close()


def main():
    args = parse_args()
    audit = UPATAudit()
    audit.build_dataset()
    audit.extract_all_spaces()

    cross_path = CSV_DIR / "cross_model_procrustes.csv"
    if not cross_path.exists():
        audit.run_cross_model_procrustes()

    raw_rows = []
    raw_rows.extend(run_within_model(audit, seed=args.seed))
    raw_rows.extend(run_cross_model(audit, seed=args.seed))

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize(raw_df)
    raw_df.to_csv(CSV_DIR / "rise_aware_comparison_raw.csv", index=False)
    summary_df.to_csv(CSV_DIR / "rise_aware_comparison_summary.csv", index=False)
    plot_summary(summary_df)

    print(f"\nSaved: {CSV_DIR / 'rise_aware_comparison_raw.csv'}")
    print(f"Saved: {CSV_DIR / 'rise_aware_comparison_summary.csv'}")


if __name__ == "__main__":
    sys.exit(main())
