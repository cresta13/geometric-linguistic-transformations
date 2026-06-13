import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


load_dotenv()

def experiment_dir(env_name, default):
    value = os.getenv(env_name, default)
    if os.path.exists(value):
        return value
    migrated = os.path.join("results", "experiments", os.path.basename(value))
    if os.path.exists(migrated):
        return migrated
    return value


DIVERSE_DIR = experiment_dir("LIE_DIVERSE_DIR", "results/experiments/lie_llm_diverse_results")
FULL_DIR = experiment_dir("LIE_FULL_DIR", "results/experiments/lie_llm_full_semantic_holdout_results")
SYNTAX_DIR = experiment_dir("LIE_SYNTAX_DIR", "results/experiments/lie_llm_syntax_results")

OUT_DIR = os.getenv("LIE_FIGURES_DIR", "paper/figures")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_FOR_PROJECTION = os.getenv("LIE_PROJECTION_MODEL", "bert-base-uncased")


def safe_name(model_name):
    return model_name.replace("/", "__")


def save_bar_plot(df, x_col, y_col, title, ylabel, filename, hue_col=None):
    plt.figure(figsize=(10, 5))

    if hue_col is None:
        plt.bar(df[x_col].astype(str), df[y_col])
        plt.xticks(rotation=30, ha="right")
    else:
        pivot = df.pivot(index=x_col, columns=hue_col, values=y_col)
        pivot.plot(kind="bar", figsize=(11, 5))
        plt.xticks(rotation=30, ha="right")

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=220)
    plt.close()


def plot_full_semantic_accuracy():
    path = os.path.join(FULL_DIR, "full_semantic_holdout_summary.csv")
    if not os.path.exists(path):
        print("Missing:", path)
        return

    df = pd.read_csv(path)

    if "classifier" in df.columns:
        df = df[df["classifier"] == "linear_svc"].copy()

    save_bar_plot(
        df=df,
        x_col="model",
        y_col="accuracy",
        title="Full Semantic Holdout Accuracy",
        ylabel="Accuracy",
        filename="full_semantic_accuracy.png",
    )

    print("Saved full_semantic_accuracy.png")


def plot_diverse_separability():
    path = os.path.join(DIVERSE_DIR, "ALL_separability.csv")
    if not os.path.exists(path):
        print("Missing:", path)
        return

    df = pd.read_csv(path)

    save_bar_plot(
        df=df,
        x_col="model",
        y_col="separation_delta",
        title="Transformation Separability by Model",
        ylabel="Within-class cosine − between-class cosine",
        filename="diverse_separability.png",
    )

    print("Saved diverse_separability.png")


def plot_class_stability():
    path = os.path.join(DIVERSE_DIR, "ALL_class_stability.csv")
    if not os.path.exists(path):
        print("Missing:", path)
        return

    df = pd.read_csv(path)

    pivot = df.pivot(
        index="class",
        columns="model",
        values="mean_cosine_to_centroid",
    )

    pivot.plot(kind="bar", figsize=(14, 7))
    plt.ylabel("Mean cosine to class centroid")
    plt.title("Class Stability by Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_stability_by_model.png"), dpi=220)
    plt.close()

    print("Saved class_stability_by_model.png")


def load_delta_data(results_dir, model_name):
    safe = safe_name(model_name)

    delta_path = os.path.join(results_dir, f"delta_vectors_metrics_{safe}.csv")
    deltas_path = os.path.join(results_dir, f"deltas_{safe}.npy")

    if not os.path.exists(delta_path) or not os.path.exists(deltas_path):
        print("Missing projection files:")
        print(delta_path)
        print(deltas_path)
        return None, None

    delta_df = pd.read_csv(delta_path)
    D = np.load(deltas_path)

    return delta_df, D


def plot_pca_all_classes():
    delta_df, D = load_delta_data(FULL_DIR, MODEL_FOR_PROJECTION)
    if delta_df is None:
        return

    labels = delta_df["class"].astype(str).to_numpy()

    X = StandardScaler().fit_transform(D)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(9, 7))

    for cls in sorted(set(labels)):
        idx = labels == cls
        plt.scatter(Z[idx, 0], Z[idx, 1], s=12, alpha=0.55, label=cls)

    plt.title(f"PCA of Delta Vectors: {MODEL_FOR_PROJECTION}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"pca_all_classes_{safe_name(MODEL_FOR_PROJECTION)}.png"), dpi=220)
    plt.close()

    # Explained variance
    pca_full = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=42)
    pca_full.fit(X)

    plt.figure(figsize=(9, 5))
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"PCA Explained Variance: {MODEL_FOR_PROJECTION}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"pca_explained_variance_{safe_name(MODEL_FOR_PROJECTION)}.png"), dpi=220)
    plt.close()

    print("Saved PCA plots")


def plot_umap_all_classes():
    if not HAS_UMAP:
        print("UMAP not installed. Run: pip install umap-learn")
        return

    delta_df, D = load_delta_data(FULL_DIR, MODEL_FOR_PROJECTION)
    if delta_df is None:
        return

    labels = delta_df["class"].astype(str).to_numpy()

    X = StandardScaler().fit_transform(D)

    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.08,
        n_components=2,
        metric="cosine",
        random_state=42,
    )

    Z = reducer.fit_transform(X)

    plt.figure(figsize=(9, 7))

    for cls in sorted(set(labels)):
        idx = labels == cls
        plt.scatter(Z[idx, 0], Z[idx, 1], s=12, alpha=0.55, label=cls)

    plt.title(f"UMAP of Delta Vectors: {MODEL_FOR_PROJECTION}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"umap_all_classes_{safe_name(MODEL_FOR_PROJECTION)}.png"), dpi=220)
    plt.close()

    print("Saved UMAP all classes")


def plot_uncertainty_formalization_projection():
    delta_df, D = load_delta_data(FULL_DIR, MODEL_FOR_PROJECTION)
    if delta_df is None:
        return

    labels = delta_df["class"].astype(str).to_numpy()
    mask = np.isin(labels, ["uncertainty", "formalization"])

    sub_labels = labels[mask]
    X = StandardScaler().fit_transform(D[mask])

    # PCA
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for cls in sorted(set(sub_labels)):
        idx = sub_labels == cls
        plt.scatter(Z[idx, 0], Z[idx, 1], s=16, alpha=0.6, label=cls)

    plt.title(f"PCA: Uncertainty vs Formalization ({MODEL_FOR_PROJECTION})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"pca_uncertainty_formalization_{safe_name(MODEL_FOR_PROJECTION)}.png"), dpi=220)
    plt.close()

    # UMAP
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=20,
            min_dist=0.08,
            n_components=2,
            metric="cosine",
            random_state=42,
        )

        Z = reducer.fit_transform(X)

        plt.figure(figsize=(8, 6))
        for cls in sorted(set(sub_labels)):
            idx = sub_labels == cls
            plt.scatter(Z[idx, 0], Z[idx, 1], s=16, alpha=0.6, label=cls)

        plt.title(f"UMAP: Uncertainty vs Formalization ({MODEL_FOR_PROJECTION})")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"umap_uncertainty_formalization_{safe_name(MODEL_FOR_PROJECTION)}.png"), dpi=220)
        plt.close()

    print("Saved uncertainty/formalization plots")


def plot_confusion_matrices():
    models = ["bert-base-uncased", "roberta-base"]
    classifiers = ["linear_svc"]

    for model in models:
        for clf in classifiers:
            path = os.path.join(FULL_DIR, f"confusion_{safe_name(model)}_{clf}.csv")

            if not os.path.exists(path):
                print("Missing:", path)
                continue

            cm = pd.read_csv(path, index_col=0)

            plt.figure(figsize=(8, 7))
            plt.imshow(cm.values)
            plt.colorbar(label="Count")
            plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
            plt.yticks(range(len(cm.index)), cm.index)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(int(cm.values[i, j])), ha="center", va="center", fontsize=8)

            plt.title(f"Confusion Matrix: {model} ({clf})")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"confusion_{safe_name(model)}_{clf}.png"), dpi=220)
            plt.close()

    print("Saved confusion matrix plots")


def main():
    print("Output directory:", OUT_DIR)
    print("Projection model:", MODEL_FOR_PROJECTION)

    plot_full_semantic_accuracy()
    plot_diverse_separability()
    plot_class_stability()

    plot_pca_all_classes()
    plot_umap_all_classes()
    plot_uncertainty_formalization_projection()
    plot_confusion_matrices()

    print("\nDone.")
    print("Figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
