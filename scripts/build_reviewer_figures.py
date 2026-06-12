from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(path)


def plot_syntax_ablation():
    df = pd.read_csv(ROOT / "syntax_representation_ablation_results" / "syntax_representation_ablation_pivot.csv")
    df = df[df["classifier"] == "linear_svc"].set_index("model")[["x_only", "y_only", "concat", "delta"]]
    ax = df.plot(kind="bar", figsize=(9, 4.8), color=["#8a8f98", "#d95f02", "#7570b3", "#1b9e77"])
    ax.set_title("Syntax holdout representation ablation")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.06)
    ax.axhline(1 / 6, color="#444444", linestyle="--", linewidth=1, label="chance")
    ax.legend(loc="lower right", ncol=2)
    savefig("syntax_representation_ablation.png")


def plot_spotcheck():
    df = pd.read_csv(ROOT / "track1_spotcheck_results" / "spotcheck_representation_ablation_pivot.csv")
    df = df.set_index("classifier")[["x_only", "y_only", "concat", "delta"]]
    ax = df.plot(kind="bar", figsize=(7, 4.5), color=["#8a8f98", "#d95f02", "#7570b3", "#1b9e77"])
    ax.set_title("DeBERTa-v3-small representation spot-check")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.axhline(1 / 6, color="#444444", linestyle="--", linewidth=1, label="chance")
    ax.legend(loc="lower right", ncol=2)
    savefig("spotcheck_deberta_v3_small.png")


def plot_decoder_signed_permutation():
    df = pd.read_csv(ROOT / "lie_algebraic_identities_decoder_results" / "csv" / "jacobi_summary.csv")
    pivot = df.pivot(index="triple", columns="model", values="mean_jacobi_to_null_mean_ratio").sort_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0.5, vmax=1.5, aspect="auto")
    ax.set_title("Decoder signed permutation ratio to null")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i, triple in enumerate(pivot.index):
        for j, model in enumerate(pivot.columns):
            ax.text(j, i, f"{pivot.loc[triple, model]:.3f}", ha="center", va="center", fontsize=10)
    ax.axhline(2.5, color="white", linewidth=1.2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Observed / permutation-null mean")
    savefig("decoder_signed_permutation_ratio.png")


def main():
    plot_syntax_ablation()
    plot_spotcheck()
    plot_decoder_signed_permutation()


if __name__ == "__main__":
    main()
