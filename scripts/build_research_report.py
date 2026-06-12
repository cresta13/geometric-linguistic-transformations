from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "2026-06-12_reviewer_revised_report.pdf"


plt.rcParams["font.family"] = "DejaVu Sans"


def add_text_page(pdf, title, paragraphs, fontsize=10):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
    ax.axis("off")

    y = 1.0
    ax.text(0, y, title, fontsize=18, fontweight="bold", va="top")
    y -= 0.06

    for para in paragraphs:
        if isinstance(para, tuple):
            header, body = para
            ax.text(0, y, header, fontsize=12, fontweight="bold", va="top")
            y -= 0.035
            lines = textwrap.wrap(body, width=96)
        else:
            lines = []
            for chunk in str(para).splitlines():
                lines.extend(textwrap.wrap(chunk, width=100) or [""])

        for line in lines:
            ax.text(0, y, line, fontsize=fontsize, va="top")
            y -= 0.022
            if y < 0.04:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
                ax.axis("off")
                y = 1.0
        y -= 0.018

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf, title, df, max_rows=30):
    visible = df.head(max_rows).copy()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.88])
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    table = ax.table(
        cellText=visible.values,
        colLabels=visible.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf, title, path):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.88])
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    img = plt.imread(path)
    ax.imshow(img)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_code_listing(pdf, path, title, lines_per_page=54):
    lines = path.read_text(encoding="utf-8").splitlines()
    numbered = [f"{i + 1:04d}: {line}" for i, line in enumerate(lines)]

    for page_idx in range(0, len(numbered), lines_per_page):
        chunk = numbered[page_idx:page_idx + lines_per_page]
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
        ax.axis("off")
        ax.text(
            0,
            1,
            f"{title} ({page_idx // lines_per_page + 1})",
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        y = 0.965
        for line in chunk:
            ax.text(0, y, line[:132], fontsize=6.6, family="monospace", va="top")
            y -= 0.0172
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def format_float_columns(df):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    jacobi = pd.read_csv(ROOT / "lie_algebraic_identities_results" / "csv" / "jacobi_summary.csv")
    jacobi = jacobi[
        [
            "model",
            "triple",
            "mean_relative_jacobi_norm",
            "relative_jacobi_norm_ci95_low",
            "relative_jacobi_norm_ci95_high",
            "mean_jacobi_to_null_mean_ratio",
            "jacobi_to_null_mean_ratio_ci95_low",
            "jacobi_to_null_mean_ratio_ci95_high",
            "mean_null_percentile_smaller_is_better",
            "n",
        ]
    ]

    antisym = pd.read_csv(ROOT / "lie_algebraic_identities_results" / "csv" / "antisymmetry_summary.csv")
    semantic = pd.read_csv(ROOT / "lie_semantic_equivalence_results" / "csv" / "semantic_equivalence_summary.csv")
    semantic_effects = pd.read_csv(ROOT / "lie_semantic_equivalence_results" / "csv" / "reviewer_semantic_effect_sizes.csv")
    composition = pd.read_csv(ROOT / "lie_composition_results" / "csv" / "lie_composition_summary.csv")
    ablation = pd.read_csv(ROOT / "results" / "reviewer_ablation_table.csv")
    syntax_ablation = pd.read_csv(ROOT / "syntax_representation_ablation_results" / "syntax_representation_ablation_pivot.csv")
    layerwise = pd.read_csv(ROOT / "layerwise_pooling_ablation_results" / "layerwise_pooling_ablation_top20.csv")
    spotcheck = pd.read_csv(ROOT / "track1_spotcheck_results" / "spotcheck_representation_ablation_pivot.csv")
    decoder_jacobi = pd.read_csv(ROOT / "lie_algebraic_identities_decoder_results" / "csv" / "jacobi_summary.csv")

    with PdfPages(OUT) as pdf:
        add_text_page(
            pdf,
            "Reviewer-Revised Research Report: Transformation Vectors and Composition Diagnostics",
            [
                ("Date", "2026-06-12"),
                (
                    "Purpose",
                    "Fix the reviewer-revised research state for external verification. This version incorporates AC-level review concerns: endpoint leakage, syntax-holdout overclaiming, McNemar tests, semantic-control statistics, and the terminology change to a third-order signed permutation coherence test.",
                ),
                (
                    "Track 1 fixed result",
                    "Delta vectors add reproducible information beyond target-only endpoints in the multiseed full-semantic setting. The syntax=1.0 result has now been directly ablated: y_only also reaches 1.0, so the syntax split is interpreted as endpoint/surface leakage rather than as geometry.",
                ),
                (
                    "Track 2 fixed result",
                    "Composition order matters. Semantic equivalence shifts noncommutativity distributions with strong statistical tests. The third-order result is now framed as signed permutation coherence, not a formal Jacobi identity. Decoder replication now supports QMT below-null cancellation for GPT-2 and DistilGPT-2.",
                ),
                (
                    "Main conclusion",
                    "The evidence supports a local QMT signed-permutation cancellation effect across tested encoder models, while negation-heavy triples expose the boundary of the phenomenon. The current work is promising but not submission-ready without the listed blocker experiments.",
                ),
            ],
        )

        add_table_page(pdf, "Delta vs endpoint ablation with McNemar evidence", format_float_columns(ablation), max_rows=10)
        add_table_page(pdf, "Syntax representation ablation: y_only solves the split", format_float_columns(syntax_ablation), max_rows=12)
        add_table_page(pdf, "Layerwise/pooling syntax sanity check: top rows", format_float_columns(layerwise), max_rows=20)
        add_table_page(pdf, "DeBERTa-v3-small modern spot-check", format_float_columns(spotcheck), max_rows=5)
        add_table_page(pdf, "Third-order signed permutation summary with bootstrap CI", format_float_columns(jacobi), max_rows=20)
        add_table_page(pdf, "Decoder signed permutation replication", format_float_columns(decoder_jacobi), max_rows=10)
        add_table_page(pdf, "Antisymmetry sanity check", format_float_columns(antisym), max_rows=30)
        add_table_page(pdf, "Semantic equivalence control", format_float_columns(semantic), max_rows=12)
        add_table_page(pdf, "Semantic equivalence effect sizes", format_float_columns(semantic_effects), max_rows=10)
        add_table_page(
            pdf,
            "Composition summary: strongest noncommutativity rows",
            format_float_columns(composition.sort_values("mean_noncommutativity", ascending=False).head(18)),
            max_rows=18,
        )

        figures = [
            ("Signed permutation relative norm", ROOT / "lie_algebraic_identities_results" / "figures" / "03_jacobi_relative_norm_heatmap.png"),
            ("Signed permutation norm versus permutation-null", ROOT / "lie_algebraic_identities_results" / "figures" / "04_jacobi_vs_permutation_null_heatmap.png"),
            ("Semantic equivalent vs non-equivalent control", ROOT / "lie_semantic_equivalence_results" / "figures" / "01_equivalent_vs_nonequivalent.png"),
            ("Composition noncommutativity heatmap", ROOT / "lie_composition_results" / "figures" / "01_noncommutativity_heatmap.png"),
            ("Original holdout accuracy comparison", ROOT / "paper" / "figures" / "holdout_accuracy_comparison.png"),
            ("Original PCA class geometry", ROOT / "paper" / "figures" / "pca_all_classes_bert-base-uncased.png"),
            ("Syntax representation ablation", ROOT / "paper" / "figures" / "syntax_representation_ablation.png"),
            ("DeBERTa-v3-small spot-check", ROOT / "paper" / "figures" / "spotcheck_deberta_v3_small.png"),
            ("Decoder signed permutation replication", ROOT / "paper" / "figures" / "decoder_signed_permutation_ratio.png"),
        ]

        for title, path in figures:
            if path.exists():
                add_image_page(pdf, title, path)

        add_text_page(
            pdf,
            "Audit notes for external verifier",
            [
                "Antisymmetry is not treated as evidence because the implementation defines [B,A] as the negative order of [A,B].",
                "A literal nested-commutator Jacobi expression over the same six endpoint vectors cancels algebraically. The reported third-order diagnostic is the non-tautological signed permutation composition sum ABC+BCA+CAB-ACB-CBA-BAC.",
                "Duplicate endpoint templates were detected and removed before finalizing today's result. The final dataset has 400 Jacobi rows and zero duplicate endpoint sets.",
                "New Track 1 result: syntax y_only=1.0 confirms target/surface leakage for the syntax split. New Track 1 spot-check: DeBERTa-v3-small supports delta superiority for Linear SVC but not for logistic regression. New Track 2 result: GPT-2 and DistilGPT-2 both keep QMT below permutation null, while negation triples remain mixed.",
                "Remaining blockers: representation-ablation tables for every non-syntax holdout split, full-semantic pooling ablation, grammar-generated templates, endpoint-only controls for Track 2, focused negation analysis, and prior-work positioning.",
            ],
        )

        add_code_listing(pdf, ROOT / "run_lie_algebraic_identities.py", "Code listing: run_lie_algebraic_identities.py")
        add_code_listing(pdf, ROOT / "run_syntax_representation_ablation.py", "Code listing: run_syntax_representation_ablation.py")
        add_code_listing(pdf, ROOT / "run_layerwise_pooling_ablation.py", "Code listing: run_layerwise_pooling_ablation.py")
        add_code_listing(pdf, ROOT / "run_track1_spotcheck.py", "Code listing: run_track1_spotcheck.py")

    print(OUT)


if __name__ == "__main__":
    main()
