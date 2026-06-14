from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "reports" / "2026-06-13_reviewer_revised_report.pdf"


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

    jacobi = pd.read_csv(EXP / "lie_algebraic_identities_results" / "csv" / "jacobi_summary.csv")
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

    semantic = pd.read_csv(EXP / "lie_semantic_equivalence_results" / "csv" / "semantic_equivalence_summary.csv")
    semantic_effects = pd.read_csv(EXP / "lie_semantic_equivalence_results" / "csv" / "reviewer_semantic_effect_sizes.csv")
    composition = pd.read_csv(EXP / "lie_composition_results" / "csv" / "lie_composition_summary.csv")
    ablation = pd.read_csv(ROOT / "results" / "reviewer_ablation_table.csv")
    syntax_ablation = pd.read_csv(EXP / "syntax_representation_ablation_results" / "syntax_representation_ablation_pivot.csv")
    layerwise = pd.read_csv(EXP / "layerwise_pooling_ablation_results" / "layerwise_pooling_ablation_top20.csv")
    spotcheck = pd.read_csv(EXP / "track1_spotcheck_results" / "spotcheck_representation_ablation_pivot.csv")
    large_spotcheck_path = EXP / "track1_spotcheck_large_results" / "spotcheck_representation_ablation_pivot.csv"
    large_spotcheck = pd.read_csv(large_spotcheck_path) if large_spotcheck_path.exists() else pd.DataFrame()
    upat_ablation = pd.read_csv(EXP / "upat_audit_results" / "csv" / "ablation.csv")
    upat_mcnemar = pd.read_csv(EXP / "upat_audit_results" / "csv" / "mcnemar_delta_vs_y.csv")
    procrustes_null_path = EXP / "upat_large_results" / "csv" / "procrustes_null_summary.csv"
    procrustes_null = pd.read_csv(procrustes_null_path) if procrustes_null_path.exists() else pd.DataFrame()
    heldout_alignment_path = EXP / "upat_large_results" / "csv" / "heldout_alignment_curve_summary.csv"
    heldout_alignment = pd.read_csv(heldout_alignment_path) if heldout_alignment_path.exists() else pd.DataFrame()
    heldout_alignment_by_direction_path = EXP / "upat_large_results" / "csv" / "heldout_alignment_curve_by_direction.csv"
    heldout_alignment_by_direction = (
        pd.read_csv(heldout_alignment_by_direction_path)
        if heldout_alignment_by_direction_path.exists()
        else pd.DataFrame()
    )
    rise_aware_path = EXP / "upat_large_results" / "csv" / "rise_aware_comparison_summary.csv"
    rise_aware = pd.read_csv(rise_aware_path) if rise_aware_path.exists() else pd.DataFrame()
    hybrid_rise_path = EXP / "upat_large_results" / "csv" / "hybrid_rise_procrustes_summary.csv"
    hybrid_rise = pd.read_csv(hybrid_rise_path) if hybrid_rise_path.exists() else pd.DataFrame()
    spherical_delta_path = EXP / "upat_large_results" / "csv" / "spherical_delta_steering_summary.csv"
    spherical_delta = pd.read_csv(spherical_delta_path) if spherical_delta_path.exists() else pd.DataFrame()
    pooling = pd.read_csv(EXP / "full_semantic_pooling_ablation_results" / "full_semantic_pooling_ablation_pivot.csv")
    confusion_neg = pd.read_csv(ROOT / "results" / "confusion_negation_summary.csv")
    decoder_jacobi = pd.read_csv(EXP / "lie_algebraic_identities_decoder_results" / "csv" / "jacobi_summary.csv")
    decoder_composition = pd.read_csv(EXP / "lie_composition_decoder_results" / "csv" / "lie_composition_summary.csv")
    multiple_testing = pd.read_csv(EXP / "lie_algebraic_identities_results" / "csv" / "signed_permutation_multiple_testing.csv")

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
                    "Delta vectors add reproducible information beyond target-only endpoints in the multiseed full-semantic setting. The syntax=1.0 result has now been directly ablated: y_only also reaches 1.0, so the syntax split is interpreted as endpoint/surface leakage rather than as geometry. BERT-large and DeBERTa-v3-base spot-checks both rank delta best under Linear SVC and logistic regression. UPAT is now reported as a hard-holdout boundary where delta is not reliably better than y_only.",
                ),
                (
                    "Track 2 fixed result",
                    "Composition order matters. Semantic equivalence shifts noncommutativity distributions with strong statistical tests. The third-order result is now framed as signed permutation coherence, not a formal Jacobi identity. Decoder pairwise composition is now included. After multiple-testing correction, QMT is the only below-null triple that is stable across all five tested models.",
                ),
                (
                    "Main conclusion",
                    "The evidence supports a local QMT signed-permutation cancellation effect across tested encoder models, while negation-heavy triples expose the boundary of the phenomenon. RISE (Freenor and Alvarez, ICLR 2026) is now treated as the closest prior work for spherical/geodesic semantic-syntactic transformations across languages and embedding models. Our Track 3 result is therefore framed as a stress test: UPAT-large cross-model Procrustes transfer survives N=1000 random-label, random-pairing, random-orthogonal, and held-out-anchor controls. The first RISE-aware comparison shows strong within-model prototype target prediction, harder cross-model target prediction, and higher aligned delta-classifier transfer on its own class-discrimination metric.",
                ),
            ],
        )

        add_table_page(pdf, "Delta vs endpoint ablation with McNemar evidence", format_float_columns(ablation), max_rows=10)
        add_table_page(pdf, "Syntax representation ablation: y_only solves the split", format_float_columns(syntax_ablation), max_rows=12)
        add_table_page(pdf, "Layerwise/pooling syntax sanity check: top rows", format_float_columns(layerwise), max_rows=20)
        add_table_page(pdf, "DeBERTa-v3-small modern spot-check", format_float_columns(spotcheck), max_rows=5)
        if not large_spotcheck.empty:
            add_table_page(pdf, "BERT-large and DeBERTa-v3-base spot-checks", format_float_columns(large_spotcheck), max_rows=8)
        add_table_page(pdf, "UPAT hard-holdout representation ablation", format_float_columns(upat_ablation), max_rows=25)
        add_table_page(pdf, "UPAT delta vs y_only McNemar tests", format_float_columns(upat_mcnemar), max_rows=10)
        if not procrustes_null.empty:
            add_table_page(pdf, "UPAT-large Procrustes N=1000 null controls", format_float_columns(procrustes_null), max_rows=30)
        if not heldout_alignment.empty:
            add_table_page(pdf, "UPAT-large held-out alignment-size curve", format_float_columns(heldout_alignment), max_rows=10)
        if not heldout_alignment_by_direction.empty:
            largest_size = heldout_alignment_by_direction["alignment_size"].max()
            largest = heldout_alignment_by_direction[
                heldout_alignment_by_direction["alignment_size"] == largest_size
            ].sort_values("mean_gap_to_full_anchor")
            add_table_page(
                pdf,
                "UPAT-large held-out alignment by direction: largest anchor size",
                format_float_columns(largest),
                max_rows=30,
            )
        if not rise_aware.empty:
            add_table_page(pdf, "UPAT RISE-aware prototype comparison", format_float_columns(rise_aware), max_rows=10)
        if not hybrid_rise.empty:
            add_table_page(pdf, "UPAT hybrid RISE-Procrustes transfer", format_float_columns(hybrid_rise), max_rows=20)
        if not spherical_delta.empty:
            add_table_page(pdf, "UPAT spherical delta steering", format_float_columns(spherical_delta), max_rows=20)
        add_table_page(pdf, "Full-semantic pooling ablation", format_float_columns(pooling), max_rows=25)
        add_table_page(pdf, "Confusion analysis: negation vs non-negation recall", format_float_columns(confusion_neg), max_rows=25)
        add_table_page(pdf, "Third-order signed permutation summary with bootstrap CI", format_float_columns(jacobi), max_rows=20)
        add_table_page(pdf, "Decoder signed permutation replication", format_float_columns(decoder_jacobi), max_rows=10)
        add_table_page(pdf, "Signed permutation multiple-testing correction", format_float_columns(multiple_testing), max_rows=25)
        add_table_page(pdf, "Decoder pairwise composition summary", format_float_columns(decoder_composition), max_rows=15)
        add_table_page(pdf, "Semantic equivalence control", format_float_columns(semantic), max_rows=12)
        add_table_page(pdf, "Semantic equivalence effect sizes", format_float_columns(semantic_effects), max_rows=10)
        add_table_page(
            pdf,
            "Composition summary: strongest noncommutativity rows",
            format_float_columns(composition.sort_values("mean_noncommutativity", ascending=False).head(18)),
            max_rows=18,
        )

        figures = [
            ("Signed permutation relative norm", EXP / "lie_algebraic_identities_results" / "figures" / "03_jacobi_relative_norm_heatmap.png"),
            ("Signed permutation norm versus permutation-null", EXP / "lie_algebraic_identities_results" / "figures" / "04_jacobi_vs_permutation_null_heatmap.png"),
            ("Semantic equivalent vs non-equivalent control", EXP / "lie_semantic_equivalence_results" / "figures" / "01_equivalent_vs_nonequivalent.png"),
            ("Composition noncommutativity heatmap", EXP / "lie_composition_results" / "figures" / "01_noncommutativity_heatmap.png"),
            ("Original PCA class geometry", ROOT / "paper" / "figures" / "pca_all_classes_bert-base-uncased.png"),
            ("Syntax representation ablation", ROOT / "paper" / "figures" / "syntax_representation_ablation.png"),
            ("DeBERTa-v3-small spot-check", ROOT / "paper" / "figures" / "spotcheck_deberta_v3_small.png"),
            ("Large/modern spot-check: Linear SVC", ROOT / "paper" / "figures" / "spotcheck_large_linear_svc.png"),
            ("Large/modern spot-check: logistic regression", ROOT / "paper" / "figures" / "spotcheck_large_logreg.png"),
            ("UPAT hard-holdout ablation", ROOT / "paper" / "figures" / "upat_representation_ablation.png"),
            ("UPAT-large Procrustes random-pairing null", EXP / "upat_large_results" / "figures" / "10_procrustes_null_random_pairing.png"),
            ("UPAT-large Procrustes random-label null", EXP / "upat_large_results" / "figures" / "10_procrustes_null_random_labels.png"),
            ("UPAT-large Procrustes random-orthogonal null", EXP / "upat_large_results" / "figures" / "10_procrustes_null_random_orthogonal.png"),
            ("UPAT-large held-out alignment-size curve", EXP / "upat_large_results" / "figures" / "11_heldout_alignment_size_curve.png"),
            ("UPAT-large held-out alignment by direction", EXP / "upat_large_results" / "figures" / "11_heldout_alignment_by_direction.png"),
            ("UPAT RISE-aware target prediction cosine", EXP / "upat_large_results" / "figures" / "12_rise_aware_target_cosine.png"),
            ("UPAT RISE-aware nearest-target class retrieval", EXP / "upat_large_results" / "figures" / "12_rise_aware_retrieval_f1.png"),
            ("UPAT hybrid RISE-Procrustes label F1", EXP / "upat_large_results" / "figures" / "13_hybrid_rise_procrustes_f1.png"),
            ("UPAT hybrid RISE-Procrustes heatmap", EXP / "upat_large_results" / "figures" / "13_hybrid_rise_procrustes_heatmap.png"),
            ("UPAT spherical delta target cosine", EXP / "upat_large_results" / "figures" / "14_spherical_delta_target_cosine.png"),
            ("UPAT spherical delta retrieval top-1", EXP / "upat_large_results" / "figures" / "14_spherical_delta_retrieval_top1.png"),
            ("UPAT spherical delta retrieval label F1", EXP / "upat_large_results" / "figures" / "14_spherical_delta_retrieval_label_f1.png"),
            ("Full-semantic pooling ablation: Linear SVC", ROOT / "paper" / "figures" / "full_semantic_pooling_linear_svc.png"),
            ("Full-semantic pooling ablation: logistic regression", ROOT / "paper" / "figures" / "full_semantic_pooling_logreg.png"),
            ("Confusion analysis: negation recall Linear SVC", ROOT / "paper" / "figures" / "confusion_negation_linear_svc.png"),
            ("Confusion analysis: negation recall logistic regression", ROOT / "paper" / "figures" / "confusion_negation_logreg.png"),
            ("Decoder signed permutation replication", ROOT / "paper" / "figures" / "decoder_signed_permutation_ratio.png"),
            ("Decoder pairwise composition", EXP / "lie_composition_decoder_results" / "figures" / "01_noncommutativity_heatmap.png"),
        ]

        for title, path in figures:
            if path.exists():
                add_image_page(pdf, title, path)

        add_text_page(
            pdf,
            "Audit notes for external verifier",
            [
                "Antisymmetry is not reported as evidence because the implementation defines [B,A] as the negative order of [A,B]. It is a tautological implementation check only.",
                "A literal nested-commutator Jacobi expression over the same six endpoint vectors cancels algebraically. The reported third-order diagnostic is the non-tautological signed permutation composition sum ABC+BCA+CAB-ACB-CBA-BAC.",
                "Duplicate endpoint templates were detected and removed before finalizing today's result. The final dataset has 400 Jacobi rows and zero duplicate endpoint sets.",
                "New Track 1 result: syntax y_only=1.0 confirms target/surface leakage for the syntax split. New Track 1 spot-check: DeBERTa-v3-small supports delta superiority for Linear SVC but not for logistic regression. New Track 2 result: GPT-2 and DistilGPT-2 both keep QMT below permutation null, while negation triples remain mixed.",
                "2026-06-13 update: after cache cleanup, BERT-large and DeBERTa-v3-base were both run successfully. Delta is best for both Linear SVC and logistic regression on both models.",
                "Second 2026-06-13 update: UPAT is now explicitly reported as a hard-holdout boundary condition. Full-semantic pooling ablation, confusion/negation analysis, decoder pairwise composition, and signed-permutation multiple-testing correction are included.",
                "2026-06-14 update: UPAT-large Procrustes null controls are scaled to N=1000 and now include random-label, random-pairing, and random-orthogonal baselines. The observed aligned F1 exceeds every null repeat across all non-identity directions.",
                "Second 2026-06-14 update: held-out alignment-size controls are now added. Procrustes maps fitted on auxiliary anchor texts disjoint from classifier train/test texts recover most of the full-anchor transfer effect.",
                "Third 2026-06-14 update: RISE is now the central related-work anchor. Track 3 is reframed as RISE-aware stress testing rather than first-discovery cross-model geometry. A RISE/MDV-style UPAT comparison is now a required gate.",
                "Fourth 2026-06-14 update: first-pass RISE-aware UPAT comparison is now added. MDV and simplified RISE-style prototype methods predict targets well within-model, but cross-model target prediction remains harder than aligned delta-classifier transfer on the class-discrimination metric.",
                "Fifth 2026-06-14 update: a non-leaky Hybrid RISE-Procrustes transfer test is now added. It scores every pair against all class prototypes, then tests prototype-score and delta+prototype-score features for cross-model transformation-label F1. The hybrid does not improve over aligned delta_only, suggesting that target-reconstructive prototype geometry and class-discriminative delta geometry are not automatically complementary under this metric.",
                "Sixth 2026-06-14 update: movement-level spherical delta steering is now added. Linear centroid steering is best for cross-model target cosine, while RISE-style prediction is best for nearest-target label F1. Uncalibrated spherical delta movement slightly improves label F1 over linear deltas but lowers target cosine.",
                "Remaining blockers: confidence intervals and anchor-domain diversity for Track 3, stronger/faithful RISE comparison if feasible, resolve or expand UPAT hard-holdout, representation-ablation tables for every non-syntax holdout split, grammar-generated templates, endpoint-only controls for Track 2, composition layer/pooling ablations, and final bibliography formatting.",
            ],
        )

        response_path = ROOT / "paper" / "reviewer_response_round3.md"
        if response_path.exists():
            add_text_page(
                pdf,
                "Round-3 Reviewer Response",
                [response_path.read_text(encoding="utf-8")],
                fontsize=8,
            )

        roadmap_path = ROOT / "paper" / "research_program.md"
        if roadmap_path.exists():
            add_text_page(
                pdf,
                "Updated Research Roadmap",
                [roadmap_path.read_text(encoding="utf-8")],
                fontsize=7.5,
            )

        related_work_path = ROOT / "paper" / "related_work_positioning.md"
        if related_work_path.exists():
            add_text_page(
                pdf,
                "RISE Related-Work Positioning",
                [related_work_path.read_text(encoding="utf-8")],
                fontsize=8,
            )

        add_code_listing(pdf, ROOT / "scripts" / "run_lie_algebraic_identities.py", "Code listing: run_lie_algebraic_identities.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_syntax_representation_ablation.py", "Code listing: run_syntax_representation_ablation.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_layerwise_pooling_ablation.py", "Code listing: run_layerwise_pooling_ablation.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_track1_spotcheck.py", "Code listing: run_track1_spotcheck.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_full_semantic_pooling_ablation.py", "Code listing: run_full_semantic_pooling_ablation.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_upat_procrustes_nulls.py", "Code listing: run_upat_procrustes_nulls.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_upat_alignment_size_heldout.py", "Code listing: run_upat_alignment_size_heldout.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_upat_rise_aware_comparison.py", "Code listing: run_upat_rise_aware_comparison.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_upat_hybrid_rise_procrustes.py", "Code listing: run_upat_hybrid_rise_procrustes.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_upat_spherical_delta_steering.py", "Code listing: run_upat_spherical_delta_steering.py")
        add_code_listing(pdf, ROOT / "scripts" / "analyze_confusion_negation.py", "Code listing: analyze_confusion_negation.py")
        add_code_listing(pdf, ROOT / "scripts" / "build_signed_permutation_multiple_testing.py", "Code listing: build_signed_permutation_multiple_testing.py")

    print(OUT)


if __name__ == "__main__":
    main()
