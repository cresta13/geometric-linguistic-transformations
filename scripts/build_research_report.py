from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "reports" / "2026-06-23_reviewer_revised_report.pdf"


plt.rcParams["font.family"] = "DejaVu Sans"

MODEL_ALIASES = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "multi-mpnet",
    "sentence-transformers/LaBSE": "LaBSE",
    "intfloat/multilingual-e5-large": "e5-large",
    "BAAI/bge-m3": "bge-m3",
    "bert-base-multilingual-cased": "mBERT",
}


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
    for col in ["model", "source_model", "target_model"]:
        if col in visible.columns:
            visible[col] = visible[col].map(lambda x: MODEL_ALIASES.get(x, x))
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
    numbered = [
        f"{i + 1:04d}: {line.encode('unicode_escape').decode('ascii')}"
        for i, line in enumerate(lines)
    ]

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
    grammar_composition_path = EXP / "lie_composition_grammar_results" / "csv" / "grammar_composition_summary.csv"
    grammar_composition = pd.read_csv(grammar_composition_path) if grammar_composition_path.exists() else pd.DataFrame()
    grammar_controls_path = EXP / "lie_composition_grammar_results" / "csv" / "grammar_endpoint_controls.csv"
    grammar_controls = pd.read_csv(grammar_controls_path) if grammar_controls_path.exists() else pd.DataFrame()
    grammar_nulls_path = EXP / "lie_composition_grammar_results" / "csv" / "grammar_commutator_nulls.csv"
    grammar_nulls = pd.read_csv(grammar_nulls_path) if grammar_nulls_path.exists() else pd.DataFrame()
    multilingual_dir = EXP / "lie_multilingual_max_results"
    multilingual_triple_path = multilingual_dir / "csv" / "triple_global_summary.csv"
    multilingual_triple = pd.read_csv(multilingual_triple_path) if multilingual_triple_path.exists() else pd.DataFrame()
    multilingual_triple_by_model_path = multilingual_dir / "csv" / "triple_by_model_summary.csv"
    multilingual_triple_by_model = (
        pd.read_csv(multilingual_triple_by_model_path)
        if multilingual_triple_by_model_path.exists()
        else pd.DataFrame()
    )
    multilingual_pair_path = multilingual_dir / "csv" / "pair_global_summary.csv"
    multilingual_pair = pd.read_csv(multilingual_pair_path) if multilingual_pair_path.exists() else pd.DataFrame()
    multilingual_endpoint_path = multilingual_dir / "csv" / "endpoint_controls_summary.csv"
    multilingual_endpoint = (
        pd.read_csv(multilingual_endpoint_path)
        if multilingual_endpoint_path.exists()
        else pd.DataFrame()
    )
    multilingual_centroid_path = multilingual_dir / "csv" / "cross_language_centroid_summary.csv"
    multilingual_centroid = (
        pd.read_csv(multilingual_centroid_path)
        if multilingual_centroid_path.exists()
        else pd.DataFrame()
    )
    ablation = pd.read_csv(ROOT / "results" / "reviewer_ablation_table.csv")
    track1_intervals = pd.read_csv(ROOT / "results" / "track1_multiseed_effect_intervals.csv")
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
                ("Date", "2026-06-23"),
                (
                    "Purpose",
                    "Fix the reviewer-revised research state for external verification. This version incorporates AC-level review concerns: endpoint leakage, syntax-holdout overclaiming, McNemar tests, semantic-control statistics, and the terminology change to a third-order signed permutation coherence test.",
                ),
                (
                    "Track 1 fixed result",
                    "Delta vectors add reproducible information beyond target-only endpoints in the multiseed full-semantic setting under Linear SVC. New seed-level effect intervals make the classifier dependence explicit: Linear SVC is positive for every original model, while logistic regression is mixed for GPT-2 and RoBERTa. The syntax=1.0 result has now been directly ablated: y_only also reaches 1.0, so the syntax split is interpreted as endpoint/surface leakage rather than as geometry. BERT-large and DeBERTa-v3-base spot-checks both rank delta best under Linear SVC and logistic regression. UPAT is now reported as a hard-holdout boundary where delta is not reliably better than y_only.",
                ),
                (
                    "Track 2 fixed result",
                    "Composition order matters. Semantic equivalence shifts noncommutativity distributions with strong statistical tests. The third-order result is now framed as signed permutation coherence, not a formal Jacobi identity. Decoder pairwise composition is now included. The 2026-06-23 multilingual max audit extends the signed-permutation probe to 7 languages and 5 multilingual encoders. All four tested triples are below signed-null in every model-language cell; therefore the older 'QMT-only' story must be narrowed to the earlier English/decoder table, not promoted as a universal claim.",
                ),
                (
                    "Main conclusion",
                    "The evidence now supports a broader but still controlled signed-permutation signal in multilingual sentence encoders, with NQM and QMT especially strong in the new audit. This is closer to the Lie-algebra direction, but still not a proof of a Lie algebra: endpoint controls remain strong and cross-language centroid consistency is only moderate with high variance. RISE (Freenor and Alvarez, ICLR 2026) remains the closest prior work for spherical/geodesic semantic-syntactic transformations across languages and embedding models.",
                ),
            ],
        )

        add_table_page(pdf, "Delta vs endpoint ablation with McNemar evidence", format_float_columns(ablation), max_rows=10)
        add_table_page(pdf, "Track 1 seed-level effect intervals", format_float_columns(track1_intervals), max_rows=20)
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
        if not grammar_composition.empty:
            add_table_page(pdf, "Grammar-generated pairwise composition summary", format_float_columns(grammar_composition), max_rows=20)
        if not grammar_controls.empty:
            add_table_page(pdf, "Grammar composition endpoint controls", format_float_columns(grammar_controls), max_rows=25)
        if not grammar_nulls.empty:
            same_pair = grammar_nulls[grammar_nulls["null_type"] == "random_pairing_same_pair"]
            add_table_page(pdf, "Grammar composition commutator nulls: same-pair shuffle", format_float_columns(same_pair), max_rows=20)
        if not multilingual_triple.empty:
            add_table_page(pdf, "Multilingual max audit: global triple summary", format_float_columns(multilingual_triple), max_rows=10)
        if not multilingual_triple_by_model.empty:
            add_table_page(pdf, "Multilingual max audit: triple summary by model", format_float_columns(multilingual_triple_by_model), max_rows=25)
        if not multilingual_pair.empty:
            add_table_page(pdf, "Multilingual max audit: pairwise commutator summary", format_float_columns(multilingual_pair), max_rows=10)
        if not multilingual_endpoint.empty:
            add_table_page(pdf, "Multilingual max audit: held-out language endpoint controls", format_float_columns(multilingual_endpoint), max_rows=30)
        if not multilingual_centroid.empty:
            add_table_page(pdf, "Multilingual max audit: cross-language centroid consistency", format_float_columns(multilingual_centroid), max_rows=10)

        figures = [
            ("Signed permutation relative norm", EXP / "lie_algebraic_identities_results" / "figures" / "03_jacobi_relative_norm_heatmap.png"),
            ("Signed permutation norm versus permutation-null", EXP / "lie_algebraic_identities_results" / "figures" / "04_jacobi_vs_permutation_null_heatmap.png"),
            ("Semantic equivalent vs non-equivalent control", EXP / "lie_semantic_equivalence_results" / "figures" / "01_equivalent_vs_nonequivalent.png"),
            ("Composition noncommutativity heatmap", EXP / "lie_composition_results" / "figures" / "01_noncommutativity_heatmap.png"),
            ("Grammar composition relative commutator heatmap", EXP / "lie_composition_grammar_results" / "figures" / "01_relative_commutator_norm_heatmap.png"),
            ("Multilingual signed-permutation ratios", EXP / "lie_multilingual_max_results" / "figures" / "01_multilingual_signed_permutation_ratios.png"),
            ("Multilingual global triple ratios", EXP / "lie_multilingual_max_results" / "figures" / "02_triple_global_ratio_summary.png"),
            ("Multilingual endpoint controls", EXP / "lie_multilingual_max_results" / "figures" / "03_endpoint_control_macro_f1.png"),
            ("Multilingual cross-language centroid consistency", EXP / "lie_multilingual_max_results" / "figures" / "04_cross_language_centroid_consistency.png"),
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
                "Seventh 2026-06-14 update: grammar-generated pairwise composition controls are now added for Track 2. Observed relative commutator norms remain below shuffled/norm-matched nulls, but endpoint-only and delta-only controls classify pair labels almost perfectly, so this is not yet endpoint-independent algebraic evidence.",
                "2026-06-23 update: the multilingual max audit completed successfully on seven languages and five multilingual encoders: paraphrase-multilingual-mpnet-base-v2, LaBSE, multilingual-e5-large, BGE-M3, and mBERT. All four third-order triples are below signed-null in every model-language cell; the global mean ratios are NQM=0.5798, QMT=0.6203, NQT=0.7014, and NMT=0.7716. This strengthens the existence of a controlled signed-permutation signal while weakening any universal claim that QMT is uniquely special.",
                "2026-06-23 caution: source-only held-out-language controls are chance-like, but endpoint/delta/commutator controls remain high. Cross-language centroid cosine is moderate and high-variance. The next Track 2 gate is endpoint-balanced multilingual generation plus third-order target-only controls, not another unqualified scale-up.",
                "2026-06-23 Track 1 cleanup: seed-level effect intervals are now added for delta-y_only and delta-concat. The Track 1 claim is narrowed to a robust Linear SVC/margin-probe result, because logistic regression is mixed for GPT-2 and RoBERTa.",
                "Remaining blockers: confidence intervals and anchor-domain diversity for Track 3, stronger/faithful RISE comparison if feasible, resolve or expand UPAT hard-holdout, representation-ablation tables for every non-syntax holdout split, endpoint-balanced grammar generation and third-order target-only controls for Track 2, composition layer/pooling ablations, and final bibliography formatting.",
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

        critical_review_path = ROOT / "research" / "critical_review_2026-06-14.md"
        if critical_review_path.exists():
            add_text_page(
                pdf,
                "Critical Review: Current Research State",
                [critical_review_path.read_text(encoding="utf-8")],
                fontsize=8,
            )

        add_code_listing(pdf, ROOT / "scripts" / "run_lie_algebraic_identities.py", "Code listing: run_lie_algebraic_identities.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_lie_composition_grammar_controls.py", "Code listing: run_lie_composition_grammar_controls.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_lie_multilingual_max_audit.py", "Code listing: run_lie_multilingual_max_audit.py")
        add_code_listing(pdf, ROOT / "scripts" / "run_syntax_representation_ablation.py", "Code listing: run_syntax_representation_ablation.py")
        add_code_listing(pdf, ROOT / "scripts" / "build_track1_effect_intervals.py", "Code listing: build_track1_effect_intervals.py")
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
