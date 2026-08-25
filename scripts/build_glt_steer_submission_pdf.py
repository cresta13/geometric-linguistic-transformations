from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "reports" / "2026-08-25_glt_steer_submission_draft.pdf"


@dataclass(frozen=True)
class FigureSpec:
    title: str
    path: Path
    caption: str


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def fmt_rate(x: float) -> str:
    return f"{x:.4f}"


def fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"[{lo:.3f}, {hi:.3f}]"


def styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="PaperTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    base.add(
        ParagraphStyle(
            name="PaperSubtitle",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=18,
        )
    )
    base.add(
        ParagraphStyle(
            name="Section",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#111827"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="Subsection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.7,
            leading=11.2,
            alignment=TA_LEFT,
            spaceAfter=5,
        )
    )
    base.add(
        ParagraphStyle(
            name="Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
            textColor=colors.HexColor("#374151"),
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.0,
            leading=8.5,
            textColor=colors.white,
            spaceAfter=0,
        )
    )
    base.add(
        ParagraphStyle(
            name="Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.2,
            leading=9,
            textColor=colors.HexColor("#4b5563"),
            spaceBefore=3,
            spaceAfter=8,
        )
    )
    return base


S = styles()


def para(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(escape(text), S[style])


def heading(text: str) -> Paragraph:
    return para(text, "Section")


def subheading(text: str) -> Paragraph:
    return para(text, "Subsection")


def bullet(items: Iterable[str]) -> list[Paragraph]:
    return [Paragraph(f"- {escape(item)}", S["Body"]) for item in items]


def code_block(lines: Iterable[str]) -> Table:
    rows = [[Paragraph(escape(line), S["Small"])] for line in lines]
    table = Table(rows, colWidths=[6.7 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f4f6")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def make_table(rows: list[list[str]], widths: list[float] | None = None, font_size: float = 6.8) -> Table:
    data = []
    for row_idx, row in enumerate(rows):
        style = S["TableHeader"] if row_idx == 0 else S["Small"]
        data.append([Paragraph(escape(str(cell)), style) for cell in row])
    if widths is not None:
        col_widths = [w * inch for w in widths]
    else:
        col_widths = None
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def image_block(spec: FigureSpec, max_width: float = 6.8 * inch, max_height: float = 3.8 * inch):
    if not spec.path.exists():
        return []
    with PILImage.open(spec.path) as img:
        w, h = img.size
    scale = min(max_width / w, max_height / h)
    return [
        KeepTogether(
            [
                subheading(spec.title),
                Image(str(spec.path), width=w * scale, height=h * scale),
                para(spec.caption, "Caption"),
            ]
        )
    ]


def confirmatory_table() -> list[list[str]]:
    raw = EXP / "glt_steer_confirmatory_fixed_params_20260825_results" / "csv" / "glt_steer_confirmatory_raw.csv"
    df = pd.read_csv(raw)
    rows = [
        [
            "model",
            "target",
            "target marker rate",
            "max control marker",
            "target marker+preserved",
            "max control marker+preserved",
        ]
    ]
    for (model, target), group in df.groupby(["model", "target_class"]):
        target_rows = group[group["control"] == "target"]
        control_rows = group[group["control"] != "target"]
        n = len(target_rows)
        marker_hits = int(target_rows["target_marker_hit"].sum())
        joint_hits = int(target_rows["target_and_preserved"].sum())
        control_marker = control_rows.groupby("control")["target_marker_hit"].mean().max()
        control_joint = control_rows.groupby("control")["target_and_preserved"].mean().max()
        rows.append(
            [
                model,
                target,
                f"{fmt_rate(marker_hits / n)} (N={n}, CI {fmt_ci(marker_hits, n)})",
                fmt_rate(float(control_marker)),
                f"{fmt_rate(joint_hits / n)} (N={n}, CI {fmt_ci(joint_hits, n)})",
                fmt_rate(float(control_joint)),
            ]
        )
    return sorted(rows[1:], key=lambda r: (r[0], r[1]))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(0.65 * inch, 0.42 * inch, "GLT-STEER submission draft - not peer reviewed")
    canvas.drawRightString(7.6 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_story():
    story = []
    story.append(para("GLT-STEER: Transformation Deltas as Activation-Space Editors for Final Markers", "PaperTitle"))
    story.append(para("Anna Simakova - submission draft - 2026-08-25 - not peer reviewed", "PaperSubtitle"))

    story.append(heading("Abstract"))
    story.append(
        para(
            "This short draft studies whether linguistic transformation vectors learned from sentence-pair hidden-state differences can be injected back into a generative transformer as activation-space editors. In GPT-2, vectors derived from controlled transformation pairs reliably steer final-position surface markers such as ?, !, and ... under matched controls. The effect is visible in generated text, next-token marker logits, position-of-intervention audits, and a fixed-parameter confirmatory rerun on fresh hard-heldout sources. The claim is intentionally bounded: current GLT-STEER supports final-marker form steering, not general semantic rewriting and not a Lie-algebra claim."
        )
    )

    story.append(heading("1. Question and Scope"))
    story.append(
        para(
            "For a source sentence x and transformed sentence y, GLT computes a hidden-state displacement delta = h(y) - h(x). Earlier GLT tracks test whether these deltas classify transformation type offline. GLT-STEER asks a stronger intervention question: if a transformation delta is injected during generation, does the model produce the corresponding transformation marker?"
        )
    )
    story.append(
        para(
            "The current defensible claim is that transformation deltas can act as activation-space editors for final-position surface markers in GPT-style residual streams. The cleanest evidence is in GPT-2, with marker-dependent transfer to DistilGPT-2. This is not a claim of robust semantic rewriting."
        )
    )

    story.append(heading("2. Method"))
    story.extend(
        bullet(
            [
                "Build synthetic source/target pairs for a transformation class.",
                "Extract hidden states at selected GPT-style residual blocks.",
                "Compute a centroid transformation vector from training pairs.",
                "During generation, add the vector to the residual stream.",
                "Compare target steering against no-steering, wrong-marker, random-norm, and negative-vector controls.",
            ]
        )
    )
    story.append(
        code_block(
            [
                "delta = h(target sentence) - h(source sentence)",
                "y_pred generation: residual[layer] += gain * centroid_delta",
            ]
        )
    )

    story.append(heading("3. Main Evidence"))
    story.append(subheading("3.1 Question steering and prompt-only control"))
    story.append(
        make_table(
            [
                ["condition", "question mark rate"],
                ["target question vector", "0.9350"],
                ["random-norm control", "0.0000"],
                ["wrong-class control", "0.0000"],
                ["negative-target control", "0.0000"],
            ],
            widths=[4.5, 2.0],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        para(
            "Copy-like prompts without steering produce 0.0000 question marks across 960 no-steering rows from GPT-2 and DistilGPT-2. This rules out the simplest prompt-only explanation for the question-marker result."
        )
    )

    story.append(subheading("3.2 Content preservation"))
    story.append(
        make_table(
            [
                ["source set", "prompt family", "target question-and-preserved"],
                ["in-template", "copy-like prompts", "0.9625-0.9750"],
                ["simple out-of-template", "copy-like prompts", "0.8250-0.9000"],
            ],
            widths=[2.1, 2.1, 2.2],
        )
    )
    story.append(
        para(
            "Hard out-of-template sources reduce preservation. GPT-2 still shows question-form steering; DistilGPT-2 is weaker and should be reported as marker-form transfer only."
        )
    )

    story.extend(
        image_block(
            FigureSpec(
                "Figure 1. Final-marker controls",
                ROOT / "paper" / "figures" / "glt_steer_final_marker_controls.png",
                "Exclamation and ellipsis controls show that the question result is part of a broader final-marker phenomenon, not only a question-specific artifact.",
            )
        )
    )

    story.append(PageBreak())
    story.append(heading("4. Mechanistic Audits"))
    story.append(subheading("4.1 Logit-level final-marker audit"))
    story.append(
        make_table(
            [
                ["target", "control", "marker rate (N, 95% CI)", "median best rank"],
                ["?", "none", "0.0000 (N=96, [0.0000, 0.0385])", "28.0"],
                ["?", "target", "0.8542 (N=96, [0.7700, 0.9111])", "1.0"],
                ["!", "none", "0.0000 (N=96, [0.0000, 0.0385])", "20.0"],
                ["!", "target", "0.9063 (N=96, [0.8313, 0.9499])", "1.0"],
                ["...", "none", "0.0000 (N=96, [0.0000, 0.0385])", "11.75"],
                ["...", "target", "0.8750 (N=96, [0.7941, 0.9270])", "1.0"],
            ],
            widths=[0.7, 1.0, 4.3, 1.0],
        )
    )
    story.append(
        para(
            "No-steering marker rates are zero, while target steering moves the intended marker token to rank 1 in most sequences. This is the strongest current evidence that the intervention changes the generation distribution itself."
        )
    )

    story.append(subheading("4.2 Position-of-intervention audit"))
    story.append(
        make_table(
            [
                ["condition", "position mode", "question rate", "question+preserved"],
                ["none", "none", "0.0000", "0.0000"],
                ["target_last_each_step", "last_each_step", "0.9604", "0.7917"],
                ["target_prompt_all_once", "prompt_all", "0.8625", "0.7896"],
                ["target_prompt_first", "prompt_first", "0.0000", "0.0000"],
                ["target_prompt_middle", "prompt_middle", "0.0000", "0.0000"],
                ["target_prompt_last_once", "prompt_last", "0.0000", "0.0000"],
            ],
            widths=[2.0, 1.8, 1.3, 1.5],
        )
    )
    story.append(
        para(
            "Single prompt-position edits do not work. Distributed prompt-state editing or repeated last-token-at-each-step editing does work. This argues against a trivial one-token prompt perturbation story."
        )
    )

    story.append(heading("5. Fixed-Parameter Confirmation"))
    confirm_rows = [["model", "target", "target marker rate", "max control marker", "target marker+preserved", "max control marker+preserved"]]
    confirm_rows.extend(confirmatory_table())
    story.append(make_table(confirm_rows, widths=[1.0, 1.0, 1.65, 1.15, 1.75, 1.25], font_size=5.7))
    story.append(
        para(
            "This run uses fresh hard-heldout sources and fixed settings: GPT-2 layers 2,3 at gain 0.75; DistilGPT-2 layer 2 at gain 1.0. No layer/gain search is performed inside the run. Target marker rates remain separated from controls across all three final markers and both tested models."
        )
    )

    story.append(PageBreak())
    story.append(heading("6. Boundary Results"))
    story.append(subheading("6.1 Negation"))
    story.append(
        para(
            "Negation is weak under the same copy-prompt recipe. A GPT-2 layer sweep over layers 0-11 does not find a clean intervention site. The best target-and-preserved row reaches only 0.1729, while matched controls remain nontrivial, with maxima up to 0.1229."
        )
    )
    story.append(
        make_table(
            [
                ["layer", "question mean pairwise cosine", "negation mean pairwise cosine"],
                ["2", "0.9693", "0.5621"],
                ["3", "0.9365", "0.5665"],
            ],
            widths=[1.0, 2.6, 2.6],
        )
    )
    story.append(subheading("6.2 DistilGPT-2"))
    story.extend(
        image_block(
            FigureSpec(
                "Figure 2. DistilGPT-2 hard-OOT boundary",
                ROOT / "paper" / "figures" / "glt_steer_distilgpt2_hard_oot_boundary.png",
                "DistilGPT-2 preserves marker-form steering but has weak strict content preservation on hard out-of-template sources.",
            ),
            max_height=3.2 * inch,
        )
    )
    story.append(subheading("6.3 Marker composition"))
    story.extend(
        image_block(
            FigureSpec(
                "Figure 3. Marker composition",
                ROOT / "paper" / "figures" / "glt_steer_composition_marker_profile.png",
                "Combined question/exclamation vectors produce mixed marker profiles and lower preservation. The correct interpretation is competition/saturation, not noncommutative algebra.",
            ),
            max_height=3.2 * inch,
        )
    )

    story.append(PageBreak())
    story.append(heading("7. Related Work Positioning"))
    story.append(
        para(
            "GLT-STEER is closest to activation-engineering and representation-steering work. Activation Addition computes activation differences from contrastive prompts and injects them at inference time. Contrastive Activation Addition averages residual-stream differences over positive and negative behavioral examples. Representation Engineering frames a broader family of methods for monitoring and manipulating model internals."
        )
    )
    story.append(
        para(
            "The contribution here is not that activation addition exists. The narrower contribution is diagnostic: GLT-STEER derives vectors from controlled linguistic transformation pairs and asks which linguistic transformations behave as reusable intervention vectors. The current answer is bounded: final-position markers are steerable; sentence-internal transformations are not solved by this recipe."
        )
    )

    story.append(heading("8. Limitations"))
    story.extend(
        bullet(
            [
                "GLT-STEER does not solve semantic rewriting.",
                "Negation and modality are not cleanly steered by the current method.",
                "Final-marker steering does not prove general linguistic transformation editing.",
                "Marker composition does not prove noncommutative algebra.",
                "The result does not establish a Lie algebra in transformer activations.",
                "Prompt wording remains a real boundary condition.",
            ]
        )
    )

    story.append(heading("9. Reproducibility"))
    story.append(
        para(
            "All reported results are archived in results/experiments/. No new model inference is required to inspect the tables in this draft. The datasets are synthetic controlled sentence-pair templates generated by repository scripts."
        )
    )
    story.append(
        make_table(
            [
                ["artifact", "path"],
                ["draft source", "paper/articles/glt-steer-activation-editors/submission_draft.md"],
                ["PDF builder", "scripts/build_glt_steer_submission_pdf.py"],
                ["generated PDF", "reports/2026-08-25_glt_steer_submission_draft.pdf"],
                ["confirmatory result", "results/experiments/glt_steer_confirmatory_fixed_params_20260825_results/"],
                ["CI audit", "results/experiments/glt_steer_headline_ci_20260825_results/"],
            ],
            widths=[1.8, 4.9],
        )
    )

    story.append(heading("References"))
    story.extend(
        bullet(
            [
                "Turner et al. 2023/2024. Steering Language Models With Activation Engineering / Activation Addition. arXiv:2308.10248.",
                "Panickssery et al. 2024. Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.06681 / ACL 2024.",
                "Zou et al. 2023. Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.",
                "Freenor and Alvarez 2026. RISE: geometric rotations for semantic-syntactic transformations.",
                "Ilharco et al. 2023. Editing Models with Task Arithmetic.",
                "Todd et al. 2024. Function Vectors in Large Language Models.",
            ]
        )
    )
    return story


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.62 * inch,
        title="GLT-STEER Submission Draft",
        author="Anna Simakova",
    )
    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
