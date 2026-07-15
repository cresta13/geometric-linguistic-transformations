"""Print a lightweight GLT-STEER demo from archived GPT-2 steering results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULT_DIR = Path("results/experiments/gpt2_question_activation_steering_focused_20260714_results")
RAW_PATH = RESULT_DIR / "csv" / "activation_steering_raw.csv"
CONTROL_DIR = Path("results/experiments/gpt2_question_steering_controls_20260714_results")
CONTROL_SUMMARY_PATH = CONTROL_DIR / "csv" / "question_steering_controls_summary.csv"
BASE_RATE_PATH = CONTROL_DIR / "csv" / "question_mark_base_rate.csv"
PROMPT_ROBUSTNESS_DIR = Path("results/experiments/gpt2_question_prompt_robustness_20260715_results")
PROMPT_ROBUSTNESS_SUMMARY_PATH = PROMPT_ROBUSTNESS_DIR / "csv" / "question_prompt_robustness_summary.csv"


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"Missing archived result file: {RAW_PATH}")

    raw = pd.read_csv(RAW_PATH)

    print("GLT-STEER demo: GPT-2 question activation steering")
    print("=" * 58)
    print()
    print("Question:")
    print("  Can a transformation vector change what GPT-2 generates?")
    print()

    gain = 0.75
    view = raw[raw["gain"].eq(gain)]
    overall = (
        view.groupby("control")
        .agg(
            target_marker_rate=("target_marker_hit", "mean"),
            question_mark_rate=("question_mark_hit", "mean"),
            rows=("question_mark_hit", "size"),
        )
        .reindex(["target", "random_norm", "wrong_class", "negative_target"])
        .reset_index()
    )

    print(f"Aggregate result at gain={gain}:")
    print()
    print(f"{'control':<18} {'target_marker':>14} {'question_mark':>14} {'rows':>7}")
    print("-" * 58)
    for row in overall.itertuples(index=False):
        print(
            f"{row.control:<18} "
            f"{fmt(row.target_marker_rate):>14} "
            f"{fmt(row.question_mark_rate):>14} "
            f"{int(row.rows):>7}"
        )

    best = (
        view[view["control"].eq("target")]
        .groupby(["layer", "gain"])
        .agg(question_mark_rate=("question_mark_hit", "mean"))
        .reset_index()
        .sort_values("question_mark_rate", ascending=False)
        .iloc[0]
    )

    print()
    print(
        "Best target-steering setting: "
        f"layer={int(best.layer)}, gain={best.gain}, "
        f"question_mark_rate={fmt(best.question_mark_rate)}"
    )

    examples = [
        "The robot opened the portal.",
        "The oracle built the tower.",
        "The ranger guarded the treasure.",
    ]

    print()
    print("Examples at layer=2, gain=0.75:")
    print()

    layer_view = raw[raw["layer"].eq(2)]
    for source in examples:
        print(f"Source: {source}")
        for control, control_gain in [
            ("none", 0.0),
            ("target", 0.75),
            ("random_norm", 0.75),
            ("wrong_class", 0.75),
            ("negative_target", 0.75),
        ]:
            row = layer_view[
                layer_view["source"].eq(source)
                & layer_view["control"].eq(control)
                & layer_view["gain"].eq(control_gain)
            ]
            if row.empty:
                continue
            generated = str(row.iloc[0]["generated"]).replace("\n", " ")[:150]
            qmark = int(row.iloc[0]["question_mark_hit"])
            print(f"  {control:<15} qmark={qmark}: {generated}")
        print()

    print("Safe interpretation:")
    print(
        "  The archived run shows question-form steering, not solved semantic rewriting "
        "and not proof of a complete linguistic algebra."
    )
    print()
    print(f"Full summary: {RESULT_DIR / 'SUMMARY.md'}")

    if CONTROL_SUMMARY_PATH.exists() and BASE_RATE_PATH.exists():
        control_summary = pd.read_csv(CONTROL_SUMMARY_PATH)
        base_rate = pd.read_csv(BASE_RATE_PATH)

        print()
        print("Reviewer controls")
        print("=" * 58)
        print()
        print("No-steering question-mark base rate:")
        print()
        print(f"{'source_set':<30} {'base_qmark_rate':>16} {'rows':>7}")
        print("-" * 58)
        for row in base_rate.itertuples(index=False):
            print(f"{row.source_set:<30} {fmt(row.base_question_mark_rate):>16} {int(row.rows):>7}")

        pivot = control_summary.pivot_table(
            index="source_set",
            columns="control",
            values="question_mark_rate",
            aggfunc="mean",
        )
        print()
        print("Question-mark rate at layer family {2,3}, gain=0.75:")
        print()
        controls = ["none", "target", "random_norm", "wrong_class", "negative_target"]
        print(f"{'source_set':<30} " + " ".join(f"{c:>15}" for c in controls))
        print("-" * 110)
        for source_set, row in pivot.iterrows():
            print(f"{source_set:<30} " + " ".join(f"{fmt(row.get(c, float('nan'))):>15}" for c in controls))

        print()
        print(f"Control summary: {CONTROL_DIR / 'SUMMARY.md'}")

    if PROMPT_ROBUSTNESS_SUMMARY_PATH.exists():
        prompt_summary = pd.read_csv(PROMPT_ROBUSTNESS_SUMMARY_PATH)
        prompt_pivot = prompt_summary.pivot_table(
            index=["source_set", "prompt_style"],
            columns="control",
            values="question_mark_rate",
            aggfunc="mean",
        )

        print()
        print("Prompt robustness controls")
        print("=" * 58)
        print()
        controls = ["none", "target", "random_norm", "wrong_class", "negative_target"]
        print(f"{'source_set':<28} {'prompt_style':<24} " + " ".join(f"{c:>15}" for c in controls))
        print("-" * 132)
        for (source_set, prompt_style), row in prompt_pivot.iterrows():
            print(
                f"{source_set:<28} {prompt_style:<24} "
                + " ".join(f"{fmt(row.get(c, float('nan'))):>15}" for c in controls)
            )

        print()
        print(f"Prompt robustness summary: {PROMPT_ROBUSTNESS_DIR / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
