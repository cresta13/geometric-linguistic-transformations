from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "experiments" / "glt_steer_headline_ci_20260825_results"
CSV_DIR = OUT_DIR / "csv"
OUT_CSV = CSV_DIR / "glt_steer_headline_ci.csv"
SUMMARY_MD = OUT_DIR / "SUMMARY.md"
STATUS_JSON = OUT_DIR / "run_status.json"


@dataclass
class RateRow:
    section: str
    source_file: str
    group: str
    metric: str
    estimate: float
    n: int
    numerator: float
    ci95_low: float
    ci95_high: float


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + (z * z / n)
    center = (p + z * z / (2.0 * n)) / denom
    half = z * sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def add_from_raw(
    rows: list[RateRow],
    section: str,
    path: Path,
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
) -> None:
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = " | ".join(f"{col}={value}" for col, value in zip(group_cols, keys))
        for metric in metrics:
            if metric not in sub.columns:
                continue
            values = sub[metric].astype(float)
            n = int(values.shape[0])
            k = float(values.sum())
            lo, hi = wilson(k, n)
            rows.append(
                RateRow(
                    section=section,
                    source_file=str(path.relative_to(ROOT)),
                    group=group,
                    metric=metric,
                    estimate=k / n if n else float("nan"),
                    n=n,
                    numerator=k,
                    ci95_low=lo,
                    ci95_high=hi,
                )
            )


def add_from_summary(
    rows: list[RateRow],
    section: str,
    path: Path,
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
) -> None:
    for _, row in df.iterrows():
        group = " | ".join(f"{col}={row[col]}" for col in group_cols)
        n = int(row["rows"])
        for metric in metrics:
            estimate = float(row[metric])
            k = estimate * n
            lo, hi = wilson(k, n)
            rows.append(
                RateRow(
                    section=section,
                    source_file=str(path.relative_to(ROOT)),
                    group=group,
                    metric=metric,
                    estimate=estimate,
                    n=n,
                    numerator=k,
                    ci95_low=lo,
                    ci95_high=hi,
                )
            )


def read_csv(rel_path: str) -> tuple[Path, pd.DataFrame]:
    path = ROOT / rel_path
    return path, pd.read_csv(path)


def build_rows() -> list[RateRow]:
    rows: list[RateRow] = []

    path, df = read_csv(
        "results/experiments/gpt2_question_copy_prompt_preservation_20260716_results/"
        "csv/question_copy_prompt_summary.csv"
    )
    df = df[
        (df["control"] == "target")
        & (df["prompt_style"].isin(["copy_sentence", "repeat_sentence", "same_sentence"]))
    ]
    add_from_summary(
        rows,
        "gpt2_question_copy_prompt_preservation",
        path,
        df,
        ["source_set", "prompt_style", "control"],
        ["question_mark_rate", "question_and_preserved_rate"],
    )

    path, df = read_csv(
        "results/experiments/gpt2_question_hard_oot_copy_prompt_steering_20260716_results/"
        "csv/question_hard_oot_question_preservation_summary.csv"
    )
    df = df[(df["control"] == "target") & (df["prompt_style"].isin(["copy_sentence", "repeat_sentence", "same_sentence"]))]
    add_from_summary(
        rows,
        "gpt2_question_hard_oot",
        path,
        df,
        ["prompt_style", "control"],
        ["question_mark_rate", "question_and_preserved_rate"],
    )

    path, df = read_csv(
        "results/experiments/distilgpt2_question_hard_oot_best_layer2_gain10_20260801_results/"
        "csv/distilgpt2_hard_oot_question_preservation_summary.csv"
    )
    df = df[(df["control"] == "target") & (df["prompt_style"].isin(["copy_sentence", "repeat_sentence", "same_sentence"]))]
    add_from_summary(
        rows,
        "distilgpt2_question_hard_oot",
        path,
        df,
        ["prompt_style", "layer", "gain", "control"],
        ["question_mark_rate", "strict_question_and_preserved_rate"],
    )

    path, df = read_csv(
        "results/experiments/gpt2_exclamation_copy_prompt_steering_20260716_results/"
        "csv/exclamation_copy_prompt_summary.csv"
    )
    df = df[
        (df["control"] == "target_exclamation")
        & (df["prompt_style"].isin(["copy_sentence", "repeat_sentence", "same_sentence"]))
    ]
    add_from_summary(
        rows,
        "gpt2_exclamation_steering",
        path,
        df,
        ["source_set", "prompt_style", "layer", "control"],
        ["exclamation_mark_rate", "exclamation_and_preserved_rate"],
    )

    path, df = read_csv(
        "results/experiments/gpt2_ellipsis_hard_oot_layer2_20260801_results/"
        "csv/final_marker_copy_prompt_summary.csv"
    )
    df = df[(df["control"] == "target_marker") & (df["prompt_style"].isin(["copy_sentence", "repeat_sentence", "same_sentence"]))]
    add_from_summary(
        rows,
        "gpt2_ellipsis_hard_oot",
        path,
        df,
        ["prompt_style", "layer", "control"],
        ["target_marker_rate", "target_and_preserved_rate"],
    )

    for section, rel_path in [
        (
            "gpt2_final_marker_logit_audit",
            "results/experiments/gpt2_final_marker_logit_audit_layer2_3_20260810_v2_results/"
            "csv/final_marker_logit_sequence_raw.csv",
        ),
        (
            "distilgpt2_final_marker_logit_audit",
            "results/experiments/distilgpt2_final_marker_logit_audit_l1_2_3_gain10_20260821_results/"
            "csv/final_marker_logit_sequence_raw.csv",
        ),
    ]:
        path, df = read_csv(rel_path)
        df = df[df["control"].isin(["none", "target"])]
        add_from_raw(rows, section, path, df, ["target_class", "control"], ["target_marker_hit", "target_top1_any"])

    path, df = read_csv(
        "results/experiments/gpt2_question_position_intervention_audit_layer2_3_20260821_results/"
        "csv/question_position_intervention_raw.csv"
    )
    position_controls = [
        "none",
        "target_last_each_step",
        "target_prompt_all_once",
        "target_prompt_first",
        "target_prompt_middle",
        "target_prompt_last_once",
        "random_last_each_step",
        "wrong_last_each_step",
        "negative_last_each_step",
    ]
    df = df[df["control"].isin(position_controls)]
    add_from_raw(rows, "gpt2_question_position_intervention", path, df, ["control", "position_mode"], ["question_mark_hit", "question_and_preserved"])

    path, df = read_csv(
        "results/experiments/gpt2_question_exclamation_marker_composition_layer2_3_20260801_results/"
        "csv/marker_composition_steering_raw.csv"
    )
    composition_controls = [
        "none",
        "a_only_late",
        "b_only_late",
        "a_plus_b_late",
        "a_then_b_layers",
        "b_then_a_layers",
        "random_sum_late",
    ]
    df = df[(df["prompt_style"] == "same_sentence") & (df["control"].isin(composition_controls))]
    add_from_raw(
        rows,
        "gpt2_question_exclamation_composition_same_sentence",
        path,
        df,
        ["control"],
        ["a_marker_hit", "b_marker_hit", "both_markers_hit", "content_preserved"],
    )

    path, df = read_csv(
        "results/experiments/gpt2_question_exclamation_marker_composition_layer2_3_20260801_results/"
        "csv/marker_composition_order_contrast_raw.csv"
    )
    add_from_raw(
        rows,
        "gpt2_question_exclamation_order_contrast",
        path,
        df,
        ["prompt_style"],
        ["ab_equals_ba", "ab_ba_marker_profile_equal", "ab_equals_plus", "ba_equals_plus"],
    )

    return rows


def write_summary(table: pd.DataFrame) -> None:
    headline = table[
        (
            table["section"].isin(
                [
                    "gpt2_final_marker_logit_audit",
                    "distilgpt2_final_marker_logit_audit",
                    "gpt2_question_position_intervention",
                    "gpt2_question_exclamation_order_contrast",
                ]
            )
        )
        & (
            table["metric"].isin(
                [
                    "target_marker_hit",
                    "question_mark_hit",
                    "question_and_preserved",
                    "ab_equals_ba",
                    "ab_ba_marker_profile_equal",
                ]
            )
        )
    ].copy()
    lines = [
        "# GLT-STEER Headline Confidence Intervals",
        "",
        "Derived CI audit for the Track 1 / GLT-STEER headline tables. The script uses Wilson 95% confidence intervals for binary rates, computed from raw rows when available and from summary rate + row counts for strict summary-only metrics.",
        "",
        "Script:",
        "",
        "- `scripts/summarize_glt_steer_headline_ci.py`",
        "",
        "Output:",
        "",
        "- `csv/glt_steer_headline_ci.csv`",
        "",
        "Key interpretation:",
        "",
        "- GPT-2 final-marker logit steering has non-overlapping qualitative separation: no-steering rows remain at `0.0000`, while target steering is high for `?`, `!`, and `...`.",
        "- DistilGPT-2 keeps the no-steering baseline at `0.0000`, but target steering is marker- and layer-sensitive, strongest for `!`, intermediate for `...`, and weakest for `?`.",
        "- The position audit has enough rows (`N=480` per aggregate condition) to show that first/middle/last single prompt-token edits are null while repeated last-token and all-prompt-token edits are strong.",
        "- Composition rows have smaller cells (`N=40` per prompt-style/control row); order contrasts should therefore remain descriptive and should not be promoted as algebraic evidence.",
        "",
        "Selected headline rows:",
        "",
        "| section | group | metric | estimate | N | 95% CI |",
        "|---|---|---|---:|---:|---|",
    ]
    for _, row in headline.iterrows():
        group = str(row["group"]).replace("|", "\\|")
        lines.append(
            f"| `{row['section']}` | `{group}` | `{row['metric']}` | "
            f"`{row['estimate']:.4f}` | `{int(row['n'])}` | "
            f"`[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]` |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row.__dict__ for row in build_rows()])
    table.sort_values(["section", "group", "metric"], inplace=True)
    table.to_csv(OUT_CSV, index=False)
    write_summary(table)
    STATUS_JSON.write_text(
        json.dumps(
            {
                "status": "finished",
                "kind": "derived_ci_audit",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "rows": int(table.shape[0]),
                "output_csv": str(OUT_CSV.relative_to(ROOT)),
                "summary": str(SUMMARY_MD.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {STATUS_JSON}")


if __name__ == "__main__":
    main()
