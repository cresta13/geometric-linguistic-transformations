from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


IN_DIR = Path("results/experiments/gpt2_question_prompt_robustness_20260715_results")
OUT_DIR = Path("results/experiments/gpt2_question_content_preservation_20260716_results")
IN_RAW = IN_DIR / "csv" / "question_prompt_robustness_raw.csv"
CSV_DIR = OUT_DIR / "csv"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def normalize_token(token: str) -> str:
    token = token.lower()
    token = re.sub(r"[^a-z0-9]+", "", token)
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_tokens(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[A-Za-z0-9']+", str(text).lower()):
        tok = normalize_token(raw)
        if tok and tok not in STOPWORDS:
            tokens.append(tok)
    return tokens


def preservation_score(source: str, generated: str) -> float:
    src = set(content_tokens(source))
    if not src:
        return 0.0
    gen = set(content_tokens(generated))
    return len(src & gen) / len(src)


def extract_until_first_question(text: str) -> str:
    text = str(text)
    idx = text.find("?")
    if idx < 0:
        return text
    return text[: idx + 1]


def main() -> None:
    if not IN_RAW.exists():
        raise SystemExit(f"Missing input: {IN_RAW}")

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(IN_RAW)

    scored = raw.copy()
    scored["source_content_tokens"] = scored["source"].map(lambda x: " ".join(content_tokens(x)))
    scored["generated_until_question"] = scored["generated"].map(extract_until_first_question)
    scored["content_preservation"] = [
        preservation_score(src, gen) for src, gen in zip(scored["source"], scored["generated"], strict=False)
    ]
    scored["content_preservation_until_question"] = [
        preservation_score(src, gen)
        for src, gen in zip(scored["source"], scored["generated_until_question"], strict=False)
    ]
    scored["steered_question_and_preserved"] = (
        scored["question_mark_hit"].eq(1.0) & scored["content_preservation"].ge(0.67)
    ).astype(float)
    scored["strict_question_and_preserved"] = (
        scored["question_mark_hit"].eq(1.0) & scored["content_preservation_until_question"].ge(0.67)
    ).astype(float)

    scored.to_csv(CSV_DIR / "question_content_preservation_raw.csv", index=False)

    group_cols = ["source_set", "prompt_style", "control"]
    summary = (
        scored.groupby(group_cols)
        .agg(
            question_mark_rate=("question_mark_hit", "mean"),
            mean_content_preservation=("content_preservation", "mean"),
            mean_content_preservation_until_question=("content_preservation_until_question", "mean"),
            steered_question_and_preserved_rate=("steered_question_and_preserved", "mean"),
            strict_question_and_preserved_rate=("strict_question_and_preserved", "mean"),
            rows=("question_mark_hit", "count"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    summary.to_csv(CSV_DIR / "question_content_preservation_summary.csv", index=False)

    target_vs_controls = []
    for (source_set, prompt_style), group in summary.groupby(["source_set", "prompt_style"]):
        target = group[group["control"].eq("target")]
        controls = group[group["control"].isin(["none", "random_norm", "wrong_class", "negative_target"])]
        if target.empty or controls.empty:
            continue
        t = target.iloc[0]
        target_vs_controls.append(
            {
                "source_set": source_set,
                "prompt_style": prompt_style,
                "target_qmark": t["question_mark_rate"],
                "best_control_qmark": controls["question_mark_rate"].max(),
                "target_preserved": t["steered_question_and_preserved_rate"],
                "best_control_preserved": controls["steered_question_and_preserved_rate"].max(),
                "target_strict_preserved": t["strict_question_and_preserved_rate"],
                "best_control_strict_preserved": controls["strict_question_and_preserved_rate"].max(),
            }
        )
    contrast = pd.DataFrame(target_vs_controls)
    contrast.to_csv(CSV_DIR / "question_content_preservation_contrast.csv", index=False)

    print("Saved:")
    print(CSV_DIR / "question_content_preservation_raw.csv")
    print(CSV_DIR / "question_content_preservation_summary.csv")
    print(CSV_DIR / "question_content_preservation_contrast.csv")
    print()
    print(
        contrast.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()
