from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "results" / "ablation_multiseed_summary.csv"
OUT_PATH = ROOT / "results" / "track1_multiseed_effect_intervals.csv"

# Two-sided 95% t critical value for n=5 seeds, df=4.
T_CRIT_95_DF4 = 2.7764451051977987


def main():
    df = pd.read_csv(IN_PATH)
    pivot = (
        df.pivot_table(
            index=["seed", "model", "classifier"],
            columns="representation",
            values="accuracy",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot["delta_minus_y_only"] = pivot["delta"] - pivot["y_only"]
    pivot["delta_minus_concat"] = pivot["delta"] - pivot["concat"]

    rows = []
    for (model, classifier), sub in pivot.groupby(["model", "classifier"], sort=True):
        for effect in ["delta_minus_y_only", "delta_minus_concat"]:
            values = sub[effect]
            n = int(values.count())
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            half_width = float(T_CRIT_95_DF4 * std / (n ** 0.5))
            rows.append(
                {
                    "model": model,
                    "classifier": classifier,
                    "effect": effect,
                    "mean": mean,
                    "std": std,
                    "n_seeds": n,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                    "all_seed_effects_positive": bool((values > 0).all()),
                }
            )

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
