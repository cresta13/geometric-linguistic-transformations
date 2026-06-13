import gc
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


class LieAlgebraicIdentitiesAudit:
    def __init__(self):
        self.out_dir = Path(os.getenv("LIE_ALGEBRA_OUT_DIR", "results/experiments/lie_algebraic_identities_results"))
        self.csv_dir = self.out_dir / "csv"
        self.fig_dir = self.out_dir / "figures"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = None
        self.pca_dim = 64
        self.n_jacobi_null = 1000
        self.n_bootstrap = 2000

        self.models = [
            model.strip()
            for model in os.getenv(
                "LIE_ALGEBRA_MODELS",
                "bert-base-uncased,distilroberta-base,roberta-base",
            ).split(",")
            if model.strip()
        ]

        self.ops = ["N", "Q", "M", "T"]

        self.subjects = [
            "scientist", "engineer", "teacher", "doctor", "programmer",
            "researcher", "analyst", "manager", "wizard", "dragon",
            "queen", "robot", "pirate", "oracle", "knight", "alien",
        ]

        self.actions = [
            ("accepted", "accept", "the explanation"),
            ("completed", "complete", "the repair"),
            ("confirmed", "confirm", "the answer"),
            ("approved", "approve", "the treatment"),
            ("fixed", "fix", "the bug"),
            ("supported", "support", "the theory"),
            ("verified", "verify", "the report"),
            ("guarded", "guard", "the treasure"),
            ("opened", "open", "the portal"),
            ("signed", "sign", "the treaty"),
        ]

    def log(self, msg):
        print(msg, flush=True)

    def base_sentence(self, subject, past, obj):
        return f"The {subject} {past} {obj}."

    def compose(self, seq, subject, past, base, obj):
        s = "".join(seq)

        mapping = {
            "": f"The {subject} {past} {obj}.",

            "N": f"The {subject} failed to {base} {obj}.",
            "Q": f"Could the {subject} {base} {obj}?",
            "M": f"The {subject} allegedly {past} {obj}.",
            "T": f"The {subject} will {base} {obj} tomorrow.",

            "NQ": f"Could the {subject} fail to {base} {obj}?",
            "QN": f"Is it false that the {subject} {past} {obj}?",

            "NM": f"The {subject} allegedly failed to {base} {obj}.",
            "MN": f"Allegedly, the {subject} failed to {base} {obj}.",

            "NT": f"The {subject} will fail to {base} {obj} tomorrow.",
            "TN": f"The {subject} failed to have {past} {obj} earlier.",

            "QM": f"Could the {subject} allegedly {base} {obj}?",
            "MQ": f"Is it alleged that the {subject} {past} {obj}?",

            "QT": f"Could the {subject} {base} {obj} tomorrow?",
            "TQ": f"Will the {subject} {base} {obj} tomorrow?",

            "MT": f"The {subject} will allegedly {base} {obj} tomorrow.",
            "TM": f"The {subject} was allegedly going to {base} {obj}.",
        }

        if s in mapping:
            return mapping[s]

        # canonical triple templates
        if s == "NQM":
            return f"Could the {subject} allegedly fail to {base} {obj}?"
        if s == "QMN":
            return f"Is it false that the {subject} allegedly {past} {obj}?"
        if s == "MNQ":
            return f"Is it alleged that the {subject} failed to {base} {obj}?"

        if s == "QNM":
            return f"Could it be false that the {subject} allegedly {past} {obj}?"
        if s == "NMQ":
            return f"Could it be alleged that the {subject} failed to {base} {obj}?"
        if s == "MQN":
            return f"Allegedly, is it false that the {subject} {past} {obj}?"

        if s == "NMT":
            return f"The {subject} will allegedly fail to {base} {obj} tomorrow."
        if s == "MTN":
            return f"The {subject} was allegedly going to fail to {base} {obj}."
        if s == "TNM":
            return f"Earlier, the {subject} allegedly failed to have {past} {obj}."

        if s == "NTM":
            return f"Allegedly, the {subject} will fail to {base} {obj} tomorrow."
        if s == "TMN":
            return f"Earlier, it was alleged that the {subject} failed to have {past} {obj}."
        if s == "MNT":
            return f"It is alleged that the {subject} will fail to {base} {obj} tomorrow."

        if s == "NQT":
            return f"Could the {subject} fail to {base} {obj} tomorrow?"
        if s == "QTN":
            return f"Is it false that the {subject} will {base} {obj} tomorrow?"
        if s == "TNQ":
            return f"Could the {subject} have failed to {base} {obj} earlier?"

        if s == "QNT":
            return f"Could it be false that the {subject} will {base} {obj} tomorrow?"
        if s == "NTQ":
            return f"Could it be that the {subject} will fail to {base} {obj} tomorrow?"
        if s == "TQN":
            return f"Will it be false that the {subject} {base} {obj} tomorrow?"

        if s == "QMT":
            return f"Could the {subject} allegedly {base} {obj} tomorrow?"
        if s == "MTQ":
            return f"Could it be alleged that the {subject} will {base} {obj} tomorrow?"
        if s == "TQM":
            return f"Allegedly, will the {subject} {base} {obj} tomorrow?"

        if s == "MQT":
            return f"Will it be alleged that the {subject} {base} {obj} tomorrow?"
        if s == "QTM":
            return f"Allegedly, could the {subject} {base} {obj} tomorrow?"
        if s == "TMQ":
            return f"Could the {subject} have allegedly planned to {base} {obj}?"

        raise ValueError(f"Unsupported composition: {s}")

    def build_dataset(self, n_templates=100, seed=42):
        rng = np.random.default_rng(seed)
        combos = [(s, a) for s in self.subjects for a in self.actions]
        rng.shuffle(combos)
        combos = combos[:n_templates]

        rows = []

        pairs = [("N", "Q"), ("N", "M"), ("N", "T"), ("Q", "M"), ("Q", "T"), ("M", "T")]
        triples = [("N", "Q", "M"), ("N", "Q", "T"), ("N", "M", "T"), ("Q", "M", "T")]

        for i, (subject, action) in enumerate(combos):
            past, base, obj = action
            source = self.base_sentence(subject, past, obj)

            for a, b in pairs:
                rows.append({
                    "template_id": i,
                    "kind": "antisymmetry",
                    "source": source,
                    "a": a,
                    "b": b,
                    "c": "",
                    "ab": self.compose([a, b], subject, past, base, obj),
                    "ba": self.compose([b, a], subject, past, base, obj),
                })

            for a, b, c in triples:
                rows.append({
                    "template_id": i,
                    "kind": "jacobi",
                    "source": source,
                    "a": a,
                    "b": b,
                    "c": c,
                    "abc": self.compose([a, b, c], subject, past, base, obj),
                    "bca": self.compose([b, c, a], subject, past, base, obj),
                    "cab": self.compose([c, a, b], subject, past, base, obj),
                    "acb": self.compose([a, c, b], subject, past, base, obj),
                    "cba": self.compose([c, b, a], subject, past, base, obj),
                    "bac": self.compose([b, a, c], subject, past, base, obj),
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.csv_dir / "lie_algebraic_identities_dataset.csv", index=False)
        return df

    @torch.no_grad()
    def get_embeddings(self, model_name, texts, batch_size=16):
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"

        model = AutoModel.from_pretrained(model_name, token=self.hf_token).to(self.device)
        model.eval()

        embs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)

            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embs.append(pooled.detach().cpu().numpy())

        del model
        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return np.vstack(embs)

    def cos(self, x, y):
        return float(cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0])

    def alternating_null_stats(self, vectors, observed_norm, scale, rng):
        null_relative = []
        n = len(vectors)

        for _ in range(self.n_jacobi_null):
            positive = set(rng.choice(n, size=n // 2, replace=False).tolist())
            signed_sum = np.zeros_like(vectors[0])

            for i, vector in enumerate(vectors):
                signed_sum += vector if i in positive else -vector

            null_relative.append(np.linalg.norm(signed_sum) / scale)

        null_relative = np.asarray(null_relative, dtype=float)
        observed_relative = observed_norm / scale

        return {
            "null_mean_relative_jacobi_norm": float(null_relative.mean()),
            "null_std_relative_jacobi_norm": float(null_relative.std(ddof=1)),
            "jacobi_to_null_mean_ratio": float(observed_relative / (null_relative.mean() + 1e-12)),
            "null_percentile_smaller_is_better": float((null_relative <= observed_relative).mean()),
        }

    def bootstrap_mean_ci(self, values, rng, alpha=0.05):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return np.nan, np.nan

        draws = rng.choice(values, size=(self.n_bootstrap, len(values)), replace=True).mean(axis=1)
        return (
            float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)),
        )

    def compute_for_model(self, model_name, df):
        self.log("\n" + "=" * 80)
        self.log(f"MODEL: {model_name}")
        self.log("=" * 80)

        text_cols = ["source", "ab", "ba", "abc", "bca", "cab", "acb", "cba", "bac"]
        all_texts = set()

        for col in text_cols:
            if col in df.columns:
                all_texts |= set(df[col].dropna().values)

        all_texts = sorted(all_texts)
        idx = {t: i for i, t in enumerate(all_texts)}

        raw = self.get_embeddings(model_name, all_texts)

        dim = min(self.pca_dim, raw.shape[0] - 1, raw.shape[1])
        pca = PCA(n_components=dim, random_state=42)
        vecs = pca.fit_transform(raw)

        anti_rows = []
        jacobi_rows = []

        anti_df = df[df["kind"] == "antisymmetry"]

        for row in anti_df.itertuples():
            x = vecs[idx[row.source]]
            ab = vecs[idx[row.ab]]
            ba = vecs[idx[row.ba]]

            delta_ab = ab - x
            delta_ba = ba - x

            comm_ab = delta_ab - delta_ba
            comm_ba = delta_ba - delta_ab

            antisym_error = np.linalg.norm(comm_ab + comm_ba)
            comm_norm = np.linalg.norm(comm_ab)

            anti_rows.append({
                "model": model_name,
                "template_id": row.template_id,
                "pair": f"{row.a}{row.b}_vs_{row.b}{row.a}",
                "a": row.a,
                "b": row.b,
                "comm_norm": float(comm_norm),
                "antisym_error": float(antisym_error),
                "relative_antisym_error": float(antisym_error / (comm_norm + 1e-12)),
                "cos_comm_ab_neg_comm_ba": self.cos(comm_ab, -comm_ba),
            })

        jacobi_df = df[df["kind"] == "jacobi"]
        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(model_name)) % (2 ** 32)
        rng = np.random.default_rng(seed)

        for row in jacobi_df.itertuples():
            x = vecs[idx[row.source]]

            abc = vecs[idx[row.abc]] - x
            bca = vecs[idx[row.bca]] - x
            cab = vecs[idx[row.cab]] - x

            acb = vecs[idx[row.acb]] - x
            cba = vecs[idx[row.cba]] - x
            bac = vecs[idx[row.bac]] - x

            # Jacobi-like alternating sum over permutations:
            # J = ABC + BCA + CAB - ACB - CBA - BAC
            jacobi = abc + bca + cab - acb - cba - bac

            positive_norm = np.linalg.norm(abc) + np.linalg.norm(bca) + np.linalg.norm(cab)
            negative_norm = np.linalg.norm(acb) + np.linalg.norm(cba) + np.linalg.norm(bac)
            scale = 0.5 * (positive_norm + negative_norm) + 1e-12

            jacobi_norm = np.linalg.norm(jacobi)
            null_stats = self.alternating_null_stats(
                [abc, bca, cab, acb, cba, bac],
                jacobi_norm,
                scale,
                rng,
            )

            jacobi_rows.append({
                "model": model_name,
                "template_id": row.template_id,
                "triple": f"{row.a}{row.b}{row.c}",
                "a": row.a,
                "b": row.b,
                "c": row.c,
                "jacobi_norm": float(jacobi_norm),
                "relative_jacobi_norm": float(jacobi_norm / scale),
                **null_stats,
            })

        anti_result = pd.DataFrame(anti_rows)
        jacobi_result = pd.DataFrame(jacobi_rows)

        anti_result.to_csv(self.csv_dir / f"antisymmetry_raw_{self.safe_name(model_name)}.csv", index=False)
        jacobi_result.to_csv(self.csv_dir / f"jacobi_raw_{self.safe_name(model_name)}.csv", index=False)

        return anti_result, jacobi_result

    def safe_name(self, model_name):
        return model_name.replace("/", "_").replace("-", "_")

    def summarize(self, anti_all, jacobi_all):
        anti_summary = (
            anti_all.groupby(["model", "pair"])
            .agg(
                mean_comm_norm=("comm_norm", "mean"),
                mean_relative_antisym_error=("relative_antisym_error", "mean"),
                mean_cos_comm_ab_neg_comm_ba=("cos_comm_ab_neg_comm_ba", "mean"),
                n=("pair", "count"),
            )
            .reset_index()
        )

        jacobi_summary = (
            jacobi_all.groupby(["model", "triple"])
            .agg(
                mean_jacobi_norm=("jacobi_norm", "mean"),
                mean_relative_jacobi_norm=("relative_jacobi_norm", "mean"),
                mean_null_relative_jacobi_norm=("null_mean_relative_jacobi_norm", "mean"),
                mean_jacobi_to_null_mean_ratio=("jacobi_to_null_mean_ratio", "mean"),
                mean_null_percentile_smaller_is_better=("null_percentile_smaller_is_better", "mean"),
                n=("triple", "count"),
            )
            .reset_index()
        )

        rng = np.random.default_rng(20260611)
        ci_rows = []

        for row in jacobi_summary.itertuples():
            group = jacobi_all[
                (jacobi_all["model"] == row.model)
                & (jacobi_all["triple"] == row.triple)
            ]

            rel_low, rel_high = self.bootstrap_mean_ci(group["relative_jacobi_norm"].values, rng)
            ratio_low, ratio_high = self.bootstrap_mean_ci(group["jacobi_to_null_mean_ratio"].values, rng)

            ci_rows.append({
                "model": row.model,
                "triple": row.triple,
                "relative_jacobi_norm_ci95_low": rel_low,
                "relative_jacobi_norm_ci95_high": rel_high,
                "jacobi_to_null_mean_ratio_ci95_low": ratio_low,
                "jacobi_to_null_mean_ratio_ci95_high": ratio_high,
            })

        jacobi_summary = jacobi_summary.merge(pd.DataFrame(ci_rows), on=["model", "triple"], how="left")

        anti_summary.to_csv(self.csv_dir / "antisymmetry_summary.csv", index=False)
        jacobi_summary.to_csv(self.csv_dir / "jacobi_summary.csv", index=False)

        return anti_summary, jacobi_summary

    def plot_heatmap(self, df, index_col, value_col, filename, title):
        rows = sorted(df["model"].unique())
        cols = sorted(df[index_col].unique())

        mat = np.zeros((len(rows), len(cols)))

        for i, model in enumerate(rows):
            for j, col in enumerate(cols):
                val = df[(df["model"] == model) & (df[index_col] == col)][value_col].values
                mat[i, j] = val[0] if len(val) else np.nan

        plt.figure(figsize=(10, 5))
        plt.imshow(mat, aspect="auto")
        plt.colorbar(label=value_col)

        plt.xticks(np.arange(len(cols)), cols, rotation=35, ha="right")
        plt.yticks(np.arange(len(rows)), rows)
        plt.title(title)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")

        path = self.fig_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()
        self.log(f"Saved figure: {path}")

    def run(self):
        self.log(f"DEVICE: {self.device}")
        self.log(f"OUT DIR: {self.out_dir}")

        df = self.build_dataset(n_templates=100, seed=42)

        anti_results = []
        jacobi_results = []

        for model in self.models:
            anti, jacobi = self.compute_for_model(model, df)
            anti_results.append(anti)
            jacobi_results.append(jacobi)

        anti_all = pd.concat(anti_results, ignore_index=True)
        jacobi_all = pd.concat(jacobi_results, ignore_index=True)

        anti_all.to_csv(self.csv_dir / "antisymmetry_raw_all_models.csv", index=False)
        jacobi_all.to_csv(self.csv_dir / "jacobi_raw_all_models.csv", index=False)

        anti_summary, jacobi_summary = self.summarize(anti_all, jacobi_all)

        self.plot_heatmap(
            anti_summary,
            "pair",
            "mean_cos_comm_ab_neg_comm_ba",
            "01_antisymmetry_cosine_heatmap.png",
            "Antisymmetry Check: cos([A,B], -[B,A])",
        )

        self.plot_heatmap(
            anti_summary,
            "pair",
            "mean_relative_antisym_error",
            "02_antisymmetry_error_heatmap.png",
            "Antisymmetry Error",
        )

        self.plot_heatmap(
            jacobi_summary,
            "triple",
            "mean_relative_jacobi_norm",
            "03_jacobi_relative_norm_heatmap.png",
            "Jacobi-like Alternating Sum Relative Norm",
        )

        self.plot_heatmap(
            jacobi_summary,
            "triple",
            "mean_jacobi_to_null_mean_ratio",
            "04_jacobi_vs_permutation_null_heatmap.png",
            "Jacobi-like Norm / Permutation Null Mean",
        )

        metadata = {
            "antisymmetry": "checks whether [A,B] ≈ -[B,A]",
            "jacobi_like": "checks alternating sum ABC+BCA+CAB-ACB-CBA-BAC",
            "jacobi_null": (
                "compares the observed alternating sum with random 3-positive/3-negative "
                "sign assignments over the same six composition vectors; lower ratio and "
                "lower percentile mean stronger Jacobi-like cancellation"
            ),
            "caution": (
                "A literal nested-commutator Jacobi expansion over the same six embedded "
                "composition endpoints cancels by construction, so this script reports a "
                "non-tautological alternating third-order cancellation diagnostic instead."
            ),
            "note": "This is an empirical Lie-style diagnostic, not a formal proof of Lie algebra.",
            "models": self.models,
            "n_jacobi_null": self.n_jacobi_null,
            "n_bootstrap": self.n_bootstrap,
            "jacobi_triples": ["NQM", "NQT", "NMT", "QMT"],
        }

        with open(self.out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.log("\nANTISYMMETRY SUMMARY")
        self.log(str(anti_summary))

        self.log("\nJACOBI SUMMARY")
        self.log(str(jacobi_summary))

        self.log("\nDONE")


if __name__ == "__main__":
    audit = LieAlgebraicIdentitiesAudit()
    audit.run()
