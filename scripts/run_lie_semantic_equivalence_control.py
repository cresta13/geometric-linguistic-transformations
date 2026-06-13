import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind, mannwhitneyu

warnings.filterwarnings("ignore")


class LieSemanticEquivalenceControl:
    def __init__(self):
        self.out_dir = Path("results/experiments/lie_semantic_equivalence_results")
        self.csv_dir = self.out_dir / "csv"
        self.fig_dir = self.out_dir / "figures"

        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = None
        self.pca_dim = 64

        self.models = [
            "bert-base-uncased",
            "distilroberta-base",
            "roberta-base",
        ]

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

    def log(self, msg: str):
        print(msg, flush=True)

    def build_dataset(self, n_templates: int = 100, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)

        combos = [(s, a) for s in self.subjects for a in self.actions]
        rng.shuffle(combos)
        combos = combos[:n_templates]

        rows = []

        for i, (subject, action) in enumerate(combos):
            past, base, obj = action
            source = f"The {subject} {past} {obj}."

            # ==================================================
            # EQUIVALENT CONTROLS
            # AB and BA are intended to be close in meaning.
            # ==================================================

            rows.append({
                "template_id": i,
                "source": source,
                "pair": "N_M_equivalent",
                "label": "equivalent",
                "ab_text": f"The {subject} allegedly failed to {base} {obj}.",
                "ba_text": f"Allegedly, the {subject} failed to {base} {obj}.",
            })

            rows.append({
                "template_id": i,
                "source": source,
                "pair": "M_T_equivalent",
                "label": "equivalent",
                "ab_text": f"The {subject} will allegedly {base} {obj} tomorrow.",
                "ba_text": f"Allegedly, the {subject} will {base} {obj} tomorrow.",
            })

            rows.append({
                "template_id": i,
                "source": source,
                "pair": "Q_M_equivalent",
                "label": "equivalent",
                "ab_text": f"Could the {subject} allegedly {base} {obj}?",
                "ba_text": f"Allegedly, could the {subject} {base} {obj}?",
            })

            rows.append({
                "template_id": i,
                "source": source,
                "pair": "Q_T_equivalent",
                "label": "equivalent",
                "ab_text": f"Will the {subject} {base} {obj} tomorrow?",
                "ba_text": f"Could the {subject} {base} {obj} tomorrow?",
            })

            # ==================================================
            # NON-EQUIVALENT CONTROLS
            # AB and BA are intended to differ in meaning/scope.
            # ==================================================

            rows.append({
                "template_id": i,
                "source": source,
                "pair": "N_Q_nonequivalent",
                "label": "non_equivalent",
                "ab_text": f"Could the {subject} fail to {base} {obj}?",
                "ba_text": f"Is it false that the {subject} {past} {obj}?",
            })

            rows.append({
                "template_id": i,
                "source": source,
                "pair": "N_T_nonequivalent",
                "label": "non_equivalent",
                "ab_text": f"The {subject} will fail to {base} {obj} tomorrow.",
                "ba_text": f"The {subject} failed to have {past} {obj} earlier.",
            })

            rows.append({
                "template_id": i,
                "source": source,
                "pair": "M_T_nonequivalent",
                "label": "non_equivalent",
                "ab_text": f"The {subject} will allegedly {base} {obj} tomorrow.",
                "ba_text": f"The {subject} was allegedly going to {base} {obj}.",
            })

            rows.append({
                "template_id": i,
                "source": source,
                "pair": "Q_M_nonequivalent",
                "label": "non_equivalent",
                "ab_text": f"Could the {subject} allegedly {base} {obj}?",
                "ba_text": f"Is it alleged that the {subject} {past} {obj}?",
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.csv_dir / "semantic_equivalence_dataset.csv", index=False)

        self.log("\nDATASET")
        self.log(str(df.groupby(["label", "pair"]).size()))
        self.log(f"Total rows: {len(df)}")

        return df

    @torch.no_grad()
    def get_embeddings(self, model_name: str, texts: list[str], batch_size: int = 16) -> np.ndarray:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"

        model = AutoModel.from_pretrained(model_name, token=self.hf_token).to(self.device)
        model.eval()

        embeddings = []

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
            embeddings.append(pooled.detach().cpu().numpy())

        del model
        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return np.vstack(embeddings)

    def compute_model_results(self, model_name: str, df: pd.DataFrame) -> pd.DataFrame:
        self.log("\n" + "=" * 80)
        self.log(f"MODEL: {model_name}")
        self.log("=" * 80)

        all_texts = sorted(
            set(df["source"])
            | set(df["ab_text"])
            | set(df["ba_text"])
        )

        idx = {text: i for i, text in enumerate(all_texts)}

        raw = self.get_embeddings(model_name, all_texts)

        dim = min(self.pca_dim, raw.shape[0] - 1, raw.shape[1])
        pca = PCA(n_components=dim, random_state=42)
        vecs = pca.fit_transform(raw)

        rows = []

        for row in df.itertuples():
            source = vecs[idx[row.source]]
            ab = vecs[idx[row.ab_text]]
            ba = vecs[idx[row.ba_text]]

            delta_ab = ab - source
            delta_ba = ba - source
            comm = delta_ab - delta_ba

            norm_ab = float(np.linalg.norm(delta_ab))
            norm_ba = float(np.linalg.norm(delta_ba))
            comm_norm = float(np.linalg.norm(comm))

            cosine = float(
                cosine_similarity(
                    delta_ab.reshape(1, -1),
                    delta_ba.reshape(1, -1),
                )[0, 0]
            )

            relative_comm_norm = comm_norm / (0.5 * (norm_ab + norm_ba) + 1e-12)
            noncomm_score = 1.0 - cosine

            rows.append({
                "model": model_name,
                "template_id": row.template_id,
                "pair": row.pair,
                "label": row.label,
                "source": row.source,
                "ab_text": row.ab_text,
                "ba_text": row.ba_text,
                "norm_ab": norm_ab,
                "norm_ba": norm_ba,
                "commutator_norm": comm_norm,
                "relative_commutator_norm": relative_comm_norm,
                "cosine_ab_ba": cosine,
                "noncommutativity_score": noncomm_score,
            })

        result = pd.DataFrame(rows)
        result.to_csv(
            self.csv_dir / f"semantic_equivalence_raw_{self.safe_name(model_name)}.csv",
            index=False,
        )

        return result

    def safe_name(self, model_name: str) -> str:
        return model_name.replace("/", "_").replace("-", "_")

    def aggregate_and_test(self, raw_df: pd.DataFrame):
        summary = (
            raw_df
            .groupby(["model", "label"])
            .agg(
                mean_noncommutativity=("noncommutativity_score", "mean"),
                std_noncommutativity=("noncommutativity_score", "std"),
                mean_relative_commutator_norm=("relative_commutator_norm", "mean"),
                std_relative_commutator_norm=("relative_commutator_norm", "std"),
                mean_cosine=("cosine_ab_ba", "mean"),
                std_cosine=("cosine_ab_ba", "std"),
                n=("label", "count"),
            )
            .reset_index()
        )

        by_pair = (
            raw_df
            .groupby(["model", "label", "pair"])
            .agg(
                mean_noncommutativity=("noncommutativity_score", "mean"),
                std_noncommutativity=("noncommutativity_score", "std"),
                mean_relative_commutator_norm=("relative_commutator_norm", "mean"),
                std_relative_commutator_norm=("relative_commutator_norm", "std"),
                mean_cosine=("cosine_ab_ba", "mean"),
                n=("pair", "count"),
            )
            .reset_index()
        )

        tests = []

        for model in raw_df["model"].unique():
            sub = raw_df[raw_df["model"] == model]

            eq = sub[sub["label"] == "equivalent"]["noncommutativity_score"].values
            neq = sub[sub["label"] == "non_equivalent"]["noncommutativity_score"].values

            t_stat, t_p = ttest_ind(neq, eq, equal_var=False)
            u_stat, u_p = mannwhitneyu(neq, eq, alternative="greater")

            tests.append({
                "model": model,
                "mean_equivalent": float(eq.mean()),
                "mean_non_equivalent": float(neq.mean()),
                "difference": float(neq.mean() - eq.mean()),
                "welch_t": float(t_stat),
                "welch_p": float(t_p),
                "mannwhitney_u": float(u_stat),
                "mannwhitney_p_greater": float(u_p),
            })

        tests_df = pd.DataFrame(tests)

        summary.to_csv(self.csv_dir / "semantic_equivalence_summary.csv", index=False)
        by_pair.to_csv(self.csv_dir / "semantic_equivalence_by_pair.csv", index=False)
        tests_df.to_csv(self.csv_dir / "semantic_equivalence_tests.csv", index=False)

        return summary, by_pair, tests_df

    def plot_summary(self, summary: pd.DataFrame):
        pivot = summary.pivot(
            index="model",
            columns="label",
            values="mean_noncommutativity",
        )

        plt.figure(figsize=(8, 5))
        x = np.arange(len(pivot.index))
        width = 0.35

        plt.bar(
            x - width / 2,
            pivot["equivalent"],
            width,
            label="Equivalent AB≈BA",
        )

        plt.bar(
            x + width / 2,
            pivot["non_equivalent"],
            width,
            label="Non-equivalent AB≠BA",
        )

        plt.xticks(x, pivot.index, rotation=25, ha="right")
        plt.ylabel("Mean noncommutativity score = 1 - cosine")
        plt.title("Semantic Equivalence Control for Lie-Style Commutators")
        plt.legend()
        plt.grid(axis="y")

        path = self.fig_dir / "01_equivalent_vs_nonequivalent.png"
        plt.tight_layout()
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()

        self.log(f"Saved figure: {path}")

    def plot_by_pair(self, by_pair: pd.DataFrame):
        for model in by_pair["model"].unique():
            sub = by_pair[by_pair["model"] == model].sort_values("mean_noncommutativity")

            plt.figure(figsize=(11, 5))
            colors = [
                "tab:blue" if label == "equivalent" else "tab:orange"
                for label in sub["label"]
            ]

            plt.bar(sub["pair"], sub["mean_noncommutativity"], color=colors)
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Mean noncommutativity score")
            plt.title(f"Semantic Equivalence Control by Pair — {model}")
            plt.grid(axis="y")

            path = self.fig_dir / f"02_by_pair_{self.safe_name(model)}.png"
            plt.tight_layout()
            plt.savefig(path, dpi=220, bbox_inches="tight")
            plt.close()

            self.log(f"Saved figure: {path}")

    def run(self):
        self.log(f"DEVICE: {self.device}")
        self.log(f"OUT DIR: {self.out_dir}")

        df = self.build_dataset(n_templates=100, seed=42)

        all_results = []

        for model in self.models:
            result = self.compute_model_results(model, df)
            all_results.append(result)

        raw_df = pd.concat(all_results, ignore_index=True)
        raw_df.to_csv(self.csv_dir / "semantic_equivalence_raw_all_models.csv", index=False)

        summary, by_pair, tests_df = self.aggregate_and_test(raw_df)

        self.plot_summary(summary)
        self.plot_by_pair(by_pair)

        metadata = {
            "goal": "Test whether commutator-like scores distinguish semantically equivalent AB/BA compositions from non-equivalent AB/BA compositions.",
            "metric": "noncommutativity_score = 1 - cosine(delta_AB, delta_BA)",
            "models": self.models,
        }

        with open(self.out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.log("\nSUMMARY")
        self.log(str(summary))

        self.log("\nSTATISTICAL TESTS")
        self.log(str(tests_df))

        self.log("\nDONE")


if __name__ == "__main__":
    audit = LieSemanticEquivalenceControl()
    audit.run()
