import gc
import os
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

from lie_composition_dataset import LieCompositionDataset, LieCompositionConfig

warnings.filterwarnings("ignore")


class LieCompositionAudit:
    def __init__(self):
        self.out_dir = Path(os.getenv("LIE_COMPOSITION_OUT_DIR", "lie_composition_results"))
        self.csv_dir = self.out_dir / "csv"
        self.fig_dir = self.out_dir / "figures"

        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = None
        self.pca_dim = 64

        self.models = [
            m.strip()
            for m in os.getenv(
                "LIE_COMPOSITION_MODELS",
                "bert-base-uncased,distilroberta-base,roberta-base",
            ).split(",")
            if m.strip()
        ]

    def log(self, msg: str):
        print(msg, flush=True)

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

    def build_dataset(self):
        dataset = LieCompositionDataset(
            LieCompositionConfig(
                n_templates=80,
                seed=42,
            )
        )

        df = dataset.save(self.csv_dir / "lie_composition_dataset.csv")

        self.log("\nDATASET")
        self.log(str(df.groupby("pair").size()))
        self.log(f"Total rows: {len(df)}")

        return df

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
            x = vecs[idx[row.source]]
            ab = vecs[idx[row.ab_text]]
            ba = vecs[idx[row.ba_text]]

            delta_ab = ab - x
            delta_ba = ba - x

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

            rows.append({
                "model": model_name,
                "template_id": row.template_id,
                "pair": row.pair,
                "op_a": row.op_a,
                "op_b": row.op_b,
                "source": row.source,
                "ab_text": row.ab_text,
                "ba_text": row.ba_text,
                "norm_ab": norm_ab,
                "norm_ba": norm_ba,
                "commutator_norm": comm_norm,
                "relative_commutator_norm": relative_comm_norm,
                "cosine_ab_ba": cosine,
                "noncommutativity_score": 1.0 - cosine,
            })

        result_df = pd.DataFrame(rows)
        result_df.to_csv(self.csv_dir / f"lie_composition_raw_{self.safe_name(model_name)}.csv", index=False)

        return result_df

    def safe_name(self, model_name: str) -> str:
        return model_name.replace("/", "_").replace("-", "_")

    def aggregate(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            raw_df
            .groupby(["model", "pair"])
            .agg(
                mean_commutator_norm=("commutator_norm", "mean"),
                std_commutator_norm=("commutator_norm", "std"),
                mean_relative_commutator_norm=("relative_commutator_norm", "mean"),
                std_relative_commutator_norm=("relative_commutator_norm", "std"),
                mean_cosine=("cosine_ab_ba", "mean"),
                std_cosine=("cosine_ab_ba", "std"),
                mean_noncommutativity=("noncommutativity_score", "mean"),
                std_noncommutativity=("noncommutativity_score", "std"),
                n=("pair", "count"),
            )
            .reset_index()
        )

        summary.to_csv(self.csv_dir / "lie_composition_summary.csv", index=False)
        return summary

    def plot_heatmap(self, summary: pd.DataFrame, value_col: str, filename: str, title: str):
        pairs = sorted(summary["pair"].unique())
        models = sorted(summary["model"].unique())

        mat = np.zeros((len(models), len(pairs)))

        for i, model in enumerate(models):
            for j, pair in enumerate(pairs):
                val = summary[
                    (summary["model"] == model)
                    & (summary["pair"] == pair)
                ][value_col].values

                mat[i, j] = val[0] if len(val) else np.nan

        plt.figure(figsize=(11, 5))
        plt.imshow(mat, aspect="auto")
        plt.colorbar(label=value_col)

        plt.xticks(np.arange(len(pairs)), pairs, rotation=35, ha="right")
        plt.yticks(np.arange(len(models)), models)

        plt.title(title)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")

        plt.tight_layout()
        path = self.fig_dir / filename
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()

        self.log(f"Saved figure: {path}")

    def plot_bar_by_pair(self, summary: pd.DataFrame):
        for model_name in summary["model"].unique():
            sub = summary[summary["model"] == model_name].sort_values("mean_noncommutativity")

            plt.figure(figsize=(10, 5))
            plt.bar(sub["pair"], sub["mean_noncommutativity"])
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("1 - cosine(delta_AB, delta_BA)")
            plt.title(f"Noncommutativity by Operation Pair — {model_name}")
            plt.grid(axis="y")

            path = self.fig_dir / f"noncommutativity_bar_{self.safe_name(model_name)}.png"
            plt.tight_layout()
            plt.savefig(path, dpi=220, bbox_inches="tight")
            plt.close()

            self.log(f"Saved figure: {path}")

    def run(self):
        self.log(f"DEVICE: {self.device}")
        self.log(f"OUT DIR: {self.out_dir}")

        df = self.build_dataset()

        all_results = []

        for model_name in self.models:
            result_df = self.compute_model_results(model_name, df)
            all_results.append(result_df)

        raw_df = pd.concat(all_results, ignore_index=True)
        raw_df.to_csv(self.csv_dir / "lie_composition_raw_all_models.csv", index=False)

        summary = self.aggregate(raw_df)

        self.plot_heatmap(
            summary,
            value_col="mean_noncommutativity",
            filename="01_noncommutativity_heatmap.png",
            title="Lie Composition Audit: Noncommutativity Score",
        )

        self.plot_heatmap(
            summary,
            value_col="mean_relative_commutator_norm",
            filename="02_relative_commutator_norm_heatmap.png",
            title="Lie Composition Audit: Relative Commutator Norm",
        )

        self.plot_heatmap(
            summary,
            value_col="mean_cosine",
            filename="03_composition_cosine_heatmap.png",
            title="Lie Composition Audit: Cosine Similarity of AB vs BA",
        )

        self.plot_bar_by_pair(summary)

        metadata = {
            "models": self.models,
            "pca_dim": self.pca_dim,
            "metrics": {
                "commutator_norm": "||delta_AB - delta_BA||",
                "relative_commutator_norm": "||delta_AB - delta_BA|| / mean(||delta_AB||, ||delta_BA||)",
                "cosine_ab_ba": "cosine(delta_AB, delta_BA)",
                "noncommutativity_score": "1 - cosine(delta_AB, delta_BA)",
            },
        }

        with open(self.out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.log("\nDONE")
        self.log(f"Results saved to: {self.out_dir}")


if __name__ == "__main__":
    audit = LieCompositionAudit()
    audit.run()
