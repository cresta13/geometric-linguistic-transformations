import os
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.linalg import orthogonal_procrustes
from scipy.stats import binomtest

from upat_dataset import UPATDataset, UPATDatasetConfig

warnings.filterwarnings("ignore")


class UPATAudit:
    def __init__(self):
        self.out_dir = Path("upat_large_results")
        self.csv_dir = self.out_dir / "csv"
        self.fig_dir = self.out_dir / "figures"

        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = None

        self.pca_dim = 64
        self.n_shuffles = 100
        self.n_repeats = 20
        self.seed = 42

        self.models = [
            "bert-base-uncased",
            "roberta-base",
            "distilroberta-base",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "microsoft/deberta-v3-base",
        ]

        self.procrustes_models = [
            "bert-base-uncased",
            "roberta-base",
            "distilroberta-base",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "microsoft/deberta-v3-base",
        ]

        self.rng = np.random.default_rng(self.seed)

        self.df = None
        self.labels = None
        self.y = None
        self.le = None
        self.train_mask = None
        self.test_mask = None
        self.baseline = None
        self.spaces = {}

    def log(self, text: str):
        print(text, flush=True)

    def build_dataset(self):
        dataset = UPATDataset(
            UPATDatasetConfig(
                train_templates=100,
                test_templates=100,
                seed_train=42,
                seed_test=123,
            )
        )

        self.df = dataset.save(self.csv_dir / "dataset.csv")

        self.log("\nDATASET")
        self.log(str(self.df.groupby(["split", "class"]).size().unstack(fill_value=0)))
        self.log(f"Total pairs: {len(self.df)}")

        self.labels = self.df["class"].values
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.labels)

        self.train_mask = self.df["split"].values == "train"
        self.test_mask = self.df["split"].values == "test"
        self.baseline = 1.0 / len(self.le.classes_)

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

    def make_spaces(self, model_name: str) -> dict:
        texts = sorted(set(self.df["source"]) | set(self.df["target"]))
        idx = {t: i for i, t in enumerate(texts)}

        raw = self.get_embeddings(model_name, texts)

        dim = min(self.pca_dim, raw.shape[0] - 1, raw.shape[1])
        pca = PCA(n_components=dim, random_state=self.seed)
        pca_vecs = pca.fit_transform(raw)

        x_raw = np.array([raw[idx[row.source]] for row in self.df.itertuples()])
        y_raw = np.array([raw[idx[row.target]] for row in self.df.itertuples()])
        delta_raw = y_raw - x_raw

        x_pca = np.array([pca_vecs[idx[row.source]] for row in self.df.itertuples()])
        y_pca = np.array([pca_vecs[idx[row.target]] for row in self.df.itertuples()])
        delta_pca = y_pca - x_pca

        return {
            "model": model_name,
            "texts": texts,
            "idx": idx,
            "raw": raw,
            "pca": pca_vecs,
            "x_pca": x_pca,
            "y_pca": y_pca,
            "delta_pca": delta_pca,
            "x_raw": x_raw,
            "y_raw": y_raw,
            "delta_raw": delta_raw,
        }

    def extract_all_spaces(self):
        for model_name in self.models:
            self.log("\n" + "=" * 80)
            self.log(f"EXTRACTING {model_name}")
            self.log("=" * 80)
            self.spaces[model_name] = self.make_spaces(model_name)

    def fit_predict(self, x_train, y_train, x_test, clf_type="logreg", seed=42):
        if clf_type == "svc":
            clf = LinearSVC(C=1.0, random_state=seed, max_iter=20000)
        else:
            clf = LogisticRegression(max_iter=5000, C=1.0, random_state=seed)

        clf.fit(x_train, y_train)
        return clf.predict(x_test)

    def eval_feature(self, x, clf_type="logreg", seed=42):
        pred = self.fit_predict(
            x[self.train_mask],
            self.y[self.train_mask],
            x[self.test_mask],
            clf_type=clf_type,
            seed=seed,
        )

        return {
            "acc": accuracy_score(self.y[self.test_mask], pred),
            "f1": f1_score(self.y[self.test_mask], pred, average="macro"),
            "pred": pred,
        }

    def mcnemar_p(self, correct_a, correct_b):
        b = int(np.sum(correct_a & ~correct_b))
        c = int(np.sum(~correct_a & correct_b))

        if b + c == 0:
            return 1.0, b, c

        return binomtest(min(b, c), n=b + c, p=0.5).pvalue, b, c

    def run_hard_holdout_and_shuffle(self):
        hard_rows = []
        shuffle_rows = []

        for model_name, sp in self.spaces.items():
            self.log(f"\nHard holdout: {model_name}")

            x_delta = normalize(sp["delta_pca"])
            true_eval = self.eval_feature(x_delta)

            hard_rows.append({
                "model": model_name,
                "acc": true_eval["acc"],
                "f1": true_eval["f1"],
                "baseline": self.baseline,
            })

            shuf_f1 = []

            for i in range(self.n_shuffles):
                y_train_shuf = self.rng.permutation(self.y[self.train_mask])

                pred = self.fit_predict(
                    x_delta[self.train_mask],
                    y_train_shuf,
                    x_delta[self.test_mask],
                    seed=1000 + i,
                )

                shuf_f1.append(f1_score(self.y[self.test_mask], pred, average="macro"))

            shuf_f1 = np.array(shuf_f1)

            shuffle_rows.append({
                "model": model_name,
                "true_f1": true_eval["f1"],
                "shuffle_mean_f1": shuf_f1.mean(),
                "shuffle_std_f1": shuf_f1.std(),
                "shuffle_max_f1": shuf_f1.max(),
                "p_empirical": ((shuf_f1 >= true_eval["f1"]).sum() + 1) / (self.n_shuffles + 1),
            })

        hard_df = pd.DataFrame(hard_rows)
        shuffle_df = pd.DataFrame(shuffle_rows)

        hard_df.to_csv(self.csv_dir / "hard_holdout.csv", index=False)
        shuffle_df.to_csv(self.csv_dir / "shuffle_control.csv", index=False)

        return hard_df, shuffle_df

    def run_ablation(self):
        rows = []
        mcnemar_rows = []

        for model_name, sp in self.spaces.items():
            self.log(f"\nAblation: {model_name}")

            features = {
                "x_only": normalize(sp["x_pca"]),
                "y_only": normalize(sp["y_pca"]),
                "delta": normalize(sp["delta_pca"]),
                "concat_xy": normalize(np.hstack([sp["x_pca"], sp["y_pca"]])),
            }

            preds = {}

            for name, x in features.items():
                ev = self.eval_feature(x, clf_type="svc")
                preds[name] = ev["pred"]

                rows.append({
                    "model": model_name,
                    "feature": name,
                    "acc": ev["acc"],
                    "f1": ev["f1"],
                    "baseline": self.baseline,
                })

            correct_delta = preds["delta"] == self.y[self.test_mask]
            correct_y = preds["y_only"] == self.y[self.test_mask]

            p, b, c = self.mcnemar_p(correct_delta, correct_y)

            mcnemar_rows.append({
                "model": model_name,
                "compare": "delta_vs_y_only",
                "delta_acc": accuracy_score(self.y[self.test_mask], preds["delta"]),
                "y_only_acc": accuracy_score(self.y[self.test_mask], preds["y_only"]),
                "diff": accuracy_score(self.y[self.test_mask], preds["delta"]) - accuracy_score(self.y[self.test_mask], preds["y_only"]),
                "b_delta_correct_y_wrong": b,
                "c_delta_wrong_y_correct": c,
                "p_value": p,
            })

        ablation_df = pd.DataFrame(rows)
        mcnemar_df = pd.DataFrame(mcnemar_rows)

        ablation_df.to_csv(self.csv_dir / "ablation.csv", index=False)
        mcnemar_df.to_csv(self.csv_dir / "mcnemar_delta_vs_y.csv", index=False)

        return ablation_df, mcnemar_df

    def run_magnitude_direction(self):
        rows = []

        for model_name, sp in self.spaces.items():
            self.log(f"\nMagnitude vs direction: {model_name}")

            deltas = sp["delta_pca"]
            norms = np.linalg.norm(deltas, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0, 1e-12, norms)

            feature_sets = {
                "norm_only": StandardScaler().fit_transform(norms),
                "unit_delta_direction_only": normalize(deltas / safe_norms),
                "full_delta_normalized": normalize(deltas),
                "raw_delta_scaled": StandardScaler().fit_transform(deltas),
            }

            for name, x in feature_sets.items():
                ev = self.eval_feature(x)

                rows.append({
                    "model": model_name,
                    "feature": name,
                    "acc": ev["acc"],
                    "f1": ev["f1"],
                    "baseline": self.baseline,
                })

        result_df = pd.DataFrame(rows)
        result_df.to_csv(self.csv_dir / "magnitude_direction.csv", index=False)

        return result_df

    def run_capacity_curve(self):
        rows = []

        train_indices = np.where(self.train_mask)[0]
        test_indices = np.where(self.test_mask)[0]

        train_sizes = [25, 50, 100, 200, 300, len(train_indices)]

        for model_name, sp in self.spaces.items():
            self.log(f"\nCapacity curve: {model_name}")

            x = normalize(sp["delta_pca"])

            for train_size in train_sizes:
                real_size = min(train_size, len(train_indices))

                for repeat in range(self.n_repeats):
                    sampled = self.rng.choice(train_indices, size=real_size, replace=False)

                    if len(np.unique(self.y[sampled])) < len(self.le.classes_):
                        continue

                    pred = self.fit_predict(
                        x[sampled],
                        self.y[sampled],
                        x[test_indices],
                        seed=2000 + repeat,
                    )

                    rows.append({
                        "model": model_name,
                        "train_size": real_size,
                        "repeat": repeat,
                        "acc": accuracy_score(self.y[test_indices], pred),
                        "f1": f1_score(self.y[test_indices], pred, average="macro"),
                    })

        raw_df = pd.DataFrame(rows)

        summary_df = (
            raw_df
            .groupby(["model", "train_size"])
            .agg(
                mean_f1=("f1", "mean"),
                std_f1=("f1", "std"),
                mean_acc=("acc", "mean"),
                n=("f1", "count"),
            )
            .reset_index()
        )

        raw_df.to_csv(self.csv_dir / "capacity_raw.csv", index=False)
        summary_df.to_csv(self.csv_dir / "capacity_summary.csv", index=False)

        return raw_df, summary_df

    def align_source_to_target(self, source_anchor, target_anchor, source_all, target_all):
        source_mean = source_anchor.mean(axis=0, keepdims=True)
        target_mean = target_anchor.mean(axis=0, keepdims=True)

        source_anchor_c = source_anchor - source_mean
        target_anchor_c = target_anchor - target_mean

        source_all_c = source_all - source_mean
        target_all_c = target_all - target_mean

        min_dim = min(
            source_anchor_c.shape[1],
            target_anchor_c.shape[1],
            source_all_c.shape[1],
            target_all_c.shape[1],
        )

        source_anchor_c = source_anchor_c[:, :min_dim]
        target_anchor_c = target_anchor_c[:, :min_dim]
        source_all_c = source_all_c[:, :min_dim]
        target_all_c = target_all_c[:, :min_dim]

        r, _ = orthogonal_procrustes(source_anchor_c, target_anchor_c)

        return source_all_c @ r, target_all_c

    def make_raw_deltas(self, model_name, raw_vectors):
        sp = self.spaces[model_name]
        idx = sp["idx"]

        return np.array([
            raw_vectors[idx[row.target]] - raw_vectors[idx[row.source]]
            for row in self.df.itertuples()
        ])

    def run_cross_model_procrustes(self):
        rows = []

        for train_model in self.procrustes_models:
            for test_model in self.procrustes_models:
                self.log(f"\nCross model: {train_model} -> {test_model}")

                source_sp = self.spaces[train_model]
                target_sp = self.spaces[test_model]

                # --------------------------------------------------
                # RAW BASELINE
                # Different models may have different embedding dims:
                # BERT=768, MiniLM=384, MPNet=768, etc.
                # So raw transfer is evaluated after separate PCA
                # into the same common dimension.
                # --------------------------------------------------

                common_dim = min(
                    self.pca_dim,
                    source_sp["delta_raw"].shape[0] - 1,
                    target_sp["delta_raw"].shape[0] - 1,
                    source_sp["delta_raw"].shape[1],
                    target_sp["delta_raw"].shape[1],
                )

                source_pca = PCA(
                    n_components=common_dim,
                    random_state=self.seed,
                )

                target_pca = PCA(
                    n_components=common_dim,
                    random_state=self.seed,
                )

                source_raw_projected = source_pca.fit_transform(
                    source_sp["delta_raw"]
                )

                target_raw_projected = target_pca.fit_transform(
                    target_sp["delta_raw"]
                )

                x_train_raw = normalize(source_raw_projected)
                x_test_raw = normalize(target_raw_projected)

                pred_raw = self.fit_predict(
                    x_train_raw[self.train_mask],
                    self.y[self.train_mask],
                    x_test_raw[self.test_mask],
                )

                raw_f1 = f1_score(
                    self.y[self.test_mask],
                    pred_raw,
                    average="macro",
                )

                raw_acc = accuracy_score(
                    self.y[self.test_mask],
                    pred_raw,
                )

                # --------------------------------------------------
                # PROCRUSTES ALIGNMENT
                # Align source raw embedding space to target raw space.
                # Works even if dimensions differ because
                # align_source_to_target trims to min_dim internally.
                # --------------------------------------------------

                common_texts = sorted(
                    set(source_sp["texts"])
                    & set(target_sp["texts"])
                )

                source_idx = [
                    source_sp["idx"][text]
                    for text in common_texts
                ]

                target_idx = [
                    target_sp["idx"][text]
                    for text in common_texts
                ]

                source_aligned, target_aligned = self.align_source_to_target(
                    source_anchor=source_sp["raw"][source_idx],
                    target_anchor=target_sp["raw"][target_idx],
                    source_all=source_sp["raw"],
                    target_all=target_sp["raw"],
                )

                source_deltas = self.make_raw_deltas(
                    train_model,
                    source_aligned,
                )

                target_deltas = self.make_raw_deltas(
                    test_model,
                    target_aligned,
                )

                x_train_aligned = normalize(source_deltas)
                x_test_aligned = normalize(target_deltas)

                pred_aligned = self.fit_predict(
                    x_train_aligned[self.train_mask],
                    self.y[self.train_mask],
                    x_test_aligned[self.test_mask],
                )

                aligned_f1 = f1_score(
                    self.y[self.test_mask],
                    pred_aligned,
                    average="macro",
                )

                aligned_acc = accuracy_score(
                    self.y[self.test_mask],
                    pred_aligned,
                )

                rows.append({
                    "train_model": train_model,
                    "test_model": test_model,
                    "raw_acc": raw_acc,
                    "raw_f1": raw_f1,
                    "aligned_acc": aligned_acc,
                    "aligned_f1": aligned_f1,
                    "gain_f1": aligned_f1 - raw_f1,
                    "common_dim_raw_pca": common_dim,
                    "source_raw_dim": source_sp["delta_raw"].shape[1],
                    "target_raw_dim": target_sp["delta_raw"].shape[1],
                })

        result_df = pd.DataFrame(rows)

        result_df.to_csv(
            self.csv_dir / "cross_model_procrustes.csv",
            index=False,
        )

        return result_df

    def run_alignment_curve(self):
        source_model = "roberta-base"
        target_model = "bert-base-uncased"

        source_sp = self.spaces[source_model]
        target_sp = self.spaces[target_model]

        train_df = self.df[self.df["split"] == "train"]
        train_anchor_texts = sorted(set(train_df["source"]) | set(train_df["target"]))

        source_anchor_all = np.array([source_sp["idx"][t] for t in train_anchor_texts])
        target_anchor_all = np.array([target_sp["idx"][t] for t in train_anchor_texts])

        sizes = [2, 4, 8, 16, 32, 64, 100, len(train_anchor_texts)]
        rows = []

        for size in sizes:
            real_size = min(size, len(train_anchor_texts))
            self.log(f"\nAlignment size: {real_size}")

            for repeat in range(self.n_repeats):
                sample_pos = self.rng.choice(
                    np.arange(len(train_anchor_texts)),
                    size=real_size,
                    replace=False,
                )

                source_anchor_idx = source_anchor_all[sample_pos]
                target_anchor_idx = target_anchor_all[sample_pos]

                source_aligned, target_aligned = self.align_source_to_target(
                    source_sp["raw"][source_anchor_idx],
                    target_sp["raw"][target_anchor_idx],
                    source_sp["raw"],
                    target_sp["raw"],
                )

                source_deltas = self.make_raw_deltas(source_model, source_aligned)
                target_deltas = self.make_raw_deltas(target_model, target_aligned)

                pred = self.fit_predict(
                    normalize(source_deltas)[self.train_mask],
                    self.y[self.train_mask],
                    normalize(target_deltas)[self.test_mask],
                    seed=3000 + repeat,
                )

                rows.append({
                    "source_model": source_model,
                    "target_model": target_model,
                    "alignment_size": real_size,
                    "repeat": repeat,
                    "acc": accuracy_score(self.y[self.test_mask], pred),
                    "f1": f1_score(self.y[self.test_mask], pred, average="macro"),
                })

        raw_df = pd.DataFrame(rows)

        summary_df = (
            raw_df
            .groupby("alignment_size")
            .agg(
                mean_f1=("f1", "mean"),
                std_f1=("f1", "std"),
                mean_acc=("acc", "mean"),
                n=("f1", "count"),
            )
            .reset_index()
        )

        raw_df.to_csv(self.csv_dir / "alignment_curve_raw.csv", index=False)
        summary_df.to_csv(self.csv_dir / "alignment_curve_summary.csv", index=False)

        return raw_df, summary_df

    def savefig(self, name):
        path = self.fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()
        self.log(f"Saved figure: {path}")

    def plot_all(self, hard_df, shuffle_df, ablation_df, magdir_df, capacity_summary, cross_df, align_summary, mcnemar_df):
        x = np.arange(len(hard_df))

        plt.figure(figsize=(9, 5))
        plt.bar(x - 0.2, hard_df["f1"], width=0.4, label="True delta F1")
        plt.bar(x + 0.2, shuffle_df["shuffle_mean_f1"], width=0.4, label="Shuffle mean F1")
        plt.axhline(self.baseline, linestyle="--", label="Random baseline")
        plt.xticks(x, hard_df["model"], rotation=25, ha="right")
        plt.ylabel("Macro F1")
        plt.title("Hard Semantic Holdout vs Label Shuffle")
        plt.legend()
        self.savefig("01_hard_holdout_vs_shuffle.png")

        ablation_df.pivot(index="model", columns="feature", values="f1").plot(kind="bar", figsize=(10, 5))
        plt.axhline(self.baseline, linestyle="--")
        plt.ylabel("Macro F1")
        plt.title("Representation Ablation")
        self.savefig("02_ablation.png")

        magdir_df.pivot(index="model", columns="feature", values="f1").plot(kind="bar", figsize=(10, 5))
        plt.axhline(self.baseline, linestyle="--")
        plt.ylabel("Macro F1")
        plt.title("Magnitude vs Direction")
        self.savefig("03_magnitude_direction.png")

        plt.figure(figsize=(9, 5))
        for model_name in capacity_summary["model"].unique():
            sub = capacity_summary[capacity_summary["model"] == model_name].sort_values("train_size")
            plt.errorbar(sub["train_size"], sub["mean_f1"], yerr=sub["std_f1"], marker="o", capsize=3, label=model_name)
        plt.axhline(self.baseline, linestyle="--")
        plt.xlabel("Training pairs")
        plt.ylabel("Macro F1")
        plt.title("Capacity Curve")
        plt.grid(True)
        plt.legend()
        self.savefig("04_capacity_curve.png")

        raw_mat = cross_df.pivot(index="train_model", columns="test_model", values="raw_f1")
        plt.figure(figsize=(7, 6))
        plt.imshow(raw_mat.values, aspect="auto")
        plt.colorbar(label="Macro F1")
        plt.xticks(np.arange(len(raw_mat.columns)), raw_mat.columns, rotation=25, ha="right")
        plt.yticks(np.arange(len(raw_mat.index)), raw_mat.index)
        plt.title("Cross-Model Transfer: Raw")
        for i in range(raw_mat.shape[0]):
            for j in range(raw_mat.shape[1]):
                plt.text(j, i, f"{raw_mat.values[i, j]:.2f}", ha="center", va="center")
        self.savefig("05_cross_model_raw.png")

        aligned_mat = cross_df.pivot(index="train_model", columns="test_model", values="aligned_f1")
        plt.figure(figsize=(7, 6))
        plt.imshow(aligned_mat.values, aspect="auto")
        plt.colorbar(label="Macro F1")
        plt.xticks(np.arange(len(aligned_mat.columns)), aligned_mat.columns, rotation=25, ha="right")
        plt.yticks(np.arange(len(aligned_mat.index)), aligned_mat.index)
        plt.title("Cross-Model Transfer: Procrustes Aligned")
        for i in range(aligned_mat.shape[0]):
            for j in range(aligned_mat.shape[1]):
                plt.text(j, i, f"{aligned_mat.values[i, j]:.2f}", ha="center", va="center")
        self.savefig("06_cross_model_aligned.png")

        plt.figure(figsize=(8, 5))
        plt.errorbar(
            align_summary["alignment_size"],
            align_summary["mean_f1"],
            yerr=align_summary["std_f1"],
            marker="o",
            capsize=4,
        )
        plt.xlabel("Aligned train text pairs")
        plt.ylabel("Cross-model Macro F1")
        plt.title("Alignment Sample Curve: RoBERTa → BERT")
        plt.grid(True)
        self.savefig("07_alignment_curve.png")

        plt.figure(figsize=(9, 5))
        plt.bar(mcnemar_df["model"], mcnemar_df["diff"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Accuracy gain: delta - y_only")
        plt.title("Delta Adds Information Beyond Target Embedding")
        plt.grid(axis="y")
        self.savefig("08_delta_gain.png")

    def run(self):
        self.log(f"DEVICE: {self.device}")
        self.log(f"OUT DIR: {self.out_dir}")

        self.build_dataset()
        self.extract_all_spaces()

        hard_df, shuffle_df = self.run_hard_holdout_and_shuffle()
        ablation_df, mcnemar_df = self.run_ablation()
        magdir_df = self.run_magnitude_direction()
        _, capacity_summary = self.run_capacity_curve()
        cross_df = self.run_cross_model_procrustes()
        _, align_summary = self.run_alignment_curve()

        self.plot_all(
            hard_df,
            shuffle_df,
            ablation_df,
            magdir_df,
            capacity_summary,
            cross_df,
            align_summary,
            mcnemar_df,
        )

        summary = {
            "dataset_size": len(self.df),
            "classes": list(self.le.classes_),
            "models": self.models,
            "baseline": self.baseline,
        }

        with open(self.out_dir / "master_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.log("\nDONE")
        self.log(f"Results saved to: {self.out_dir}")


if __name__ == "__main__":
    audit = UPATAudit()
    audit.run()