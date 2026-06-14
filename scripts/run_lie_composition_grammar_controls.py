import gc
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import percentileofscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")


class GrammarCompositionDataset:
    SUBJECTS = [
        "scientist", "engineer", "teacher", "doctor", "programmer",
        "researcher", "analyst", "manager", "architect", "designer",
        "editor", "planner", "auditor", "coordinator", "operator", "consultant",
    ]

    ACTIONS = [
        ("accepted", "accept", "the explanation"),
        ("completed", "complete", "the repair"),
        ("confirmed", "confirm", "the answer"),
        ("approved", "approve", "the treatment"),
        ("fixed", "fix", "the bug"),
        ("supported", "support", "the theory"),
        ("verified", "verify", "the report"),
        ("reviewed", "review", "the proposal"),
        ("updated", "update", "the schedule"),
        ("submitted", "submit", "the request"),
    ]

    CONTEXTS = [
        "during the meeting",
        "after the review",
        "before the deadline",
        "in the morning",
        "at the office",
        "for the committee",
        "under the policy",
        "with the team",
    ]

    NEG_FORMS = ["failed to {base}", "did not {base}", "refused to {base}"]
    MOD_FORMS = ["allegedly", "reportedly", "apparently"]
    FUTURE_FORMS = ["will {base}", "is going to {base}", "plans to {base}"]
    PAIRS = [("N", "Q"), ("N", "M"), ("N", "T"), ("Q", "M"), ("Q", "T"), ("M", "T")]

    def __init__(self, n_templates=120, seed=20260614):
        self.n_templates = n_templates
        self.rng = np.random.default_rng(seed)

    def sample_form(self):
        return {
            "neg": self.rng.choice(self.NEG_FORMS),
            "mod": self.rng.choice(self.MOD_FORMS),
            "future": self.rng.choice(self.FUTURE_FORMS),
            "question": int(self.rng.integers(0, 3)),
            "context": self.rng.choice(self.CONTEXTS),
        }

    def base_sentence(self, subject, past, obj, context):
        return f"The {subject} {past} {obj} {context}."

    def phrase(self, ops, subject, past, base, obj, form):
        ops = list(ops)
        context = form["context"]

        negated = "N" in ops
        modal = "M" in ops
        future = "T" in ops
        question = "Q" in ops

        if future:
            vp = form["future"].format(base=base)
            vp_past = f"was going to {base}"
        elif negated:
            vp = form["neg"].format(base=base)
            vp_past = f"failed to {base}"
        else:
            vp = base
            vp_past = past

        if negated and future:
            if ops.index("N") < ops.index("T"):
                vp = f"will fail to {base}"
                vp_past = f"would fail to {base}"
            else:
                vp = f"will not {base}"
                vp_past = f"would not {base}"

        if modal:
            marker = form["mod"]
            if ops.index("M") == 0:
                vp = f"{marker} {vp}"
                vp_past = f"{marker} {vp_past}"
            elif negated and not future:
                vp = f"did not {marker} {base}"
                vp_past = f"did not {marker} {base}"
            elif future:
                vp = f"will {marker} {base}"
                vp_past = f"was going to {marker} {base}"
            else:
                vp = f"{marker} {vp}"
                vp_past = f"{marker} {vp_past}"

        if question:
            q_before_neg = negated and ops.index("Q") < ops.index("N")
            q_before_mod = modal and ops.index("Q") < ops.index("M")
            q_before_future = future and ops.index("Q") < ops.index("T")

            if q_before_neg:
                positive_vp = form["future"].format(base=base) if future else past
                if modal:
                    positive_vp = f"{form['mod']} {positive_vp}"
                return f"Is it false that the {subject} {positive_vp} {obj} {context}?"

            if q_before_mod:
                positive_vp = form["future"].format(base=base) if future else (f"failed to {base}" if negated else past)
                return f"Is it {form['mod']} true that the {subject} {positive_vp} {obj} {context}?"

            if q_before_future:
                positive_vp = f"fail to {base}" if negated else form["future"].format(base=base)
                if modal:
                    positive_vp = f"{form['mod']} {positive_vp}"
                return f"Will it be true that the {subject} {positive_vp} {obj} {context}?"

            if future:
                q_vp = f"fail to {base}" if negated else base
                if modal:
                    q_vp = f"{form['mod']} {q_vp}"
                return f"Will the {subject} {q_vp} {obj} {context}?"

            if negated:
                q_vp = f"fail to {base}"
                if modal:
                    q_vp = f"{form['mod']} {q_vp}"
                return f"Did the {subject} {q_vp} {obj} {context}?"

            if modal:
                return f"Did the {subject} {form['mod']} {base} {obj} {context}?"

            return f"Did the {subject} {base} {obj} {context}?"

        sentence = f"The {subject} {vp} {obj} {context}."
        return " ".join(sentence.split())

    def build(self):
        combos = [(s, a) for s in self.SUBJECTS for a in self.ACTIONS]
        self.rng.shuffle(combos)
        combos = combos[: self.n_templates]

        rows = []
        for template_id, (subject, action) in enumerate(combos):
            past, base, obj = action
            form = self.sample_form()
            source = self.base_sentence(subject, past, obj, form["context"])

            for a, b in self.PAIRS:
                ab = self.phrase([a, b], subject, past, base, obj, form)
                ba = self.phrase([b, a], subject, past, base, obj, form)
                rows.append({
                    "template_id": template_id,
                    "subject": subject,
                    "action_base": base,
                    "object": obj,
                    "context": form["context"],
                    "op_a": a,
                    "op_b": b,
                    "pair": f"{a}{b}_vs_{b}{a}",
                    "source": source,
                    "ab_text": ab,
                    "ba_text": ba,
                })

        return pd.DataFrame(rows)


class GrammarCompositionAudit:
    def __init__(self):
        self.out_dir = Path(os.getenv(
            "LIE_GRAMMAR_COMPOSITION_OUT_DIR",
            "results/experiments/lie_composition_grammar_results",
        ))
        self.csv_dir = self.out_dir / "csv"
        self.fig_dir = self.out_dir / "figures"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = None
        self.pca_dim = 64
        self.seed = 20260614
        self.n_null = int(os.getenv("LIE_GRAMMAR_COMPOSITION_N_NULL", "1000"))
        self.models = [
            model.strip()
            for model in os.getenv(
                "LIE_GRAMMAR_COMPOSITION_MODELS",
                "bert-base-uncased,distilroberta-base,roberta-base",
            ).split(",")
            if model.strip()
        ]
        self.rng = np.random.default_rng(self.seed)

    def log(self, msg):
        print(msg, flush=True)

    def safe_name(self, model_name):
        return model_name.replace("/", "_").replace("-", "_")

    @torch.no_grad()
    def get_embeddings(self, model_name, texts, batch_size=16):
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
        df = GrammarCompositionDataset().build()
        df.to_csv(self.csv_dir / "grammar_composition_dataset.csv", index=False)
        self.log("\nDATASET")
        self.log(str(df.groupby("pair").size()))
        self.log(f"Total rows: {len(df)}")
        return df

    def compute_model_vectors(self, model_name, df):
        self.log("\n" + "=" * 80)
        self.log(f"MODEL: {model_name}")
        self.log("=" * 80)

        texts = sorted(set(df["source"]) | set(df["ab_text"]) | set(df["ba_text"]))
        idx = {text: i for i, text in enumerate(texts)}
        raw = self.get_embeddings(model_name, texts)

        dim = min(self.pca_dim, raw.shape[0] - 1, raw.shape[1])
        vecs = PCA(n_components=dim, random_state=42).fit_transform(raw)

        rows = []
        features = {
            "source": [],
            "ab_endpoint": [],
            "ba_endpoint": [],
            "ab_delta": [],
            "ba_delta": [],
            "commutator": [],
            "pair": [],
            "template_id": [],
        }

        for row in df.itertuples(index=False):
            source = vecs[idx[row.source]]
            ab = vecs[idx[row.ab_text]]
            ba = vecs[idx[row.ba_text]]
            delta_ab = ab - source
            delta_ba = ba - source
            comm = delta_ab - delta_ba

            norm_ab = float(np.linalg.norm(delta_ab))
            norm_ba = float(np.linalg.norm(delta_ba))
            comm_norm = float(np.linalg.norm(comm))
            scale = 0.5 * (norm_ab + norm_ba) + 1e-12
            cosine = float(cosine_similarity(delta_ab.reshape(1, -1), delta_ba.reshape(1, -1))[0, 0])

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
                "relative_commutator_norm": comm_norm / scale,
                "cosine_ab_ba": cosine,
                "noncommutativity_score": 1.0 - cosine,
            })

            features["source"].append(source)
            features["ab_endpoint"].append(ab)
            features["ba_endpoint"].append(ba)
            features["ab_delta"].append(delta_ab)
            features["ba_delta"].append(delta_ba)
            features["commutator"].append(comm)
            features["pair"].append(row.pair)
            features["template_id"].append(row.template_id)

        result = pd.DataFrame(rows)
        for key in ["source", "ab_endpoint", "ba_endpoint", "ab_delta", "ba_delta", "commutator"]:
            features[key] = np.vstack(features[key])
        features["pair"] = np.array(features["pair"])
        features["template_id"] = np.array(features["template_id"])
        return result, features

    def endpoint_controls(self, model_name, features):
        pairs = sorted(set(features["pair"]))
        pair_to_id = {pair: i for i, pair in enumerate(pairs)}
        y = np.array([pair_to_id[pair] for pair in features["pair"]])
        template_ids = features["template_id"]

        train_mask = template_ids < np.median(template_ids)
        test_mask = ~train_mask

        feature_sets = {
            "source_only": features["source"],
            "ab_endpoint_only": features["ab_endpoint"],
            "ba_endpoint_only": features["ba_endpoint"],
            "ab_delta_only": features["ab_delta"],
            "ba_delta_only": features["ba_delta"],
            "commutator_delta": features["commutator"],
            "endpoint_concat": np.hstack([features["ab_endpoint"], features["ba_endpoint"]]),
        }

        rows = []
        for name, x in feature_sets.items():
            clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
            clf.fit(normalize(x[train_mask]), y[train_mask])
            pred = clf.predict(normalize(x[test_mask]))
            rows.append({
                "model": model_name,
                "feature": name,
                "accuracy": accuracy_score(y[test_mask], pred),
                "macro_f1": f1_score(y[test_mask], pred, average="macro"),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "chance": 1.0 / len(pairs),
            })

        return pd.DataFrame(rows)

    def commutator_nulls_from_features(self, model_name, features):
        rows = []
        for pair in sorted(set(features["pair"])):
            mask = features["pair"] == pair
            delta_ab = features["ab_delta"][mask]
            delta_ba = features["ba_delta"][mask]
            norm_ab = np.linalg.norm(delta_ab, axis=1)
            norm_ba = np.linalg.norm(delta_ba, axis=1)
            observed = np.linalg.norm(delta_ab - delta_ba, axis=1) / (0.5 * (norm_ab + norm_ba) + 1e-12)
            observed_mean = float(observed.mean())

            null_specs = {
                "random_pairing_same_pair": lambda: (delta_ab, delta_ba[self.rng.permutation(len(delta_ba))]),
                "random_pairing_any_pair": lambda: (
                    features["ab_delta"][self.rng.choice(len(features["ab_delta"]), size=len(delta_ab), replace=False)],
                    features["ba_delta"][self.rng.choice(len(features["ba_delta"]), size=len(delta_ba), replace=False)],
                ),
                "random_direction_norm_matched": lambda: self.random_direction_vectors(norm_ab, norm_ba),
            }

            for null_type, make_null in null_specs.items():
                null_means = []
                for _ in range(self.n_null):
                    random_ab, random_ba = make_null()
                    random_norm_ab = np.linalg.norm(random_ab, axis=1)
                    random_norm_ba = np.linalg.norm(random_ba, axis=1)
                    random_comm = np.linalg.norm(random_ab - random_ba, axis=1)
                    random_scale = 0.5 * (random_norm_ab + random_norm_ba) + 1e-12
                    null_means.append(float(np.mean(random_comm / random_scale)))

                null = np.array(null_means)
                rows.append({
                    "model": model_name,
                    "pair": pair,
                    "null_type": null_type,
                    "observed_mean_relative_commutator_norm": observed_mean,
                    "null_mean": float(null.mean()),
                    "null_std": float(null.std(ddof=1)),
                    "null_p05": float(np.percentile(null, 5)),
                    "null_p95": float(np.percentile(null, 95)),
                    "observed_percentile_smaller_is_more_coherent": float(percentileofscore(null, observed_mean, kind="mean")),
                    "p_smaller_or_equal_null": float(((null <= observed_mean).sum() + 1) / (len(null) + 1)),
                    "p_larger_or_equal_null": float(((null >= observed_mean).sum() + 1) / (len(null) + 1)),
                    "n_null": len(null),
                })

        return pd.DataFrame(rows)

    def random_direction_vectors(self, norm_ab, norm_ba):
        dim = self.pca_dim
        u = self.rng.normal(size=(len(norm_ab), dim))
        v = self.rng.normal(size=(len(norm_ba), dim))
        u = normalize(u)
        v = normalize(v)
        return u * norm_ab[:, None], v * norm_ba[:, None]

    def summarize(self, raw_df):
        return (
            raw_df
            .groupby(["model", "pair"])
            .agg(
                mean_relative_commutator_norm=("relative_commutator_norm", "mean"),
                std_relative_commutator_norm=("relative_commutator_norm", "std"),
                mean_noncommutativity=("noncommutativity_score", "mean"),
                std_noncommutativity=("noncommutativity_score", "std"),
                mean_cosine=("cosine_ab_ba", "mean"),
                n=("pair", "count"),
            )
            .reset_index()
        )

    def plot_summary(self, summary):
        pairs = sorted(summary["pair"].unique())
        models = sorted(summary["model"].unique())
        mat = np.zeros((len(models), len(pairs)))
        for i, model in enumerate(models):
            for j, pair in enumerate(pairs):
                val = summary[(summary["model"] == model) & (summary["pair"] == pair)]["mean_relative_commutator_norm"]
                mat[i, j] = val.iloc[0] if len(val) else np.nan

        plt.figure(figsize=(11, 5))
        plt.imshow(mat, aspect="auto")
        plt.colorbar(label="Mean relative commutator norm")
        plt.xticks(np.arange(len(pairs)), pairs, rotation=35, ha="right")
        plt.yticks(np.arange(len(models)), models)
        plt.title("Grammar-generated composition: relative commutator norm")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "01_relative_commutator_norm_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close()

    def run(self):
        self.log(f"DEVICE: {self.device}")
        self.log(f"OUT DIR: {self.out_dir}")
        df = self.build_dataset()

        raw_parts = []
        control_parts = []
        null_parts = []
        for model_name in self.models:
            raw_model, features = self.compute_model_vectors(model_name, df)
            raw_model.to_csv(self.csv_dir / f"grammar_composition_raw_{self.safe_name(model_name)}.csv", index=False)
            raw_parts.append(raw_model)
            control_parts.append(self.endpoint_controls(model_name, features))
            null_parts.append(self.commutator_nulls_from_features(model_name, features))

        raw_df = pd.concat(raw_parts, ignore_index=True)
        controls = pd.concat(control_parts, ignore_index=True)
        summary = self.summarize(raw_df)
        nulls = pd.concat(null_parts, ignore_index=True)

        raw_df.to_csv(self.csv_dir / "grammar_composition_raw_all_models.csv", index=False)
        controls.to_csv(self.csv_dir / "grammar_endpoint_controls.csv", index=False)
        summary.to_csv(self.csv_dir / "grammar_composition_summary.csv", index=False)
        nulls.to_csv(self.csv_dir / "grammar_commutator_nulls.csv", index=False)
        self.plot_summary(summary)

        metadata = {
            "models": self.models,
            "pca_dim": self.pca_dim,
            "n_null": self.n_null,
            "note": "Grammar-generated Track 2 control: endpoint controls plus random-direction commutator norm nulls.",
        }
        (self.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self.log("\nDONE")
        self.log(f"Results saved to: {self.out_dir}")


if __name__ == "__main__":
    GrammarCompositionAudit().run()
