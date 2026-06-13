# ============================================================
# UPAT MASTER AUDIT
# Universal Procrustes Alignment Test for Transformation Geometry
# ============================================================

import os, gc, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.linalg import orthogonal_procrustes
from scipy.stats import binomtest

# ----------------------------
# CONFIG
# ----------------------------

OUT_DIR = "results/experiments/upat_audit_results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUT_DIR}/csv", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = None

PCA_DIM = 64
N_SHUFFLES = 100
N_REPEATS = 30
RANDOM_SEED = 42

MODELS = [
    "bert-base-uncased",
    "distilroberta-base",
    "roberta-base",
    "gpt2",
    "distilgpt2",
]

PROCRUSTES_MODELS = [
    "bert-base-uncased",
    "distilroberta-base",
    "roberta-base",
]

rng = np.random.default_rng(RANDOM_SEED)

print("DEVICE:", DEVICE)
print("OUT_DIR:", OUT_DIR)


# ============================================================
# DATASET
# ============================================================

HARD_TRAIN = [
    {
        "source": "The scientist accepted the explanation.",
        "negation": "The scientist rejected the explanation.",
        "question": "Could the scientist accept the explanation?",
        "modality": "The scientist apparently accepted the explanation.",
        "tense_shift": "The scientist used to accept the explanation.",
        "passive": "The explanation received acceptance from the scientist.",
    },
    {
        "source": "The engineer completed the repair.",
        "negation": "The engineer failed to complete the repair.",
        "question": "Could the engineer complete the repair?",
        "modality": "The engineer reportedly completed the repair.",
        "tense_shift": "The engineer had completed the repair earlier.",
        "passive": "The repair received completion by the engineer.",
    },
    {
        "source": "The teacher confirmed the answer.",
        "negation": "The teacher denied the answer.",
        "question": "Could the teacher confirm the answer?",
        "modality": "The teacher apparently confirmed the answer.",
        "tense_shift": "The teacher had confirmed the answer earlier.",
        "passive": "The answer received confirmation from the teacher.",
    },
    {
        "source": "The doctor approved the treatment.",
        "negation": "The doctor refused to approve the treatment.",
        "question": "Could the doctor approve the treatment?",
        "modality": "The doctor reportedly approved the treatment.",
        "tense_shift": "The doctor used to approve the treatment.",
        "passive": "The treatment received approval from the doctor.",
    },
    {
        "source": "The programmer fixed the bug.",
        "negation": "The programmer failed to fix the bug.",
        "question": "Could the programmer fix the bug?",
        "modality": "The programmer apparently fixed the bug.",
        "tense_shift": "The programmer had fixed the bug earlier.",
        "passive": "The bug received a fix from the programmer.",
    },
    {
        "source": "The researcher supported the theory.",
        "negation": "The researcher rejected the theory.",
        "question": "Could the researcher support the theory?",
        "modality": "The researcher reportedly supported the theory.",
        "tense_shift": "The researcher used to support the theory.",
        "passive": "The theory received support from the researcher.",
    },
    {
        "source": "The analyst verified the report.",
        "negation": "The analyst failed to verify the report.",
        "question": "Could the analyst verify the report?",
        "modality": "The analyst apparently verified the report.",
        "tense_shift": "The analyst had verified the report earlier.",
        "passive": "The report received verification from the analyst.",
    },
    {
        "source": "The manager approved the plan.",
        "negation": "The manager declined to approve the plan.",
        "question": "Could the manager approve the plan?",
        "modality": "The manager reportedly approved the plan.",
        "tense_shift": "The manager used to approve the plan.",
        "passive": "The plan received approval from the manager.",
    },
]

HARD_TEST = [
    {
        "source": "The dragon guarded the treasure.",
        "negation": "The dragon avoided guarding the treasure.",
        "question": "I wonder whether the dragon guarded the treasure.",
        "modality": "The dragon allegedly guarded the treasure.",
        "tense_shift": "The dragon will guard the treasure tomorrow.",
        "passive": "The treasure came under the dragon's guard.",
    },
    {
        "source": "The wizard opened the portal.",
        "negation": "The wizard avoided opening the portal.",
        "question": "I wonder whether the wizard opened the portal.",
        "modality": "The wizard allegedly opened the portal.",
        "tense_shift": "The wizard will open the portal tomorrow.",
        "passive": "The portal came open through the wizard's action.",
    },
    {
        "source": "The queen signed the treaty.",
        "negation": "The queen declined to sign the treaty.",
        "question": "I wonder whether the queen signed the treaty.",
        "modality": "The queen allegedly signed the treaty.",
        "tense_shift": "The queen will sign the treaty tomorrow.",
        "passive": "The treaty gained the queen's signature.",
    },
    {
        "source": "The robot repaired the satellite.",
        "negation": "The robot avoided repairing the satellite.",
        "question": "I wonder whether the robot repaired the satellite.",
        "modality": "The robot allegedly repaired the satellite.",
        "tense_shift": "The robot will repair the satellite tomorrow.",
        "passive": "The satellite came back repaired through the robot.",
    },
    {
        "source": "The pirate found the island.",
        "negation": "The pirate failed to find the island.",
        "question": "I wonder whether the pirate found the island.",
        "modality": "The pirate allegedly found the island.",
        "tense_shift": "The pirate will find the island tomorrow.",
        "passive": "The island became the pirate's discovery.",
    },
    {
        "source": "The oracle predicted the storm.",
        "negation": "The oracle refused to predict the storm.",
        "question": "I wonder whether the oracle predicted the storm.",
        "modality": "The oracle allegedly predicted the storm.",
        "tense_shift": "The oracle will predict the storm tomorrow.",
        "passive": "The storm became the oracle's prediction.",
    },
    {
        "source": "The alien built the tower.",
        "negation": "The alien avoided building the tower.",
        "question": "I wonder whether the alien built the tower.",
        "modality": "The alien allegedly built the tower.",
        "tense_shift": "The alien will build the tower tomorrow.",
        "passive": "The tower became the alien's construction.",
    },
    {
        "source": "The knight protected the village.",
        "negation": "The knight failed to protect the village.",
        "question": "I wonder whether the knight protected the village.",
        "modality": "The knight allegedly protected the village.",
        "tense_shift": "The knight will protect the village tomorrow.",
        "passive": "The village came under the knight's protection.",
    },
]


def build_df(train_items, test_items):
    rows = []
    classes = ["negation", "question", "modality", "tense_shift", "passive"]
    for split_name, items in [("train", train_items), ("test", test_items)]:
        for item in items:
            for cls in classes:
                rows.append({
                    "source": item["source"],
                    "target": item[cls],
                    "class": cls,
                    "split": split_name,
                })
    return pd.DataFrame(rows)

df = build_df(HARD_TRAIN, HARD_TEST)
df.to_csv(f"{OUT_DIR}/csv/dataset.csv", index=False)

print("Dataset")
print(df.groupby(["split", "class"]).size().unstack(fill_value=0))


# ============================================================
# EMBEDDINGS
# ============================================================

@torch.no_grad()
def get_embeddings(model_name, texts, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"

    model = AutoModel.from_pretrained(model_name, token=HF_TOKEN).to(DEVICE)
    model.eval()

    embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(DEVICE)

        out = model(**enc)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embs.append(pooled.detach().cpu().numpy())

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return np.vstack(embs)


def make_spaces(model_name, df):
    texts = sorted(set(df["source"]) | set(df["target"]))
    idx = {t: i for i, t in enumerate(texts)}

    raw = get_embeddings(model_name, texts)

    dim = min(PCA_DIM, raw.shape[0] - 1, raw.shape[1])
    pca = PCA(n_components=dim, random_state=RANDOM_SEED)
    pca_vecs = pca.fit_transform(raw)

    def deltas_from(vecs):
        return np.array([
            vecs[idx[row.target]] - vecs[idx[row.source]]
            for row in df.itertuples()
        ])

    x_raw = np.array([raw[idx[row.source]] for row in df.itertuples()])
    y_raw = np.array([raw[idx[row.target]] for row in df.itertuples()])
    delta_raw = y_raw - x_raw

    x_pca = np.array([pca_vecs[idx[row.source]] for row in df.itertuples()])
    y_pca = np.array([pca_vecs[idx[row.target]] for row in df.itertuples()])
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


spaces = {}

for model_name in MODELS:
    print("\n" + "=" * 80)
    print("EXTRACTING:", model_name)
    print("=" * 80)
    spaces[model_name] = make_spaces(model_name, df)


# ============================================================
# EVALUATION HELPERS
# ============================================================

labels = df["class"].values
splits = df["split"].values

le = LabelEncoder()
y = le.fit_transform(labels)

train_mask = splits == "train"
test_mask = splits == "test"
baseline = 1.0 / len(le.classes_)


def fit_predict(X_train, y_train, X_test, clf_type="logreg", seed=42):
    if clf_type == "svc":
        clf = LinearSVC(C=1.0, random_state=seed, max_iter=10000)
    else:
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def eval_feature(X, clf_type="logreg", seed=42):
    pred = fit_predict(
        X[train_mask],
        y[train_mask],
        X[test_mask],
        clf_type=clf_type,
        seed=seed,
    )
    return {
        "acc": accuracy_score(y[test_mask], pred),
        "f1": f1_score(y[test_mask], pred, average="macro"),
        "pred": pred,
    }


def mcnemar_p(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return 1.0, b, c
    return binomtest(min(b, c), n=b+c, p=0.5).pvalue, b, c


# ============================================================
# EXPERIMENT 1: HARD HOLDOUT + SHUFFLE CONTROL
# ============================================================

hard_rows = []
shuffle_rows = []

for model_name, sp in spaces.items():
    X_delta = normalize(sp["delta_pca"])
    true_eval = eval_feature(X_delta)

    hard_rows.append({
        "model": model_name,
        "acc": true_eval["acc"],
        "f1": true_eval["f1"],
        "baseline": baseline,
    })

    shuf_f1 = []
    shuf_acc = []

    for i in range(N_SHUFFLES):
        y_train_shuf = rng.permutation(y[train_mask])
        pred = fit_predict(
            X_delta[train_mask],
            y_train_shuf,
            X_delta[test_mask],
            seed=1000+i,
        )
        shuf_f1.append(f1_score(y[test_mask], pred, average="macro"))
        shuf_acc.append(accuracy_score(y[test_mask], pred))

    shuf_f1 = np.array(shuf_f1)

    shuffle_rows.append({
        "model": model_name,
        "true_f1": true_eval["f1"],
        "shuffle_mean_f1": shuf_f1.mean(),
        "shuffle_std_f1": shuf_f1.std(),
        "shuffle_max_f1": shuf_f1.max(),
        "p_empirical": ((shuf_f1 >= true_eval["f1"]).sum() + 1) / (N_SHUFFLES + 1),
    })

hard_df = pd.DataFrame(hard_rows)
shuffle_df = pd.DataFrame(shuffle_rows)

hard_df.to_csv(f"{OUT_DIR}/csv/hard_holdout.csv", index=False)
shuffle_df.to_csv(f"{OUT_DIR}/csv/shuffle_control.csv", index=False)


# ============================================================
# EXPERIMENT 2: ABLATION
# x_only, y_only, delta, concat
# ============================================================

ablation_rows = []
mcnemar_rows = []

for model_name, sp in spaces.items():
    features = {
        "x_only": normalize(sp["x_pca"]),
        "y_only": normalize(sp["y_pca"]),
        "delta": normalize(sp["delta_pca"]),
        "concat_xy": normalize(np.hstack([sp["x_pca"], sp["y_pca"]])),
    }

    preds = {}

    for feat_name, X in features.items():
        ev = eval_feature(X, clf_type="svc")
        preds[feat_name] = ev["pred"]

        ablation_rows.append({
            "model": model_name,
            "feature": feat_name,
            "acc": ev["acc"],
            "f1": ev["f1"],
            "baseline": baseline,
        })

    correct_delta = preds["delta"] == y[test_mask]
    correct_y = preds["y_only"] == y[test_mask]

    p, b, c = mcnemar_p(correct_delta, correct_y)

    mcnemar_rows.append({
        "model": model_name,
        "compare": "delta_vs_y_only",
        "delta_acc": accuracy_score(y[test_mask], preds["delta"]),
        "y_only_acc": accuracy_score(y[test_mask], preds["y_only"]),
        "diff": accuracy_score(y[test_mask], preds["delta"]) - accuracy_score(y[test_mask], preds["y_only"]),
        "b_delta_correct_y_wrong": b,
        "c_delta_wrong_y_correct": c,
        "p_value": p,
    })

ablation_df = pd.DataFrame(ablation_rows)
mcnemar_df = pd.DataFrame(mcnemar_rows)

ablation_df.to_csv(f"{OUT_DIR}/csv/ablation.csv", index=False)
mcnemar_df.to_csv(f"{OUT_DIR}/csv/mcnemar_delta_vs_y.csv", index=False)


# ============================================================
# EXPERIMENT 3: MAGNITUDE VS DIRECTION
# ============================================================

magdir_rows = []

for model_name, sp in spaces.items():
    deltas = sp["delta_pca"]
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1e-12, norms)

    feature_sets = {
        "norm_only": StandardScaler().fit_transform(norms),
        "unit_delta_direction_only": normalize(deltas / safe_norms),
        "full_delta_normalized": normalize(deltas),
        "raw_delta_scaled": StandardScaler().fit_transform(deltas),
    }

    for feat_name, X in feature_sets.items():
        ev = eval_feature(X)
        magdir_rows.append({
            "model": model_name,
            "feature": feat_name,
            "acc": ev["acc"],
            "f1": ev["f1"],
            "baseline": baseline,
        })

magdir_df = pd.DataFrame(magdir_rows)
magdir_df.to_csv(f"{OUT_DIR}/csv/magnitude_direction.csv", index=False)


# ============================================================
# EXPERIMENT 4: CAPACITY CURVE
# ============================================================

capacity_rows = []
train_indices_all = np.where(train_mask)[0]
test_indices = np.where(test_mask)[0]

train_sizes = [10, 15, 20, 25, len(train_indices_all)]

for model_name, sp in spaces.items():
    X = normalize(sp["delta_pca"])

    for train_size in train_sizes:
        real_size = min(train_size, len(train_indices_all))

        for repeat in range(N_REPEATS):
            sampled = rng.choice(train_indices_all, size=real_size, replace=False)

            if len(np.unique(y[sampled])) < len(le.classes_):
                continue

            pred = fit_predict(
                X[sampled],
                y[sampled],
                X[test_indices],
                seed=2000+repeat,
            )

            capacity_rows.append({
                "model": model_name,
                "train_size": real_size,
                "repeat": repeat,
                "acc": accuracy_score(y[test_indices], pred),
                "f1": f1_score(y[test_indices], pred, average="macro"),
                "baseline": baseline,
            })

capacity_df = pd.DataFrame(capacity_rows)
capacity_summary = (
    capacity_df.groupby(["model", "train_size"])
    .agg(mean_f1=("f1", "mean"), std_f1=("f1", "std"), mean_acc=("acc", "mean"), n=("f1", "count"))
    .reset_index()
)

capacity_df.to_csv(f"{OUT_DIR}/csv/capacity_raw.csv", index=False)
capacity_summary.to_csv(f"{OUT_DIR}/csv/capacity_summary.csv", index=False)


# ============================================================
# EXPERIMENT 5: CROSS-MODEL RAW + PROCRUSTES
# ============================================================

def align_source_to_target(source_anchor, target_anchor, source_all, target_all):
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

    R, scale = orthogonal_procrustes(source_anchor_c, target_anchor_c)

    return source_all_c @ R, target_all_c


def make_raw_deltas_for_model(model_name, raw_vectors):
    texts = spaces[model_name]["texts"]
    idx = {t: i for i, t in enumerate(texts)}
    return np.array([
        raw_vectors[idx[row.target]] - raw_vectors[idx[row.source]]
        for row in df.itertuples()
    ])


cross_rows = []

for train_model in PROCRUSTES_MODELS:
    for test_model in PROCRUSTES_MODELS:
        source_sp = spaces[train_model]
        target_sp = spaces[test_model]

        X_train_raw = normalize(source_sp["delta_raw"])
        X_test_raw = normalize(target_sp["delta_raw"])

        pred_raw = fit_predict(
            X_train_raw[train_mask],
            y[train_mask],
            X_test_raw[test_mask],
        )

        raw_f1 = f1_score(y[test_mask], pred_raw, average="macro")
        raw_acc = accuracy_score(y[test_mask], pred_raw)

        common_texts = sorted(set(source_sp["texts"]) & set(target_sp["texts"]))

        source_idx = [source_sp["idx"][t] for t in common_texts]
        target_idx = [target_sp["idx"][t] for t in common_texts]

        source_aligned, target_aligned = align_source_to_target(
            source_sp["raw"][source_idx],
            target_sp["raw"][target_idx],
            source_sp["raw"],
            target_sp["raw"],
        )

        source_deltas_aligned = make_raw_deltas_for_model(train_model, source_aligned)
        target_deltas_aligned = make_raw_deltas_for_model(test_model, target_aligned)

        pred_aligned = fit_predict(
            normalize(source_deltas_aligned)[train_mask],
            y[train_mask],
            normalize(target_deltas_aligned)[test_mask],
        )

        aligned_f1 = f1_score(y[test_mask], pred_aligned, average="macro")
        aligned_acc = accuracy_score(y[test_mask], pred_aligned)

        cross_rows.append({
            "train_model": train_model,
            "test_model": test_model,
            "raw_acc": raw_acc,
            "raw_f1": raw_f1,
            "aligned_acc": aligned_acc,
            "aligned_f1": aligned_f1,
            "gain_f1": aligned_f1 - raw_f1,
        })

cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(f"{OUT_DIR}/csv/cross_model_procrustes.csv", index=False)


# ============================================================
# EXPERIMENT 6: ALIGNMENT SAMPLE CURVE
# RoBERTa -> BERT
# ============================================================

source_model = "roberta-base"
target_model = "bert-base-uncased"

source_sp = spaces[source_model]
target_sp = spaces[target_model]

train_df = df[df["split"] == "train"]
train_anchor_texts = sorted(set(train_df["source"]) | set(train_df["target"]))

source_anchor_all = np.array([source_sp["idx"][t] for t in train_anchor_texts])
target_anchor_all = np.array([target_sp["idx"][t] for t in train_anchor_texts])

align_sizes = [2, 4, 6, 8, 10, 15, 20, len(train_anchor_texts)]
align_curve_rows = []

for size in align_sizes:
    real_size = min(size, len(train_anchor_texts))

    for repeat in range(N_REPEATS):
        sample_pos = rng.choice(np.arange(len(train_anchor_texts)), size=real_size, replace=False)

        source_anchor_idx = source_anchor_all[sample_pos]
        target_anchor_idx = target_anchor_all[sample_pos]

        source_aligned, target_aligned = align_source_to_target(
            source_sp["raw"][source_anchor_idx],
            target_sp["raw"][target_anchor_idx],
            source_sp["raw"],
            target_sp["raw"],
        )

        source_deltas = make_raw_deltas_for_model(source_model, source_aligned)
        target_deltas = make_raw_deltas_for_model(target_model, target_aligned)

        pred = fit_predict(
            normalize(source_deltas)[train_mask],
            y[train_mask],
            normalize(target_deltas)[test_mask],
            seed=3000+repeat,
        )

        align_curve_rows.append({
            "source_model": source_model,
            "target_model": target_model,
            "alignment_size": real_size,
            "repeat": repeat,
            "acc": accuracy_score(y[test_mask], pred),
            "f1": f1_score(y[test_mask], pred, average="macro"),
        })

align_curve_df = pd.DataFrame(align_curve_rows)
align_curve_summary = (
    align_curve_df.groupby("alignment_size")
    .agg(mean_f1=("f1", "mean"), std_f1=("f1", "std"), mean_acc=("acc", "mean"), n=("f1", "count"))
    .reset_index()
)

align_curve_df.to_csv(f"{OUT_DIR}/csv/alignment_curve_raw.csv", index=False)
align_curve_summary.to_csv(f"{OUT_DIR}/csv/alignment_curve_summary.csv", index=False)


# ============================================================
# PLOTS
# ============================================================

def savefig(name):
    path = f"{OUT_DIR}/figures/{name}"
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.show()
    print("saved:", path)


# 1. Hard holdout vs shuffle
plt.figure(figsize=(9, 5))
x = np.arange(len(hard_df))
plt.bar(x - 0.2, hard_df["f1"], width=0.4, label="True delta F1")
plt.bar(x + 0.2, shuffle_df["shuffle_mean_f1"], width=0.4, label="Shuffle mean F1")
plt.axhline(baseline, linestyle="--", label="Random baseline")
plt.xticks(x, hard_df["model"], rotation=25, ha="right")
plt.ylabel("Macro F1")
plt.title("Hard Semantic Holdout vs Label Shuffle Control")
plt.legend()
savefig("01_hard_holdout_vs_shuffle.png")


# 2. Ablation
pivot_ablation = ablation_df.pivot(index="model", columns="feature", values="f1")
pivot_ablation.plot(kind="bar", figsize=(10, 5))
plt.axhline(baseline, linestyle="--", label="Random baseline")
plt.ylabel("Macro F1")
plt.title("Representation Ablation: x_only vs y_only vs delta vs concat")
plt.legend()
savefig("02_representation_ablation.png")


# 3. Magnitude vs direction
pivot_magdir = magdir_df.pivot(index="model", columns="feature", values="f1")
pivot_magdir.plot(kind="bar", figsize=(10, 5))
plt.axhline(baseline, linestyle="--", label="Random baseline")
plt.ylabel("Macro F1")
plt.title("Magnitude vs Direction Decomposition")
plt.legend()
savefig("03_magnitude_vs_direction.png")


# 4. Capacity curve
plt.figure(figsize=(9, 5))
for model_name in capacity_summary["model"].unique():
    sub = capacity_summary[capacity_summary["model"] == model_name].sort_values("train_size")
    plt.errorbar(sub["train_size"], sub["mean_f1"], yerr=sub["std_f1"], marker="o", capsize=3, label=model_name)
plt.axhline(baseline, linestyle="--", label="Random baseline")
plt.xlabel("Number of training pairs")
plt.ylabel("Macro F1")
plt.title("Capacity Curve: Emergence of Transformation Geometry")
plt.legend()
plt.grid(True)
savefig("04_capacity_curve.png")


# 5. Cross-model raw heatmap
raw_mat = cross_df.pivot(index="train_model", columns="test_model", values="raw_f1")
plt.figure(figsize=(7, 6))
plt.imshow(raw_mat.values, aspect="auto")
plt.colorbar(label="Macro F1")
plt.xticks(np.arange(len(raw_mat.columns)), raw_mat.columns, rotation=25, ha="right")
plt.yticks(np.arange(len(raw_mat.index)), raw_mat.index)
plt.title("Cross-Model Transfer — Raw Spaces")
for i in range(raw_mat.shape[0]):
    for j in range(raw_mat.shape[1]):
        plt.text(j, i, f"{raw_mat.values[i, j]:.2f}", ha="center", va="center")
savefig("05_cross_model_raw_heatmap.png")


# 6. Cross-model aligned heatmap
aligned_mat = cross_df.pivot(index="train_model", columns="test_model", values="aligned_f1")
plt.figure(figsize=(7, 6))
plt.imshow(aligned_mat.values, aspect="auto")
plt.colorbar(label="Macro F1")
plt.xticks(np.arange(len(aligned_mat.columns)), aligned_mat.columns, rotation=25, ha="right")
plt.yticks(np.arange(len(aligned_mat.index)), aligned_mat.index)
plt.title("Cross-Model Transfer — Procrustes Aligned Spaces")
for i in range(aligned_mat.shape[0]):
    for j in range(aligned_mat.shape[1]):
        plt.text(j, i, f"{aligned_mat.values[i, j]:.2f}", ha="center", va="center")
savefig("06_cross_model_aligned_heatmap.png")


# 7. Procrustes gain
cross_df_plot = cross_df.copy()
cross_df_plot["pair"] = cross_df_plot["train_model"] + " → " + cross_df_plot["test_model"]
plt.figure(figsize=(10, 5))
plt.bar(cross_df_plot["pair"], cross_df_plot["gain_f1"])
plt.xticks(rotation=35, ha="right")
plt.ylabel("F1 gain after alignment")
plt.title("Procrustes Alignment Gain")
plt.grid(axis="y")
savefig("07_procrustes_gain.png")


# 8. Alignment sample curve
plt.figure(figsize=(8, 5))
plt.errorbar(
    align_curve_summary["alignment_size"],
    align_curve_summary["mean_f1"],
    yerr=align_curve_summary["std_f1"],
    marker="o",
    capsize=4,
)
plt.xlabel("Number of aligned train text pairs")
plt.ylabel("Cross-model Macro F1")
plt.title("Alignment Sample Curve — RoBERTa → BERT")
plt.grid(True)
savefig("08_alignment_sample_curve.png")


# 9. McNemar delta gain
plt.figure(figsize=(9, 5))
plt.bar(mcnemar_df["model"], mcnemar_df["diff"])
plt.xticks(rotation=25, ha="right")
plt.ylabel("Accuracy gain: delta - y_only")
plt.title("Delta Adds Information Beyond Target Embedding")
plt.grid(axis="y")
savefig("09_mcnemar_delta_gain.png")


# ============================================================
# MASTER SUMMARY
# ============================================================

summary = {
    "dataset_size": len(df),
    "classes": list(le.classes_),
    "models": MODELS,
    "baseline": baseline,
    "main_results": {
        "hard_holdout": hard_df.to_dict(orient="records"),
        "shuffle_control": shuffle_df.to_dict(orient="records"),
        "ablation": ablation_df.to_dict(orient="records"),
        "mcnemar": mcnemar_df.to_dict(orient="records"),
        "magnitude_direction": magdir_df.to_dict(orient="records"),
        "cross_model_procrustes": cross_df.to_dict(orient="records"),
    }
}

with open(f"{OUT_DIR}/master_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nDONE.")
print("Results saved to:", OUT_DIR)
print("Figures:")
for f in sorted(os.listdir(f"{OUT_DIR}/figures")):
    print(" -", f)
