from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


DEFAULT_OUT_DIR = "results/experiments/glt_build_01_explicit_operator_smoke_20260911_results"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<sep>", "<eos>", "<I>", "<T>", "<N>", "<V>", "<Q>", "<S>"]
OPS = ["T", "N", "V", "Q", "S"]
TRAIN_PAIR_OPS = [("T", "N"), ("T", "Q"), ("N", "Q"), ("V", "Q")]
UNSEEN_PAIR_OPS = [("N", "V"), ("T", "V"), ("S", "Q")]


@dataclass(frozen=True)
class Verb:
    base: str
    present: str
    past: str
    participle: str


@dataclass(frozen=True)
class State:
    subject: str
    verb_idx: int
    object: str
    tense: str = "present"
    negated: bool = False
    voice: str = "active"
    mood: str = "statement"


@dataclass(frozen=True)
class Example:
    split: str
    family: str
    op_seq: tuple[str, ...]
    scene_id: int
    source_state: State
    target_state: State
    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]

    @property
    def op_name(self) -> str:
        return "I" if not self.op_seq else "".join(self.op_seq)


VERBS = [
    Verb("open", "opens", "opened", "opened"),
    Verb("close", "closes", "closed", "closed"),
    Verb("move", "moves", "moved", "moved"),
    Verb("lift", "lifts", "lifted", "lifted"),
    Verb("paint", "paints", "painted", "painted"),
    Verb("repair", "repairs", "repaired", "repaired"),
    Verb("inspect", "inspects", "inspected", "inspected"),
    Verb("carry", "carries", "carried", "carried"),
]

ENTITIES = ["mira", "nora", "liam", "omar", "sofia", "tariq", "vera", "yuri"]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def write_status(out_dir: Path, **kwargs: object) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_status.json"
    current: dict[str, object] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(kwargs)
    current["updated_at"] = now()
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_op(state: State, op: str) -> State:
    if op == "T":
        return replace(state, tense="past" if state.tense == "present" else "present")
    if op == "N":
        return replace(state, negated=not state.negated)
    if op == "V":
        return replace(state, voice="passive" if state.voice == "active" else "active")
    if op == "Q":
        return replace(state, mood="question" if state.mood == "statement" else "statement")
    if op == "S":
        return replace(state, subject=state.object, object=state.subject)
    raise ValueError(f"Unknown operation: {op}")


def apply_ops(state: State, ops: tuple[str, ...]) -> State:
    out = state
    for op in ops:
        out = apply_op(out, op)
    return out


def render_state(state: State) -> tuple[str, ...]:
    verb = VERBS[state.verb_idx]
    actor = state.subject
    patient = state.object

    if state.voice == "active":
        if state.mood == "question":
            aux = "does" if state.tense == "present" else "did"
            tokens = [aux, actor]
            if state.negated:
                tokens.append("not")
            tokens.extend([verb.base, patient, "?"])
            return tuple(tokens)
        if state.negated:
            aux = "does" if state.tense == "present" else "did"
            return (actor, aux, "not", verb.base, patient, ".")
        verb_form = verb.present if state.tense == "present" else verb.past
        return (actor, verb_form, patient, ".")

    be = "is" if state.tense == "present" else "was"
    if state.mood == "question":
        tokens = [be, patient]
        if state.negated:
            tokens.append("not")
        tokens.extend([verb.participle, "by", actor, "?"])
        return tuple(tokens)
    tokens = [patient, be]
    if state.negated:
        tokens.append("not")
    tokens.extend([verb.participle, "by", actor, "."])
    return tuple(tokens)


def all_base_scenes() -> list[tuple[int, State]]:
    rows: list[tuple[int, State]] = []
    scene_id = 0
    for subject in ENTITIES:
        for verb_idx in range(len(VERBS)):
            for obj in ENTITIES:
                if obj == subject:
                    continue
                rows.append((scene_id, State(subject=subject, verb_idx=verb_idx, object=obj)))
                scene_id += 1
    return rows


def scene_is_heldout(scene_id: int, state: State) -> bool:
    subject_idx = ENTITIES.index(state.subject)
    object_idx = ENTITIES.index(state.object)
    return (subject_idx + 2 * state.verb_idx + 3 * object_idx + scene_id) % 5 == 0


def build_examples(eval_limit: int) -> tuple[list[Example], dict[str, list[Example]]]:
    train_examples: list[Example] = []
    eval_groups: dict[str, list[Example]] = {
        "train_seen": [],
        "heldout_scene_single": [],
        "heldout_scene_seen_pair": [],
        "heldout_scene_reverse_seen_pair": [],
        "heldout_unseen_pair": [],
    }

    scenes = all_base_scenes()
    train_scenes = [(sid, s) for sid, s in scenes if not scene_is_heldout(sid, s)]
    heldout_scenes = [(sid, s) for sid, s in scenes if scene_is_heldout(sid, s)]

    train_op_seqs: list[tuple[str, ...]] = [tuple()] + [(op,) for op in OPS] + TRAIN_PAIR_OPS
    for scene_id, state in train_scenes:
        for op_seq in train_op_seqs:
            target = apply_ops(state, op_seq)
            ex = Example(
                split="train",
                family="train_seen",
                op_seq=tuple(op_seq),
                scene_id=scene_id,
                source_state=state,
                target_state=target,
                source_tokens=render_state(state),
                target_tokens=render_state(target),
            )
            train_examples.append(ex)
            if len(eval_groups["train_seen"]) < eval_limit:
                eval_groups["train_seen"].append(ex)

    for scene_id, state in heldout_scenes:
        for op in OPS:
            target = apply_op(state, op)
            eval_groups["heldout_scene_single"].append(
                Example(
                    split="eval",
                    family="heldout_scene_single",
                    op_seq=(op,),
                    scene_id=scene_id,
                    source_state=state,
                    target_state=target,
                    source_tokens=render_state(state),
                    target_tokens=render_state(target),
                )
            )
        for op_seq in TRAIN_PAIR_OPS:
            target = apply_ops(state, op_seq)
            eval_groups["heldout_scene_seen_pair"].append(
                Example(
                    split="eval",
                    family="heldout_scene_seen_pair",
                    op_seq=tuple(op_seq),
                    scene_id=scene_id,
                    source_state=state,
                    target_state=target,
                    source_tokens=render_state(state),
                    target_tokens=render_state(target),
                )
            )
            rev = tuple(reversed(op_seq))
            eval_groups["heldout_scene_reverse_seen_pair"].append(
                Example(
                    split="eval",
                    family="heldout_scene_reverse_seen_pair",
                    op_seq=rev,
                    scene_id=scene_id,
                    source_state=state,
                    target_state=apply_ops(state, rev),
                    source_tokens=render_state(state),
                    target_tokens=render_state(apply_ops(state, rev)),
                )
            )
        for op_seq in UNSEEN_PAIR_OPS:
            eval_groups["heldout_unseen_pair"].append(
                Example(
                    split="eval",
                    family="heldout_unseen_pair",
                    op_seq=tuple(op_seq),
                    scene_id=scene_id,
                    source_state=state,
                    target_state=apply_ops(state, op_seq),
                    source_tokens=render_state(state),
                    target_tokens=render_state(apply_ops(state, op_seq)),
                )
            )

    for key in list(eval_groups):
        eval_groups[key] = eval_groups[key][:eval_limit]
    return train_examples, eval_groups


def op_tokens(op_seq: tuple[str, ...]) -> list[str]:
    if not op_seq:
        return ["<I>"]
    return [f"<{op}>" for op in op_seq]


def build_vocab(examples: list[Example], eval_groups: dict[str, list[Example]]) -> tuple[dict[str, int], list[str]]:
    vocab = list(SPECIAL_TOKENS)
    seen = set(vocab)
    all_examples = list(examples)
    for group in eval_groups.values():
        all_examples.extend(group)
    for ex in all_examples:
        for token in list(ex.source_tokens) + list(ex.target_tokens):
            if token not in seen:
                seen.add(token)
                vocab.append(token)
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return token_to_id, vocab


def encode_example(ex: Example, token_to_id: dict[str, int]) -> tuple[list[int], list[int], int]:
    prompt_tokens = ["<bos>"] + op_tokens(ex.op_seq) + ["<sep>"] + list(ex.source_tokens) + ["<sep>"]
    full_tokens = prompt_tokens + list(ex.target_tokens) + ["<eos>"]
    ids = [token_to_id[t] for t in full_tokens]
    inputs = ids[:-1]
    labels = ids[1:]
    ignore_until = len(prompt_tokens) - 1
    labels = [(-100 if i < ignore_until else y) for i, y in enumerate(labels)]
    return inputs, labels, len(prompt_tokens)


def encode_prompt(ex: Example, token_to_id: dict[str, int], force_identity: bool = False) -> list[int]:
    seq = tuple() if force_identity else ex.op_seq
    prompt_tokens = ["<bos>"] + op_tokens(seq) + ["<sep>"] + list(ex.source_tokens) + ["<sep>"]
    return [token_to_id[t] for t in prompt_tokens]


def encode_sentence_prompt(tokens: tuple[str, ...], token_to_id: dict[str, int]) -> list[int]:
    prompt_tokens = ["<bos>", "<I>", "<sep>"] + list(tokens) + ["<sep>"]
    return [token_to_id[t] for t in prompt_tokens]


def pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    arr = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    lengths = torch.zeros(len(seqs), dtype=torch.long)
    for i, seq in enumerate(seqs):
        arr[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        lengths[i] = len(seq)
    return arr, lengths


def sample_batch(
    encoded: list[tuple[list[int], list[int], int]],
    batch_size: int,
    pad_id: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = [encoded[rng.randrange(len(encoded))] for _ in range(batch_size)]
    input_seqs = [row[0] for row in batch]
    label_seqs = [row[1] for row in batch]
    max_len = max(len(s) for s in input_seqs)
    inputs = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for i, (inp, lab) in enumerate(zip(input_seqs, label_seqs)):
        inputs[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
    return inputs, labels


class TinyDecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
        injection: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        hidden: dict[int, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers, start=1):
            x = layer(x, src_mask=causal_mask)
            if injection and layer_idx in injection:
                vec = injection[layer_idx].to(device=device, dtype=x.dtype)
                x[:, -1, :] = x[:, -1, :] + vec
            if return_hidden:
                hidden[layer_idx] = x
        x = self.ln(x)
        logits = self.head(x)
        return logits, hidden


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def decode_tokens(ids: list[int], vocab: list[str], eos_id: int) -> tuple[str, ...]:
    tokens: list[str] = []
    for idx in ids:
        if idx == eos_id:
            break
        tokens.append(vocab[idx])
    return tuple(tokens)


@torch.no_grad()
def generate(
    model: TinyDecoderOnlyTransformer,
    prompt_ids: list[int],
    vocab: list[str],
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
    device: torch.device,
    injection: dict[int, torch.Tensor] | None = None,
) -> tuple[str, ...]:
    model.eval()
    seq = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits, _ = model(seq, return_hidden=False, injection=injection)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        generated.append(next_id)
        if next_id == eos_id:
            break
        seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    _ = pad_id
    return decode_tokens(generated, vocab, eos_id)


def token_text(tokens: tuple[str, ...]) -> str:
    text = " ".join(tokens)
    return text.replace(" .", ".").replace(" ?", "?")


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return float(np.mean(np.sum(a_norm * b_norm, axis=1)))


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def fit_ridge_map(x: np.ndarray, y: np.ndarray, alpha: float, affine: bool) -> tuple[np.ndarray, bool]:
    if affine:
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        reg = np.eye(x_aug.shape[1], dtype=x.dtype) * alpha
        reg[-1, -1] = 0.0
        weights = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ y
        return weights, True
    reg = np.eye(x.shape[1], dtype=x.dtype) * alpha
    weights = np.linalg.pinv(x.T @ x + reg) @ x.T @ y
    return weights, False


def apply_ridge_map(x: np.ndarray, weights: np.ndarray, affine: bool) -> np.ndarray:
    if affine:
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        return x_aug @ weights
    return x @ weights


@torch.no_grad()
def hidden_for_prompts(
    model: TinyDecoderOnlyTransformer,
    prompts: list[list[int]],
    layers: list[int],
    pad_id: int,
    device: torch.device,
    batch_size: int = 128,
) -> dict[int, np.ndarray]:
    model.eval()
    out = {layer: [] for layer in layers}
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        ids, lengths = pad_batch(batch_prompts, pad_id)
        ids = ids.to(device)
        _, hidden = model(ids, return_hidden=True)
        for layer in layers:
            h = hidden[layer]
            rows = h[torch.arange(h.shape[0], device=device), lengths.to(device) - 1, :]
            out[layer].append(rows.cpu().numpy())
    return {layer: np.vstack(parts) for layer, parts in out.items()}


def evaluate_behavior(
    model: TinyDecoderOnlyTransformer,
    eval_groups: dict[str, list[Example]],
    token_to_id: dict[str, int],
    vocab: list[str],
    max_new_tokens: int,
    device: torch.device,
    checkpoint_step: int,
    seed: int,
) -> pd.DataFrame:
    pad_id = token_to_id["<pad>"]
    eos_id = token_to_id["<eos>"]
    rows = []
    for family, examples in eval_groups.items():
        for ex in examples:
            prompt_ids = encode_prompt(ex, token_to_id)
            generated = generate(model, prompt_ids, vocab, eos_id, pad_id, max_new_tokens, device)
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_step": checkpoint_step,
                    "family": family,
                    "scene_id": ex.scene_id,
                    "op_name": ex.op_name,
                    "target_text": token_text(ex.target_tokens),
                    "generated_text": token_text(generated),
                    "exact_match": int(generated == ex.target_tokens),
                    "target_len": len(ex.target_tokens),
                    "generated_len": len(generated),
                }
            )
    return pd.DataFrame(rows)


def representation_examples() -> tuple[list[Example], list[Example], list[Example]]:
    scenes = all_base_scenes()
    train_scenes = [(sid, s) for sid, s in scenes if not scene_is_heldout(sid, s)]
    heldout_scenes = [(sid, s) for sid, s in scenes if scene_is_heldout(sid, s)]

    def build(scene_rows: list[tuple[int, State]], op_seqs: list[tuple[str, ...]], family: str) -> list[Example]:
        examples = []
        for scene_id, state in scene_rows:
            for op_seq in op_seqs:
                target = apply_ops(state, op_seq)
                examples.append(
                    Example(
                        split="representation",
                        family=family,
                        op_seq=tuple(op_seq),
                        scene_id=scene_id,
                        source_state=state,
                        target_state=target,
                        source_tokens=render_state(state),
                        target_tokens=render_state(target),
                    )
                )
        return examples

    train = build(train_scenes, [(op,) for op in OPS], "train_single")
    test = build(heldout_scenes, [(op,) for op in OPS], "heldout_single")
    pair_test = build(heldout_scenes, list(TRAIN_PAIR_OPS) + list(UNSEEN_PAIR_OPS), "heldout_pair")
    return train, test, pair_test


def collect_sentence_hidden(
    model: TinyDecoderOnlyTransformer,
    examples: list[Example],
    token_to_id: dict[str, int],
    layers: list[int],
    pad_id: int,
    device: torch.device,
) -> dict[str, dict[int, np.ndarray]]:
    sentence_to_tokens: dict[str, tuple[str, ...]] = {}
    for ex in examples:
        sentence_to_tokens[token_text(ex.source_tokens)] = ex.source_tokens
        sentence_to_tokens[token_text(ex.target_tokens)] = ex.target_tokens
    keys = sorted(sentence_to_tokens)
    prompts = [encode_sentence_prompt(sentence_to_tokens[key], token_to_id) for key in keys]
    hidden = hidden_for_prompts(model, prompts, layers, pad_id, device)
    return {key: {layer: hidden[layer][idx] for layer in layers} for idx, key in enumerate(keys)}


def run_representation_diagnostics(
    model: TinyDecoderOnlyTransformer,
    token_to_id: dict[str, int],
    layers: list[int],
    device: torch.device,
    checkpoint_step: int,
    seed: int,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test, pair_test = representation_examples()
    all_examples = train + test + pair_test
    hidden = collect_sentence_hidden(model, all_examples, token_to_id, layers, token_to_id["<pad>"], device)

    fit_rows = []
    composition_rows = []
    rng = np.random.default_rng(seed + checkpoint_step + 17)

    for layer in layers:
        train_by_op = {op: [ex for ex in train if ex.op_seq == (op,)] for op in OPS}
        test_by_op = {op: [ex for ex in test if ex.op_seq == (op,)] for op in OPS}
        centroids: dict[str, np.ndarray] = {}
        source_train_by_op: dict[str, np.ndarray] = {}
        target_train_by_op: dict[str, np.ndarray] = {}

        for op, examples in train_by_op.items():
            sx = np.vstack([hidden[token_text(ex.source_tokens)][layer] for ex in examples])
            ty = np.vstack([hidden[token_text(ex.target_tokens)][layer] for ex in examples])
            source_train_by_op[op] = sx
            target_train_by_op[op] = ty
            centroids[op] = (ty - sx).mean(axis=0)

        all_test_rows = []
        for op, examples in test_by_op.items():
            sx = np.vstack([hidden[token_text(ex.source_tokens)][layer] for ex in examples])
            ty = np.vstack([hidden[token_text(ex.target_tokens)][layer] for ex in examples])
            delta = ty - sx
            centroid_names = list(centroids)
            centroid_mat = np.vstack([centroids[name] for name in centroid_names])
            delta_unit = delta / np.clip(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12, None)
            centroid_unit = centroid_mat / np.clip(np.linalg.norm(centroid_mat, axis=1, keepdims=True), 1e-12, None)
            predicted = [centroid_names[i] for i in np.argmax(delta_unit @ centroid_unit.T, axis=1)]
            nearest_acc = float(np.mean([p == op for p in predicted]))

            add_pred = sx + centroids[op]
            random_vec = rng.normal(size=centroids[op].shape)
            random_vec = random_vec / max(np.linalg.norm(random_vec), 1e-12) * np.linalg.norm(centroids[op])
            random_pred = sx + random_vec
            wrong_op = OPS[(OPS.index(op) + 1) % len(OPS)]
            wrong_pred = sx + centroids[wrong_op]
            linear_w, linear_aff = fit_ridge_map(source_train_by_op[op], target_train_by_op[op], ridge_alpha, False)
            affine_w, affine_aff = fit_ridge_map(source_train_by_op[op], target_train_by_op[op], ridge_alpha, True)
            linear_pred = apply_ridge_map(sx, linear_w, linear_aff)
            affine_pred = apply_ridge_map(sx, affine_w, affine_aff)
            fit_rows.append(
                {
                    "seed": seed,
                    "checkpoint_step": checkpoint_step,
                    "layer": layer,
                    "operation": op,
                    "n_train": len(train_by_op[op]),
                    "n_test": len(examples),
                    "nearest_centroid_acc": nearest_acc,
                    "source_target_cosine": mean_cosine(sx, ty),
                    "additive_target_cosine": mean_cosine(add_pred, ty),
                    "linear_target_cosine": mean_cosine(linear_pred, ty),
                    "affine_target_cosine": mean_cosine(affine_pred, ty),
                    "wrong_additive_target_cosine": mean_cosine(wrong_pred, ty),
                    "random_additive_target_cosine": mean_cosine(random_pred, ty),
                    "centroid_norm": float(np.linalg.norm(centroids[op])),
                    "test_delta_norm": float(np.linalg.norm(delta, axis=1).mean()),
                }
            )
            all_test_rows.extend(
                {
                    "op": op,
                    "delta": row,
                }
                for row in delta
            )

        for pair in list(TRAIN_PAIR_OPS) + list(UNSEEN_PAIR_OPS):
            examples = [ex for ex in pair_test if ex.op_seq == pair]
            sx = np.vstack([hidden[token_text(ex.source_tokens)][layer] for ex in examples])
            ty = np.vstack([hidden[token_text(ex.target_tokens)][layer] for ex in examples])
            pair_delta = (ty - sx).mean(axis=0)
            primitive_sum = centroids[pair[0]] + centroids[pair[1]]
            reverse_examples = []
            for ex in examples:
                rev_state = apply_ops(ex.source_state, tuple(reversed(pair)))
                reverse_examples.append(render_state(rev_state))
            reverse_hidden = np.vstack([hidden[token_text(tokens)][layer] for tokens in reverse_examples])
            reverse_delta = (reverse_hidden - sx).mean(axis=0)
            residual = pair_delta - primitive_sum
            composition_rows.append(
                {
                    "seed": seed,
                    "checkpoint_step": checkpoint_step,
                    "layer": layer,
                    "pair": "".join(pair),
                    "pair_seen_in_training": int(pair in TRAIN_PAIR_OPS),
                    "n_test": len(examples),
                    "pair_delta_norm": float(np.linalg.norm(pair_delta)),
                    "primitive_sum_norm": float(np.linalg.norm(primitive_sum)),
                    "pair_vs_sum_cosine": vector_cosine(pair_delta, primitive_sum),
                    "sum_residual_ratio": float(np.linalg.norm(residual) / max(np.linalg.norm(pair_delta), 1e-12)),
                    "reverse_order_delta_cosine": vector_cosine(pair_delta, reverse_delta),
                    "reverse_order_delta_norm_diff": float(np.linalg.norm(pair_delta - reverse_delta)),
                }
            )

    return pd.DataFrame(fit_rows), pd.DataFrame(composition_rows)


def run_causal_intervention(
    model: TinyDecoderOnlyTransformer,
    token_to_id: dict[str, int],
    vocab: list[str],
    layer: int,
    device: torch.device,
    checkpoint_step: int,
    seed: int,
    eval_limit: int,
    max_new_tokens: int,
) -> pd.DataFrame:
    train, test, _ = representation_examples()
    hidden = collect_sentence_hidden(model, train + test[: eval_limit * len(OPS)], token_to_id, [layer], token_to_id["<pad>"], device)
    centroids = {}
    for op in OPS:
        rows = [ex for ex in train if ex.op_seq == (op,)]
        sx = np.vstack([hidden[token_text(ex.source_tokens)][layer] for ex in rows])
        ty = np.vstack([hidden[token_text(ex.target_tokens)][layer] for ex in rows])
        centroids[op] = torch.tensor((ty - sx).mean(axis=0), dtype=torch.float32, device=device)

    pad_id = token_to_id["<pad>"]
    eos_id = token_to_id["<eos>"]
    rng = np.random.default_rng(seed + checkpoint_step + 101)
    rows = []
    by_op = {op: [ex for ex in test if ex.op_seq == (op,)][:eval_limit] for op in OPS}
    for op, examples in by_op.items():
        wrong_op = OPS[(OPS.index(op) + 1) % len(OPS)]
        random_vec = rng.normal(size=centroids[op].shape[0]).astype(np.float32)
        random_vec = random_vec / max(float(np.linalg.norm(random_vec)), 1e-12) * float(torch.linalg.norm(centroids[op]).item())
        conditions = {
            "none": None,
            "target_vector": {layer: centroids[op]},
            "wrong_vector": {layer: centroids[wrong_op]},
            "negative_vector": {layer: -centroids[op]},
            "random_norm": {layer: torch.tensor(random_vec, dtype=torch.float32, device=device)},
        }
        for ex in examples:
            identity_prompt = encode_prompt(ex, token_to_id, force_identity=True)
            for condition, injection in conditions.items():
                generated = generate(model, identity_prompt, vocab, eos_id, pad_id, max_new_tokens, device, injection=injection)
                rows.append(
                    {
                        "seed": seed,
                        "checkpoint_step": checkpoint_step,
                        "layer": layer,
                        "operation": op,
                        "condition": condition,
                        "scene_id": ex.scene_id,
                        "target_text": token_text(ex.target_tokens),
                        "generated_text": token_text(generated),
                        "exact_match": int(generated == ex.target_tokens),
                        "identity_match": int(generated == ex.source_tokens),
                    }
                )
    return pd.DataFrame(rows)


def write_figures(out_dir: Path, behavior_df: pd.DataFrame, rep_df: pd.DataFrame, causal_df: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not behavior_df.empty:
        summary = (
            behavior_df.groupby(["checkpoint_step", "family"], as_index=False)["exact_match"]
            .mean()
            .sort_values(["family", "checkpoint_step"])
        )
        plt.figure(figsize=(9, 5))
        for family, grp in summary.groupby("family"):
            plt.plot(grp["checkpoint_step"], grp["exact_match"], marker="o", label=family)
        plt.xlabel("training step")
        plt.ylabel("exact-match rate")
        plt.ylim(-0.05, 1.05)
        plt.title("GLT-BUILD-01A behavior over training")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "behavior_exact_match_over_time.png", dpi=160)
        plt.close()

    if not rep_df.empty:
        summary = (
            rep_df.groupby(["checkpoint_step", "layer"], as_index=False)["nearest_centroid_acc"]
            .mean()
            .sort_values(["layer", "checkpoint_step"])
        )
        plt.figure(figsize=(8, 5))
        for layer, grp in summary.groupby("layer"):
            plt.plot(grp["checkpoint_step"], grp["nearest_centroid_acc"], marker="o", label=f"layer {layer}")
        plt.xlabel("training step")
        plt.ylabel("heldout nearest-centroid accuracy")
        plt.ylim(-0.05, 1.05)
        plt.title("Transformation separability over training")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "representation_centroid_accuracy_over_time.png", dpi=160)
        plt.close()

    if not causal_df.empty:
        final_step = int(causal_df["checkpoint_step"].max())
        summary = (
            causal_df[causal_df["checkpoint_step"].eq(final_step)]
            .groupby(["operation", "condition"], as_index=False)["exact_match"]
            .mean()
        )
        pivot = summary.pivot(index="operation", columns="condition", values="exact_match").fillna(0.0)
        plt.figure(figsize=(9, 5))
        pivot.plot(kind="bar", ax=plt.gca())
        plt.ylabel("exact-match rate")
        plt.ylim(-0.05, 1.05)
        plt.title(f"Causal intervention exact match at step {final_step}")
        plt.tight_layout()
        plt.savefig(fig_dir / "causal_intervention_final.png", dpi=160)
        plt.close()


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        rows.append(values)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(values) + " |" for values in rows]
    return "\n".join([header, separator] + body)


def write_summary(
    out_dir: Path,
    config: dict[str, object],
    behavior_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    causal_df: pd.DataFrame,
) -> None:
    lines = [
        "# GLT-BUILD-01A Explicit-Operator Smoke Test",
        "",
        "Status: completed smoke test. This is an implementation and pipeline validation run, not a promoted result.",
        "",
        "## Configuration",
        "",
    ]
    for key, value in config.items():
        lines.append(f"- `{key}`: `{value}`")

    final_step = int(behavior_df["checkpoint_step"].max()) if not behavior_df.empty else -1
    lines.extend(["", "## Final Behavioral Exact Match", ""])
    if not behavior_df.empty:
        final_behavior = (
            behavior_df[behavior_df["checkpoint_step"].eq(final_step)]
            .groupby(["family"], as_index=False)
            .agg(exact_match=("exact_match", "mean"), n=("exact_match", "size"))
        )
        lines.append(df_to_markdown(final_behavior))

    lines.extend(["", "## Final Representation Diagnostics", ""])
    if not rep_df.empty:
        final_rep = (
            rep_df[rep_df["checkpoint_step"].eq(final_step)]
            .groupby(["layer"], as_index=False)
            .agg(
                nearest_centroid_acc=("nearest_centroid_acc", "mean"),
                source_target_cosine=("source_target_cosine", "mean"),
                additive_target_cosine=("additive_target_cosine", "mean"),
                linear_target_cosine=("linear_target_cosine", "mean"),
                affine_target_cosine=("affine_target_cosine", "mean"),
                wrong_additive_target_cosine=("wrong_additive_target_cosine", "mean"),
                random_additive_target_cosine=("random_additive_target_cosine", "mean"),
            )
        )
        lines.append(df_to_markdown(final_rep))

    lines.extend(["", "## Final Composition Diagnostics", ""])
    if not comp_df.empty:
        final_comp = (
            comp_df[comp_df["checkpoint_step"].eq(final_step)]
            .groupby(["pair", "pair_seen_in_training"], as_index=False)
            .agg(
                pair_vs_sum_cosine=("pair_vs_sum_cosine", "mean"),
                sum_residual_ratio=("sum_residual_ratio", "mean"),
                reverse_order_delta_cosine=("reverse_order_delta_cosine", "mean"),
                n=("pair_vs_sum_cosine", "size"),
            )
        )
        lines.append(df_to_markdown(final_comp))

    lines.extend(["", "## Final Causal Intervention Audit", ""])
    if not causal_df.empty:
        final_causal = (
            causal_df[causal_df["checkpoint_step"].eq(final_step)]
            .groupby(["operation", "condition"], as_index=False)
            .agg(exact_match=("exact_match", "mean"), identity_match=("identity_match", "mean"), n=("exact_match", "size"))
        )
        lines.append(df_to_markdown(final_causal))

    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "This smoke test validates the GLT-BUILD pipeline shape: synthetic data generation, tiny-model training, checkpointed behavioral evaluation, representation diagnostics, composition diagnostics, and a first causal intervention audit.",
            "",
            "The run should not be cited as evidence that a stable transformation algebra has emerged. A promoted GLT-BUILD result requires multiple seeds, a larger fixed protocol, and explicit null comparisons.",
            "",
        ]
    )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def save_examples(out_dir: Path, train_examples: list[Example], eval_groups: dict[str, list[Example]]) -> None:
    rows = []
    for ex in train_examples:
        rows.append(
            {
                "split": ex.split,
                "family": ex.family,
                "scene_id": ex.scene_id,
                "op_name": ex.op_name,
                "source": token_text(ex.source_tokens),
                "target": token_text(ex.target_tokens),
            }
        )
    for family, examples in eval_groups.items():
        for ex in examples:
            rows.append(
                {
                    "split": ex.split,
                    "family": family,
                    "scene_id": ex.scene_id,
                    "op_name": ex.op_name,
                    "source": token_text(ex.source_tokens),
                    "target": token_text(ex.target_tokens),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "csv" / "dataset_examples.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GLT-BUILD-01A explicit-operator smoke test.")
    parser.add_argument("--out-dir", default=os.getenv("GLT_BUILD_OUT_DIR", DEFAULT_OUT_DIR))
    parser.add_argument("--seeds", default=os.getenv("GLT_BUILD_SEEDS", "0,1,2"))
    parser.add_argument("--steps", type=int, default=int(os.getenv("GLT_BUILD_STEPS", "800")))
    parser.add_argument("--checkpoints", default=os.getenv("GLT_BUILD_CHECKPOINTS", "0,100,250,500,800"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("GLT_BUILD_BATCH_SIZE", "64")))
    parser.add_argument("--eval-limit", type=int, default=int(os.getenv("GLT_BUILD_EVAL_LIMIT", "96")))
    parser.add_argument("--causal-limit", type=int, default=int(os.getenv("GLT_BUILD_CAUSAL_LIMIT", "24")))
    parser.add_argument("--d-model", type=int, default=int(os.getenv("GLT_BUILD_D_MODEL", "64")))
    parser.add_argument("--layers", type=int, default=int(os.getenv("GLT_BUILD_LAYERS", "2")))
    parser.add_argument("--heads", type=int, default=int(os.getenv("GLT_BUILD_HEADS", "4")))
    parser.add_argument("--dropout", type=float, default=float(os.getenv("GLT_BUILD_DROPOUT", "0.0")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("GLT_BUILD_LR", "0.003")))
    parser.add_argument("--ridge-alpha", type=float, default=float(os.getenv("GLT_BUILD_RIDGE_ALPHA", "1.0")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("GLT_BUILD_MAX_NEW_TOKENS", "12")))
    parser.add_argument("--threads", type=int, default=int(os.getenv("GLT_BUILD_THREADS", "4")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"
    for directory in [csv_dir, fig_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.threads))
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    checkpoints = sorted({int(x.strip()) for x in args.checkpoints.split(",") if x.strip()} | {0, args.steps})

    train_examples, eval_groups = build_examples(args.eval_limit)
    token_to_id, vocab = build_vocab(train_examples, eval_groups)
    save_examples(out_dir, train_examples, eval_groups)

    encoded_train = [encode_example(ex, token_to_id) for ex in train_examples]
    max_train_len = max(len(row[0]) for row in encoded_train) + args.max_new_tokens + 4
    max_seq_len = max(max_train_len, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "started_at": now(),
        "out_dir": str(out_dir),
        "seeds": args.seeds,
        "steps": args.steps,
        "checkpoints": ",".join(str(x) for x in checkpoints),
        "batch_size": args.batch_size,
        "eval_limit_per_group": args.eval_limit,
        "causal_limit_per_operation": args.causal_limit,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "ridge_alpha": args.ridge_alpha,
        "vocab_size": len(vocab),
        "train_examples": len(train_examples),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (out_dir / "metadata.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")

    write_status(out_dir, status="running", phase="initializing", config=config, completed_seeds=0, total_seeds=len(seeds))

    all_training_rows = []
    all_behavior_rows = []
    all_rep_rows = []
    all_comp_rows = []
    all_causal_rows = []
    layers = list(range(1, args.layers + 1))
    pad_id = token_to_id["<pad>"]

    for seed_index, seed in enumerate(seeds, start=1):
        write_status(out_dir, status="running", phase="seed_start", current_seed=seed, completed_seeds=seed_index - 1)
        set_seed(seed)
        rng = random.Random(seed)
        model = TinyDecoderOnlyTransformer(
            vocab_size=len(vocab),
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            max_seq_len=max_seq_len,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        def evaluate_checkpoint(step: int, last_loss: float | None) -> None:
            write_status(out_dir, status="running", phase="evaluating", current_seed=seed, current_step=step)
            behavior = evaluate_behavior(model, eval_groups, token_to_id, vocab, args.max_new_tokens, device, step, seed)
            rep, comp = run_representation_diagnostics(model, token_to_id, layers, device, step, seed, args.ridge_alpha)
            causal = run_causal_intervention(
                model,
                token_to_id,
                vocab,
                layer=args.layers,
                device=device,
                checkpoint_step=step,
                seed=seed,
                eval_limit=args.causal_limit,
                max_new_tokens=args.max_new_tokens,
            )
            all_behavior_rows.append(behavior)
            all_rep_rows.append(rep)
            all_comp_rows.append(comp)
            all_causal_rows.append(causal)
            if last_loss is not None:
                all_training_rows.append({"seed": seed, "checkpoint_step": step, "train_loss": last_loss})
            pd.concat(all_behavior_rows, ignore_index=True).to_csv(csv_dir / "behavior_eval.csv", index=False)
            pd.concat(all_rep_rows, ignore_index=True).to_csv(csv_dir / "representation_fit.csv", index=False)
            pd.concat(all_comp_rows, ignore_index=True).to_csv(csv_dir / "composition_diagnostics.csv", index=False)
            pd.concat(all_causal_rows, ignore_index=True).to_csv(csv_dir / "causal_intervention.csv", index=False)
            if all_training_rows:
                pd.DataFrame(all_training_rows).to_csv(csv_dir / "training_log.csv", index=False)

        evaluate_checkpoint(0, None)
        last_loss = math.nan
        for step in range(1, args.steps + 1):
            model.train()
            inputs, labels = sample_batch(encoded_train, args.batch_size, pad_id, rng)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.item())

            if step % 25 == 0 or step == args.steps:
                write_status(
                    out_dir,
                    status="running",
                    phase="training",
                    current_seed=seed,
                    current_step=step,
                    latest_loss=last_loss,
                    completed_seeds=seed_index - 1,
                )
            if step in checkpoints:
                evaluate_checkpoint(step, last_loss)

        write_status(out_dir, status="running", phase="seed_done", current_seed=seed, completed_seeds=seed_index)

    behavior_df = pd.concat(all_behavior_rows, ignore_index=True) if all_behavior_rows else pd.DataFrame()
    rep_df = pd.concat(all_rep_rows, ignore_index=True) if all_rep_rows else pd.DataFrame()
    comp_df = pd.concat(all_comp_rows, ignore_index=True) if all_comp_rows else pd.DataFrame()
    causal_df = pd.concat(all_causal_rows, ignore_index=True) if all_causal_rows else pd.DataFrame()

    write_figures(out_dir, behavior_df, rep_df, causal_df)
    write_summary(out_dir, config, behavior_df, rep_df, comp_df, causal_df)
    write_status(
        out_dir,
        status="finished",
        phase="complete",
        completed_seeds=len(seeds),
        total_seeds=len(seeds),
        finished_at=now(),
        summary=str(out_dir / "SUMMARY.md"),
    )


if __name__ == "__main__":
    main()
