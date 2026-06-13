from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class LieCompositionConfig:
    n_templates: int = 80
    seed: int = 42


class LieCompositionDataset:
    OPS = ["N", "Q", "M", "T"]

    SUBJECTS = [
        "scientist", "engineer", "teacher", "doctor", "programmer",
        "researcher", "analyst", "manager", "architect", "designer",
        "wizard", "dragon", "queen", "robot", "pirate", "oracle",
    ]

    ACTIONS = [
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

    def __init__(self, config: LieCompositionConfig | None = None):
        self.config = config or LieCompositionConfig()

    def base_sentence(self, subject: str, past: str, obj: str) -> str:
        return f"The {subject} {past} {obj}."

    def apply_op(self, op: str, subject: str, past: str, base: str, obj: str, state: str = "base") -> str:
        if op == "N":
            return f"The {subject} failed to {base} {obj}."

        if op == "Q":
            return f"Could the {subject} {base} {obj}?"

        if op == "M":
            return f"The {subject} allegedly {past} {obj}."

        if op == "T":
            return f"The {subject} will {base} {obj} tomorrow."

        raise ValueError(f"Unknown op: {op}")

    def apply_composition(self, first: str, second: str, subject: str, past: str, base: str, obj: str) -> str:
        # Hand-written canonical composed forms.
        # first then second = second(first(x)) in natural language form.

        pair = first + second

        if pair == "NQ":
            return f"Could the {subject} fail to {base} {obj}?"
        if pair == "QN":
            return f"Is it false that the {subject} {past} {obj}?"

        if pair == "NM":
            return f"The {subject} allegedly failed to {base} {obj}."
        if pair == "MN":
            return f"The {subject} failed to allegedly {base} {obj}."

        if pair == "NT":
            return f"The {subject} will fail to {base} {obj} tomorrow."
        if pair == "TN":
            return f"The {subject} failed to have {past} {obj} earlier."

        if pair == "QM":
            return f"Could the {subject} allegedly {base} {obj}?"
        if pair == "MQ":
            return f"Is it alleged that the {subject} {past} {obj}?"

        if pair == "QT":
            return f"Could the {subject} {base} {obj} tomorrow?"
        if pair == "TQ":
            return f"Will the {subject} {base} {obj} tomorrow?"

        if pair == "MT":
            return f"The {subject} will allegedly {base} {obj} tomorrow."
        if pair == "TM":
            return f"The {subject} was allegedly going to {base} {obj}."

        raise ValueError(f"Unsupported composition: {pair}")

    def build(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.config.seed)

        combos = [(s, a) for s in self.SUBJECTS for a in self.ACTIONS]
        rng.shuffle(combos)
        combos = combos[: self.config.n_templates]

        rows = []

        op_pairs = [
            ("N", "Q"),
            ("N", "M"),
            ("N", "T"),
            ("Q", "M"),
            ("Q", "T"),
            ("M", "T"),
        ]

        for i, (subject, action) in enumerate(combos):
            past, base, obj = action
            source = self.base_sentence(subject, past, obj)

            for a, b in op_pairs:
                ab = self.apply_composition(a, b, subject, past, base, obj)
                ba = self.apply_composition(b, a, subject, past, base, obj)

                rows.append({
                    "template_id": i,
                    "source": source,
                    "op_a": a,
                    "op_b": b,
                    "pair": f"{a}{b}_vs_{b}{a}",
                    "ab_text": ab,
                    "ba_text": ba,
                })

        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.build()
        df.to_csv(path, index=False)
        return df