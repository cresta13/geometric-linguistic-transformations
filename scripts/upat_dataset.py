from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class UPATDatasetConfig:
    train_templates: int = 100
    test_templates: int = 100
    seed_train: int = 42
    seed_test: int = 123


class UPATDataset:
    CLASSES = ["negation", "question", "modality", "tense_shift", "passive"]

    TRAIN_SUBJECTS = [
        "scientist", "engineer", "teacher", "doctor", "programmer",
        "researcher", "analyst", "manager", "architect", "designer",
        "lawyer", "chemist", "biologist", "mathematician", "historian",
        "translator", "editor", "journalist", "consultant", "developer",
    ]

    TRAIN_ACTIONS = [
        ("accepted", "accept", "the explanation", "acceptance"),
        ("completed", "complete", "the repair", "completion"),
        ("confirmed", "confirm", "the answer", "confirmation"),
        ("approved", "approve", "the treatment", "approval"),
        ("fixed", "fix", "the bug", "a fix"),
        ("supported", "support", "the theory", "support"),
        ("verified", "verify", "the report", "verification"),
        ("reviewed", "review", "the document", "review"),
        ("designed", "design", "the system", "design"),
        ("tested", "test", "the prototype", "testing"),
    ]

    TEST_SUBJECTS = [
        "dragon", "wizard", "queen", "robot", "pirate",
        "oracle", "alien", "knight", "phoenix", "giant",
        "mermaid", "sphinx", "goblin", "vampire", "griffin",
        "druid", "ranger", "titan", "spirit", "sorcerer",
    ]

    TEST_ACTIONS = [
        ("guarded", "guard", "the treasure", "guardianship"),
        ("opened", "open", "the portal", "opening"),
        ("signed", "sign", "the treaty", "signature"),
        ("repaired", "repair", "the satellite", "repair"),
        ("found", "find", "the island", "discovery"),
        ("predicted", "predict", "the storm", "prediction"),
        ("built", "build", "the tower", "construction"),
        ("protected", "protect", "the village", "protection"),
        ("revealed", "reveal", "the secret", "revelation"),
        ("created", "create", "the artifact", "creation"),
    ]

    def __init__(self, config: UPATDatasetConfig | None = None):
        self.config = config or UPATDatasetConfig()

    def generate_item(self, subject: str, past: str, base: str, obj: str, nominal: str, split: str) -> dict:
        source = f"The {subject} {past} {obj}."

        if split == "train":
            negation = [
                f"The {subject} failed to {base} {obj}.",
                f"The {subject} refused to {base} {obj}.",
                f"The {subject} rejected {obj}.",
                f"The {subject} declined to {base} {obj}.",
            ]

            question = [
                f"Could the {subject} {base} {obj}?",
                f"Was the {subject} able to {base} {obj}?",
                f"Did anyone see the {subject} {base} {obj}?",
                f"Could anyone confirm whether the {subject} {past} {obj}?",
            ]

            modality = [
                f"The {subject} apparently {past} {obj}.",
                f"The {subject} reportedly {past} {obj}.",
                f"The {subject} seemingly {past} {obj}.",
                f"It appears that the {subject} {past} {obj}.",
            ]

            tense = [
                f"The {subject} had {past} {obj} earlier.",
                f"The {subject} used to {base} {obj}.",
                f"The {subject} previously {past} {obj}.",
                f"The {subject} once {past} {obj}.",
            ]

            passive = [
                f"{obj.capitalize()} received {nominal} from the {subject}.",
                f"{obj.capitalize()} was subject to {nominal} by the {subject}.",
                f"{obj.capitalize()} came under the {subject}'s {nominal}.",
                f"{obj.capitalize()} gained {nominal} through the {subject}.",
            ]

        else:
            negation = [
                f"The {subject} avoided {base}ing {obj}.",
                f"The {subject} failed to {base} {obj}.",
                f"The {subject} declined to {base} {obj}.",
                f"The {subject} never managed to {base} {obj}.",
            ]

            question = [
                f"I wonder whether the {subject} {past} {obj}.",
                f"Someone asked whether the {subject} {past} {obj}.",
                f"It is unclear whether the {subject} {past} {obj}.",
                f"Nobody knew whether the {subject} had {past} {obj}.",
            ]

            modality = [
                f"The {subject} allegedly {past} {obj}.",
                f"The {subject} supposedly {past} {obj}.",
                f"The {subject} was rumored to have {past} {obj}.",
                f"According to witnesses, the {subject} {past} {obj}.",
            ]

            tense = [
                f"The {subject} will {base} {obj} tomorrow.",
                f"The {subject} is going to {base} {obj} soon.",
                f"The {subject} may {base} {obj} later.",
                f"The {subject} plans to {base} {obj} next week.",
            ]

            passive = [
                f"{obj.capitalize()} became the {subject}'s {nominal}.",
                f"{obj.capitalize()} came under the {subject}'s {nominal}.",
                f"{obj.capitalize()} ended up marked by the {subject}'s {nominal}.",
                f"{obj.capitalize()} was left with signs of the {subject}'s {nominal}.",
            ]

        return {
            "source": source,
            "negation": np.random.choice(negation),
            "question": np.random.choice(question),
            "modality": np.random.choice(modality),
            "tense_shift": np.random.choice(tense),
            "passive": np.random.choice(passive),
        }

    def generate_items(self, subjects, actions, split: str, max_items: int, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        combos = [(s, a) for s in subjects for a in actions]
        rng.shuffle(combos)

        items = []
        for subject, action in combos[:max_items]:
            past, base, obj, nominal = action
            items.append(self.generate_item(subject, past, base, obj, nominal, split))

        return items

    def build(self) -> pd.DataFrame:
        train_items = self.generate_items(
            self.TRAIN_SUBJECTS,
            self.TRAIN_ACTIONS,
            "train",
            self.config.train_templates,
            self.config.seed_train,
        )

        test_items = self.generate_items(
            self.TEST_SUBJECTS,
            self.TEST_ACTIONS,
            "test",
            self.config.test_templates,
            self.config.seed_test,
        )

        rows = []

        for split_name, items in [("train", train_items), ("test", test_items)]:
            for item in items:
                for cls in self.CLASSES:
                    rows.append({
                        "source": item["source"],
                        "target": item[cls],
                        "class": cls,
                        "split": split_name,
                    })

        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> pd.DataFrame:
        df = self.build()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df