"""Download UCI Wine Quality and write the three lab CSV splits.

Expected sizes:
    train_phase1.csv : 2998 samples
    eval.csv         :  500 samples
    train_phase2.csv : 2998 samples
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

RED_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
WHITE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-white.csv"
)

FEATURE_ORDER = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type",
]


def _quality_to_target(quality: int) -> int:
    if quality <= 5:
        return 0
    if quality == 6:
        return 1
    return 2


def load_wine() -> pd.DataFrame:
    red = pd.read_csv(RED_URL, sep=";")
    white = pd.read_csv(WHITE_URL, sep=";")
    red["wine_type"] = 0
    white["wine_type"] = 1
    df = pd.concat([red, white], ignore_index=True)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["target"] = df["quality"].map(_quality_to_target)
    return df[FEATURE_ORDER + ["target"]]


def main() -> None:
    os.makedirs("data", exist_ok=True)
    df = load_wine()

    eval_df, rest = train_test_split(
        df, train_size=500, random_state=42, stratify=df["target"]
    )
    rest = rest.iloc[:5996]
    phase1, phase2 = train_test_split(
        rest, train_size=2998, random_state=42, stratify=rest["target"]
    )

    phase1.to_csv("data/train_phase1.csv", index=False)
    eval_df.to_csv("data/eval.csv", index=False)
    phase2.to_csv("data/train_phase2.csv", index=False)

    print(f"train_phase1.csv : {len(phase1):5d} samples")
    print(f"eval.csv         : {len(eval_df):5d} samples")
    print(f"train_phase2.csv : {len(phase2):5d} samples")


if __name__ == "__main__":
    main()
