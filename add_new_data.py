"""Append train_phase2 onto train_phase1 to simulate newly collected data."""

import pandas as pd

PHASE1 = "data/train_phase1.csv"
PHASE2 = "data/train_phase2.csv"


def main() -> None:
    phase1 = pd.read_csv(PHASE1)
    phase2 = pd.read_csv(PHASE2)
    n_before = len(phase1)
    combined = pd.concat([phase1, phase2], ignore_index=True)
    combined.to_csv(PHASE1, index=False)
    print(f"Cập nhật dữ liệu: {n_before} -> {len(combined)} mẫu")


if __name__ == "__main__":
    main()
