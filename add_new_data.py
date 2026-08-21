"""Simulate a new data arrival by replacing train_phase1 with train_phase2.

After running this script:
    dvc add data/train_phase1.csv
    git add data/train_phase1.csv.dvc
    git commit -m "data: add phase-2 training samples"
    git push
The GitHub Actions pipeline then retrains, evaluates, and deploys automatically.
"""

import shutil

SRC = "data/train_phase2.csv"
DST = "data/train_phase1.csv"


def main() -> None:
    shutil.copyfile(SRC, DST)
    print(f"Copied {SRC} -> {DST}")
    print("Next: dvc add data/train_phase1.csv && git add data/train_phase1.csv.dvc")


if __name__ == "__main__":
    main()
