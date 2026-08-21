import json
import os

import numpy as np
import pandas as pd
import pytest

from src.train import train

FEATURES = [
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


def _write_synthetic_csvs(tmp_path):
    rng = np.random.default_rng(42)
    n_train, n_eval = 80, 20

    def make_frame(n):
        data = {
            "fixed_acidity": rng.uniform(4.0, 15.0, n),
            "volatile_acidity": rng.uniform(0.1, 1.2, n),
            "citric_acid": rng.uniform(0.0, 1.0, n),
            "residual_sugar": rng.uniform(0.5, 15.0, n),
            "chlorides": rng.uniform(0.01, 0.2, n),
            "free_sulfur_dioxide": rng.uniform(1.0, 70.0, n),
            "total_sulfur_dioxide": rng.uniform(6.0, 250.0, n),
            "density": rng.uniform(0.98, 1.01, n),
            "pH": rng.uniform(2.8, 3.8, n),
            "sulphates": rng.uniform(0.3, 1.5, n),
            "alcohol": rng.uniform(8.0, 14.5, n),
            "wine_type": rng.integers(0, 2, n),
            "target": rng.integers(0, 3, n),
        }
        return pd.DataFrame(data)[FEATURES + ["target"]]

    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    make_frame(n_train).to_csv(train_path, index=False)
    make_frame(n_eval).to_csv(eval_path, index=False)
    return str(train_path), str(eval_path)


@pytest.fixture
def synthetic_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "mlartifacts"))
    return _write_synthetic_csvs(tmp_path)


def test_train_returns_accuracy_float(synthetic_data):
    train_path, eval_path = synthetic_data
    params = {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2}
    acc = train(params, data_path=train_path, eval_path=eval_path)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_train_writes_metrics_json(synthetic_data):
    train_path, eval_path = synthetic_data
    params = {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2}
    acc = train(params, data_path=train_path, eval_path=eval_path)

    metrics_path = "outputs/metrics.json"
    assert os.path.exists(metrics_path)
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] == pytest.approx(acc)


def test_train_writes_model_pkl(synthetic_data):
    train_path, eval_path = synthetic_data
    params = {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2}
    train(params, data_path=train_path, eval_path=eval_path)
    assert os.path.exists("models/model.pkl")
