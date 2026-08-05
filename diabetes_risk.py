"""Shared utilities for the BRFSS Stage 1/2 diabetes-risk workflow."""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

TARGET = "Diabetes_012"
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
ARTIFACTS = Path("artifacts")


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the known BRFSS CSV schema."""
    frame = pd.read_csv(path)
    expected = set(FEATURES + [TARGET])
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    frame = frame[FEATURES + [TARGET]].dropna().copy()
    target = frame.pop(TARGET).astype(int)
    if not set(target.unique()).issubset(RISK_LABELS):
        raise ValueError("Diabetes_012 must contain only 0, 1, and 2.")
    return frame, target


def validate_patient(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(frame.columns)
    extra = set(frame.columns) - set(FEATURES)
    if missing or extra:
        raise ValueError(
            f"Patient CSV must contain exactly the BRFSS feature columns. "
            f"Missing: {sorted(missing)}; extra: {sorted(extra)}"
        )
    if len(frame) != 1:
        raise ValueError("Patient CSV must contain exactly one row.")
    if frame[FEATURES].isna().any().any():
        raise ValueError("Patient CSV cannot have missing values.")
    return frame[FEATURES].astype(float)


def save_json(value: dict, path: str | Path) -> None:
    def converter(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"Not JSON serializable: {type(obj)}")
    Path(path).write_text(json.dumps(value, indent=2, default=converter), encoding="utf-8")


def load_model(path: str | Path = ARTIFACTS / "diabetes_risk_random_forest.joblib"):
    return joblib.load(path)
