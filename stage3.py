"""Prediction, explanation, and manual what-if simulation for Stages 1-3.

This module deliberately implements the scope in the project proposal through
Stage 3.  It does not recommend treatment (that is a Stage 4 knowledge-graph
responsibility); it only shows the model's changed estimate after a user edits
one or more survey variables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from diabetes_risk import FEATURES


MODEL_CANDIDATES = (
    Path("artifacts_notebook/diabetes_risk_random_forest.joblib"),
)
RISK_LABELS = {0: "Low", 1: "Medium (prediabetes)", 2: "High (diabetes)"}
SMPL_MODELS_DIR = Path("models/smpl")


def find_model_path() -> Path:
    """Return the preferred available trained-model path."""
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(f"No trained model found. Expected one of: {searched}")


class DiabetesDigitalTwin:
    """A lightweight Stage 3 twin backed by the trained risk model."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or find_model_path()
        self.model = joblib.load(self.model_path)

    @staticmethod
    def patient_frame(values: dict[str, float]) -> pd.DataFrame:
        """Put a validated set of BRFSS values into model feature order."""
        missing = [feature for feature in FEATURES if feature not in values]
        if missing:
            raise ValueError(f"Missing values for: {', '.join(missing)}")
        return pd.DataFrame([{feature: float(values[feature]) for feature in FEATURES}])

    def predict(self, values: dict[str, float]) -> dict[str, Any]:
        """Predict three class probabilities and expose high-diabetes probability."""
        patient = self.patient_frame(values)
        probabilities = self.model.predict_proba(patient)[0]
        class_ids = [int(class_id) for class_id in self.model.classes_]
        probability_by_class = dict(zip(class_ids, map(float, probabilities)))
        predicted_class = int(self.model.predict(patient)[0])
        return {
            "predicted_class": predicted_class,
            "label": RISK_LABELS[predicted_class],
            "high_risk_probability": probability_by_class.get(2, 0.0),
            "probabilities": [
                {
                    "class_id": class_id,
                    "label": RISK_LABELS[class_id],
                    "value": probability_by_class[class_id],
                }
                for class_id in class_ids
            ],
        }

    def explain(
        self, values: dict[str, float], max_factors: int | None = 5
    ) -> list[dict[str, Any]]:
        """Return SHAP factors for the class predicted for this one patient.

        SHAP output conventions differ between versions, so this supports both
        list-per-class and three-dimensional ndarray formats.
        """
        import shap

        patient = self.patient_frame(values)
        predicted_class = int(self.model.predict(patient)[0])
        class_index = list(self.model.classes_).index(predicted_class)
        values_out = shap.TreeExplainer(self.model).shap_values(patient)

        if isinstance(values_out, list):
            contributions = np.asarray(values_out[class_index])[0]
        else:
            array = np.asarray(values_out)
            if array.ndim != 3:
                raise RuntimeError("Unexpected SHAP output shape.")
            if array.shape[1] == len(FEATURES):
                contributions = array[0, :, class_index]
            elif array.shape[2] == len(FEATURES):
                contributions = array[0, class_index, :]
            else:
                raise RuntimeError(f"Unexpected SHAP output shape: {array.shape}")

        factors = pd.DataFrame({
            "feature": FEATURES,
            "value": patient.iloc[0].to_numpy(),
            "shap_value": contributions,
        })
        factors["absolute_shap_value"] = factors["shap_value"].abs()
        factors = factors.sort_values("absolute_shap_value", ascending=False)
        if max_factors is not None:
            factors = factors.head(max_factors)
        return factors[["feature", "value", "shap_value"]].to_dict(orient="records")

    def simulate(self, baseline: dict[str, float], scenario: dict[str, float]) -> dict[str, Any]:
        """Compare a current patient profile with a manually edited scenario."""
        baseline_result = self.predict(baseline)
        scenario_result = self.predict(scenario)
        changes = [
            {"feature": feature, "from": baseline[feature], "to": scenario[feature]}
            for feature in FEATURES
            if float(baseline[feature]) != float(scenario[feature])
        ]
        return {
            "baseline": baseline_result,
            "scenario": scenario_result,
            "changes": changes,
            "high_risk_change": scenario_result["high_risk_probability"] - baseline_result["high_risk_probability"],
        }


def smpl_twin_descriptor(bmi: float, high_risk_probability: float) -> dict[str, Any]:
    """Describe the SMPL representation for a Stage 3 dashboard result.

    The licensed SMPL weights are deliberately not shipped with this repository.
    Once a user places a licensed `.pkl` file in ``models/smpl/``, the same BMI
    and risk values can be exported through ``src.export_smpl``.
    """
    risk_percent = float(high_risk_probability) * 100
    if risk_percent > 70:
        color, band = "#FF4D4D", "High"
    elif risk_percent >= 40:
        color, band = "#FFA500", "Moderate"
    else:
        color, band = "#2ECC71", "Low"
    has_weights = any(SMPL_MODELS_DIR.glob("*.pkl"))
    return {
        "bmi": float(bmi),
        "beta0": (float(bmi) - 22.0) * 0.5,
        "risk_percent": risk_percent,
        "color": color,
        "band": band,
        "has_weights": has_weights,
    }
