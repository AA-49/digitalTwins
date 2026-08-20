"""Behavior and cache tests for the optimized Stage 1-3 model wrapper."""
from __future__ import annotations

from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from diabetes_risk import FEATURES
from stage3 import DiabetesDigitalTwin


class FakeModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self):
        self.probability_calls = 0

    def predict_proba(self, frame):
        self.probability_calls += 1
        bmi = float(frame.iloc[0]["BMI"])
        high = min(0.8, bmi / 100)
        return np.array([[0.9 - high, 0.1, high]])

    def predict(self, frame):
        return self.classes_[np.argmax(self.predict_proba(frame), axis=1)]


class FakeExplainer:
    builds = 0
    calls = 0

    def __init__(self, _model):
        type(self).builds += 1

    def shap_values(self, _frame):
        type(self).calls += 1
        values = np.zeros((1, len(FEATURES), 3))
        values[0, :, 0] = np.linspace(-0.1, 0.1, len(FEATURES))
        values[0, :, 2] = np.linspace(0.3, -0.3, len(FEATURES))
        return values


def profile(bmi=30.0):
    values = {feature: 0.0 for feature in FEATURES}
    values.update(BMI=bmi, Age=7.0, GenHlth=3.0, Education=5.0, Income=6.0)
    return values


def twin_with_fake_model():
    twin = DiabetesDigitalTwin.__new__(DiabetesDigitalTwin)
    twin.model_path = Path("fake.joblib")
    twin.model = FakeModel()
    twin._explainer = None
    twin._explainer_lock = RLock()
    twin.clear_caches()
    return twin


class Stage3EfficiencyTests(unittest.TestCase):
    def setUp(self):
        FakeExplainer.builds = 0
        FakeExplainer.calls = 0

    def test_prediction_uses_one_probability_call_and_is_cached(self):
        twin = twin_with_fake_model()
        first = twin.predict(profile())
        second = twin.predict(profile())
        changed = twin.predict(profile(40.0))
        self.assertEqual(first, second)
        self.assertNotEqual(first["high_risk_probability"], changed["high_risk_probability"])
        self.assertEqual(2, twin.model.probability_calls)
        self.assertEqual(1, twin.cache_info()["prediction"]["hits"])
        twin.clear_caches()

    def test_explainer_and_exact_results_are_reused(self):
        twin = twin_with_fake_model()
        fake_shap = SimpleNamespace(TreeExplainer=FakeExplainer)
        with patch.dict(sys.modules, {"shap": fake_shap}):
            first = twin.explain(profile(), max_factors=None)
            second = twin.explain(profile(), max_factors=None)
        self.assertEqual(first, second)
        self.assertEqual(1, FakeExplainer.builds)
        self.assertEqual(1, FakeExplainer.calls)
        self.assertEqual(1, twin.model.probability_calls)
        self.assertEqual(1, twin.cache_info()["explanation"]["hits"])
        twin.clear_caches()

    def test_requested_class_two_shap_is_used_when_class_zero_is_predicted(self):
        twin = twin_with_fake_model()
        fake_shap = SimpleNamespace(TreeExplainer=FakeExplainer)
        self.assertEqual(0, twin.predict(profile())["predicted_class"])
        with patch.dict(sys.modules, {"shap": fake_shap}):
            predicted_class = twin.explain(profile(), max_factors=None)
            high_class = twin.explain(profile(), max_factors=None, class_id=2)

        predicted_by_feature = {item["feature"]: item["shap_value"] for item in predicted_class}
        high_by_feature = {item["feature"]: item["shap_value"] for item in high_class}
        self.assertAlmostEqual(-0.1, predicted_by_feature[FEATURES[0]])
        self.assertAlmostEqual(0.3, high_by_feature[FEATURES[0]])
        self.assertNotEqual(predicted_by_feature, high_by_feature)
        self.assertEqual(1, FakeExplainer.builds)
        self.assertEqual(2, FakeExplainer.calls)
        twin.clear_caches()


if __name__ == "__main__":
    unittest.main()
