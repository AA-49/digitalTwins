"""Tests for the local Ollama recommendation boundary."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ollama_recommendations import (
    OllamaRecommendationError,
    build_recommendation_input,
    generate_local_guidance,
)


def evidence():
    attributes = [
        {
            "key": f"feature_{index}", "label": f"Feature {index}",
            "domain": "Research", "value": str(index), "state": f"State {index}",
            "shap_value": (index - 10) / 100,
        }
        for index in range(21)
    ]
    prediction = {
        "label": "High (diabetes)", "predicted_class": 2,
        "high_risk_probability": 0.7,
        "probabilities": [
            {"class_id": 0, "label": "Low", "value": 0.2},
            {"class_id": 1, "label": "Medium", "value": 0.1},
            {"class_id": 2, "label": "High", "value": 0.7},
        ],
    }
    graph = {"attributes": attributes, "warning": "Medium recall is 0.0."}
    twin = {
        "bmi": 32.0, "beta0": 0.4, "risk_percent": 70.0,
        "band": "high", "color": "#d72845",
    }
    return prediction, graph, twin


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({
            "response": (
                "This research explanation is non-causal. Medium/prediabetes recall is 0.0. "
                "It is not medical advice."
            )
        }).encode()


class UnsafeResponse(FakeResponse):
    def read(self):
        return json.dumps({
            "response": "# Advice\n1. This factor increases the risk and has a protective effect."
        }).encode()


class OllamaRecommendationTests(unittest.TestCase):
    def test_payload_preserves_complete_patient_evidence(self):
        prediction, graph, twin = evidence()
        payload = build_recommendation_input(7, prediction, graph, twin)
        self.assertEqual(21, len(payload["observations_and_shap_evidence"]))
        self.assertEqual(3, len(payload["prediction"]["probabilities"]))
        self.assertEqual("Medium recall is 0.0.", payload["model_limitation"])

    def test_incomplete_evidence_is_rejected(self):
        prediction, graph, twin = evidence()
        graph["attributes"].pop()
        with self.assertRaises(OllamaRecommendationError):
            build_recommendation_input(1, prediction, graph, twin)

    @patch("ollama_recommendations.urlopen", return_value=FakeResponse())
    def test_generate_uses_local_non_streaming_api(self, urlopen_mock):
        prediction, graph, twin = evidence()
        with patch.dict(
            "ollama_recommendations.os.environ",
            {"OLLAMA_BASE_URL": "http://local-ollama:11434", "OLLAMA_MODEL": "test-model"},
        ):
            result = generate_local_guidance(7, prediction, graph, twin)

        self.assertIn("This research explanation is non-causal", result)
        request = urlopen_mock.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual("http://local-ollama:11434/api/generate", request.full_url)
        self.assertEqual("test-model", body["model"])
        self.assertFalse(body["stream"])
        self.assertIn("Medium recall is 0.0.", body["prompt"])
        self.assertEqual(21, body["prompt"].count('"shap_value"'))

    @patch("ollama_recommendations.urlopen", return_value=UnsafeResponse())
    def test_unsafe_draft_is_replaced_with_deterministic_summary(self, _urlopen_mock):
        prediction, graph, twin = evidence()
        result = generate_local_guidance(7, prediction, graph, twin)
        self.assertIn("did not pass the research-safety checks", result)
        self.assertIn("Low 20.0%, Medium 10.0%, High 70.0%", result)
        self.assertIn("not medical advice", result)
        self.assertNotIn("protective effect", result)

    def test_dashboard_renders_local_guidance(self):
        import app as dashboard

        prediction, graph, twin_data = evidence()
        graph.update({"connected": False, "message": "Test graph", "nodes": [], "edges": [], "legend": []})
        profile = {name: 0.0 for name, *_rest in dashboard.FIELDS}
        profile.update(BMI=32.0, Age=7.0, GenHlth=3.0, Education=5.0, Income=6.0)

        class FakeDataset:
            source_name = "test.csv"
            frame = [profile]

            def patient(self, _number):
                return profile.copy()

            def patient_window(self, _number):
                return []

            def actual_label(self, _number):
                return None

        fake_twin = SimpleNamespace(
            model_path=Path("fake-model.joblib"),
            predict=lambda _profile: prediction,
            explain=lambda _profile, max_factors=None: graph["attributes"],
        )

        with (
            patch.object(dashboard, "get_patient_dataset", return_value=FakeDataset()),
            patch.object(dashboard, "get_twin", return_value=fake_twin),
            patch.object(dashboard.KNOWLEDGE_GRAPH, "explain", return_value=graph),
            patch.object(dashboard, "smpl_twin_descriptor", return_value=twin_data),
            patch.object(dashboard, "refresh_smpl_twin", return_value="Test twin"),
            patch.object(dashboard, "generate_local_guidance", return_value="Safe local explanation."),
        ):
            response = dashboard.app.test_client().post(
                "/", data={"action": "local_guidance", "patient_number": "1"}
            )

        self.assertEqual(200, response.status_code)
        self.assertIn(b"Safe local explanation.", response.data)
        self.assertIn(b"Generate local research guidance", response.data)


if __name__ == "__main__":
    unittest.main()
