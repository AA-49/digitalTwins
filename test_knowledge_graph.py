"""Focused tests for the temporary patient-centric Stage 4 graph."""
from __future__ import annotations

import unittest

from diabetes_risk import FEATURES
from knowledge_graph import (
    ATTRIBUTE_DEFINITIONS,
    BMI_CATEGORIES,
    DOMAIN_DEFINITIONS,
    VALUE_MAPS,
    PatientKnowledgeGraph,
    classify_bmi,
    decode_feature,
)
from stage3 import MODEL_CANDIDATES


class FakeResult:
    def __init__(self, rows=None):
        self.rows = rows or []

    def __iter__(self):
        return iter(self.rows)

    def consume(self):
        return self


class FakeSession:
    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        if "RETURN a.key AS key" in query:
            domains = {item["key"]: item["label"] for item in DOMAIN_DEFINITIONS}
            return FakeResult([
                {
                    "key": item["key"], "label": item["label"], "kind": item["kind"],
                    "displayOrder": index, "domainKey": item["domain"],
                    "domainLabel": domains[item["domain"]],
                }
                for index, item in enumerate(ATTRIBUTE_DEFINITIONS)
            ])
        return FakeResult()


class FakeDriver:
    def __init__(self):
        self.fake_session = FakeSession()

    def session(self):
        return self.fake_session

    def close(self):
        pass


def example_profile(bmi=32.0):
    profile = {feature: 0.0 for feature in FEATURES}
    profile.update(BMI=bmi, Age=7, GenHlth=3, Education=5, Income=6)
    return profile


def example_prediction():
    return {
        "predicted_class": 2, "label": "High (diabetes)", "high_risk_probability": 0.61,
        "probabilities": [
            {"class_id": 0, "label": "Low", "value": 0.29},
            {"class_id": 1, "label": "Medium (prediabetes)", "value": 0.10},
            {"class_id": 2, "label": "High (diabetes)", "value": 0.61},
        ],
    }


def example_contributions():
    values = [0.2, -0.1, 0.0] + [0.01] * (len(FEATURES) - 3)
    return [
        {"feature": feature, "value": example_profile()[feature], "shap_value": value}
        for feature, value in zip(FEATURES, values)
    ]


def explain(graph):
    return graph.explain(
        example_profile(), example_prediction(), example_contributions(),
        "balanced_random_forest.joblib",
    )


class KnowledgeGraphTests(unittest.TestCase):
    def test_stage3_model_candidates_remain_iterable(self):
        self.assertIsInstance(MODEL_CANDIDATES, tuple)
        self.assertGreaterEqual(len(MODEL_CANDIDATES), 1)

    def test_reusable_schema_covers_every_feature_and_domain(self):
        self.assertEqual(21, len(ATTRIBUTE_DEFINITIONS))
        self.assertEqual(set(FEATURES), {item["key"] for item in ATTRIBUTE_DEFINITIONS})
        self.assertEqual(6, len(DOMAIN_DEFINITIONS))
        self.assertEqual(4, len(BMI_CATEGORIES))

    def test_bmi_boundaries(self):
        for bmi, expected in [(18.49, "Underweight"), (18.5, "Healthy range"),
                              (24.99, "Healthy range"), (25.0, "Overweight"),
                              (29.99, "Overweight"), (30.0, "Obesity")]:
            with self.subTest(bmi=bmi):
                self.assertEqual(expected, classify_bmi(bmi))

    def test_every_categorical_code_decodes(self):
        for feature, mapping in VALUE_MAPS.items():
            for code, label in mapping.items():
                with self.subTest(feature=feature, code=code):
                    self.assertEqual(label, decode_feature(feature, code))
        self.assertEqual("1 mentally unhealthy day in past 30 days", decode_feature("MentHlth", 1))
        self.assertEqual("12 physically unhealthy days in past 30 days", decode_feature("PhysHlth", 12))

    def test_graph_contains_complete_temporary_patient_model(self):
        driver = FakeDriver()
        result = explain(PatientKnowledgeGraph(driver=driver))
        self.assertTrue(result["connected"])
        self.assertEqual(21, len(result["attributes"]))
        self.assertEqual(97, len(result["nodes"]))
        self.assertEqual(21, sum(node["type"] == "Observation" for node in result["nodes"]))
        self.assertEqual(21, sum(node["type"] == "ShapContribution" for node in result["nodes"]))
        self.assertEqual(3, sum(node["type"] == "RiskProbability" for node in result["nodes"]))
        bmi_observation = next(node for node in result["nodes"] if node["id"] == "observation-bmi")
        self.assertEqual("domain-clinical", bmi_observation["parent"])
        labels = {node["label"] for node in result["nodes"]}
        self.assertIn("Obesity", labels)
        self.assertNotIn("Insulin Resistance", labels)
        self.assertNotIn("Increased T2DM Risk", labels)
        self.assertFalse(any("PatientSnapshot" in query or "(:Patient" in query
                             for query, _ in driver.fake_session.queries))

    def test_class_two_shap_relationships_colors_and_probabilities_are_preserved(self):
        result = explain(PatientKnowledgeGraph(driver=FakeDriver()))
        relationships = {edge["relationship"] for edge in result["edges"]}
        self.assertTrue({
            "INCREASES_MODEL_HIGH_RISK_ESTIMATE",
            "DECREASES_MODEL_HIGH_RISK_ESTIMATE",
            "NEUTRAL_FOR_MODEL_HIGH_RISK_ESTIMATE",
        } <= relationships)
        self.assertEqual({"class_id": 2, "label": "High (diabetes)"}, result["shap_target"])
        shap_edges = [edge for edge in result["edges"] if "HIGH_RISK_ESTIMATE" in edge["relationship"]]
        self.assertEqual(21, len(shap_edges))
        self.assertTrue(all(edge["target"] == "probability-2" for edge in shap_edges))
        legend = {item["key"]: item["color"] for item in result["legend"]}
        self.assertEqual("#c92a3a", legend["positive_shap"])
        self.assertEqual("#087f5b", legend["negative_shap"])
        self.assertEqual("#7b8794", legend["neutral_shap"])
        probabilities = {node["details"]["class_id"]: node["details"]["probability"]
                         for node in result["nodes"] if node["type"] == "RiskProbability"}
        self.assertEqual({0: 0.29, 1: 0.10, 2: 0.61}, probabilities)
        self.assertFalse(any(node["type"] == "DigitalTwin" for node in result["nodes"]))
        self.assertFalse(any(edge["group"] == "twin" for edge in result["edges"]))

    def test_initialization_is_idempotent_and_removes_legacy_graph_nodes(self):
        driver = FakeDriver()
        graph = PatientKnowledgeGraph(driver=driver)
        explain(graph)
        explain(graph)
        constraint_queries = [query for query, _ in driver.fake_session.queries if "CREATE CONSTRAINT" in query]
        self.assertEqual(6, len(constraint_queries))
        definition_queries = [query for query, _ in driver.fake_session.queries if "RETURN a.key AS key" in query]
        model_queries = [query for query, _ in driver.fake_session.queries if "MERGE (m:ModelVersion" in query]
        self.assertEqual(1, len(definition_queries))
        self.assertEqual(1, len(model_queries))
        migration = "\n".join(query for query, _ in driver.fake_session.queries)
        self.assertIn("node:DigitalTwin", migration)
        self.assertIn("Insulin Resistance", migration)
        self.assertIn("Increased T2DM Risk", migration)

    def test_unavailable_database_keeps_all_attributes_visible(self):
        class UnavailableDriver:
            def session(self):
                raise RuntimeError("database stopped")

        result = explain(PatientKnowledgeGraph(driver=UnavailableDriver()))
        self.assertFalse(result["connected"])
        self.assertEqual(21, len(result["attributes"]))
        self.assertIn("database stopped", result["message"])
        self.assertEqual(97, len(result["nodes"]))
        self.assertEqual(132, len(result["edges"]))


if __name__ == "__main__":
    unittest.main()
