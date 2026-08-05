"""Patient-centric, model-derived Neo4j knowledge graph for Stage 4.

Neo4j stores only reusable domains, feature definitions, value/category
definitions, and model evaluation metadata. Patient observations, predictions,
SHAP contributions, and Digital Twin nodes are assembled in memory for the
current request and are never persisted.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from diabetes_risk import FEATURES

try:
    from neo4j import GraphDatabase
except ImportError:  # Keep Stages 1-3 usable before the optional driver is installed.
    GraphDatabase = None


DOMAIN_DEFINITIONS = [
    {"key": "clinical", "label": "Clinical"},
    {"key": "lifestyle", "label": "Lifestyle"},
    {"key": "health-status", "label": "Health Status"},
    {"key": "demographic", "label": "Demographic"},
    {"key": "socioeconomic", "label": "Socioeconomic"},
    {"key": "healthcare-access", "label": "Healthcare Access"},
]

ATTRIBUTE_DEFINITIONS = [
    {"key": "HighBP", "label": "High blood pressure", "kind": "binary", "domain": "clinical"},
    {"key": "HighChol", "label": "High cholesterol", "kind": "binary", "domain": "clinical"},
    {"key": "CholCheck", "label": "Cholesterol checked in past 5 years", "kind": "binary", "domain": "healthcare-access"},
    {"key": "BMI", "label": "Body mass index (BMI)", "kind": "number", "domain": "clinical"},
    {"key": "Smoker", "label": "Smoked at least 100 cigarettes", "kind": "binary", "domain": "lifestyle"},
    {"key": "Stroke", "label": "History of stroke", "kind": "binary", "domain": "clinical"},
    {"key": "HeartDiseaseorAttack", "label": "Heart disease or heart attack", "kind": "binary", "domain": "clinical"},
    {"key": "PhysActivity", "label": "Recent physical activity", "kind": "binary", "domain": "lifestyle"},
    {"key": "Fruits", "label": "Fruit consumed daily", "kind": "binary", "domain": "lifestyle"},
    {"key": "Veggies", "label": "Vegetables consumed daily", "kind": "binary", "domain": "lifestyle"},
    {"key": "HvyAlcoholConsump", "label": "Heavy alcohol consumption", "kind": "binary", "domain": "lifestyle"},
    {"key": "AnyHealthcare", "label": "Has healthcare coverage", "kind": "binary", "domain": "healthcare-access"},
    {"key": "NoDocbcCost", "label": "Could not see doctor due to cost", "kind": "binary", "domain": "healthcare-access"},
    {"key": "GenHlth", "label": "General health category", "kind": "ordinal", "domain": "health-status"},
    {"key": "MentHlth", "label": "Mentally unhealthy days", "kind": "number", "domain": "health-status"},
    {"key": "PhysHlth", "label": "Physically unhealthy days", "kind": "number", "domain": "health-status"},
    {"key": "DiffWalk", "label": "Serious difficulty walking", "kind": "binary", "domain": "health-status"},
    {"key": "Sex", "label": "Sex", "kind": "binary", "domain": "demographic"},
    {"key": "Age", "label": "Age category", "kind": "ordinal", "domain": "demographic"},
    {"key": "Education", "label": "Education category", "kind": "ordinal", "domain": "socioeconomic"},
    {"key": "Income", "label": "Income category", "kind": "ordinal", "domain": "socioeconomic"},
]

BMI_CATEGORIES = [
    {"id": "BMI:Underweight", "feature": "BMI", "code": "underweight", "label": "Underweight", "minimum": 0.0, "maximum": 18.5, "order": 1},
    {"id": "BMI:Healthy range", "feature": "BMI", "code": "healthy", "label": "Healthy range", "minimum": 18.5, "maximum": 25.0, "order": 2},
    {"id": "BMI:Overweight", "feature": "BMI", "code": "overweight", "label": "Overweight", "minimum": 25.0, "maximum": 30.0, "order": 3},
    {"id": "BMI:Obesity", "feature": "BMI", "code": "obesity", "label": "Obesity", "minimum": 30.0, "maximum": None, "order": 4},
]

BINARY_LABELS = {
    "HighBP": {0: "High blood pressure not reported", 1: "High blood pressure reported"},
    "HighChol": {0: "High cholesterol not reported", 1: "High cholesterol reported"},
    "CholCheck": {0: "No cholesterol check in past 5 years", 1: "Cholesterol checked in past 5 years"},
    "Smoker": {0: "Has not smoked 100 cigarettes", 1: "Has smoked at least 100 cigarettes"},
    "Stroke": {0: "No stroke history reported", 1: "Stroke history reported"},
    "HeartDiseaseorAttack": {0: "No heart disease or heart attack reported", 1: "Heart disease or heart attack reported"},
    "PhysActivity": {0: "No recent leisure-time physical activity", 1: "Recent leisure-time physical activity"},
    "Fruits": {0: "Fruit not consumed daily", 1: "Fruit consumed daily"},
    "Veggies": {0: "Vegetables not consumed daily", 1: "Vegetables consumed daily"},
    "HvyAlcoholConsump": {0: "Heavy alcohol consumption not reported", 1: "Heavy alcohol consumption reported"},
    "AnyHealthcare": {0: "No healthcare coverage", 1: "Has healthcare coverage"},
    "NoDocbcCost": {0: "No cost-related doctor access barrier reported", 1: "Could not see doctor due to cost"},
    "DiffWalk": {0: "No serious walking difficulty reported", 1: "Serious walking difficulty reported"},
    "Sex": {0: "Female", 1: "Male"},
}

AGE_LABELS = {
    1: "Age 18 to 24", 2: "Age 25 to 29", 3: "Age 30 to 34",
    4: "Age 35 to 39", 5: "Age 40 to 44", 6: "Age 45 to 49",
    7: "Age 50 to 54", 8: "Age 55 to 59", 9: "Age 60 to 64",
    10: "Age 65 to 69", 11: "Age 70 to 74", 12: "Age 75 to 79",
    13: "Age 80 or older",
}

GENERAL_HEALTH_LABELS = {
    1: "Excellent general health", 2: "Very good general health",
    3: "Good general health", 4: "Fair general health", 5: "Poor general health",
}

EDUCATION_LABELS = {
    1: "Never attended school or kindergarten only",
    2: "Grades 1 through 8",
    3: "Grades 9 through 11",
    4: "High school graduate",
    5: "Some college or technical school",
    6: "College or technical school graduate",
}

INCOME_LABELS = {
    1: "Annual household income below $10,000",
    2: "Annual household income $10,000 to $14,999",
    3: "Annual household income $15,000 to $19,999",
    4: "Annual household income $20,000 to $24,999",
    5: "Annual household income $25,000 to $34,999",
    6: "Annual household income $35,000 to $49,999",
    7: "Annual household income $50,000 to $74,999",
    8: "Annual household income $75,000 or more",
}

VALUE_MAPS = {
    **BINARY_LABELS,
    "Age": AGE_LABELS,
    "GenHlth": GENERAL_HEALTH_LABELS,
    "Education": EDUCATION_LABELS,
    "Income": INCOME_LABELS,
}

VALUE_DEFINITIONS = [
    {
        "id": f"{feature}:{code}",
        "feature": feature,
        "code": code,
        "label": label,
    }
    for feature, mapping in VALUE_MAPS.items()
    for code, label in mapping.items()
]

_ATTRIBUTE_KEYS = [definition["key"] for definition in ATTRIBUTE_DEFINITIONS]
if len(ATTRIBUTE_DEFINITIONS) != 21 or set(_ATTRIBUTE_KEYS) != set(FEATURES):
    raise RuntimeError("Knowledge-graph attribute definitions must match all 21 model features.")


def classify_bmi(bmi: float) -> str:
    """Apply the adult BMI ranges used by the current prototype."""
    value = float(bmi)
    for category in BMI_CATEGORIES:
        if value >= category["minimum"] and (
            category["maximum"] is None or value < category["maximum"]
        ):
            return str(category["label"])
    raise ValueError("BMI must be zero or greater.")


def _integer_code(value: Any) -> int:
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"Expected a whole-number category code, received {value}.")
    return int(numeric)


def decode_feature(feature: str, value: Any) -> str:
    """Decode one BRFSS model value without inventing medical severity."""
    if feature == "BMI":
        return classify_bmi(float(value))
    if feature == "MentHlth":
        days = _integer_code(value)
        return f"{days} mentally unhealthy day{'s' if days != 1 else ''} in past 30 days"
    if feature == "PhysHlth":
        days = _integer_code(value)
        return f"{days} physically unhealthy day{'s' if days != 1 else ''} in past 30 days"
    mapping = VALUE_MAPS.get(feature)
    if mapping is None:
        return _format_value(value)
    code = _integer_code(value)
    if code not in mapping:
        raise ValueError(f"Unknown {feature} category code: {code}.")
    return str(mapping[code])


def _format_value(value: Any) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"


def _safe_id(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")


class PatientKnowledgeGraph:
    """Lazy Neo4j client returning a Cytoscape-ready temporary patient graph."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        driver: Any = None,
        metrics_path: str | Path = "artifacts_notebook/metrics.json",
        importance_path: str | Path = "artifacts_notebook/global_permutation_importance.csv",
    ) -> None:
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "training3graph")
        self._driver = driver
        self._driver_supplied = driver is not None
        self._initialized = False
        self._lock = Lock()
        self.metrics_path = Path(metrics_path)
        self.importance_path = Path(importance_path)
        self.metrics = self._load_metrics()
        self.importance = self._load_importance()

    def _load_metrics(self) -> dict[str, Any]:
        try:
            return json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_importance(self) -> dict[str, dict[str, float | int]]:
        try:
            with self.importance_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            return {}
        return {
            row["feature"]: {
                "rank": index,
                "mean": float(row["importance_mean"]),
                "std": float(row["importance_std"]),
            }
            for index, row in enumerate(rows, start=1)
        }

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        if GraphDatabase is None:
            raise RuntimeError("The Neo4j Python driver is not installed.")
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            connection_timeout=3,
            connection_acquisition_timeout=3,
        )
        return self._driver

    def initialize(self) -> None:
        """Create reusable definitions and remove the obsolete causal prototype."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            with self._get_driver().session() as session:
                for query in (
                    "CREATE CONSTRAINT domain_key IF NOT EXISTS FOR (d:Domain) REQUIRE d.key IS UNIQUE",
                    "CREATE CONSTRAINT attribute_key IF NOT EXISTS FOR (a:AttributeDefinition) REQUIRE a.key IS UNIQUE",
                    "CREATE CONSTRAINT value_definition_id IF NOT EXISTS FOR (v:ValueDefinition) REQUIRE v.id IS UNIQUE",
                    "CREATE CONSTRAINT category_definition_id IF NOT EXISTS FOR (c:CategoryDefinition) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT model_version_name IF NOT EXISTS FOR (m:ModelVersion) REQUIRE m.name IS UNIQUE",
                    "CREATE CONSTRAINT model_evaluation_id IF NOT EXISTS FOR (e:ModelEvaluation) REQUIRE e.id IS UNIQUE",
                ):
                    session.run(query).consume()

                # Narrow migration from the earlier BMI causal demo. Those nodes
                # are not learned by this project's data or model.
                session.run(
                    "MATCH (node) WHERE node:BmiCategory OR "
                    "(node:HealthConcept AND node.name IN ['Insulin Resistance', 'Increased T2DM Risk']) "
                    "DETACH DELETE node"
                ).consume()

                session.run(
                    """
                    UNWIND $domains AS domain
                    MERGE (d:Domain {key: domain.key})
                    SET d.label = domain.label
                    """,
                    domains=DOMAIN_DEFINITIONS,
                ).consume()
                session.run(
                    """
                    UNWIND $attributes AS attribute
                    MERGE (a:AttributeDefinition {key: attribute.key})
                    SET a.label = attribute.label,
                        a.kind = attribute.kind,
                        a.displayOrder = attribute.displayOrder
                    WITH a, attribute
                    MATCH (d:Domain {key: attribute.domain})
                    MERGE (a)-[:BELONGS_TO]->(d)
                    """,
                    attributes=[
                        {**definition, "displayOrder": index}
                        for index, definition in enumerate(ATTRIBUTE_DEFINITIONS)
                    ],
                ).consume()
                session.run(
                    """
                    UNWIND $values AS value
                    MERGE (v:ValueDefinition {id: value.id})
                    SET v.code = value.code, v.label = value.label
                    WITH v, value
                    MATCH (a:AttributeDefinition {key: value.feature})
                    MERGE (a)-[:DECODES_TO]->(v)
                    """,
                    values=VALUE_DEFINITIONS,
                ).consume()
                session.run(
                    """
                    UNWIND $categories AS category
                    MERGE (c:CategoryDefinition {id: category.id})
                    SET c.code = category.code,
                        c.label = category.label,
                        c.minimum = category.minimum,
                        c.maximum = category.maximum,
                        c.displayOrder = category.order
                    WITH c, category
                    MATCH (a:AttributeDefinition {key: category.feature})
                    MERGE (a)-[:CLASSIFIED_AS]->(c)
                    """,
                    categories=BMI_CATEGORIES,
                ).consume()
            self._initialized = True

    def _ensure_model(self, model_name: str) -> None:
        report = self.metrics.get("classification_report", {})
        medium = report.get("Medium (prediabetes)", {})
        high = report.get("High (diabetes)", {})
        low = report.get("Low", {})
        evaluation = {
            "id": f"{model_name}:evaluation",
            "accuracy": self.metrics.get("accuracy"),
            "balancedAccuracy": self.metrics.get("balanced_accuracy"),
            "macroF1": self.metrics.get("macro_f1"),
            "macroRocAuc": self.metrics.get("macro_ovr_roc_auc"),
            "lowRecall": low.get("recall"),
            "mediumRecall": medium.get("recall"),
            "highRecall": high.get("recall"),
        }
        with self._get_driver().session() as session:
            session.run(
                """
                MERGE (m:ModelVersion {name: $modelName})
                MERGE (e:ModelEvaluation {id: $evaluation.id})
                SET e += $evaluation
                MERGE (m)-[:EVALUATED_BY]->(e)
                """,
                modelName=model_name,
                evaluation=evaluation,
            ).consume()

    def explain(
        self,
        profile: dict[str, float],
        prediction: dict[str, Any],
        contributions: list[dict[str, Any]],
        twin: dict[str, Any],
        model_name: str,
    ) -> dict[str, Any]:
        """Join Neo4j definitions with live model outputs without patient writes."""
        fallback_attributes = self._local_attributes(profile, contributions)
        try:
            self.initialize()
            self._ensure_model(model_name)
            with self._get_driver().session() as session:
                rows = list(
                    session.run(
                        """
                        MATCH (a:AttributeDefinition)-[:BELONGS_TO]->(d:Domain)
                        WHERE a.key IN $keys
                        RETURN a.key AS key, a.label AS label, a.kind AS kind,
                               a.displayOrder AS displayOrder,
                               d.key AS domainKey, d.label AS domainLabel
                        ORDER BY a.displayOrder
                        """,
                        keys=FEATURES,
                    )
                )
            if len(rows) != len(FEATURES):
                raise RuntimeError("Neo4j did not return all 21 attribute definitions.")
            definitions = {row["key"]: dict(row) for row in rows}
            nodes, edges = self._build_graph(
                profile, prediction, contributions, twin, model_name, definitions
            )
            return {
                "connected": True,
                "message": "Reusable definitions loaded from Neo4j; this patient graph is temporary and was not stored.",
                "warning": self._model_warning(),
                "attributes": self._attributes(profile, contributions, definitions),
                "nodes": nodes,
                "edges": edges,
                "legend": [
                    {"key": "profile", "label": "Profile meaning", "color": "#1769aa"},
                    {"key": "supports", "label": "Supports prediction", "color": "#087f5b"},
                    {"key": "opposes", "label": "Opposes prediction", "color": "#c92a3a"},
                    {"key": "twin", "label": "Digital Twin", "color": "#6f42c1"},
                ],
            }
        except Exception as exc:
            return {
                "connected": False,
                "message": f"Knowledge graph unavailable: {exc}",
                "warning": self._model_warning(),
                "attributes": fallback_attributes,
                "nodes": [],
                "edges": [],
                "legend": [],
            }

    def _model_warning(self) -> str:
        medium = (
            self.metrics.get("classification_report", {})
            .get("Medium (prediabetes)", {})
            .get("recall")
        )
        if medium == 0:
            return "Model limitation: Medium (prediabetes) recall is 0.0 in the checked test metrics. Do not interpret this prototype as a clinical screening system."
        return "Model results are research outputs, not clinical diagnoses."

    def _attributes(
        self,
        profile: dict[str, float],
        contributions: list[dict[str, Any]],
        definitions: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        contribution_by_feature = {item["feature"]: item for item in contributions}
        return [
            {
                "key": key,
                "label": definitions[key]["label"],
                "kind": definitions[key]["kind"],
                "domain": definitions[key]["domainLabel"],
                "value": _format_value(profile[key]),
                "state": decode_feature(key, profile[key]),
                "shap_value": float(contribution_by_feature[key]["shap_value"]),
            }
            for key in FEATURES
        ]

    def _local_attributes(
        self, profile: dict[str, float], contributions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        definitions = {item["key"]: item for item in ATTRIBUTE_DEFINITIONS}
        domain_labels = {item["key"]: item["label"] for item in DOMAIN_DEFINITIONS}
        local = {
            key: {
                "key": key,
                "label": definition["label"],
                "kind": definition["kind"],
                "domainKey": definition["domain"],
                "domainLabel": domain_labels[definition["domain"]],
            }
            for key, definition in definitions.items()
        }
        return self._attributes(profile, contributions, local)

    def _build_graph(
        self,
        profile: dict[str, float],
        prediction: dict[str, Any],
        contributions: list[dict[str, Any]],
        twin: dict[str, Any],
        model_name: str,
        definitions: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        contribution_by_feature = {item["feature"]: item for item in contributions}
        missing = sorted(set(FEATURES) - set(contribution_by_feature))
        if missing:
            raise ValueError(f"Missing SHAP contributions for: {', '.join(missing)}")

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        def add_node(node_id: str, label: str, node_type: str, groups: list[str], **details):
            node = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "groups": groups,
                "summary": details.pop("summary", label),
                "details": details,
            }
            parent = node["details"].pop("parent", None)
            if parent:
                node["parent"] = parent
            nodes.append(node)

        def add_edge(source: str, target: str, relationship: str, label: str, group: str, **details):
            edges.append({
                "id": f"edge-{len(edges) + 1}",
                "source": source,
                "target": target,
                "relationship": relationship,
                "label": label,
                "group": group,
                **details,
            })

        add_node(
            "patient-current", "Current Patient", "PatientSnapshot", ["profile"],
            summary="Temporary patient snapshot built from the selected dataset row.",
            persistence="Not stored in Neo4j", feature_count=21,
        )

        for domain in DOMAIN_DEFINITIONS:
            domain_id = f"domain-{domain['key']}"
            add_node(domain_id, domain["label"], "Domain", ["profile", "explanation"],
                     summary=f"Groups current-profile observations in the {domain['label']} domain.")

        for feature in FEATURES:
            definition = definitions[feature]
            value = profile[feature]
            formatted_value = _format_value(value)
            state = decode_feature(feature, value)
            importance = self.importance.get(feature, {})
            contribution = float(contribution_by_feature[feature]["shap_value"])
            direction = "supports" if contribution > 0 else "opposes" if contribution < 0 else "neutral"
            observation_id = f"observation-{_safe_id(feature)}"
            definition_id = f"attribute-{_safe_id(feature)}"
            state_id = f"state-{_safe_id(feature)}"
            contribution_id = f"contribution-{_safe_id(feature)}"
            domain_id = f"domain-{definition['domainKey']}"

            add_node(
                observation_id, f"{definition['label']}: {formatted_value}", "Observation", ["profile", "explanation"],
                summary=f"Current {definition['label'].lower()} value is {formatted_value}.",
                parent=domain_id,
                feature=feature, raw_value=float(value), decoded_state=state,
                domain=definition["domainLabel"], global_importance=importance,
                shap_value=contribution, shap_direction=direction,
            )
            add_node(
                definition_id, definition["label"], "AttributeDefinition", ["profile"],
                summary=f"Reusable definition for the {feature} model input.",
                feature=feature, kind=definition["kind"], domain=definition["domainLabel"],
            )
            add_node(
                state_id, state, "State", ["profile"],
                summary=f"Dataset-defined meaning of {definition['label'].lower()} value {formatted_value}.",
                parent=domain_id,
                feature=feature, raw_value=float(value), rule="Dataset decoding only; not a causal medical claim.",
            )
            add_node(
                contribution_id, f"{feature} SHAP {contribution:+.3f}", "ShapContribution", ["explanation"],
                summary=f"{feature} {direction} the predicted class in this model output.",
                parent=domain_id,
                feature=feature, shap_value=contribution, direction=direction,
                meaning="Model support, not medical causation.",
            )
            add_edge("patient-current", observation_id, "HAS_OBSERVATION", "has observation", "profile")
            add_edge(observation_id, definition_id, "INSTANCE_OF", "instance of", "profile")
            add_edge(definition_id, domain_id, "BELONGS_TO", "belongs to", "profile")
            add_edge(observation_id, state_id, "HAS_STATE", "decoded as", "profile")
            add_edge(observation_id, contribution_id, "HAS_CONTRIBUTION", "has SHAP", "explanation")

        predicted_label = prediction["label"]
        prediction_id = "prediction-current"
        add_node(
            prediction_id, f"Prediction: {predicted_label}", "Prediction", ["prediction", "explanation"],
            summary=f"The model predicted {predicted_label} for the current patient.",
            predicted_class=prediction["predicted_class"], high_risk_probability=prediction["high_risk_probability"],
        )
        add_edge("patient-current", prediction_id, "HAS_PREDICTION", "has prediction", "prediction")

        probability_ids: dict[int, str] = {}
        for probability in prediction["probabilities"]:
            class_id = int(probability["class_id"])
            probability_id = f"probability-{class_id}"
            probability_ids[class_id] = probability_id
            value = float(probability["value"])
            add_node(
                probability_id, f"{probability['label']}: {value:.1%}", "RiskProbability", ["prediction"],
                summary=f"Model probability for {probability['label']} is {value:.1%}.",
                class_id=class_id, probability=value,
                selected=class_id == int(prediction["predicted_class"]),
            )
            add_edge(prediction_id, probability_id, "ESTIMATES", "estimates", "prediction", weight=value)

        for feature in FEATURES:
            contribution = float(contribution_by_feature[feature]["shap_value"])
            relationship = "SUPPORTS_PREDICTION" if contribution > 0 else "OPPOSES_PREDICTION" if contribution < 0 else "NEUTRAL_FOR_PREDICTION"
            group = "supports" if contribution > 0 else "opposes" if contribution < 0 else "neutral"
            add_edge(
                f"contribution-{_safe_id(feature)}", prediction_id, relationship,
                "supports" if contribution > 0 else "opposes" if contribution < 0 else "neutral",
                group, weight=abs(contribution), shap_value=contribution,
            )

        report = self.metrics.get("classification_report", {})
        add_node(
            "model-current", model_name, "ModelVersion", ["prediction", "explanation"],
            summary="Trained Random Forest artifact used for this prediction.",
            artifact=model_name,
        )
        add_node(
            "model-evaluation", "Model Evaluation", "ModelEvaluation", ["prediction"],
            summary=self._model_warning(),
            accuracy=self.metrics.get("accuracy"), balanced_accuracy=self.metrics.get("balanced_accuracy"),
            macro_f1=self.metrics.get("macro_f1"), macro_roc_auc=self.metrics.get("macro_ovr_roc_auc"),
            low_recall=report.get("Low", {}).get("recall"),
            medium_recall=report.get("Medium (prediabetes)", {}).get("recall"),
            high_recall=report.get("High (diabetes)", {}).get("recall"),
            warning=self._model_warning(),
        )
        add_edge(prediction_id, "model-current", "GENERATED_BY", "generated by", "prediction")
        add_edge("model-current", "model-evaluation", "EVALUATED_BY", "evaluated by", "prediction")

        add_node(
            "digital-twin-current", "Current 3D Digital Twin", "DigitalTwin", ["twin"],
            summary="3D representation linked to this patient snapshot.",
            bmi=twin["bmi"], beta0=twin["beta0"], risk_percent=twin["risk_percent"],
            risk_band=twin["band"], color=twin["color"],
        )
        add_edge("digital-twin-current", "patient-current", "REPRESENTS", "represents", "twin")
        add_edge("digital-twin-current", "observation-bmi", "SHAPED_BY", "shaped by", "twin")
        add_edge(
            "digital-twin-current", probability_ids.get(2, "probability-2"),
            "COLORED_BY", "colored by", "twin",
        )
        return nodes, edges

    def close(self) -> None:
        if self._driver is not None and hasattr(self._driver, "close"):
            self._driver.close()
        if not self._driver_supplied:
            self._driver = None
        self._initialized = False
