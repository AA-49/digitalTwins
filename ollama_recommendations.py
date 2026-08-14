"""Local Ollama guidance generated from temporary Stage 4 patient evidence."""
from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:1.5b"
SYSTEM_PROMPT = """
Select references for a diabetes-risk research explanation. Return one JSON
object only, with exactly these keys: supporting_features, opposing_features,
and discussion_topics. Feature lists may contain only supplied feature keys;
supporting features must have positive SHAP values and opposing features must
have negative SHAP values. Choose at most three supporting and two opposing
features. discussion_topics may contain at most three codes from the supplied
allowed_discussion_topics object. Do not write prose, diagnoses, advice,
treatment, medication instructions, or additional keys.
""".strip()

DISCUSSION_TOPICS: dict[str, tuple[str, str]] = {
    "blood_pressure": ("HighBP", "the recorded blood-pressure status"),
    "cholesterol": ("HighChol", "the recorded cholesterol status"),
    "physical_activity": ("PhysActivity", "the recorded physical-activity response"),
    "nutrition": ("Fruits", "the recorded fruit and vegetable responses"),
    "smoking": ("Smoker", "the recorded smoking-history response"),
    "mobility": ("DiffWalk", "the recorded walking-difficulty response"),
    "general_health": ("GenHlth", "the recorded general-health response"),
    "healthcare_access": ("AnyHealthcare", "the recorded healthcare-access responses"),
    "wellbeing": ("MentHlth", "the recorded physical and mental health-day responses"),
    "alcohol": ("HvyAlcoholConsump", "the recorded alcohol-consumption response"),
}


class OllamaRecommendationError(RuntimeError):
    """A local-model failure with a safe dashboard message."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.public_message = message


def configured_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def build_recommendation_input(
    patient_number: int,
    prediction: dict[str, Any],
    knowledge_graph: dict[str, Any],
    twin: dict[str, Any],
) -> dict[str, Any]:
    """Build the complete, temporary evidence package for the local model."""
    attributes = knowledge_graph.get("attributes", [])
    probabilities = prediction.get("probabilities", [])
    if len(attributes) != 21:
        raise OllamaRecommendationError("All 21 patient attributes are required for local guidance.")
    if len(probabilities) != 3:
        raise OllamaRecommendationError("All three model probabilities are required for local guidance.")

    attribute_by_key = {item["key"]: item for item in attributes}
    allowed_topics = {
        code: description
        for code, (_feature, description) in DISCUSSION_TOPICS.items()
        if _topic_is_supported(code, attribute_by_key)
    }
    return {
        "local_patient_reference": f"Patient #{patient_number}",
        "prediction": {
            "label": prediction.get("label"),
            "predicted_class": prediction.get("predicted_class"),
            "probabilities": probabilities,
        },
        "observations_and_shap_evidence": attributes,
        "digital_twin": {
            "bmi": twin.get("bmi"),
            "risk_percent": twin.get("risk_percent"),
            "risk_band": twin.get("band"),
        },
        "model_limitation": knowledge_graph.get("warning"),
        "allowed_discussion_topics": allowed_topics,
        "evidence_boundary": (
            "BRFSS observations and SHAP values are model-based associations, not clinical causes."
        ),
    }


def _topic_is_supported(
    code: str, attributes: dict[str, dict[str, Any]]
) -> bool:
    values = {key: float(item["value"]) for key, item in attributes.items()}
    checks = {
        "blood_pressure": values.get("HighBP") == 1,
        "cholesterol": values.get("HighChol") == 1,
        "physical_activity": values.get("PhysActivity") == 0,
        "nutrition": values.get("Fruits") == 0 or values.get("Veggies") == 0,
        "smoking": values.get("Smoker") == 1,
        "mobility": values.get("DiffWalk") == 1,
        "general_health": values.get("GenHlth", 0) >= 3,
        "healthcare_access": values.get("AnyHealthcare") == 0 or values.get("NoDocbcCost") == 1,
        "wellbeing": values.get("MentHlth", 0) > 0 or values.get("PhysHlth", 0) > 0,
        "alcohol": values.get("HvyAlcoholConsump") == 1,
    }
    return bool(checks.get(code, False))


def _validate_selection(
    raw: str, evidence: dict[str, Any]
) -> dict[str, list[str]]:
    try:
        selection = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaRecommendationError("The local model returned invalid structured output.") from exc
    required_keys = {"supporting_features", "opposing_features", "discussion_topics"}
    if not isinstance(selection, dict) or set(selection) != required_keys:
        raise OllamaRecommendationError("The local model returned an unexpected selection schema.")
    if not all(isinstance(selection[key], list) for key in required_keys):
        raise OllamaRecommendationError("The local model selection fields must be lists.")

    attributes = {
        item["key"]: item for item in evidence["observations_and_shap_evidence"]
    }
    supports = selection["supporting_features"]
    opposes = selection["opposing_features"]
    topics = selection["discussion_topics"]
    if len(supports) > 3 or len(opposes) > 2 or len(topics) > 3:
        raise OllamaRecommendationError("The local model selected too many evidence references.")
    if any(not isinstance(item, str) for item in supports + opposes + topics):
        raise OllamaRecommendationError("The local model selected an invalid reference type.")
    if len(set(supports + opposes)) != len(supports + opposes):
        raise OllamaRecommendationError("The local model repeated an evidence reference.")
    if any(key not in attributes or float(attributes[key]["shap_value"]) <= 0 for key in supports):
        raise OllamaRecommendationError("The local model selected unsupported positive evidence.")
    if any(key not in attributes or float(attributes[key]["shap_value"]) >= 0 for key in opposes):
        raise OllamaRecommendationError("The local model selected unsupported negative evidence.")
    allowed_topics = evidence["allowed_discussion_topics"]
    if len(set(topics)) != len(topics) or any(code not in allowed_topics for code in topics):
        raise OllamaRecommendationError("The local model selected an unsupported discussion topic.")
    return {key: list(selection[key]) for key in required_keys}


def _render_selection(
    patient_number: int,
    prediction: dict[str, Any],
    knowledge_graph: dict[str, Any],
    selection: dict[str, list[str]],
) -> str:
    attributes = {item["key"]: item for item in knowledge_graph["attributes"]}

    def describe(keys: list[str]) -> str:
        return "; ".join(
            f"{attributes[key]['label']} ({attributes[key]['state']}, SHAP {float(attributes[key]['shap_value']):+.3f})"
            for key in keys
        ) or "no factors selected"

    probabilities = ", ".join(
        f"{item['label']} {float(item['value']):.1%}"
        for item in prediction["probabilities"]
    )
    topics = "; ".join(
        DISCUSSION_TOPICS[code][1] for code in selection["discussion_topics"]
    ) or "the recorded survey observations in their full context"
    warning = knowledge_graph.get("warning") or "Model results are research outputs."
    return (
        f"For Patient #{patient_number}, the research model predicted {prediction['label']}. "
        f"Its estimated probabilities were {probabilities}. These are model estimates, not a diagnosis.\n\n"
        f"The validated positive SHAP references selected were {describe(selection['supporting_features'])}. "
        f"The validated negative SHAP references selected were {describe(selection['opposing_features'])}. "
        "SHAP values describe support for or opposition to this model output; they do not establish medical causes.\n\n"
        f"A qualified healthcare professional can help interpret {topics}. No raw local-model wording is displayed. "
        f"{warning} This explanation is model-based, non-causal, for research only, and not medical advice."
    )


def _deterministic_evidence_summary(
    patient_number: int,
    prediction: dict[str, Any],
    knowledge_graph: dict[str, Any],
    intro: str | None = None,
) -> str:
    probabilities = prediction["probabilities"]
    probability_text = ", ".join(
        f"{item['label']} {float(item['value']):.1%}" for item in probabilities
    )
    attributes = sorted(
        knowledge_graph["attributes"], key=lambda item: float(item["shap_value"]), reverse=True
    )
    supports = [item for item in attributes if float(item["shap_value"]) > 0][:3]
    opposes = [item for item in reversed(attributes) if float(item["shap_value"]) < 0][:2]

    def describe(items: list[dict[str, Any]]) -> str:
        return "; ".join(
            f"{item['label']} ({item['state']}, SHAP {float(item['shap_value']):+.3f})"
            for item in items
        ) or "no factors"

    warning = knowledge_graph.get("warning") or "Model results are research outputs."
    summary_intro = intro or (
        "The local model draft did not pass the research-safety checks, so this deterministic "
    )
    return (
        summary_intro
        + f"evidence summary is shown instead. For Patient #{patient_number}, the research model "
        f"predicted {prediction['label']}. Its estimated probabilities were {probability_text}. "
        "These are model estimates, not a diagnosis.\n\n"
        f"The largest positive SHAP contributions were {describe(supports)}. The largest negative "
        f"SHAP contributions were {describe(opposes)}. Positive and negative SHAP values describe "
        "support for or opposition to this model output; they do not mean that a feature caused "
        "or prevented diabetes.\n\n"
        "A qualified healthcare professional can help interpret the recorded survey observations "
        "in their full context. Do not use this prototype to start, stop, or change medication or "
        f"treatment. {warning} This summary is model-based, non-causal, for research only, and not "
        "medical advice."
    )


def generate_local_guidance(
    patient_number: int,
    prediction: dict[str, Any],
    knowledge_graph: dict[str, Any],
    twin: dict[str, Any],
) -> str:
    """Generate one non-streaming response from the local Ollama API."""
    evidence = build_recommendation_input(patient_number, prediction, knowledge_graph, twin)
    base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = configured_model()
    try:
        timeout = max(10.0, float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180")))
    except ValueError as exc:
        raise OllamaRecommendationError("OLLAMA_TIMEOUT_SECONDS must be a number.") from exc

    body = json.dumps({
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": (
            "Select only valid references from this temporary evidence.\n\n"
            + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        ),
        "format": "json",
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.1, "num_predict": 320, "num_ctx": 8192},
    }).encode("utf-8")
    request = Request(
        f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise OllamaRecommendationError(
                f"The Ollama model '{model}' is not installed. Pull it and try again."
            ) from exc
        raise OllamaRecommendationError(f"Ollama rejected the request with HTTP {exc.code}.") from exc
    except (URLError, ConnectionError) as exc:
        raise OllamaRecommendationError(
            "The dashboard cannot reach local Ollama. Start Ollama and confirm port 11434 is available."
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise OllamaRecommendationError(
            "The local model timed out. Keep Ollama running and try again after it finishes loading."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OllamaRecommendationError("Ollama returned an unreadable response.") from exc

    raw_selection = str(result.get("response", "")).strip()
    if not raw_selection:
        raise OllamaRecommendationError("Ollama returned an empty response.")
    try:
        selection = _validate_selection(raw_selection, evidence)
    except OllamaRecommendationError:
        return _deterministic_evidence_summary(patient_number, prediction, knowledge_graph)
    return _render_selection(patient_number, prediction, knowledge_graph, selection)


def deterministic_evidence_summary(
    patient_number: int,
    prediction: dict[str, Any],
    knowledge_graph: dict[str, Any],
) -> str:
    """Return the safety-checked fallback for hosts that cannot reach Ollama."""
    return _deterministic_evidence_summary(
        patient_number,
        prediction,
        knowledge_graph,
        intro="This hosted deployment uses a deterministic, safety-checked ",
    )


def guidance_error_message(exc: Exception) -> str:
    if isinstance(exc, OllamaRecommendationError):
        return exc.public_message
    return "The local guidance could not be generated. Check the dashboard and Ollama logs."
