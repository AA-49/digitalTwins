"""Local Ollama guidance generated from temporary Stage 4 patient evidence."""
from __future__ import annotations

import json
import os
import re
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:1.5b"
SYSTEM_PROMPT = """
You are writing a patient-friendly explanation for a diabetes-risk research
prototype. Use only the supplied BRFSS observations, decoded states, three
model probabilities, patient-specific SHAP values, Digital Twin summary, and
model limitation.

Write three or four short connected paragraphs in clear, simple English. Do
not use bullet points or numbered lists. First explain the prediction and the
strongest model-supporting and model-opposing factors. Then suggest practical,
low-risk topics the person could discuss with a qualified healthcare
professional, but only when supported by the supplied observations. Do not
diagnose, prescribe treatment, recommend medication changes, invent laboratory
results, or claim that SHAP values or graph links are medical causes. Do not
promise that changing a factor will reduce risk. Explicitly state the supplied
model limitation and finish by saying the output is model-based, non-causal,
for research only, and not medical advice.
""".strip()

PROHIBITED_CLAIMS = (
    "causes diabetes",
    "prevent diabetes",
    "protective effect",
    "reduces the risk",
    "reduce the risk",
    "increases the risk",
    "increase the risk",
    "risk of developing diabetes",
    "chances of diabetes",
    "strong predictor",
    "contribute to an increased likelihood",
)


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
        "evidence_boundary": (
            "BRFSS observations and SHAP values are model-based associations, not clinical causes."
        ),
    }


def _passes_research_safety_checks(text: str) -> bool:
    lowered = text.lower()
    required = (
        "research" in lowered,
        "not medical advice" in lowered,
        "0.0" in lowered and ("medium" in lowered or "prediabetes" in lowered),
        "non-causal" in lowered or "not a cause" in lowered or "does not mean" in lowered,
    )
    if not all(required):
        return False
    if any(claim in lowered for claim in PROHIBITED_CLAIMS):
        return False
    if re.search(r"(?m)^\s*(?:#{1,6}\s|[-*]\s|\d+\.\s)", text):
        return False
    return True


def _deterministic_evidence_summary(
    patient_number: int,
    prediction: dict[str, Any],
    knowledge_graph: dict[str, Any],
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
    return (
        "The local model draft did not pass the research-safety checks, so this deterministic "
        f"evidence summary is shown instead. For Patient #{patient_number}, the research model "
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
            "Prepare the research-only explanation from this temporary patient evidence. "
            "Do not infer any fact that is absent.\n\n"
            + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        ),
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

    guidance = str(result.get("response", "")).strip()
    if not guidance:
        raise OllamaRecommendationError("Ollama returned an empty response.")
    if not _passes_research_safety_checks(guidance):
        return _deterministic_evidence_summary(patient_number, prediction, knowledge_graph)
    return guidance


def guidance_error_message(exc: Exception) -> str:
    if isinstance(exc, OllamaRecommendationError):
        return exc.public_message
    return "The local guidance could not be generated. Check the dashboard and Ollama logs."
