"""Stage 1-4 local dashboard for the diabetes-risk research prototype."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, render_template, request, send_file, url_for

from diabetes_risk import FEATURES
from knowledge_graph import PatientKnowledgeGraph
from ollama_recommendations import (
    configured_model,
    generate_local_guidance,
    guidance_error_message,
)
from stage3 import DiabetesDigitalTwin, RISK_LABELS, smpl_twin_descriptor
from twin_assets import TwinAssetResult, TwinAssetService

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
TWIN: DiabetesDigitalTwin | None = None
KNOWLEDGE_GRAPH = PatientKnowledgeGraph(
    metrics_path=os.environ.get("MODEL_METRICS_PATH", "artifacts_notebook/metrics.json")
)
TWIN_GLB_PATH = Path("artifacts_notebook") / "digital_twin.glb"
TWIN_METADATA_PATH = TWIN_GLB_PATH.with_suffix(".json")
SCENARIO_TWIN_GLB_PATH = Path("artifacts_notebook") / "digital_twin_scenario.glb"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"
TARGET = "Diabetes_012"


FIELDS = [
    ("BMI", "Body mass index (BMI)", "number", 40, {"min": 12, "max": 100, "step": "0.1"}),
    ("Age", "Age category (1 = 18-24, 13 = 80+)", "number", 9, {"min": 1, "max": 13, "step": 1}),
    ("GenHlth", "General health (1 = excellent, 5 = poor)", "number", 3, {"min": 1, "max": 5, "step": 1}),
    ("PhysHlth", "Physically unhealthy days in the past 30 days", "number", 5, {"min": 0, "max": 30, "step": 1}),
    ("MentHlth", "Mentally unhealthy days in the past 30 days", "number", 5, {"min": 0, "max": 30, "step": 1}),
    ("Education", "Education category (1 to 6)", "number", 4, {"min": 1, "max": 6, "step": 1}),
    ("Income", "Income category (1 to 8)", "number", 4, {"min": 1, "max": 8, "step": 1}),
    ("HighBP", "High blood pressure", "binary", 0, {}),
    ("HighChol", "High cholesterol", "binary", 0, {}),
    ("CholCheck", "Cholesterol checked in past 5 years", "binary", 1, {}),
    ("Smoker", "Smoked at least 100 cigarettes in lifetime", "binary", 0, {}),
    ("Stroke", "Ever had a stroke", "binary", 0, {}),
    ("HeartDiseaseorAttack", "Coronary heart disease or heart attack", "binary", 0, {}),
    ("PhysActivity", "Leisure-time physical activity in past 30 days", "binary", 1, {}),
    ("Fruits", "Consumes fruit at least once per day", "binary", 1, {}),
    ("Veggies", "Consumes vegetables at least once per day", "binary", 1, {}),
    ("HvyAlcoholConsump", "Heavy alcohol consumption", "binary", 0, {}),
    ("AnyHealthcare", "Has any healthcare coverage", "binary", 1, {}),
    ("NoDocbcCost", "Could not see a doctor because of cost", "binary", 0, {}),
    ("DiffWalk", "Serious difficulty walking", "binary", 0, {}),
    ("Sex", "Sex (0 = female, 1 = male)", "binary", 0, {}),
]

# These are editable controls in the concise manual Stage 3 simulation panel.
SCENARIO_FEATURES = ["BMI", "GenHlth", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "HighBP", "HighChol"]
FIELD_BY_NAME = {field[0]: field for field in FIELDS}
FIELD_KINDS = {name: kind for name, _, kind, _, _ in FIELDS}
FIELD_STEPS = {name: attrs.get("step") for name, _, _, _, attrs in FIELDS}
FIELD_LIMITS = {
    name: (attrs.get("min"), attrs.get("max"))
    for name, _, _, _, attrs in FIELDS
}
TWIN_ASSETS = TwinAssetService()
LATEST_TWIN_ASSET_KEYS: dict[str, str | None] = {"current": None, "scenario": None}


def validate_feature_value(feature: str, value: object) -> float:
    """Validate one BRFSS value using the same rules for uploads and scenarios."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{feature} must be a finite number.")
    if FIELD_KINDS[feature] == "binary" and numeric not in (0.0, 1.0):
        raise ValueError(f"{feature} must be 0 or 1.")
    if str(FIELD_STEPS[feature]) == "1" and not numeric.is_integer():
        raise ValueError(f"{feature} must be a whole-number category or count.")
    minimum, maximum = FIELD_LIMITS[feature]
    if minimum is not None and numeric < float(minimum):
        raise ValueError(f"{feature} must be at least {minimum}.")
    if maximum is not None and numeric > float(maximum):
        raise ValueError(f"{feature} must be at most {maximum}.")
    return numeric


def validate_feature_series(feature: str, values: pd.Series) -> None:
    """Vectorized counterpart to ``validate_feature_value`` for large CSVs."""
    numeric = values.to_numpy(dtype=float, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{feature} must contain only finite numbers.")
    if FIELD_KINDS[feature] == "binary" and not values.isin([0, 1]).all():
        raise ValueError(f"{feature} must contain only 0 or 1.")
    if str(FIELD_STEPS[feature]) == "1" and not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{feature} must contain whole-number categories or counts.")
    minimum, maximum = FIELD_LIMITS[feature]
    if minimum is not None and (values < float(minimum)).any():
        raise ValueError(f"{feature} contains values below {minimum}.")
    if maximum is not None and (values > float(maximum)).any():
        raise ValueError(f"{feature} contains values above {maximum}.")


class PatientDataset:
    """Validated, numbered patient records used by every dashboard stage."""

    def __init__(self, frame: pd.DataFrame, source_name: str) -> None:
        missing = sorted(set(FEATURES) - set(frame.columns))
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        columns = FEATURES + ([TARGET] if TARGET in frame.columns else [])
        clean = frame[columns].copy()
        clean[FEATURES] = clean[FEATURES].apply(pd.to_numeric, errors="coerce")
        clean = clean.dropna(subset=FEATURES).reset_index(drop=True)
        if clean.empty:
            raise ValueError("CSV contains no complete patient rows.")
        if len(clean) > 500_000:
            raise ValueError("CSV contains more than the 500,000-row prototype limit.")

        for feature in FEATURES:
            validate_feature_series(feature, clean[feature])

        if TARGET in clean:
            clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce")
            invalid_target = clean[TARGET].notna() & ~clean[TARGET].isin(RISK_LABELS)
            if invalid_target.any():
                raise ValueError(f"{TARGET} must contain only 0, 1, 2, or blank values.")
        self.frame = clean
        self.source_name = source_name

    @classmethod
    def from_path(cls, path: Path) -> "PatientDataset":
        return cls(pd.read_csv(path), path.name)

    @classmethod
    def from_upload(cls, upload) -> "PatientDataset":
        filename = Path(upload.filename or "uploaded-patients.csv").name
        if not filename.lower().endswith(".csv"):
            raise ValueError("Please import a CSV file.")
        return cls(pd.read_csv(upload.stream), filename)

    def patient(self, patient_number: int) -> dict[str, float]:
        if patient_number < 1 or patient_number > len(self.frame):
            raise ValueError(f"Patient number must be between 1 and {len(self.frame):,}.")
        row = self.frame.iloc[patient_number - 1]
        return {feature: float(row[feature]) for feature in FEATURES}

    def actual_label(self, patient_number: int) -> str | None:
        if TARGET not in self.frame:
            return None
        value = self.frame.iloc[patient_number - 1][TARGET]
        if pd.isna(value) or int(value) not in RISK_LABELS:
            return None
        return RISK_LABELS[int(value)]

    def patient_window(self, patient_number: int, size: int = 9) -> list[dict]:
        half = size // 2
        start = max(0, min(patient_number - 1 - half, len(self.frame) - size))
        end = min(len(self.frame), start + size)
        records = []
        for index in range(start, end):
            row = self.frame.iloc[index]
            target = None
            if TARGET in self.frame and not pd.isna(row[TARGET]) and int(row[TARGET]) in RISK_LABELS:
                target = RISK_LABELS[int(row[TARGET])]
            records.append({
                "number": index + 1,
                "bmi": float(row["BMI"]),
                "age": int(row["Age"]),
                "sex": "Male" if int(row["Sex"]) == 1 else "Female",
                "actual_label": target,
            })
        return records


PATIENT_DATASET: PatientDataset | None = None


def get_patient_dataset() -> PatientDataset:
    global PATIENT_DATASET
    if PATIENT_DATASET is None:
        PATIENT_DATASET = PatientDataset.from_path(DEFAULT_DATASET_PATH)
    return PATIENT_DATASET


def get_twin() -> DiabetesDigitalTwin:
    """Load the large model only when a user requests a prediction."""
    global TWIN
    if TWIN is None:
        TWIN = DiabetesDigitalTwin()
    return TWIN


def twin_metadata(glb_path: Path = TWIN_GLB_PATH) -> dict | None:
    metadata_path = glb_path.with_suffix(".json")
    if not glb_path.exists():
        return None
    if not metadata_path.exists():
        return {"mesh_file": glb_path.name, "version": glb_path.stat().st_mtime_ns}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["version"] = glb_path.stat().st_mtime_ns
        return metadata
    except json.JSONDecodeError:
        return {"mesh_file": glb_path.name, "version": glb_path.stat().st_mtime_ns}


@app.get("/digital-twin.glb")
def digital_twin_asset():
    cached_path = TWIN_ASSETS.asset_path(LATEST_TWIN_ASSET_KEYS["current"] or "")
    if cached_path is not None:
        return send_file(cached_path, mimetype="model/gltf-binary", conditional=True)
    if not TWIN_GLB_PATH.exists():
        abort(404)
    return send_file(TWIN_GLB_PATH, mimetype="model/gltf-binary", conditional=True)


@app.get("/digital-twin-scenario.glb")
def scenario_digital_twin_asset():
    cached_path = TWIN_ASSETS.asset_path(LATEST_TWIN_ASSET_KEYS["scenario"] or "")
    if cached_path is not None:
        return send_file(cached_path, mimetype="model/gltf-binary", conditional=True)
    if not SCENARIO_TWIN_GLB_PATH.exists():
        abort(404)
    return send_file(SCENARIO_TWIN_GLB_PATH, mimetype="model/gltf-binary", conditional=True)


@app.get("/digital-twin/<asset_key>.glb")
def cached_digital_twin_asset(asset_key: str):
    asset_path = TWIN_ASSETS.asset_path(asset_key)
    if asset_path is None:
        abort(404)
    return send_file(asset_path, mimetype="model/gltf-binary", conditional=True)


def refresh_smpl_twin(
    values: dict[str, float],
    prediction: dict,
    glb_path: Path = TWIN_GLB_PATH,
    profile_name: str = "3D twin",
) -> TwinAssetResult:
    """Return a cached or newly generated content-addressed SMPL asset."""
    if not any((PROJECT_DIR / "models" / "smpl").glob("*.pkl")):
        return TwinAssetResult(
            None, "SMPL model weights are not available, so the 3D twin was not updated."
        )

    gender = "male" if int(values["Sex"]) == 1 else "female"
    risk_percent = prediction["high_risk_probability"] * 100
    result = TWIN_ASSETS.get_or_create(
        gender=gender,
        bmi=values["BMI"],
        risk_percent=risk_percent,
        profile_name=profile_name,
    )
    if result.metadata is not None:
        slot = "scenario" if glb_path == SCENARIO_TWIN_GLB_PATH else "current"
        LATEST_TWIN_ASSET_KEYS[slot] = result.metadata["asset_key"]
    return result


def parse_scenario(baseline: dict[str, float]) -> dict[str, float]:
    """Reload the baseline and apply only validated, allowlisted form changes."""
    scenario = baseline.copy()
    for feature in SCENARIO_FEATURES:
        scenario[feature] = validate_feature_value(
            feature, request.form[f"scenario_{feature}"]
        )
    return scenario


def request_asset_metadata(result: TwinAssetResult) -> dict | None:
    if result.metadata is None:
        return None
    metadata = dict(result.metadata)
    metadata["asset_url"] = url_for(
        "cached_digital_twin_asset", asset_key=metadata["asset_key"]
    )
    metadata["version"] = metadata["asset_key"]
    return metadata


def empty_analysis(values: dict[str, float]) -> dict:
    return {
        "values": values,
        "current": None,
        "explanation": None,
        "simulation": None,
        "smpl": None,
        "smpl_status": None,
        "twin_metadata": None,
        "scenario_twin_metadata": None,
        "scenario_smpl": None,
        "scenario_smpl_status": None,
        "knowledge_graph": None,
        "local_guidance": None,
        "local_guidance_error": None,
    }


def run_patient_action(
    patient_number: int, baseline: dict[str, float], action: str
) -> dict:
    """Run one validated patient workflow and return its template context."""
    result = empty_analysis(baseline)
    twin = get_twin()
    scenario = None
    if action == "simulate":
        scenario = parse_scenario(baseline)
        result["simulation"] = twin.simulate(baseline, scenario)
        result["current"] = result["simulation"]["baseline"]
        result["scenario_smpl"] = smpl_twin_descriptor(
            scenario["BMI"], result["simulation"]["scenario"]["high_risk_probability"]
        )
    else:
        result["current"] = twin.predict(baseline)

    all_contributions = twin.explain(baseline, max_factors=None)
    result["explanation"] = all_contributions[:5]
    result["smpl"] = smpl_twin_descriptor(
        baseline["BMI"], result["current"]["high_risk_probability"]
    )
    result["knowledge_graph"] = KNOWLEDGE_GRAPH.explain(
        baseline,
        result["current"],
        all_contributions,
        result["smpl"],
        twin.model_path.name,
    )
    if action == "local_guidance":
        try:
            result["local_guidance"] = generate_local_guidance(
                patient_number,
                result["current"],
                result["knowledge_graph"],
                result["smpl"],
            )
        except Exception as exc:
            result["local_guidance_error"] = guidance_error_message(exc)

    current_asset = refresh_smpl_twin(baseline, result["current"])
    result["smpl_status"] = current_asset.status
    result["twin_metadata"] = request_asset_metadata(current_asset)
    if scenario is not None:
        scenario_asset = refresh_smpl_twin(
            scenario,
            result["simulation"]["scenario"],
            SCENARIO_TWIN_GLB_PATH,
            "Scenario 3D twin",
        )
        result["scenario_smpl_status"] = scenario_asset.status
        result["scenario_twin_metadata"] = request_asset_metadata(scenario_asset)
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    global PATIENT_DATASET
    dataset = get_patient_dataset()
    patient_number = 1
    values = dataset.patient(patient_number)
    analysis = empty_analysis(values)
    error = None
    notice = None
    if request.method == "POST":
        try:
            action = request.form.get("action", "predict")
            if action == "import_dataset":
                upload = request.files.get("dataset_file")
                if upload is None or not upload.filename:
                    raise ValueError("Choose a CSV file to import.")
                PATIENT_DATASET = PatientDataset.from_upload(upload)
                dataset = PATIENT_DATASET
                patient_number = 1
                values = dataset.patient(patient_number)
                analysis = empty_analysis(values)
                notice = (
                    f"Imported {dataset.source_name}: {len(dataset.frame):,} valid patient rows. "
                    "Patient numbering now follows this file."
                )
            else:
                patient_number = int(request.form.get("patient_number", 1))
                values = dataset.patient(patient_number)
                analysis = run_patient_action(patient_number, values, action)
        except (KeyError, ValueError) as exc:
            error = f"Please provide valid patient data. ({exc})"
        except Exception as exc:
            error = f"The model could not complete this request. ({exc})"
    return render_template(
        "index.html", fields=FIELDS, scenario_fields=[FIELD_BY_NAME[name] for name in SCENARIO_FEATURES],
        **analysis, error=error,
        model_name=TWIN.model_path.name if TWIN else "Loads when you predict",
        notice=notice,
        ollama_model=configured_model(),
        dataset_name=dataset.source_name, patient_count=len(dataset.frame), patient_number=patient_number,
        patient_rows=dataset.patient_window(patient_number), actual_label=dataset.actual_label(patient_number),
    )


if __name__ == "__main__":
    # A development reloader starts two processes, which would load the large
    # trained model twice and can exhaust a laptop's memory.
    app.run(host=os.environ.get("FLASK_HOST", "127.0.0.1"), port=5000, debug=False)
