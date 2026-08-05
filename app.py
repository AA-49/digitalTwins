"""Stage 1-4 local dashboard for the diabetes-risk research prototype."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
from flask import Flask, abort, render_template, request, send_file

from diabetes_risk import FEATURES
from knowledge_graph import PatientKnowledgeGraph
from ollama_recommendations import (
    configured_model,
    generate_local_guidance,
    guidance_error_message,
)
from stage3 import DiabetesDigitalTwin, RISK_LABELS, smpl_twin_descriptor

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
FIELD_LIMITS = {
    name: (attrs.get("min"), attrs.get("max"))
    for name, _, _, _, attrs in FIELDS
}


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

        for feature, (minimum, maximum) in FIELD_LIMITS.items():
            if FIELD_KINDS[feature] == "binary" and not clean[feature].isin([0, 1]).all():
                raise ValueError(f"{feature} must contain only 0 or 1.")
            if minimum is not None and (clean[feature] < float(minimum)).any():
                raise ValueError(f"{feature} contains values below {minimum}.")
            if maximum is not None and (clean[feature] > float(maximum)).any():
                raise ValueError(f"{feature} contains values above {maximum}.")

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
    if not TWIN_GLB_PATH.exists():
        abort(404)
    return send_file(TWIN_GLB_PATH, mimetype="model/gltf-binary", conditional=True)


@app.get("/digital-twin-scenario.glb")
def scenario_digital_twin_asset():
    if not SCENARIO_TWIN_GLB_PATH.exists():
        abort(404)
    return send_file(SCENARIO_TWIN_GLB_PATH, mimetype="model/gltf-binary", conditional=True)


def refresh_smpl_twin(
    values: dict[str, float],
    prediction: dict,
    glb_path: Path = TWIN_GLB_PATH,
    profile_name: str = "3D twin",
) -> str:
    """Regenerate the saved GLB inside the app, with Docker as a host fallback."""
    if not any((PROJECT_DIR / "models" / "smpl").glob("*.pkl")):
        return "SMPL model weights are not available, so the 3D twin was not updated."

    gender = "male" if int(values["Sex"]) == 1 else "female"
    risk_percent = prediction["high_risk_probability"] * 100
    metadata_path = glb_path.with_suffix(".json")
    if glb_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                float(metadata.get("bmi", -1)) == float(values["BMI"])
                and abs(float(metadata.get("risk_percent", -1)) - risk_percent) < 0.0001
                and metadata.get("gender") == gender
            ):
                return f"{profile_name} already matches its profile."
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    export_arguments = [
        "-m", "src.export_smpl",
        "--bmi", str(values["BMI"]),
        "--risk", str(risk_percent),
        "--gender", gender,
        "--out", glb_path.as_posix(),
    ]

    # In the Docker dashboard, all SMPL dependencies are already installed, so
    # run the exporter in this container instead of trying to invoke Docker from
    # inside Docker. On a local host with fewer Python dependencies, Docker
    # remains a useful fallback.
    direct_error = None
    try:
        completed = subprocess.run(
            [sys.executable, *export_arguments],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if completed.returncode == 0:
            return (
                f"{profile_name} updated from this profile: BMI {values['BMI']}, {gender}, "
                f"high-risk probability {prediction['high_risk_probability']:.1%}."
            )
        details = (completed.stderr or completed.stdout).strip().splitlines()
        direct_error = details[-1] if details else "direct SMPL export failed"
    except subprocess.TimeoutExpired:
        direct_error = "direct SMPL export timed out"
    except OSError as exc:
        direct_error = str(exc)

    if shutil.which("docker") is None:
        return f"The 3D twin could not be updated in this environment: {direct_error}."

    docker_command = [
        "docker", "compose", "--profile", "smpl", "run", "--rm", "smpl-export",
        "python", *export_arguments,
    ]
    try:
        completed = subprocess.run(
            docker_command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"The 3D twin could not be updated: {exc}"
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip().splitlines()
        return "The 3D twin could not be updated: " + (
            details[-1] if details else "Docker export failed."
        )
    return (
        f"{profile_name} updated from this profile: BMI {values['BMI']}, {gender}, "
        f"high-risk probability {prediction['high_risk_probability']:.1%}."
    )


@app.route("/", methods=["GET", "POST"])
def index():
    global PATIENT_DATASET
    dataset = get_patient_dataset()
    patient_number = 1
    values = dataset.patient(patient_number)
    current = None
    explanation = None
    simulation = None
    smpl = None
    smpl_status = None
    error = None
    notice = None
    current_twin_metadata = None
    scenario_twin_metadata = None
    scenario_smpl = None
    scenario_smpl_status = None
    knowledge_graph = None
    local_guidance = None
    local_guidance_error = None
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
                notice = (
                    f"Imported {dataset.source_name}: {len(dataset.frame):,} valid patient rows. "
                    "Patient numbering now follows this file."
                )
            else:
                patient_number = int(request.form.get("patient_number", 1))
                baseline = dataset.patient(patient_number)
                values = baseline
                twin = get_twin()
                if action == "simulate":
                    scenario = baseline.copy()
                    for feature in SCENARIO_FEATURES:
                        scenario[feature] = float(request.form[f"scenario_{feature}"])
                    simulation = twin.simulate(baseline, scenario)
                    current = simulation["baseline"]
                    scenario_smpl = smpl_twin_descriptor(
                        scenario["BMI"], simulation["scenario"]["high_risk_probability"]
                    )
                else:
                    current = twin.predict(baseline)
                all_contributions = twin.explain(baseline, max_factors=None)
                explanation = all_contributions[:5]
                smpl = smpl_twin_descriptor(baseline["BMI"], current["high_risk_probability"])
                knowledge_graph = KNOWLEDGE_GRAPH.explain(
                    baseline,
                    current,
                    all_contributions,
                    smpl,
                    twin.model_path.name,
                )
                if action == "local_guidance":
                    try:
                        local_guidance = generate_local_guidance(
                            patient_number, current, knowledge_graph, smpl
                        )
                    except Exception as exc:
                        local_guidance_error = guidance_error_message(exc)
                smpl_status = refresh_smpl_twin(baseline, current)
                metadata = twin_metadata()
                if metadata and (
                    float(metadata.get("bmi", -1)) == float(baseline["BMI"])
                    and abs(
                        float(metadata.get("risk_percent", -1))
                        - current["high_risk_probability"] * 100
                    ) < 0.0001
                ):
                    current_twin_metadata = metadata
                if action == "simulate":
                    scenario_smpl_status = refresh_smpl_twin(
                        scenario,
                        simulation["scenario"],
                        SCENARIO_TWIN_GLB_PATH,
                        "Scenario 3D twin",
                    )
                    scenario_metadata = twin_metadata(SCENARIO_TWIN_GLB_PATH)
                    if scenario_metadata and (
                        float(scenario_metadata.get("bmi", -1)) == float(scenario["BMI"])
                        and abs(
                            float(scenario_metadata.get("risk_percent", -1))
                            - simulation["scenario"]["high_risk_probability"] * 100
                        ) < 0.0001
                    ):
                        scenario_twin_metadata = scenario_metadata
        except (KeyError, ValueError) as exc:
            error = f"Please provide valid patient data. ({exc})"
        except Exception as exc:
            error = f"The model could not complete this request. ({exc})"
    return render_template(
        "index.html", fields=FIELDS, scenario_fields=[FIELD_BY_NAME[name] for name in SCENARIO_FEATURES],
        values=values, current=current, explanation=explanation, simulation=simulation, smpl=smpl, error=error,
        model_name=TWIN.model_path.name if TWIN else "Loads when you predict",
        twin_metadata=current_twin_metadata, smpl_status=smpl_status, notice=notice,
        scenario_twin_metadata=scenario_twin_metadata, scenario_smpl=scenario_smpl,
        scenario_smpl_status=scenario_smpl_status,
        knowledge_graph=knowledge_graph,
        local_guidance=local_guidance, local_guidance_error=local_guidance_error,
        ollama_model=configured_model(),
        dataset_name=dataset.source_name, patient_count=len(dataset.frame), patient_number=patient_number,
        patient_rows=dataset.patient_window(patient_number), actual_label=dataset.actual_label(patient_number),
    )


if __name__ == "__main__":
    # A development reloader starts two processes, which would load the large
    # trained model twice and can exhaust a laptop's memory.
    app.run(host=os.environ.get("FLASK_HOST", "127.0.0.1"), port=5000, debug=False)
