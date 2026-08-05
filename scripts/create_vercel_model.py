"""Create the smaller, reproducible Random Forest artifact used on Vercel.

The local Docker workflow keeps the checked 400-tree model. Vercel uses a
subset of those already-trained trees so the deployed artifact stays below
GitHub's single-file limit and fits comfortably in serverless memory.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from diabetes_risk import FEATURES, TARGET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=75)
    parser.add_argument(
        "--source-model",
        type=Path,
        default=Path("artifacts_notebook/diabetes_risk_random_forest.joblib"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("diabetes_012_health_indicators_BRFSS2015.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_vercel"))
    args = parser.parse_args()

    full_model = joblib.load(args.source_model)
    if args.trees < 1 or args.trees > len(full_model.estimators_):
        raise ValueError(f"--trees must be between 1 and {len(full_model.estimators_)}")

    deployment_model = copy.copy(full_model)
    deployment_model.estimators_ = list(full_model.estimators_[: args.trees])
    deployment_model.n_estimators = args.trees

    frame = pd.read_csv(args.dataset, usecols=FEATURES + [TARGET]).dropna()
    x = frame[FEATURES]
    y = frame[TARGET].astype(int)
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    prediction = deployment_model.predict(x_test)
    probabilities = deployment_model.predict_proba(x_test)
    report = classification_report(
        y_test,
        prediction,
        labels=[0, 1, 2],
        target_names=["Low", "Medium (prediabetes)", "High (diabetes)"],
        output_dict=True,
        zero_division=0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "diabetes_risk_random_forest.joblib"
    metrics_path = args.output_dir / "metrics.json"
    joblib.dump(deployment_model, model_path, compress=3)

    metrics = {
        "deployment_variant": "Vercel 75-tree subset of the checked 400-tree model",
        "source_model": args.source_model.name,
        "n_estimators": args.trees,
        "n_rows": int(len(frame)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "macro_ovr_roc_auc": float(
            roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")
        ),
        "classification_report": report,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": str(model_path),
                "size_mb": round(model_path.stat().st_size / 1024 / 1024, 1),
                "metrics": str(metrics_path),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    )


if __name__ == "__main__":
    main()
