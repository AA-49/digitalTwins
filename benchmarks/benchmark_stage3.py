"""Reproducible Stage 1-3 latency and prediction-equivalence benchmark."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from diabetes_risk import FEATURES  # noqa: E402
from stage3 import DiabetesDigitalTwin  # noqa: E402


RECORDED_BASELINE_WARM_SECONDS = 15.13


def timed(callable_):
    started = perf_counter()
    result = callable_()
    return result, perf_counter() - started


def main() -> None:
    dataset = pd.read_csv(
        PROJECT_DIR / "diabetes_012_health_indicators_BRFSS2015.csv",
        usecols=FEATURES,
        nrows=1000,
    ).astype(float)
    twin, model_load_seconds = timed(DiabetesDigitalTwin)

    probabilities = twin.model.predict_proba(dataset)
    expected = twin.model.predict(dataset).astype(int)
    class_ids = np.asarray(twin.model.classes_, dtype=int)
    optimized = class_ids[np.argmax(probabilities, axis=1)]
    if not np.array_equal(expected, optimized):
        raise AssertionError("Probability argmax changed at least one of 1,000 model classes.")

    warmup = dataset.iloc[0].to_dict()
    unique = dataset.iloc[1].to_dict()
    twin.predict(warmup)
    twin.explain(warmup, max_factors=None)

    def analyze(values):
        return twin.predict(values), twin.explain(values, max_factors=None)

    _unique_result, warm_unique_seconds = timed(lambda: analyze(unique))
    _repeat_result, repeated_seconds = timed(lambda: analyze(unique))
    scenario = unique.copy()
    scenario["BMI"] = min(100.0, scenario["BMI"] + 1.0)
    _simulation, scenario_seconds = timed(lambda: twin.simulate(unique, scenario))

    warm_improvement = 1 - (warm_unique_seconds / RECORDED_BASELINE_WARM_SECONDS)
    repeat_improvement = 1 - (repeated_seconds / RECORDED_BASELINE_WARM_SECONDS)
    report = {
        "model_load_seconds": model_load_seconds,
        "prediction_equivalence_rows": len(dataset),
        "warm_unique_analysis_seconds": warm_unique_seconds,
        "repeated_analysis_seconds": repeated_seconds,
        "scenario_seconds": scenario_seconds,
        "warm_unique_improvement_percent": warm_improvement * 100,
        "repeated_improvement_percent": repeat_improvement * 100,
        "acceptance": {
            "warm_unique_at_least_30_percent": warm_improvement >= 0.30,
            "repeated_at_least_80_percent": repeat_improvement >= 0.80,
        },
        "cache_info": twin.cache_info(),
    }
    print(json.dumps(report, indent=2))
    if not all(report["acceptance"].values()):
        raise SystemExit("Performance acceptance target was not met.")


if __name__ == "__main__":
    main()
