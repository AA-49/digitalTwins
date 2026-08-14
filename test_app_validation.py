"""Server-side validation tests shared by CSV imports and scenarios."""
from __future__ import annotations

import unittest

import pandas as pd

import app as dashboard


class AppValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = pd.read_csv(dashboard.DEFAULT_DATASET_PATH, nrows=1)

    def test_fractional_category_is_rejected_during_import(self):
        frame = self.row.copy()
        frame.loc[0, "Age"] = 1.5
        with self.assertRaisesRegex(ValueError, "whole-number"):
            dashboard.PatientDataset(frame, "fractional.csv")

    def test_non_finite_and_out_of_range_values_are_rejected(self):
        for feature, value in [("BMI", float("nan")), ("BMI", -999), ("HighBP", 2)]:
            with self.subTest(feature=feature, value=value):
                with self.assertRaises(ValueError):
                    dashboard.validate_feature_value(feature, value)

    def test_scenario_parser_reloads_and_validates_baseline(self):
        baseline = dashboard.PatientDataset(self.row, "one.csv").patient(1)
        form = {
            f"scenario_{feature}": str(baseline[feature])
            for feature in dashboard.SCENARIO_FEATURES
        }
        form["scenario_HighBP"] = "2"
        with dashboard.app.test_request_context("/", method="POST", data=form):
            with self.assertRaisesRegex(ValueError, "HighBP"):
                dashboard.parse_scenario(baseline)


if __name__ == "__main__":
    unittest.main()
