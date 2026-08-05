# Website and research-stage review

## Outcome

The dashboard is now designed around a single numbered row from the active dataset. That row is the server-side source for Stage 1 prediction, Stage 2 SHAP explanation, the Stage 3 baseline, and the current-patient SMPL metadata.

## Supervisor feedback addressed

| Feedback | Previous state | Updated state |
|---|---|---|
| Test with training-data patients | Hard-coded initial profile | Default BRFSS file is loaded as numbered patient records; a compatible CSV can be imported |
| Make the 3D model respond to the current patient | A saved mesh could remain visible even when stale | Mesh regeneration uses the selected row's BMI, sex, and predicted probability; stale metadata is hidden |
| Number the current patient list | No dataset patient list | The list and direct selector use one-based patient numbers tied to the active CSV |
| Make stages easier to share | Stage 3 logic was outside the Stage 1-2 notebook | `diabetes_risk_stages_1_2.ipynb` now continues through Stage 3 and states the Stage 4 boundary |

## Alignment with `Knowledge Graph-driven Digital Twin 1.pdf`

- Stage 1: three-class risk prediction and probabilities.
- Stage 2: SHAP explanation.
- Stage 3: manual scenario editing and model re-prediction.
- Stage 4: not implemented. Neo4j reasoning, automatic intervention generation, and treatment recommendations must not be claimed yet.

## Important research gaps

1. The implementation uses the cross-sectional BRFSS 2015 dataset, while the proposal text describes longitudinal ShanghaiT2DM clinical records.
2. BRFSS has no HbA1c or MeanCGM, so examples and claims involving those measurements are unsupported.
3. A changed model estimate after edited inputs is not a causal treatment effect.
4. The BMI-to-SMPL beta mapping is a prototype visualization rule and needs calibration or validation before it can be described as anatomically faithful.

## Validation criteria

- Import rejects a CSV missing required model features.
- Patient #N always resolves to row N of the active cleaned dataset.
- Scenario comparison reloads the baseline from the dataset instead of trusting browser-hidden fields.
- The 3D viewer appears only when its metadata matches the current patient's BMI and predicted risk.
- The combined notebook executes from top to bottom and produces Stage 1-3 artifacts.
