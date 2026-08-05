# Supervisor hand-off: Knowledge Graph-driven Diabetes Digital Twin

## What to share now

Share a ZIP copy of this project folder **without** `.venv/`, the raw BRFSS CSV, licensed SMPL files, or patient-identifiable data. Include:

- `README.md` and this hand-off note
- `WEBSITE_REVIEW.md`, `research-question-card.md`, and the executed `diabetes_risk_stages_1_2.ipynb`
- `app.py`, `stage3.py`, `diabetes_risk.py`, `src/`, `templates/`, `static/`, and `requirements.txt`
- the small JSON, CSV, and PNG evidence in `artifacts_notebook/`, including `stage3_case_study.json`
- a short screen recording of the Stage 3 comparison page

The trained model is large. Send it separately by OneDrive/Google Drive only if your supervisor needs to run the dashboard; do not email it as an attachment.

## One-minute demo sequence

1. Start the dashboard with `python app.py` and open `http://127.0.0.1:5000`.
2. Select a numbered BRFSS dataset patient (or import a compatible CSV) and run Stages 1-3.
3. Point out the three probabilities and the top SHAP factors.
4. Change one or more editable values in **What-if simulation** (for example BMI or physical activity) and select **Compare scenario**.
5. Show the baseline and scenario high-diabetes probabilities and the exact variables changed.
6. State that Stage 3 accepts manual scenarios; automated intervention recommendation is intentionally reserved for Stage 4, when the knowledge graph is added.

## Honest project status against the proposal

| Proposal stage | Current evidence | Status |
| --- | --- | --- |
| Stage 1 - risk prediction | Random Forest artifacts and evaluation metrics | Implemented |
| Stage 2 - explanation | SHAP outputs and dashboard factors | Implemented |
| Stage 3 - manual what-if Digital Twin | Executed notebook case study and dashboard comparison | Implemented |
| Stage 4 - knowledge graph reasoning | Neo4j, recommendations, and automatic interventions | Not started |

## Limitations to state clearly

- This is a research prototype, not a diagnostic or treatment system.
- The current data is BRFSS survey data, not the proposed longitudinal ShanghaiT2DM clinical dataset. It has no HbA1c or MeanCGM fields.
- The baseline metrics show very weak detection of the minority prediabetes class (F1 around 0.002). Accuracy alone must not be used as evidence of a clinically reliable model.
- Simulation is a model re-prediction after user edits; it does **not** prove that the edited intervention causes that health outcome.

## Suggested supervisor email

Subject: Stage 1-3 prototype - Explainable Diabetes Risk Digital Twin

Dear [Supervisor Name],

I am sharing my current prototype covering Stages 1-3 of the proposed workflow: diabetes risk prediction, SHAP-based explanation, and manual what-if Digital Twin simulation. The dashboard now selects a numbered patient directly from the active dataset, and the same record drives the prediction, explanation, 3D metadata, and scenario baseline. The attached hand-off note explains the run steps, evidence, and limitations. Stage 4 (Neo4j knowledge-graph reasoning and automated recommendations) remains future work.

I would appreciate feedback on whether the Stage 3 simulation scope and evaluation plan are appropriate before I begin the knowledge-graph stage.

Kind regards,
[Your name]
