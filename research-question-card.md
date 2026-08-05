## Research Question Card

Question: Can a dataset-linked diabetes Digital Twin consistently use one real patient record across risk prediction, patient-level explanation, 3D representation, and manual what-if simulation?

Type: applied

Hypothesis: Replacing the hard-coded web profile with a validated, numbered dataset record will make every Stage 1-3 output traceable to the same patient and make the prototype easier to reproduce and share.

Why it matters: A Digital Twin cannot be evaluated as patient-specific if its prediction, explanation, simulation baseline, and 3D representation are derived from different or untraceable inputs.

Current evidence:

- ER-20260729-project-code-01: The original dashboard initialized a hard-coded profile in `app.py`.
- ER-20260729-project-paper-01: The project document defines Stage 3 as Patient Data -> prediction -> SHAP -> virtual patient -> manual what-if simulation.
- ER-20260729-brfss-data-01: The local BRFSS CSV contains 253,680 rows and the 21 model input columns.

Missing evidence:

- End-to-end results from multiple held-out patients after the trained model artifact is regenerated.
- A usability check showing that another researcher can import a compatible CSV and reproduce a numbered patient case.
- Evidence from longitudinal ShanghaiT2DM data; the current BRFSS data is cross-sectional.

What would support it:

- The selected patient number maps to the same row in the dataset and notebook.
- Prediction, SHAP factors, simulation baseline, and 3D metadata all use that row.
- Changing the selected patient changes the BMI/body-shape coefficient and predicted-risk colour.
- Invalid or incompatible imports are rejected with a clear message.

What would falsify it:

- Any stage silently falls back to hard-coded values or a stale 3D mesh.
- Patient numbering changes independently of the active dataset.
- The simulation baseline can be altered by browser-submitted hidden values.

Minimal next action: Run the combined notebook to regenerate the trained model and Stage 1-3 evidence, then test at least one Low-, Medium-, and High-class dataset patient in the dashboard.

Decision: run experiment

