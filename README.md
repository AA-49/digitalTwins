# Explainable Diabetes Risk Digital Twin

An interactive research prototype that connects a numbered CDC BRFSS 2015 patient record to diabetes-risk prediction, patient-specific SHAP explanation, a manual what-if Digital Twin, a temporary patient-centric knowledge graph, and research-safe explanatory guidance.

## Local dashboard

Run the complete research environment locally with Docker, Neo4j, Jupyter, Ollama, and optional licensed SMPL assets. The dashboard is available only on this computer at <http://127.0.0.1:5000>.

> **Research only:** This project is not a medical diagnosis, treatment recommendation, causal forecast, or clinical decision-support system. BRFSS is cross-sectional survey data. A changed prediction after editing a scenario does not prove that the change caused or prevented diabetes.

## System stages

| Stage | Output |
| --- | --- |
| 1 - Prediction | Low, Medium (prediabetes), or High (diabetes), with all three probabilities |
| 2 - Explanation | Patient-specific SHAP contributions showing model support or opposition |
| 3 - Digital Twin | Current-versus-scenario prediction and optional patient-specific SMPL meshes |
| 4 - Knowledge graph | All 21 observations, decoded states, 21 SHAP contributions, three probabilities, model evaluation, and current twin |
| Guidance | Local Ollama explanation with a deterministic safety fallback |

The selected one-based patient number always maps to the same row in the active cleaned dataset. The server reloads that row for prediction and scenario baselines instead of trusting browser-submitted baseline values.

## Model limitations

The default dataset contains 253,680 records and is highly imbalanced. Accuracy must never be reported alone.

| Model | Trees | Accuracy | Balanced accuracy | Macro-F1 | Medium recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local Docker model | 400 | 81.96% | 45.09% | 44.34% | 0.00% |

The local result does not demonstrate reliable Medium/prediabetes screening performance.

## Run locally with Docker

### Requirements

- Docker Desktop using Linux containers
- A web browser
- Ollama for optional local guidance
- Official licensed SMPL model files for live 3D mesh generation

### Start the application

1. Start Docker Desktop and wait for the engine to become ready.
2. Start Ollama and install the configured local model:

   ```powershell
   ollama pull qwen2.5-coder:1.5b
   ollama list
   ```

3. From this project directory, build and start the dashboard with Neo4j:

   ```powershell
   docker compose up -d --build dashboard
   docker compose ps
   ```

4. Open:

   - Dashboard: <http://127.0.0.1:5000>
   - Neo4j Browser: <http://127.0.0.1:7474>

5. Check logs when needed:

   ```powershell
   docker compose logs --tail 100 dashboard neo4j
   ```

The dashboard container connects to Neo4j at `bolt://neo4j:7687` and reaches Windows Ollama at `http://host.docker.internal:11434`. Local defaults can be overridden in `.env`.

Do not start the normal workflow with `python app.py`. A host process uses different networking, may fail to reach the Compose Neo4j service, and can conflict with the dashboard container on port 5000.

### Stop or restart

```powershell
docker compose restart dashboard neo4j
docker compose down
```

`docker compose down` preserves the Neo4j volume. Do not add `-v` unless deleting that volume is intentional.

## Use the dashboard

1. Enter a patient number and choose **Load patient**, or choose **Use** in the patient table.
2. Review the Stage 1 category and all three probabilities.
3. Review the Stage 2 SHAP factors. They explain this model output; they do not establish medical causes.
4. Explore the Stage 4 graph, filters, keyboard selector, connected evidence, and model limitations.
5. Choose **Generate local research guidance** to use Ollama on this computer.
6. Inspect the Stage 3 current twin when a matching SMPL mesh is available.
7. Edit the permitted scenario fields and choose **Compare scenario**.
8. Review the baseline and scenario probabilities, exact edited values, and separate twins.

A compatible imported CSV must contain all 21 model input columns. `Diabetes_012` is optional. Imported data remains in memory only and resets when the application restarts.

Patient observations, predictions, SHAP values, and Digital Twin graph nodes are temporary and are not persisted in Neo4j. The reusable definition schema may be stored there, but the selected patient's data is assembled for the current request only.

## Train and reproduce artifacts

Start JupyterLab in Docker:

```powershell
docker compose up -d jupyter
```

Open <http://127.0.0.1:8888>. The notebook `diabetes_risk_stages_1_2.ipynb` is the combined training and experiment path.

Execute it non-interactively:

```powershell
docker compose --profile training run --rm train-notebook
```

This reads `diabetes_012_health_indicators_BRFSS2015.csv` and updates the local evidence under `artifacts_notebook/`.

## Optional SMPL 3D setup

SMPL weights are licensed and are not included in this repository. Obtain them from the official SMPL provider and place the appropriate files in `models/smpl/`, for example:

```text
models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl
models/smpl/basicModel_m_lbs_10_207_0_v1.0.0.pkl
```

Never commit or redistribute these files.

Export the default mesh:

```powershell
docker compose --profile smpl run --rm smpl-export
```

Export a chosen profile:

```powershell
docker compose --profile smpl run --rm smpl-export python -m src.export_smpl --bmi 27 --risk 35 --gender female --out artifacts_notebook/digital_twin.glb
```

The prototype maps BMI to the first SMPL shape coefficient and maps predicted High probability to mesh colour. This visualization rule is not an anatomically validated clinical model. When weights or matching metadata are unavailable, the dashboard hides stale meshes instead of displaying an incorrect patient twin.

## Tests

Run the automated checks with the project virtual environment:

```powershell
.\.venv-stage3\Scripts\python.exe -m unittest discover -v
node --check static/js/knowledge_graph.js
docker compose config --quiet
```

The Python suite checks the complete patient graph, categorical decoding, temporary fallback behavior, Ollama evidence boundary, deterministic summary, and research-safety fallback.

## Troubleshooting

### Docker daemon or named-pipe error

Start Docker Desktop, wait until the engine is ready, verify with `docker info`, and rerun:

```powershell
docker compose up -d --build dashboard
```

### Port 5000 is already in use

Stop any host Flask process with `Ctrl+C`, then restart the Compose dashboard.

### Knowledge graph shows fallback mode locally

```powershell
docker compose ps
docker compose logs --tail 100 neo4j dashboard
```

Even in fallback mode, the page keeps all 21 patient observations and their complete Stage 4 evidence available.

### Local guidance is unavailable

```powershell
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/tags
docker compose exec -T dashboard python -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://host.docker.internal:11434/api/tags')))"
```

The first response can take one or two minutes while Ollama loads the model. Unsafe or incomplete drafts are replaced with a deterministic evidence summary.

### 3D twin is unavailable

Confirm that the licensed SMPL files exist under `models/smpl/`, then restart the dashboard:

```powershell
docker compose restart dashboard
```

## Research and sharing boundaries

- The current dataset is BRFSS survey data, not longitudinal ShanghaiT2DM clinical records.
- BRFSS does not provide HbA1c or MeanCGM, so the application must not invent those measurements.
- SHAP is model explanation, not causal evidence.
- Manual scenarios are model re-predictions, not intervention effects.
- Report balanced accuracy, macro-F1, confusion matrix, and per-class recall alongside accuracy.
- Do not publish secrets, raw patient-identifiable data, virtual environments, licensed SMPL files, the full local model, or the recoverable `delete/` archive.

## Key files

| Path | Purpose |
| --- | --- |
| `app.py` | Flask routes and dashboard orchestration |
| `stage3.py` | Dataset, prediction, SHAP, scenario, and twin helpers |
| `knowledge_graph.py` | Reusable Neo4j schema and temporary patient graph assembly |
| `ollama_recommendations.py` | Local evidence prompt, safety checks, and deterministic fallback |
| `diabetes_risk_stages_1_2.ipynb` | Reproducible training and experiment notebook |
| `docker-compose.yml` | Dashboard, Neo4j, Jupyter, training, and SMPL services |
| `artifacts_notebook/` | Local model and research evidence |

## License and data

The repository does not grant permission to redistribute the BRFSS-derived data, trained artifacts, or SMPL assets. Confirm the terms of each upstream dataset, model, and dependency before public redistribution.
