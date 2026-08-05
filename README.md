# Explainable Diabetes Risk Digital Twin — Docker Workflow

This research prototype implements diabetes-risk prediction, patient-specific SHAP explanation, manual Digital Twin simulation, 3D SMPL visualisation, and a patient-centric Neo4j knowledge graph using the CDC BRFSS 2015 diabetes health indicators dataset.

| Stage | Output |
| --- | --- |
| 1 — Prediction | Low, Medium, or High model category with three class probabilities |
| 2 — Explanation | Global feature importance and patient-specific SHAP evidence |
| 3 — Digital Twin | Current-versus-scenario model comparison and separate 3D twins |
| 4 — Knowledge graph and local guidance | All 21 observations, decoded states, SHAP contributions, probabilities, model evaluation, current twin, and optional Ollama explanation |

> **Research only:** BRFSS is a cross-sectional survey dataset. These outputs are not diagnoses, treatment recommendations, causal forecasts, or clinical decision support. The checked model has 0.0 recall for the Medium/prediabetes class, so accuracy must not be interpreted as reliable screening performance.

## Required software

- Docker Desktop for Windows with the Linux container engine running.
- Ollama for Windows running locally with `qwen2.5-coder:1.5b` installed.
- A web browser.
- Optional licensed SMPL `.pkl` files under `models/smpl/` for 3D mesh generation.

Python and project packages are installed inside the Docker image. A host virtual environment is not required for the normal workflow.

## Start the complete system

1. Open Docker Desktop and wait until it reports that the engine is running.
2. Open PowerShell in this project directory.
3. Confirm that Docker is ready:

   ```powershell
   docker info
   ```

4. Confirm that Ollama is running and the configured model is installed:

   ```powershell
   ollama list
   ollama pull qwen2.5-coder:1.5b
   ```

5. Build and start the dashboard and Neo4j:

   ```powershell
   docker compose up -d --build dashboard
   ```

   The dashboard service depends on a healthy Neo4j service, so Compose starts both in the correct order.

6. Check both containers:

   ```powershell
   docker compose ps
   docker compose logs --tail 100 dashboard neo4j
   ```

7. Open:

   - Dashboard: <http://127.0.0.1:5000>
   - Neo4j Browser: <http://127.0.0.1:7474>

The default local Neo4j login is `neo4j` / `training3graph`. Values in `.env` override the default local credentials.

The dashboard container reaches the Windows Ollama service through `http://host.docker.internal:11434`. The defaults can be changed in `.env` with `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_TIMEOUT_SECONDS`.

## Important: do not run Flask on the host

Do not run:

```powershell
python app.py
```

A host-run Flask process uses `127.0.0.1:7687` for Neo4j and may attempt to call the Docker API for 3D export. If Docker Desktop or Neo4j is unavailable, this produces the “Knowledge graph unavailable” and Docker named-pipe errors. The Compose dashboard instead uses the internal address `bolt://neo4j:7687` and runs the 3D exporter directly inside its container.

If a previous host Flask process is still running, return to that terminal and press `Ctrl+C` before starting Compose. Only the Compose dashboard should own port 5000.

## Use the dashboard

1. Select a numbered patient from the active BRFSS dataset or import a compatible CSV.
2. Review the Stage 1 category and all three probabilities.
3. Review Stage 2 SHAP evidence; SHAP describes model support, not medical causation.
4. Explore Stage 4 and confirm the graph reports that reusable definitions were loaded from Neo4j.
5. Choose **Generate local research guidance** to ask Ollama for a simple-language explanation of the complete temporary Stage 4 evidence.
6. Inspect the current patient’s 3D twin.
7. Edit the permitted Stage 3 scenario fields and compare the current and scenario predictions and twins.

The selected dataset row remains the server-side source of truth. Patient observations, predictions, SHAP values, and Digital Twin nodes are temporary and are never persisted in Neo4j. Local guidance sends this temporary evidence only to Ollama on this computer; it does not call an external AI API. Every Ollama draft must pass checks for the research boundary, model limitation, paragraph-only format, and prohibited causal or treatment-like claims. A rejected draft is replaced with a deterministic summary of the verified probabilities and SHAP evidence.

## Run Jupyter entirely in Docker

Start JupyterLab:

```powershell
docker compose up -d jupyter
```

Open <http://127.0.0.1:8888>. The notebook `diabetes_risk_stages_1_2.ipynb` is the single training path.

To execute every notebook cell non-interactively and save the executed notebook:

```powershell
docker compose --profile training run --rm train-notebook
```

Training reads `diabetes_012_health_indicators_BRFSS2015.csv` and updates `artifacts_notebook/`. The full 400-tree workflow and permutation importance can take several minutes.

## Container management

```powershell
# Show service health
docker compose ps

# Follow dashboard and Neo4j logs
docker compose logs -f dashboard neo4j

# Restart the application stack
docker compose restart dashboard neo4j

# Stop containers while preserving Neo4j data
docker compose down
```

Do not use `docker compose down -v` unless you intentionally want to delete the Neo4j volume.

## Troubleshooting

### Docker named-pipe or daemon error

Example: `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`.

Docker Desktop is not ready. Start Docker Desktop, wait for the engine, run `docker info`, and then rerun:

```powershell
docker compose up -d --build dashboard
```

### Knowledge graph tries 127.0.0.1:7687

The dashboard was started with `python app.py` instead of Compose, or Neo4j is stopped. Stop the host Flask process with `Ctrl+C` and start the Compose dashboard. Inside Compose, the configured address is `bolt://neo4j:7687`.

### Port 5000 is already in use

Stop the terminal running `python app.py`, then run:

```powershell
docker compose up -d dashboard
```

### Knowledge graph fallback remains visible

Check Neo4j health and dashboard logs:

```powershell
docker compose ps
docker compose logs --tail 100 neo4j dashboard
```

The page intentionally keeps all 21 accessible patient attributes visible when Neo4j is unavailable.

### 3D twin is unavailable

Confirm that licensed SMPL weights exist under `models/smpl/`, then restart the dashboard:

```powershell
docker compose restart dashboard
```

The dashboard container runs `src.export_smpl` directly. It does not need access to the host Docker socket.

### Local Ollama guidance is unavailable

Confirm Ollama is running on Windows and that the configured model exists:

```powershell
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/tags
docker compose exec -T dashboard python -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://host.docker.internal:11434/api/tags')))"
```

The first generation can be slow while Ollama loads the model. The dashboard waits up to 180 seconds by default. After changing the model or connection settings in `.env`, recreate the dashboard container with `docker compose up -d --force-recreate dashboard`.

## Research and sharing boundaries

- Report balanced accuracy, macro-F1, confusion matrix, and per-class recall alongside accuracy.
- The dataset does not contain HbA1c or MeanCGM, so the system must not invent those measurements.
- The graph contains no direct causal edges from observations to diabetes risk.
- Ollama output is generated text from a small code-oriented model. It can be incomplete or inaccurate and remains research-only, non-causal, and not medical advice.
- Do not share raw survey data, patient-identifiable data, `.venv` folders, model binaries, licensed SMPL files, secrets, or the recoverable `delete/` folder.
- Use `SUPERVISOR_HANDOFF.md` when preparing a supervisor demonstration or project archive.
