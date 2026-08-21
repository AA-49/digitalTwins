# Explainable Diabetes Risk Digital Twin

A local, Docker-based research workflow for training and exploring a three-class diabetes-risk model with CDC BRFSS 2015 survey data. The project combines a reproducible Jupyter notebook, patient-specific SHAP explanations, manual what-if scenarios, a Neo4j knowledge graph, a Flask dashboard, optional SMPL 3D twins, and optional local Ollama guidance.

> **Research only:** This project is not a medical device, diagnostic tool, treatment recommendation, causal model, or clinical decision-support system. BRFSS is cross-sectional survey data. Predictions, SHAP values, and scenario differences describe model behavior; they do not demonstrate medical causes or intervention effects.

## What the pipeline does

The canonical workflow is [`digital_twin_full_pipeline.ipynb`](digital_twin_full_pipeline.ipynb):

1. **Stage 1 — Train and evaluate:** clean the 21-feature BRFSS dataset, train the three-class model, and report class-aware metrics.
2. **Stage 2 — Explain:** calculate exact patient-specific SHAP evidence for the model's predicted class.
3. **Stage 3 — Simulate:** compare a baseline profile with manually edited what-if scenarios. These comparisons are non-causal.
4. **Stage 4 — Connect:** assemble the temporary patient knowledge graph using class-2 High/diabetes SHAP evidence.
5. **Dashboard — Explore:** write fresh files to `artifacts_notebook/`, then start Neo4j and the Flask dashboard from a host terminal and verify <http://127.0.0.1:5000>.

Stage 2 and Stage 4 intentionally use different SHAP targets: Stage 2 explains the predicted class, while Stage 4 consistently describes class 2, **High (diabetes)**. The optional 3D twin is a visual aid and is not part of the analytical evidence.

## Requirements

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://docs.docker.com/desktop/) using Linux containers
- A modern web browser
- [Ollama](https://ollama.com/download) only for optional local research guidance
- Licensed SMPL weights only for optional patient-specific 3D mesh generation

The examples below use Windows Command Prompt. Run them from the repository directory.

## Quick start

### 1. Clone the repository

```bat
cd /d "%USERPROFILE%\Documents"
git clone https://github.com/AA-49/digitalTwins.git
cd digitalTwins
```

### 2. Verify Docker Desktop

Start Docker Desktop and wait for the engine to report that it is running, then check:

```bat
docker info
docker compose version
```

Do not continue until `docker info` displays a **Server** section. A missing `dockerDesktopLinuxEngine` named pipe means Docker Desktop is not ready or is not using Linux containers.

### 3. Open the training notebook

For the first run, build the image and start JupyterLab:

```bat
docker compose up -d --build jupyter
```

For later runs, when the Jupyter image already exists, start it without rebuilding:

```bat
docker compose up -d jupyter
```

Open <http://127.0.0.1:8888> and select `digital_twin_full_pipeline.ipynb`. Use **Restart Kernel and Run All Cells**.

The notebook trains the model when no local model exists and creates the model and evidence files under `artifacts_notebook/`. Wait for all cells to finish. The trained `joblib` file is large and intentionally excluded from Git, so each fresh clone creates its own local copy from the tracked BRFSS dataset.

The final notebook cell verifies that the model and Stage 3/4 artifacts were created. JupyterLab itself runs inside a container and does not contain the Docker CLI, so the notebook does not try to start other containers.

After the notebook finishes, return to **Windows Command Prompt on the host**, in the cloned repository, and run:

```bat
docker compose up -d --build dashboard
docker compose ps
curl.exe --fail --retry 30 --retry-delay 2 --retry-connrefused --head http://127.0.0.1:5000/
```

Open <http://127.0.0.1:5000> after the request returns HTTP 200. Compose starts Neo4j automatically and waits for its health check before starting the dashboard.

### 4. Stop the services when finished

Run this manually from a terminal:

```bat
docker compose down
```

This preserves the Neo4j named volume. Add `-v` only when permanent deletion of that local database is intentional.

## Run without JupyterLab

To execute the complete notebook non-interactively:

```bat
docker compose --profile training run --rm train-notebook
```

After the notebook command completes, start the dashboard from the same host terminal:

```bat
docker compose up -d --build dashboard
docker compose ps
curl.exe --fail --retry 30 --retry-delay 2 --retry-connrefused --head http://127.0.0.1:5000/
```

Expected Compose state:

- `neo4j` is `Up` and `healthy`;
- `dashboard` is `Up` with `127.0.0.1:5000->5000/tcp`;
- the dashboard URL returns HTTP 200.

Neo4j Browser is available at <http://127.0.0.1:7474>. All published ports bind to `127.0.0.1`, so they are accessible only from the current computer by default.

## Using the dashboard

1. Select a one-based patient number from the cleaned dataset.
2. Review Stage 1 probabilities for Low, Medium (prediabetes), and High (diabetes).
3. Review Stage 2 SHAP evidence for the predicted class.
4. Edit permitted Stage 3 fields and compare the baseline with a what-if scenario.
5. Explore Stage 4 observations, class-2 High/diabetes evidence, evaluation metadata, and the current Digital Twin.
6. Optionally generate local research guidance through Ollama.

Imported patient CSV data remains in memory and resets when the application restarts. Patient observations, predictions, SHAP values, and patient graph nodes are temporary and are not persisted in Neo4j.

## Optional local Ollama guidance

The model, SHAP, scenarios, graph, and dashboard work without Ollama. To enable local guidance, start Ollama on Windows and install the configured model:

```bat
ollama pull qwen2.5-coder:1.5b
ollama list
```

The dashboard container reaches the host service at `http://host.docker.internal:11434`. The server validates model-selected features and discussion topics before rendering deterministic research-safe text; raw local-model wording is not displayed.

## Optional SMPL 3D twins

Licensed SMPL weights are not included and must never be committed or redistributed. Place authorized weights under `models\smpl\`, for example:

```text
models\smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl
models\smpl\basicModel_m_lbs_10_207_0_v1.0.0.pkl
```

Export the default mesh with:

```bat
docker compose --profile smpl run --rm smpl-export
```

The prediction and explanation workflow remains usable without these weights.

## Development and validation

Run the same focused checks used by continuous integration:

```bat
python -m unittest test_app_validation.py test_full_pipeline_notebook.py test_knowledge_graph.py test_ollama_recommendations.py test_stage3_efficiency.py test_twin_assets.py
node --check static\js\knowledge_graph.js
docker compose config --quiet
```

Run the Stage 3 benchmark inside the dashboard container:

```bat
docker compose exec -T dashboard python benchmarks/benchmark_stage3.py
```

The notebook contract tests verify the canonical filename, clean saved state, artifact-readiness handoff to host Compose, artifact paths, and Stage 4 evidence semantics.

## Model limitations

The checked local Random Forest contains 400 trees and was evaluated on the imbalanced 253,680-row BRFSS dataset.

| Accuracy | Balanced accuracy | Macro-F1 | Medium recall |
| ---: | ---: | ---: | ---: |
| 81.96% | 45.06% | 44.37% | 0.11% |

Accuracy alone is misleading. In particular, the very low Medium/prediabetes recall means this model is unsuitable for Medium-class screening. Report all results as model-based, non-causal research evidence.

## Troubleshooting

### Docker is unavailable

Start Docker Desktop, select Linux containers, and confirm both commands succeed:

```bat
docker info
docker compose version
```

Run these commands in Windows Command Prompt or PowerShell, not in a notebook cell. If `docker` is not found there, Docker Desktop is not installed or its command-line tools are not on `PATH`. Install or restart Docker Desktop, open a new terminal, and repeat the checks.

### The dashboard does not open

```bat
docker compose ps
docker compose logs --tail=50 dashboard
```

If `dashboard` is not `Up`, resolve the first error in its logs. If only Neo4j is running, wait for it to become healthy and retry `docker compose up -d dashboard`.

### The dashboard shows stale notebook data

Flask loads the twin and knowledge graph when its process starts. Restart the service after new artifact files are written:

```bat
docker compose restart dashboard
```

If the dashboard was already running while the notebook generated new artifacts, run the restart command above so Flask reloads them.

### No trained model is found

Confirm that this file exists:

```bat
dir artifacts_notebook\diabetes_risk_random_forest.joblib
```

If it is missing, run `digital_twin_full_pipeline.ipynb` from the beginning and restart the dashboard afterward.

### Port 5000 is already in use

Do not run `python app.py` alongside the container. Stop the other process or Compose project, then retry:

```bat
docker compose down
docker compose up -d dashboard
```

### Local Ollama guidance is unavailable

```bat
ollama list
curl.exe http://127.0.0.1:11434/api/tags
docker compose exec -T dashboard python -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://host.docker.internal:11434/api/tags')))"
```

## Architecture

```text
JupyterLab container
    |
    | writes model and evidence
    v
artifacts_notebook/
    |
    | bind-mounted at /app
    v
Flask dashboard container <---- host.docker.internal:11434 ---- optional Ollama
    |
    `---- bolt://neo4j:7687 ---- Neo4j container
    |
    `---- http://127.0.0.1:5000 ---- local browser
```

Exact predictions and SHAP results use a bounded in-memory LRU cache. Generated SMPL assets use a bounded, content-addressed disk cache under `artifacts_notebook/twin_cache/`. These caches are temporary and ignored by Git.

## Key files

| Path | Purpose |
| --- | --- |
| `digital_twin_full_pipeline.ipynb` | Canonical Stage 1–4 training, explanation, simulation, graph, and Docker dashboard workflow |
| `test_full_pipeline_notebook.py` | Notebook structure, privacy, artifact, and host-startup handoff contract tests |
| `docker-compose.yml` | Neo4j, dashboard, Jupyter, notebook execution, and SMPL services |
| `app.py` | Flask routes and the interactive four-stage workflow |
| `stage3.py` | Model loading, prediction, exact SHAP, scenarios, and caching |
| `knowledge_graph.py` | Neo4j schema and temporary patient graph assembly |
| `ollama_recommendations.py` | Validated local-model selections and research-safe rendering |
| `twin_assets.py` | Optional SMPL generation and bounded asset reuse |
| `artifacts_notebook/` | Generated local model and research evidence; large or patient-specific outputs are ignored |

## Privacy, data, and licensing

- Do not commit `.env` files, API keys, local patient files, notebook outputs, or personal filesystem paths.
- Do not redistribute licensed SMPL weights.
- The repository does not grant permission to redistribute BRFSS-derived data or trained artifacts; confirm all upstream terms first.
- If distributing the large trained model separately, use a trusted channel and publish its checksum.
- Never describe predictions or scenarios as medical advice, diagnoses, or causal treatment effects.
