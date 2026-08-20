# Explainable Diabetes Risk Digital Twin

A local, Docker-based research prototype for exploring diabetes-risk model predictions using CDC BRFSS 2015 survey records. It combines three-class prediction, patient-specific SHAP explanation, manual what-if comparison, optional SMPL 3D twins, a temporary patient-centric knowledge graph, and optional local Ollama guidance.

> **Research only:** This system is not a medical diagnosis, treatment recommendation, causal forecast, or clinical decision-support tool. BRFSS is cross-sectional survey data. SHAP values and changed scenario predictions do not prove medical causes or intervention effects.

## Start here: Windows quick start

These instructions are written for **Windows Command Prompt (`cmd.exe`)**. In Command Prompt, use `dir`, not `ls`.

### 1. Install the required software

Install:

- [Git for Windows](https://git-scm.com/download/win)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
- A web browser
- [Ollama for Windows](https://ollama.com/download/windows) only if local AI guidance is wanted

Docker Desktop must use **Linux containers**. Complete any WSL 2 setup requested by Docker Desktop and restart Windows if the installer asks.

### 2. Start Docker Desktop before running Docker commands

Open Docker Desktop from the Windows Start menu. Wait until it reports that the engine is running.

Then open Command Prompt and verify Docker:

```bat
docker version
docker info
docker compose version
```

Do not continue until `docker info` shows a **Server** section.

If the command reports either of these errors:

```text
failed to connect to the docker API
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified
```

Docker Desktop is not ready. Start it, wait for the engine, confirm that Linux containers are selected, and run `docker info` again. This is a Docker Desktop problem, not a project-code problem.

### 3. Clone the repository

Choose a folder and run:

```bat
cd /d "%USERPROFILE%\Documents"
git clone https://github.com/AA-49/digitalTwins.git
cd digitalTwins
dir
```

`cd /d` changes both the drive and directory. You may replace the example with any folder where you want to keep the project.

### 4. Prepare the trained model

The application expects:

```text
artifacts_notebook\diabetes_risk_random_forest.joblib
```

The trained file is approximately 405 MB and is deliberately excluded from normal Git history. A fresh clone therefore needs one of these options before patient prediction will work:

Start JupyterLab:

```bat
docker compose up -d jupyter
```

Open <http://127.0.0.1:8888>. The combined Stage 1-4 notebook is `digital_twin_full_pipeline.ipynb`.

Stop JupyterLab with the other services:

```bat
docker compose down
```

### 5. Start the local dashboard

Build the dashboard image with detailed progress:

```bat
docker compose --progress plain build dashboard
```

The first build can take a while because it installs the machine-learning and notebook dependencies. Do not continue if the build ends with `ERROR` or `failed to solve`.

After the build reports `Built`, start the dashboard and Neo4j:

```bat
docker compose up -d dashboard
docker compose ps
```

`docker compose ps` must show both of these conditions:

- `neo4j` is `Up` and `healthy`.
- `dashboard` is `Up` with `127.0.0.1:5000->5000/tcp`.

An empty table means nothing is running. Do not open the browser yet; inspect the terminal error and logs.

Wait for an actual dashboard response:

```bat
curl.exe --fail --retry 30 --retry-delay 2 --retry-connrefused --head http://127.0.0.1:5000/
```

Open the dashboard only after the command returns an HTTP `200` response:

- Dashboard: <http://127.0.0.1:5000>
- Neo4j Browser: <http://127.0.0.1:7474>

Only the current computer can access these loopback addresses.

### 6. Optional: enable local Ollama guidance

The prediction, SHAP, scenario, and knowledge-graph stages work without Ollama. Ollama is needed only for **Generate local research guidance**.

Start Ollama and run:

```bat
ollama pull qwen2.5-coder:1.5b
ollama list
```

The dashboard container reaches Windows Ollama through `http://host.docker.internal:11434`. The first generated response may take one or two minutes while the model loads.

## Everyday commands

Run these commands from the cloned `digitalTwins` directory.

### Check status

```bat
docker compose ps
docker compose logs --tail 100 neo4j dashboard
```

### Restart

```bat
docker compose restart neo4j dashboard
```

### Stop

```bat
docker compose down
```

`docker compose down` preserves the Neo4j volume. Do not add `-v` unless permanently deleting that local database volume is intentional.

### Update a clone

Before pulling, make sure personal work is saved or committed:

```bat
git status
git pull
docker compose up -d --build dashboard
```

## Using the dashboard

1. Enter a patient number and choose **Load patient**, or choose **Use** in the patient table.
2. Review Stage 1: Low, Medium (prediabetes), or High (diabetes), with all three model probabilities.
3. Review Stage 2: patient-specific SHAP contributions. They explain this prediction; they do not establish causes.
4. Explore Stage 4: all 21 decoded observations, all 21 SHAP contributions, three probabilities, model evaluation, and the current Digital Twin.
5. Optionally choose **Generate local research guidance** to use Ollama on this computer.
6. Edit the permitted Stage 3 scenario fields and choose **Compare scenario**.
7. Compare baseline and scenario probabilities and, when available, their separate 3D twins.

The selected one-based patient number always maps to the same row in the active cleaned dataset. Imported CSV data remains in memory only and resets when the application restarts. Patient observations, predictions, SHAP values, and patient graph nodes are not persisted in Neo4j.

## Optional SMPL 3D setup

Licensed SMPL weights are not included. Obtain them from the official SMPL provider and place the appropriate files under `models\smpl\`, for example:

```text
models\smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl
models\smpl\basicModel_m_lbs_10_207_0_v1.0.0.pkl
```

Never commit or redistribute these licensed files.

Export the default mesh:

```bat
docker compose --profile smpl run --rm smpl-export
```

Without licensed weights, the main prediction and explanation workflow still works, but the application cannot generate a new patient-specific SMPL mesh.

## Troubleshooting

### `'ls' is not recognized`

You are using Windows Command Prompt. Use:

```bat
dir
```

`ls` is normally available in PowerShell, Git Bash, WSL, Linux, and macOS—not standard Command Prompt.

### Docker named-pipe or daemon error

Example:

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

Fix:

1. Start Docker Desktop.
2. Wait until the Docker engine is running.
3. Make sure Docker Desktop is using Linux containers.
4. Run `docker info` and confirm it shows a Server section.
5. Retry `docker compose up -d --build dashboard`.

If Docker Desktop reports a WSL problem, run these commands in an Administrator terminal, restart Docker Desktop, and follow any Windows restart prompt:

```bat
wsl --status
wsl --update
```

### No trained model found

If prediction reports:

```text
No trained model found
```

Check:

```bat
dir artifacts_notebook\diabetes_risk_random_forest.joblib
```

If the file is absent, complete **Prepare the trained model** above. Restart the dashboard after placing or training the model:

```bat
docker compose restart dashboard
```

### Build appears stuck

Show every build step:

```bat
docker compose --progress plain build dashboard
```

Keep Docker Desktop open. Initial dependency installation is much slower than later cached builds.

### Build fails with `invalid file request .tmp-cytoscape`

This means an old generated npm cache entered Docker's build context. The current `.dockerignore` excludes it. Update the clone and rebuild without using the failed context:

```bat
git pull
docker compose --progress plain build dashboard
```

If the clone has local changes that prevent `git pull`, save or commit those changes first. Do not continue to the browser until the build reports success.

### Port 5000 is already in use

Do not run `python app.py` alongside Docker. Stop the other process or container, then run:

```bat
docker compose down
docker compose up -d dashboard
```

### Port 8888 is already allocated

This affects the optional JupyterLab service, not the dashboard. Find the container already using the port:

```bat
docker ps --filter publish=8888
```

Stop that Jupyter container from the project that started it, or keep it running and skip `docker compose up -d jupyter`. The non-interactive `train-notebook` service does not publish port 8888.

### Dashboard does not open

`ERR_CONNECTION_REFUSED` means no process is accepting connections on port 5000. It is not a browser-cache problem. Run:

```bat
docker compose ps
docker compose logs --tail 100 neo4j dashboard
```

Interpret the result:

- Empty `docker compose ps`: the build or `up` command failed; return to the first terminal error.
- `dashboard` is `Exited`: inspect the dashboard logs for the Python startup error.
- Only `neo4j` is running: wait for it to become healthy, then run `docker compose up -d dashboard`.
- Port `5000` is absent: the dashboard is not published and the browser will refuse the connection.

When the service is running, verify it before opening the browser:

```bat
curl.exe --fail --retry 30 --retry-delay 2 --retry-connrefused --head http://127.0.0.1:5000/
```

### Local guidance is unavailable

Run:

```bat
ollama list
curl.exe http://127.0.0.1:11434/api/tags
docker compose exec -T dashboard python -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://host.docker.internal:11434/api/tags')))"
```

Ollama returns only structured feature and discussion-topic selections. The server validates those references and renders deterministic research-safe prose; raw local-model wording is never displayed.

### Knowledge graph reports fallback mode

Check Neo4j:

```bat
docker compose ps
docker compose logs --tail 100 neo4j dashboard
```

Even in fallback mode, the dashboard retains all 21 current observations and their patient-specific model evidence for the current request.

## Model evaluation and limitations

The checked local Random Forest contains 400 trees and was evaluated on the imbalanced 253,680-row BRFSS dataset.

| Accuracy | Balanced accuracy | Macro-F1 | Medium recall |
| ---: | ---: | ---: | ---: |
| 81.96% | 45.06% | 44.37% | 0.11% |

Accuracy alone is misleading for this problem. The 0.11% Medium/prediabetes recall means this model is not reliable for Medium-class screening. Results must be described as model-based, non-causal research evidence—not clinical guidance.

## Tests

Run the Python suite inside Docker and validate the Compose file:

```bat
docker compose exec -T dashboard python -m unittest discover -v
docker compose config --quiet
docker compose exec -T dashboard python benchmarks/benchmark_stage3.py
```

If the dashboard is not running, start it first with `docker compose up -d dashboard`. JavaScript syntax can also be checked on a development computer with Node.js installed:

```bat
node --check static\js\knowledge_graph.js
```

## Sharing checklist for the project owner

Before giving the repository to another person:

- Confirm the recipient knows Docker Desktop must be open before running Compose.
- Decide how the 405 MB trained `joblib` model will be supplied. It is not in normal Git history.
- Provide a checksum and trusted download channel if distributing the trained model separately.
- Do not share `.env` files containing secrets.
- Do not distribute licensed SMPL `.pkl` files.
- Do not claim that the model predicts causal treatment benefits or provides medical advice.
- Ask the recipient to send `docker compose ps` and `docker compose logs --tail 100 neo4j dashboard` when reporting a startup problem.

## Architecture

```text
Windows browser
    |
    | http://127.0.0.1:5000
    v
Flask dashboard container
    |-- local trained model + SHAP
    |-- temporary patient knowledge graph
    |-- optional SMPL export
    |-- host.docker.internal:11434 --> Windows Ollama
    |
    `-- bolt://neo4j:7687 --> Neo4j container
```

Docker Compose binds all exposed ports to `127.0.0.1`, so the services are local to the Windows computer by default.

Predictions and exact SHAP results use a bounded in-memory LRU cache (`ANALYSIS_CACHE_SIZE`, default 128). Generated SMPL assets use a content-addressed disk cache (`TWIN_CACHE_DIR`, default `artifacts_notebook/twin_cache`; `TWIN_CACHE_SIZE`, default 32). Cache data is temporary, ignored by Git, and never persisted to Neo4j.

## Key files

| Path | Purpose |
| --- | --- |
| `README.md` | Windows-first installation, operation, and troubleshooting guide |
| `docker-compose.yml` | Dashboard, Neo4j, Jupyter, training, and SMPL services |
| `Dockerfile` | Separate core, training/Jupyter, dashboard, and SMPL runtime targets |
| `app.py` | Flask routes and four-stage dashboard workflow |
| `stage3.py` | Model loading, prediction, SHAP, scenarios, and twin helpers |
| `twin_assets.py` | Bounded content-addressed SMPL generation and asset reuse |
| `knowledge_graph.py` | Reusable Neo4j schema and temporary patient graph assembly |
| `ollama_recommendations.py` | Validated local-model selections and deterministic safe rendering |
| `digital_twin_full_pipeline.ipynb` | Reproducible Stage 1-4 training, explanation, simulation, graph, and Docker dashboard notebook |
| `artifacts_notebook\` | Generated local model and research evidence |

## Data and licensing

The repository does not grant permission to redistribute BRFSS-derived data, trained artifacts, or SMPL assets. Confirm the terms of every upstream dataset, trained model, and dependency before redistribution.
