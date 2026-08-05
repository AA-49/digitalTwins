# SMPL Stage 3 setup and export

The project now includes the SMPL Stage 3 pathway required by `plan.md`:

- `src/twin_generator.py` maps BMI to the first SMPL shape coefficient: `beta0 = (BMI - 22) * 0.5`.
- It maps the model's high-diabetes probability to the mesh colour: green below 40%, orange from 40% to 70%, and red above 70%.
- `src/export_smpl.py` exports the generated mesh as GLB or OBJ using `trimesh`.
- `mesh_with_color()` returns a `pyvista.PolyData` mesh for local scientific rendering.

## One required licensed asset

SMPL weights cannot be included in this project. Download the appropriate licensed SMPL `.pkl` file from the official SMPL provider and place it in:

```text
models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl
```

or the corresponding male model. Do not upload or include these files in the supervisor ZIP.

## Docker export

After Docker Desktop is running and the SMPL model file is present, export a default twin:

```powershell
docker compose --profile smpl run --rm smpl-export
```

To choose patient-specific values, override the command:

```powershell
docker compose --profile smpl run --rm smpl-export python -m src.export_smpl --bmi 27 --risk 35 --gender female --out artifacts_notebook/digital_twin.glb
```

The GLB file can be opened in a 3D viewer such as Windows 3D Viewer, Blender, or a web GLB viewer. For a local PyVista window instead:

```powershell
docker compose --profile smpl run --rm smpl-export python src/twin_generator.py --bmi 27 --risk 35
```

The web dashboard displays the exact BMI-derived `beta0` and risk colour for each prediction. It does not fabricate a 3D mesh when the licensed SMPL asset is absent.

After export, the dashboard automatically displays `artifacts_notebook/digital_twin.glb` in an interactive 3D viewer at http://127.0.0.1:5000. The export also writes a sidecar `digital_twin.json` file containing the BMI, risk percentage, sex, and mesh colour.

When a user submits the **Current patient profile** in the dashboard, it now automatically runs the Docker SMPL export with that profile's BMI, the model's high-diabetes probability, and the recorded sex. The 3D viewer reloads the updated GLB. Docker Desktop must be running for this automatic update.

## Rebuild after dependency changes

The Docker image pins NumPy below version 2 because the included CPU PyTorch wheel is compiled against NumPy 1.x, and installs the legacy Chumpy package required by `basicModel_*.pkl`. Build the image once before export:

```powershell
docker compose build --no-cache smpl-export
```
