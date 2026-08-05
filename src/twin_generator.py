"""SMPL-based patient twin generator.

Implements BMI -> SMPL beta mapping and risk->color mapping, runs a forward pass
to obtain vertices/faces, and displays the mesh with PyVista for quick local testing.

Requirements: torch, smplx, pyvista, trimesh, numpy
Place SMPL model `.pkl` files under `models/smpl/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import inspect

import numpy as np

try:
    import torch
    import smplx
    import trimesh
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("Missing runtime dependency: run `pip install -r requirements.txt`.") from exc


MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "smpl"


def bmi_to_beta0(bmi: float) -> float:
    """Map patient BMI to SMPL first shape coefficient (beta_0).

    Formula: beta_0 = (BMI_patient - 22.0) * 0.5
    """
    return (float(bmi) - 22.0) * 0.5


def risk_to_color(prob: float) -> str:
    """Map risk probability (0-100) to hex color.

    - >70% -> red
    - 40-70% -> orange
    - <40% -> green
    """
    p = float(prob)
    if p > 70.0:
        return "#FF4D4D"
    if p >= 40.0:
        return "#FFA500"
    return "#2ECC71"


class TwinGenerator:
    def __init__(self, model_folder: Path | str = MODELS_DIR, gender: str = "female", num_betas: int = 10):
        self.model_folder = Path(model_folder)
        self.gender = gender
        self.num_betas = num_betas
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_folder.exists():
            raise FileNotFoundError(f"SMPL models folder not found: {self.model_folder}")
        gender_code = "f" if self.gender.lower().startswith("f") else "m"
        expected_name = f"basicModel_{gender_code}_lbs_10_207_0_v1.0.0.pkl"
        candidates = [path for path in self.model_folder.glob("*.pkl") if path.name.lower() == expected_name.lower()]
        if not candidates:
            available = ", ".join(path.name for path in self.model_folder.glob("*.pkl")) or "none"
            raise FileNotFoundError(
                f"No SMPL {self.gender} model found in {self.model_folder}. "
                f"Expected {expected_name}; available: {available}"
            )

        # Pass the exact licensed model file. smplx.create() expects a newer
        # model_type subdirectory (for example models/smpl/SMPL_FEMALE.pkl),
        # whereas the licensed basicModel_*.pkl files used by this project are
        # stored directly in models/smpl/.
        # Legacy basicModel files contain Chumpy objects. Python 3.11 removed
        # inspect.getargspec, which Chumpy still imports, so provide the
        # backwards-compatible alias before importing it for unpickling.
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
        # Chumpy 0.70 also imports aliases removed by NumPy 1.24. The values
        # below reproduce the historical aliases only for legacy unpickling.
        for alias, replacement in {
            "bool": np.bool_, "int": int, "float": float, "complex": complex,
            "object": object, "unicode": str, "str": str,
        }.items():
            if alias not in np.__dict__:
                setattr(np, alias, replacement)
        try:
            import chumpy  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "This legacy SMPL model requires chumpy. Rebuild the Docker image with `docker compose build smpl-export`."
            ) from exc

        self._model = smplx.SMPL(
            model_path=str(candidates[0]), gender=self.gender, num_betas=self.num_betas
        )
        return self._model

    def generate(self, bmi: float, risk_prob: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh vertices and faces from BMI and risk probability.

        Returns (vertices (N,3), faces (M,3)).
        """
        model = self._load_model()
        beta0 = bmi_to_beta0(bmi)
        betas = torch.zeros([1, self.num_betas], dtype=torch.float32)
        betas[0, 0] = float(beta0)

        # Run forward pass to obtain vertices
        device = torch.device("cpu")
        model = model.to(device)
        betas = betas.to(device)

        # Some smplx versions accept call(), others require forward(); try both.
        try:
            out = model(betas=betas, return_verts=True)
        except TypeError:
            out = model.forward(betas=betas, return_verts=True)

        verts = out.vertices.detach().cpu().numpy()[0]
        faces = model.faces.astype(np.int64)
        return verts, faces

    def mesh_with_color(self, bmi: float, risk_prob: float) -> pv.PolyData:
        """Return a PyVista mesh colored according to risk.

        This function lazily imports PyVista to avoid requiring a graphical
        environment when only mesh export is needed.
        """
        verts, faces = self.generate(bmi, risk_prob)
        tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        try:
            import pyvista as pv  # imported lazily
        except Exception:
            raise RuntimeError("PyVista is required for display. Install pyvista and vtk, or use export_smpl.py to export files.")
        mesh = pv.wrap(tri)
        color = risk_to_color(risk_prob)
        mesh.point_data["risk_color"] = np.tile(np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)], dtype=np.uint8), (mesh.n_points, 1))
        return mesh


def display_demo(bmi: float = 30.0, risk_prob: float = 65.0, gender: str = "female") -> None:
    """Generate a mesh and show it in a PyVista window for quick inspection."""
    gen = TwinGenerator(gender=gender)
    mesh = gen.mesh_with_color(bmi, risk_prob)
    import pyvista as pv
    p = pv.Plotter()
    p.add_mesh(mesh, scalars="risk_color", rgb=True)
    p.add_text(f"BMI: {bmi}  Risk: {risk_prob:.1f}%", position="upper_left")
    p.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SMPL Twin generator demo")
    parser.add_argument("--bmi", type=float, default=30.0)
    parser.add_argument("--risk", type=float, default=65.0)
    parser.add_argument("--gender", type=str, default="female")
    args = parser.parse_args()
    display_demo(bmi=args.bmi, risk_prob=args.risk, gender=args.gender)
