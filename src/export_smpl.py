"""Export SMPL mesh to GLB/OBJ without requiring PyVista.

Usage:
    python -m src.export_smpl --bmi 30 --risk 65 --out smpl_out.glb

Mount your local `models/smpl` into the container or place the .pkl files there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.twin_generator import TwinGenerator, risk_to_color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bmi", type=float, default=30.0)
    parser.add_argument("--risk", type=float, default=65.0)
    parser.add_argument("--gender", type=str, default="female")
    parser.add_argument("--out", type=str, default="smpl_out.glb")
    args = parser.parse_args()

    gen = TwinGenerator(gender=args.gender)
    verts, faces = gen.generate(args.bmi, args.risk)

    import trimesh

    color = risk_to_color(args.risk)
    rgb = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
    vertex_colors = np.tile(np.array([*rgb, 255], dtype=np.uint8), (len(verts), 1))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors, process=False)
    out_path = Path(args.out)
    mesh.export(out_path)
    metadata_path = out_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({
        "bmi": args.bmi,
        "risk_percent": args.risk,
        "gender": args.gender,
        "risk_color": color,
        "mesh_file": out_path.name,
    }, indent=2), encoding="utf-8")
    print(f"Exported mesh to: {out_path.resolve()}")
    print(f"Twin metadata saved to: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
