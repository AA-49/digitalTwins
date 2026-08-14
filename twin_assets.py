"""Bounded, content-addressed SMPL assets for the Stage 3 dashboard."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4


EXPORTER_VERSION = "2"
DEFAULT_CACHE_DIR = Path(
    os.environ.get("TWIN_CACHE_DIR", "artifacts_notebook/twin_cache")
)
DEFAULT_CACHE_SIZE = max(1, int(os.environ.get("TWIN_CACHE_SIZE", "32")))


def _risk_to_color(risk_percent: float) -> str:
    if risk_percent > 70.0:
        return "#FF4D4D"
    if risk_percent >= 40.0:
        return "#FFA500"
    return "#2ECC71"


@dataclass(frozen=True)
class TwinAssetResult:
    metadata: dict[str, Any] | None
    status: str


class TwinAssetService:
    """Generate each unique sex/BMI/risk twin once and reuse it safely."""

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        max_entries: int = DEFAULT_CACHE_SIZE,
        generator_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_entries = max(1, int(max_entries))
        self._generator_factory = generator_factory
        self._generators: dict[str, Any] = {}
        self._lock = RLock()

    @staticmethod
    def cache_key(gender: str, bmi: float, risk_percent: float) -> str:
        payload = json.dumps(
            {
                "exporter_version": EXPORTER_VERSION,
                "gender": gender.lower(),
                "bmi": float(bmi).hex(),
                "risk_percent": float(risk_percent).hex(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _generator(self, gender: str):
        if gender not in self._generators:
            if self._generator_factory is None:
                from src.twin_generator import TwinGenerator

                self._generators[gender] = TwinGenerator(gender=gender)
            else:
                self._generators[gender] = self._generator_factory(gender)
        return self._generators[gender]

    def _paths(self, asset_key: str) -> tuple[Path, Path]:
        return (
            self.cache_dir / f"{asset_key}.glb",
            self.cache_dir / f"{asset_key}.json",
        )

    def asset_path(self, asset_key: str) -> Path | None:
        if len(asset_key) != 64 or any(char not in "0123456789abcdef" for char in asset_key):
            return None
        glb_path, _metadata_path = self._paths(asset_key)
        return glb_path if glb_path.is_file() else None

    def _read_cached(self, asset_key: str) -> dict[str, Any] | None:
        glb_path, metadata_path = self._paths(asset_key)
        if not glb_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if metadata.get("asset_key") != asset_key:
            return None
        os.utime(glb_path, None)
        os.utime(metadata_path, None)
        return metadata

    def _export(
        self,
        generator: Any,
        glb_path: Path,
        gender: str,
        bmi: float,
        risk_percent: float,
        asset_key: str,
    ) -> dict[str, Any]:
        import numpy as np
        import trimesh

        vertices, faces = generator.generate(bmi, risk_percent)
        color = _risk_to_color(risk_percent)
        rgb = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
        vertex_colors = np.tile(
            np.array([*rgb, 255], dtype=np.uint8), (len(vertices), 1)
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False,
        )
        mesh.export(glb_path)
        return {
            "asset_key": asset_key,
            "exporter_version": EXPORTER_VERSION,
            "bmi": bmi,
            "risk_percent": risk_percent,
            "gender": gender,
            "risk_color": color,
            "mesh_file": f"{asset_key}.glb",
        }

    def _prune(self, keep_key: str) -> None:
        metadata_files = sorted(
            self.cache_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        retained = {keep_key}
        for metadata_path in metadata_files:
            asset_key = metadata_path.stem
            if asset_key in retained:
                continue
            if len(retained) < self.max_entries:
                retained.add(asset_key)
                continue
            glb_path, _ = self._paths(asset_key)
            glb_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)

    def get_or_create(
        self,
        *,
        gender: str,
        bmi: float,
        risk_percent: float,
        profile_name: str = "3D twin",
    ) -> TwinAssetResult:
        gender = "male" if gender.lower().startswith("m") else "female"
        bmi = float(bmi)
        risk_percent = float(risk_percent)
        asset_key = self.cache_key(gender, bmi, risk_percent)

        with self._lock:
            cached = self._read_cached(asset_key)
            if cached is not None:
                return TwinAssetResult(cached, f"{profile_name} loaded from the bounded cache.")

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            glb_path, metadata_path = self._paths(asset_key)
            temporary_glb = self.cache_dir / f".{asset_key}.{uuid4().hex}.glb"
            temporary_metadata = temporary_glb.with_suffix(".json")
            try:
                metadata = self._export(
                    self._generator(gender),
                    temporary_glb,
                    gender,
                    bmi,
                    risk_percent,
                    asset_key,
                )
                temporary_metadata.write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
                temporary_glb.replace(glb_path)
                temporary_metadata.replace(metadata_path)
                self._prune(asset_key)
                return TwinAssetResult(
                    metadata,
                    f"{profile_name} generated for BMI {bmi:g}, {gender}, risk {risk_percent:.1f}%.",
                )
            except Exception as exc:
                temporary_glb.unlink(missing_ok=True)
                temporary_metadata.unlink(missing_ok=True)
                return TwinAssetResult(
                    None, f"The {profile_name.lower()} could not be updated: {exc}"
                )
