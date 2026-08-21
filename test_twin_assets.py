"""Content addressing, reuse, and bounded eviction for Stage 3 twins."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from twin_assets import TwinAssetService


class FakeGenerator:
    def __init__(self, gender):
        self.gender = gender
        self.calls = 0

    def generate(self, _bmi, _risk):
        self.calls += 1
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        return vertices, faces


class FakeTwinAssetService(TwinAssetService):
    def _export(self, generator, glb_path, gender, bmi, risk_percent, asset_key):
        generator.generate(bmi, risk_percent)
        glb_path.write_bytes(b"test-glb")
        return {
            "asset_key": asset_key,
            "exporter_version": "test",
            "bmi": bmi,
            "risk_percent": risk_percent,
            "gender": gender,
            "risk_color": "#FFA500",
            "mesh_file": f"{asset_key}.glb",
        }


class TwinAssetTests(unittest.TestCase):
    def test_key_includes_gender_bmi_and_risk(self):
        base = TwinAssetService.cache_key("female", 30, 50)
        self.assertNotEqual(base, TwinAssetService.cache_key("male", 30, 50))
        self.assertNotEqual(base, TwinAssetService.cache_key("female", 31, 50))
        self.assertNotEqual(base, TwinAssetService.cache_key("female", 30, 51))

    def test_repeated_and_concurrent_requests_generate_once(self):
        with TemporaryDirectory() as directory:
            generators = {}

            def factory(gender):
                generators[gender] = FakeGenerator(gender)
                return generators[gender]

            service = FakeTwinAssetService(directory, max_entries=3, generator_factory=factory)
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(
                    lambda _index: service.get_or_create(
                        gender="female", bmi=30, risk_percent=50
                    ),
                    range(4),
                ))
            self.assertEqual(1, generators["female"].calls)
            self.assertEqual(1, len({item.metadata["asset_key"] for item in results}))
            self.assertTrue(all(service.asset_path(item.metadata["asset_key"]) for item in results))

    def test_cache_evicts_to_configured_bound(self):
        with TemporaryDirectory() as directory:
            service = FakeTwinAssetService(
                directory,
                max_entries=2,
                generator_factory=lambda gender: FakeGenerator(gender),
            )
            for bmi in (20, 30, 40):
                service.get_or_create(gender="female", bmi=bmi, risk_percent=50)
            self.assertEqual(2, len(list(Path(directory).glob("*.json"))))
            self.assertEqual(2, len(list(Path(directory).glob("*.glb"))))


if __name__ == "__main__":
    unittest.main()
