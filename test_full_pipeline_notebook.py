"""Contract tests for the canonical Stage 1-4 notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "digital_twin_full_pipeline.ipynb"
OLD_NOTEBOOK_PATH = ROOT / "diabetes_risk_stages_1_2.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


class FullPipelineNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_all_python_cells_compile(self):
        notebook = load_notebook()
        self.assertEqual(4, notebook["nbformat"])
        self.assertGreaterEqual(notebook["nbformat_minor"], 5)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                with self.subTest(cell=index):
                    ast.parse("".join(cell.get("source", [])))
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual([], cell.get("outputs", []))

    def test_full_pipeline_replaces_the_old_notebook(self):
        self.assertTrue(NOTEBOOK_PATH.is_file())
        self.assertFalse(OLD_NOTEBOOK_PATH.exists())
        for relative_path in (
            "README.md",
            "docker-compose.yml",
            "share_supervisor.ps1",
            "research_report_stage_experiments.md",
            "experiment_plan_stage1_stage2_tech_debt.md",
        ):
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("diabetes_risk_stages_1_2.ipynb", text, relative_path)

    def test_stage_three_and_four_preserve_evidence_contracts(self):
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in load_notebook()["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn('class_id=2', source)
        self.assertIn('important_limitation', source)
        self.assertIn('ARTIFACTS / "knowledge_graph.json"', source)
        self.assertNotIn('ARTIFACTS / "digital_twin.json"', source)
        self.assertNotIn("types.MethodType", source)
        self.assertNotIn("patient1.csv", source)

    def test_dashboard_cell_is_container_managed_and_bounded(self):
        notebook = load_notebook()
        dashboard_cell = next(
            cell for cell in notebook["cells"] if cell.get("id") == "4fc054af"
        )
        source = "".join(dashboard_cell["source"])
        self.assertNotIn("subprocess.Popen", source)
        self.assertIn('compose_base = ["docker", "compose"]', source)
        self.assertIn('compose_base + ["up", "-d", "neo4j"]', source)
        self.assertIn('compose_base + ["restart", "dashboard"]', source)
        self.assertIn('"training3-dashboard:latest"', source)
        self.assertIn('capture_output=True', source)
        self.assertIn('Dashboard running (Docker): http://127.0.0.1:5000', source)
        self.assertIn('compose_base + ["logs", "--tail=50", "dashboard"]', source)

        shutdown_cell = notebook["cells"][notebook["cells"].index(dashboard_cell) + 1]
        self.assertEqual(
            "Run `docker compose down` in a terminal to stop the dashboard and Neo4j containers when finished.",
            "".join(shutdown_cell["source"]),
        )


if __name__ == "__main__":
    unittest.main()
