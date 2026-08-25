import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data"
NOTEBOOKS = ROOT / "notebooks"
WORKERS = ROOT / "workers"


class ScalarNotebookContractTests(unittest.TestCase):
    def test_scalar_notebook_accepts_bundle_with_additional_native_heads(self):
        notebook = json.loads(
            (NOTEBOOKS / "02_multigpu_boundary_1xt4.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn(
            "set(EXPECTED_OUTPUT_DIMS).issubset(set(manifest.get(\"output_dims\", [])))",
            source,
        )
        self.assertNotIn(
            "sorted(manifest.get(\"output_dims\", [])) != list(EXPECTED_OUTPUT_DIMS)",
            source,
        )

    def test_multigpu_worker_discovers_the_exact_model_or_uses_the_release(self):
        source = (WORKERS / "multigpu_beam_search.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Path('/kaggle/input').rglob(checkpoint_name)", source)
        self.assertNotIn("Path('/kaggle/input/models').rglob('*.pth')", source)
        self.assertIn("MultiGPUBeamSearchPaper/releases/download/v1.0.0", source)
        self.assertIn("checkpoint SHA256 mismatch", source)

    def test_public_dataset_contains_native_runner_inputs(self):
        self.assertTrue((DATASET / "puzzle_info.json").is_file())
        self.assertTrue((DATASET / "test.csv").is_file())


if __name__ == "__main__":
    unittest.main()
