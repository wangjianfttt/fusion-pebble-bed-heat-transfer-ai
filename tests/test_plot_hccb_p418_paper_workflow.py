import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from plot_hccb_p418_paper_workflow import build_figure  # noqa: E402


class P418PaperWorkflowFigureTest(unittest.TestCase):
    def test_vector_and_raster_outputs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = build_figure(Path(directory) / "workflow")
            self.assertEqual({path.suffix for path in outputs}, {".pdf", ".svg", ".png"})
            for path in outputs:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)
            svg = (Path(directory) / "workflow.svg").read_text(encoding="utf-8")
            self.assertIn("Graph--Transformer", svg)
            self.assertIn("Diffusion correction", svg)
            self.assertIn("Independent test", svg)
            self.assertIn("Shared finite-volume physics", svg)


if __name__ == "__main__":
    unittest.main()
