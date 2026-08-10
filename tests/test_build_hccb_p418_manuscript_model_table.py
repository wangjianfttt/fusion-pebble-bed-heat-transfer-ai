import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_manuscript_model_table import build_table  # noqa: E402


class ManuscriptModelTableTest(unittest.TestCase):
    def test_table_is_built_from_registered_settings(self) -> None:
        text = build_table(
            ROOT / "parameters" / "hccb_p418_ai_architecture_sources.json",
            ROOT / "parameters" / "hccb_p418_model_numerical_settings.csv",
        )
        self.assertIn("Graph--Transformer", text)
        self.assertIn("iparraguirre2026mgnt", text)
        self.assertIn("wang2024hccb_pinn", text)
        self.assertIn("Diffusion correction", text)
        self.assertIn("Observable DMDc", text)
        self.assertIn("candidate ranks 1, 2, 3, 4, 5, 6", text)
        self.assertIn("Regional DMDc", text)
        self.assertIn(
            "candidate ranks 1, 2, 3, 4, 8, 12, 16, 24, 32", text
        )
        self.assertIn("not pebble-bed material properties", text)
        self.assertIn("projection-aware energy RMSE", text)
        self.assertNotIn("energy-operator difference", text)
        self.assertIn("all independent results retained", text)
        self.assertIn("local refinement graph blocks", text)
        self.assertIn("fixed and ReLoBRaLo physics-loss candidates", text)
        self.assertIn("common unweighted mean", text)
        self.assertIn("one independent-test evaluation", text)
        self.assertNotIn("Minimum validation solid-temperature", text)
        self.assertNotIn("final use requires", text)


if __name__ == "__main__":
    unittest.main()
