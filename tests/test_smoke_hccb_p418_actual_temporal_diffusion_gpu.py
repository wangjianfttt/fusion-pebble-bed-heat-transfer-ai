import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "smoke_hccb_p418_actual_temporal_diffusion_gpu.py"


class ActualTemporalDiffusionResourceScriptTest(unittest.TestCase):
    def test_script_uses_formal_architecture_and_marks_resource_scope(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE", source)
        self.assertIn('"curve_batch_size": 1', source)
        self.assertIn('choices=("float32", "bfloat16")', source)
        self.assertIn("torch.autocast", source)
        self.assertIn('"new_physical_parameters": []', source)
        self.assertIn("not a temperature-accuracy result", source)
        self.assertIn("loss.backward()", source)
        self.assertIn("validate_graph_provenance", source)
        self.assertIn('"input_provenance": input_provenance', source)
        self.assertIn('"step_plan_sha256": sha256', source)


if __name__ == "__main__":
    unittest.main()
