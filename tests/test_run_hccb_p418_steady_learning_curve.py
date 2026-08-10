import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_steady_learning_curve.sh"


class P418SteadyLearningCurveRunnerTest(unittest.TestCase):
    def test_shell_syntax_and_fixed_comparison_design(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        self.assertIn("learning_curve_n09", text)
        self.assertIn("learning_curve_n36", text)
        self.assertIn("pinn_data_only pinn graph transolver", text)
        self.assertIn("summarize_hccb_p418_learning_curve_efficiency.py", text)
        self.assertIn("plot_hccb_p418_learning_curve.py", text)


if __name__ == "__main__":
    unittest.main()
