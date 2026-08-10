import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class P418DiffusionRunnerTest(unittest.TestCase):
    def test_full_runner_uses_registered_architecture_values(self):
        text = (
            ROOT / "code" / "run_hccb_p418_60_diffusion_refiner.sh"
        ).read_text(encoding="utf-8")
        for expected in (
            "--epochs 500",
            "--batch-size 8",
            "--hidden-dim 256",
            "--layers 5",
            "--attention-heads 8",
            "--physics-slices 32",
            "--num-refinement-steps 3",
            "--min-noise-std 4e-7",
            "--ema-decay 0.995",
        ):
            self.assertIn(expected, text)
        self.assertIn("validation field RMSE", text)
        self.assertIn('(\"response_surface\", \"pinn\", \"graph\", \"transolver\")', text)
        self.assertNotIn("test_state_normalized_rmse", text)


if __name__ == "__main__":
    unittest.main()
