from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from capture_hccb_p418_runtime_environment import PACKAGES, package_version


def test_required_packages_are_declared() -> None:
    assert {"numpy", "pandas", "scipy", "scikit_learn", "torch"} <= set(PACKAGES)
    assert isinstance(package_version("json"), str)
    assert package_version("module_that_does_not_exist_p418") is None


def test_p418_requirements_is_separate_from_physical_parameters() -> None:
    text = (ROOT / "requirements-p418.txt").read_text(encoding="utf-8")
    assert "torch==" in text
    assert "numpy==" in text
    assert "thermal_conductivity" not in text
