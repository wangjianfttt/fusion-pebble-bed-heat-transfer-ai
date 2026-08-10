import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/import_hccb_p418_parallel_history.py"
RUNNER = ROOT / "code/run_hccb_p418_step_responses.sh"


def load_module():
    spec = importlib.util.spec_from_file_location("parallel_history", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_state(root: Path, rank: int, time_name: str) -> None:
    time_root = root / f"processor{rank}" / time_name
    for field in (
        "fluid/T",
        "fluid/U",
        "fluid/p",
        "fluid/p_rgh",
        "fluid/phi",
        "solid/T",
        "uniform/time",
    ):
        path = time_root / field
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rank} {time_name} {field}\n", encoding="utf-8")


def test_imports_only_requested_complete_early_history(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    metadata = {
        "sequence_id": "temperature_up",
        "source_condition_id": "cold",
        "target_condition_id": "hot",
        "snapshot_times_s": [0.0, 0.5, 1.0, 2.0],
    }
    for root in (source, destination):
        root.mkdir()
        (root / "step_case_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    for rank in range(2):
        (destination / f"processor{rank}/constant").mkdir(parents=True)
        for time_name in ("0.5", "1", "2"):
            write_state(source, rank, time_name)
    history = source / "postProcessing/fluid/outletTemperature/0/surfaceFieldValue.dat"
    history.parent.mkdir(parents=True)
    history.write_text("# Time value\n0.0 300\n0.5 400\n1.0 500\n1.5 550\n", encoding="utf-8")

    record = module.import_history(source, destination, 1.0, 2)

    assert record["imported_snapshot_times_s"] == [0.5, 1.0]
    for rank in range(2):
        assert (destination / f"processor{rank}/0.5/fluid/T").is_file()
        assert (destination / f"processor{rank}/1/fluid/T").is_file()
        assert not (destination / f"processor{rank}/2").exists()
    copied = (
        destination
        / "postProcessing/fluid/outletTemperature/0/surfaceFieldValue.dat"
    ).read_text(encoding="utf-8")
    assert "1.0 500" in copied
    assert "1.5 550" not in copied
    assert (destination / "parallel_history_import_complete.json").is_file()


def test_formal_runner_exposes_parallel_history_resume() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "RESUME_PARALLEL_HISTORY_ROOT" in text
    assert "import_hccb_p418_parallel_history.py" in text
    assert "parallel_history_import_complete.json" in text
