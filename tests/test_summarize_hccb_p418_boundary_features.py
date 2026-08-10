import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/summarize_hccb_p418_boundary_features.py"


def load_module():
    spec = importlib.util.spec_from_file_location("boundary_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_roles(path: Path) -> list[str]:
    names = [
        "inlet",
        "outlet",
        "cooling_wall",
        "symmetry",
        "fluid_solid_interface",
    ]
    path.write_text(json.dumps({"role_order": names}), encoding="utf-8")
    return names


def test_boundary_summary_reports_every_registered_role(tmp_path):
    module = load_module()
    roles_path = tmp_path / "roles.json"
    names = write_roles(roles_path)
    geometry = tmp_path / "geometry.npz"
    values = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        geometry,
        level_5_boundary_volume_fraction=values,
        boundary_role_names=np.asarray(names, dtype="U"),
    )

    payload = module.summarize(geometry, roles_path)

    assert payload["status"] == "p418_boundary_features_present_on_regional_graph"
    assert payload["feature_key"] == "level_5_boundary_volume_fraction"
    assert payload["regional_node_count"] == 4
    assert [row["role"] for row in payload["roles"]] == names
    assert all(row["nonzero_regional_nodes"] > 0 for row in payload["roles"])
    assert "not a boundary-area fraction" in payload["definition"]


def test_boundary_summary_rejects_missing_role_column(tmp_path):
    module = load_module()
    roles_path = tmp_path / "roles.json"
    write_roles(roles_path)
    geometry = tmp_path / "geometry.npz"
    np.savez_compressed(
        geometry,
        level_0_boundary_volume_fraction=np.ones((3, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="columns"):
        module.summarize(geometry, roles_path)
