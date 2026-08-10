from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hccb_p418_parametric_regional_operator",
    ROOT / "code" / "hccb_p418_parametric_regional_operator.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PINN_SPEC = importlib.util.spec_from_file_location(
    "hccb_p418_coordinate_pinn",
    ROOT / "code" / "hccb_p418_coordinate_pinn.py",
)
PINN_MODULE = importlib.util.module_from_spec(PINN_SPEC)
assert PINN_SPEC.loader is not None
sys.path.insert(0, str(ROOT / "code"))
sys.modules[PINN_SPEC.name] = PINN_MODULE
PINN_SPEC.loader.exec_module(PINN_MODULE)


def make_mesh(fine_count: int, region_count: int):
    if (fine_count, region_count) == (4, 2):
        fine_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        parent = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        region_type = torch.tensor([0, 1], dtype=torch.long)
    elif (fine_count, region_count) == (6, 3):
        fine_type = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        parent = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
        region_type = torch.tensor([0, 0, 1], dtype=torch.long)
    else:
        raise ValueError("unsupported software-test mesh")

    fine_x = torch.linspace(0.0, 1.0, fine_count)
    fine_centroid = torch.stack(
        (fine_x, 0.1 * torch.sin(fine_x), 0.1 * torch.cos(fine_x)), dim=1
    )
    region_centroid = torch.stack(
        (
            torch.linspace(0.0, 1.0, region_count),
            torch.zeros(region_count),
            torch.zeros(region_count),
        ),
        dim=1,
    )
    source = []
    target = []
    for index in range(region_count - 1):
        source.extend((index, index + 1))
        target.extend((index + 1, index))
    edge_source = torch.tensor(source, dtype=torch.long)
    edge_target = torch.tensor(target, dtype=torch.long)
    edge_count = len(edge_source)
    level = MODULE.P418RegionalLevel(
        centroid_m=region_centroid,
        volume_m3=torch.full((region_count,), 1.0 / region_count),
        node_type=region_type,
        boundary_fraction=torch.zeros((region_count, 2)),
        parent_from_finer=parent,
        edge_source=edge_source,
        edge_target=edge_target,
        edge_kind=torch.zeros(edge_count, dtype=torch.long),
        edge_area_m2=torch.ones(edge_count),
        edge_area_vector_m2=torch.ones((edge_count, 3)),
        edge_face_centroid_m=0.5
        * (region_centroid[edge_source] + region_centroid[edge_target]),
    )
    mesh = MODULE.P418RegionalMesh(
        fine_centroid_m=fine_centroid,
        fine_volume_m3=torch.full((fine_count,), 1.0 / fine_count),
        fine_node_type=fine_type,
        fine_boundary_role=torch.zeros((fine_count, 2)),
        coordinate_center_m=torch.tensor([0.5, 0.0, 0.0]),
        coordinate_scale_m=torch.tensor([1.0, 1.0, 1.0]),
        volume_scale_m3=torch.tensor(1.0),
        levels=(level,),
    )
    MODULE.validate_mesh(mesh)
    return mesh


@pytest.mark.parametrize(
    "processor_kind", ["message_passing", "hybrid_physics_attention"]
)
def test_one_weight_set_runs_two_different_packing_meshes(processor_kind: str):
    torch.manual_seed(17)
    mesh_a = make_mesh(4, 2)
    mesh_b = make_mesh(6, 3)
    model = MODULE.HCCBP418ParametricRegionalOperator(
        boundary_role_count=2,
        hidden_dim=8,
        processor_steps=2,
        active_levels=1,
        condition_dim=5,
        output_dim=5,
        processor_kind=processor_kind,
        attention_heads=2,
        attention_start_level=0,
        physics_slices=4,
    )
    condition = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.5]])

    prediction_a = model(condition, mesh_a, chunk_size=3)
    prediction_b = model(condition, mesh_b, chunk_size=4)

    assert prediction_a.shape == (1, 4, 5)
    assert prediction_b.shape == (1, 6, 5)
    assert torch.isfinite(prediction_a).all()
    assert torch.isfinite(prediction_b).all()

    (prediction_a.square().mean() + prediction_b.square().mean()).backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_runtime_mesh_rejects_incompatible_boundary_definition():
    mesh = make_mesh(4, 2)
    model = MODULE.HCCBP418ParametricRegionalOperator(
        boundary_role_count=3,
        hidden_dim=8,
        processor_steps=1,
        active_levels=1,
    )
    with pytest.raises(ValueError, match="boundary-role count"):
        model(torch.zeros((1, 5)), mesh)


def test_one_coordinate_pinn_runs_two_different_packing_meshes():
    torch.manual_seed(23)
    mesh_a = make_mesh(4, 2)
    mesh_b = make_mesh(6, 3)
    model = PINN_MODULE.HCCBP418CoordinatePINNOperator(
        boundary_role_count=2,
        hidden_dim=8,
        hidden_layers=3,
    )
    condition = torch.tensor([[0.2, -0.1, 0.4, -0.3, 0.5]])
    prediction_a = model(condition, mesh_a, chunk_size=3)
    prediction_b = model(condition, mesh_b, chunk_size=4)
    assert prediction_a.shape == (1, 4, 5)
    assert prediction_b.shape == (1, 6, 5)
    assert torch.isfinite(prediction_a).all()
    assert torch.isfinite(prediction_b).all()
    (prediction_a.square().mean() + prediction_b.square().mean()).backward()
    state_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("condition_encoder")
    ]
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in state_parameters
    )
