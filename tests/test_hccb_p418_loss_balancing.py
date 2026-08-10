from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_loss_balancing import (
    FixedLossBalancer,
    ReLoBRaLoLossBalancer,
    common_validation_score,
    weighted_group_loss,
)


def groups(values: tuple[float, float, float]) -> dict[str, torch.Tensor]:
    return {
        "state_data": torch.tensor(values[0], dtype=torch.float64),
        "face_flux_data": torch.tensor(values[1], dtype=torch.float64),
        "physics": torch.tensor(values[2], dtype=torch.float64),
    }


def test_common_validation_score_is_independent_of_training_weights() -> None:
    values = groups((1.0, 2.0, 6.0))
    assert float(common_validation_score(values)) == 3.0
    fixed = FixedLossBalancer(
        {"state_data": 2.0, "face_flux_data": 3.0, "physics": 4.0}
    )
    weights = fixed.update(values)
    assert float(weighted_group_loss(values, weights)) == 32.0
    assert float(common_validation_score(values)) == 3.0


def test_relobralo_matches_equation_11_and_official_two_step_schedule() -> None:
    balancer = ReLoBRaLoLossBalancer(
        temperature=1.0,
        alpha=0.9,
        expected_rho=1.0,
        seed=17,
    )
    first = balancer.update(groups((2.0, 1.0, 0.5)))
    np.testing.assert_allclose(
        [float(first[name]) for name in first],
        np.ones(3),
        rtol=0.0,
        atol=0.0,
    )

    second_losses = torch.tensor((1.0, 1.0, 1.0), dtype=torch.float64)
    expected_second = 3.0 * torch.softmax(
        second_losses / torch.tensor((2.0, 1.0, 0.5), dtype=torch.float64),
        dim=0,
    )
    second = balancer.update(groups((1.0, 1.0, 1.0)))
    torch.testing.assert_close(
        torch.stack(list(second.values())),
        expected_second,
    )

    third_losses = torch.tensor((0.5, 2.0, 1.0), dtype=torch.float64)
    relative_previous = 3.0 * torch.softmax(third_losses, dim=0)
    expected_third = 0.9 * expected_second + 0.1 * relative_previous
    third = balancer.update(groups((0.5, 2.0, 1.0)))
    torch.testing.assert_close(
        torch.stack(list(third.values())),
        expected_third,
    )
    torch.testing.assert_close(
        torch.stack(list(third.values())).sum(),
        torch.tensor(3.0, dtype=torch.float64),
    )


def test_relobralo_random_lookback_sequence_resumes_exactly() -> None:
    original = ReLoBRaLoLossBalancer(
        temperature=0.1,
        alpha=0.999,
        expected_rho=0.5,
        seed=20260723,
    )
    for values in ((1.0, 2.0, 3.0), (0.9, 1.7, 2.8), (0.8, 1.4, 2.5)):
        original.update(groups(values))
    state = original.state_dict()

    resumed = ReLoBRaLoLossBalancer(
        temperature=0.1,
        alpha=0.999,
        expected_rho=0.5,
        seed=20260723,
    )
    resumed.load_state_dict(state)
    for values in ((0.7, 1.2, 2.1), (0.6, 1.0, 1.9), (0.5, 0.8, 1.7)):
        expected = original.update(groups(values))
        actual = resumed.update(groups(values))
        torch.testing.assert_close(
            torch.stack(list(actual.values())),
            torch.stack(list(expected.values())),
            rtol=0.0,
            atol=0.0,
        )
    assert resumed.state_dict()["step"] == original.state_dict()["step"]
    assert resumed.state_dict()["last_rho"] == original.state_dict()["last_rho"]


def test_formal_candidates_are_published_settings_and_add_no_physics() -> None:
    path = ROOT / "parameters/hccb_p418_loss_balancing_sources.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["physical_parameter_status"]["new_physical_parameters"] == []
    candidates = {
        row["candidate_id"]: row for row in payload["formal_candidates"]
    }
    assert candidates["fixed_equal_dimensionless"]["method"] == "fixed"
    assert (
        candidates["relobralo_burgers_table_viii"]["temperature"],
        candidates["relobralo_burgers_table_viii"]["alpha"],
        candidates["relobralo_burgers_table_viii"]["expected_rho"],
    ) == (0.1, 0.999, 0.9999)
    assert (
        candidates["relobralo_kirchhoff_table_viii"]["temperature"],
        candidates["relobralo_kirchhoff_table_viii"]["alpha"],
        candidates["relobralo_kirchhoff_table_viii"]["expected_rho"],
    ) == (0.01, 0.999, 0.9999)
    assert (
        candidates["relobralo_helmholtz_table_viii"]["temperature"],
        candidates["relobralo_helmholtz_table_viii"]["alpha"],
        candidates["relobralo_helmholtz_table_viii"]["expected_rho"],
    ) == (1.0e-5, 0.99, 0.99)
