from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code/verify_hccb_p418_relobralo_primary_sources.py"
SPEC = importlib.util.spec_from_file_location("verify_relobralo_sources", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


TABLE_TEXT = """
 Hyperparameter                 Burgers     Kirchhoff     Helmholtz
 Exponential Decay Rate alpha       0.999         0.999             0.99
 Temperature T                   10−1         10−2              10−5
 Expected Saudade E[rho]          0.9999       0.9999             0.99
 Table VIII: Final choices of hyperparameters
"""


def test_parse_table_viii_values() -> None:
    assert MODULE.parse_table_viii(TABLE_TEXT) == MODULE.EXPECTED_TABLE


def test_candidate_values_must_match_table_viii() -> None:
    payload = {
        "physical_parameter_status": {"new_physical_parameters": []},
        "formal_candidates": [
            {
                "candidate_id": f"relobralo_{problem}_table_viii",
                **settings,
            }
            for problem, settings in MODULE.EXPECTED_TABLE.items()
        ],
    }
    result = MODULE.verify_candidate_file(payload, MODULE.EXPECTED_TABLE)
    assert result["candidate_count"] == 3
    assert all(result["table_viii_candidates_match"].values())


def test_official_code_semantics_are_required() -> None:
    update = """
    losses[i]/(args['l'+str(i)]*T+1e-12)
    losses[i]/(args['l0'+str(i)]*T+1e-12)
    args['rho']*args['alpha']*args['lam'+str(i)] + \
    (1-args['rho'])*args['alpha']*lambs0_hat[i] + \
    (1-args['alpha'])*lambs_hat[i]
    """
    train = """
    rho = (np.random.uniform(size=meta_args.epochs+1) < meta_args.rho).astype(int).astype(np.float32)
    if (meta_args.update_rule == 'gradnorm' or meta_args.update_rule == 'relobralo') and epoch == 1:
        args['l0'+str(i)] = ([f_loss]+b_losses)[i]
    args['rho'] = rho[1]
    """
    checks = MODULE.verify_official_code(update, train)
    assert all(checks.values())
