#!/usr/bin/env python3
"""Verify recorded ReLoBRaLo candidates against the paper and official code."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


PAPER_PDF_SHA256 = (
    "24583618240a400d4d0c1993dc8d25a46c0d3ebefcb0c9dedfc2cb32e850f04b"
)
OFFICIAL_UPDATE_RULES_SHA256 = (
    "e8dc6bec9aca0f0821dc805f57c8ab4ce728c77bdad49d0e033833613b4e42d1"
)
OFFICIAL_TRAIN_SHA256 = (
    "1af5a43211aed20e3eadc61ff61f2f1acffb39f9c0f065673d173ab80e9e2551"
)
OFFICIAL_COMMIT = "b3c76d2bed7c6bebb2e2628575008a04858472cf"
EXPECTED_TABLE = {
    "burgers": {"temperature": 0.1, "alpha": 0.999, "expected_rho": 0.9999},
    "kirchhoff": {
        "temperature": 0.01,
        "alpha": 0.999,
        "expected_rho": 0.9999,
    },
    "helmholtz": {"temperature": 1.0e-5, "alpha": 0.99, "expected_rho": 0.99},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(token: str) -> float:
    token = token.strip().replace("−", "-")
    if token.startswith("10-"):
        return 10.0 ** int(token[2:])
    return float(token)


def parse_table_viii(text: str) -> dict[str, dict[str, float]]:
    """Extract T, alpha and E[rho] from the three Table VIII columns."""
    start = text.find("Hyperparameter                 Burgers")
    end = text.find("Table VIII:", start)
    if start < 0 or end < 0:
        raise ValueError("Table VIII block was not found in the paper text")
    block = text[start:end]
    patterns = {
        "alpha": r"Exponential Decay Rate[^\n]*?([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        "temperature": r"Temperature T\s+(10[−-][0-9]+)\s+(10[−-][0-9]+)\s+(10[−-][0-9]+)",
        "expected_rho": r"Expected Saudade[^\n]*?([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
    }
    columns = ("burgers", "kirchhoff", "helmholtz")
    parsed = {name: {} for name in columns}
    for setting, pattern in patterns.items():
        match = re.search(pattern, block)
        if not match:
            raise ValueError(f"Table VIII row is missing: {setting}")
        for column, token in zip(columns, match.groups()):
            parsed[column][setting] = _number(token)
    return parsed


def verify_official_code(update_rules: str, train: str) -> dict[str, bool]:
    compact_update = re.sub(r"\s+", "", update_rules)
    compact_train = re.sub(r"\s+", "", train)
    checks = {
        "current_to_previous_relative_loss": (
            "losses[i]/(args['l'+str(i)]*T+1e-12)" in compact_update
        ),
        "current_to_initial_relative_loss": (
            "losses[i]/(args['l0'+str(i)]*T+1e-12)" in compact_update
        ),
        "equation_11_rho_alpha_combination": (
            "args['rho']*args['alpha']*args['lam'+str(i)]"
            "+(1-args['rho'])*args['alpha']*lambs0_hat[i]"
            "+(1-args['alpha'])*lambs_hat[i]" in compact_update
        ),
        "bernoulli_rho_sampling": (
            "(np.random.uniform(size=meta_args.epochs+1)<meta_args.rho)"
            ".astype(int).astype(np.float32)" in compact_train
        ),
        "rho_advanced_each_epoch": "args['rho']=rho[1]" in compact_train,
        "initial_losses_recorded_after_second_update": (
            "if(meta_args.update_rule=='gradnorm'or"
            "meta_args.update_rule=='relobralo')andepoch==1:" in compact_train
            and "args['l0'+str(i)]=([f_loss]+b_losses)[i]" in compact_train
        ),
    }
    if not all(checks.values()):
        missing = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"official-code checks failed: {missing}")
    return checks


def verify_candidate_file(
    payload: dict[str, object],
    table: dict[str, dict[str, float]],
) -> dict[str, object]:
    if payload["physical_parameter_status"]["new_physical_parameters"] != []:
        raise ValueError("loss-balancing candidates introduced physical parameters")
    candidates = {
        str(row["candidate_id"]): row for row in payload["formal_candidates"]
    }
    matches: dict[str, bool] = {}
    for problem, expected in table.items():
        candidate_id = f"relobralo_{problem}_table_viii"
        if candidate_id not in candidates:
            raise ValueError(f"candidate is missing: {candidate_id}")
        candidate = candidates[candidate_id]
        for setting, value in expected.items():
            actual = float(candidate[setting])
            if not math.isclose(actual, value, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(
                    f"{candidate_id} {setting}: candidate={actual}, paper={value}"
                )
        matches[candidate_id] = True
    return {
        "candidate_count": len(payload["formal_candidates"]),
        "table_viii_candidates_match": matches,
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--paper-pdf", type=Path, required=True)
    parser.add_argument("--paper-text", type=Path, required=True)
    parser.add_argument("--official-update-rules", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "candidate_file": args.candidate_file.resolve(),
        "paper_pdf": args.paper_pdf.resolve(),
        "paper_text": args.paper_text.resolve(),
        "official_update_rules": args.official_update_rules.resolve(),
        "official_train": args.official_train.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    expected_hashes = {
        "paper_pdf": PAPER_PDF_SHA256,
        "official_update_rules": OFFICIAL_UPDATE_RULES_SHA256,
        "official_train": OFFICIAL_TRAIN_SHA256,
    }
    for name, expected in expected_hashes.items():
        if source_hashes[name] != expected:
            raise ValueError(
                f"source SHA differs for {name}: {source_hashes[name]} != {expected}"
            )

    paper_text = paths["paper_text"].read_text(encoding="utf-8")
    table = parse_table_viii(paper_text)
    if table != EXPECTED_TABLE:
        raise ValueError(f"parsed Table VIII differs from registered values: {table}")
    code_checks = verify_official_code(
        paths["official_update_rules"].read_text(encoding="utf-8"),
        paths["official_train"].read_text(encoding="utf-8"),
    )
    candidate_payload = json.loads(
        paths["candidate_file"].read_text(encoding="utf-8")
    )
    candidate_checks = verify_candidate_file(candidate_payload, table)

    output = {
        "status": "verified_against_relobralo_primary_sources",
        "paper": {
            "title": "Multi-Objective Loss Balancing for Physics-Informed Deep Learning",
            "authors": "Rafael Bischof and Michael A. Kraus",
            "arxiv": "2110.09813",
            "doi": "10.1016/j.cma.2025.117914",
            "version_checked": "arXiv version dated 16 November 2022",
            "table_checked": "Table VIII",
            "equation_checked": "Equation (11)",
        },
        "official_repository": {
            "url": "https://github.com/rbischof/relative_balancing",
            "commit": OFFICIAL_COMMIT,
            "update_rules_file": "src/update_rules.py",
            "training_file": "src/train.py",
        },
        "source_files": {name: path.name for name, path in paths.items()},
        "retrieval_urls": {
            "paper": "https://arxiv.org/pdf/2110.09813",
            "official_repository": (
                "https://github.com/rbischof/relative_balancing/tree/"
                f"{OFFICIAL_COMMIT}"
            ),
        },
        "source_sha256": source_hashes,
        "table_viii": table,
        "official_code_checks": code_checks,
        "candidate_file_checks": candidate_checks,
        "interpretation": (
            "The three published settings are optimisation candidates only. "
            "They do not change the pebble-bed geometry, material properties, "
            "boundary conditions or OpenFOAM data, and they still require the "
            "predeclared validation-only comparison on the P418 problem."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
