#!/usr/bin/env python3
"""Build compact, path-free data tables used by three manuscript figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PHYSICAL_SOURCE = Path(
    "results/hccb_p418_sourceflow_complete_physics_60/completed_case_physics.csv"
)
SEED202_CSV_SOURCE = Path(
    "results/hccb_p418_cross_packing_seed202_integral_9/paired_integral_differences.csv"
)
SEED202_SUMMARY_SOURCE = Path(
    "results/hccb_p418_cross_packing_seed202_integral_9/summary.json"
)
STEADY_SOURCE = Path(
    "results/hccb_p418_60_corrected_20260731_model_comparison_100epoch/model_comparison.csv"
)
TRANSPORT_CHECK_SOURCE = Path(
    "results/hccb_p418_helium_transport_lookup_20260802/openfoam13_direct_transport_build.json"
)
SCOPE_SUMMARY_SOURCE = Path(
    "results/hccb_p418_scope_limits_20260730/scope_limits_summary.json"
)
PRIVATE_TEXT = (
    "/" + "Users/",
    "/" + "data2/",
    "/" + "n96pfs/",
    "192" + ".168.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if not fieldnames or not rows:
        raise ValueError(f"empty processed table: {path}")
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ensure_no_private_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    found = [token for token in PRIVATE_TEXT if token in text]
    if found:
        raise ValueError(f"private machine path remains in {path}: {found}")


def scrub_private_paths(value):
    if isinstance(value, dict):
        return {key: scrub_private_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_private_paths(item) for item in value]
    if isinstance(value, str) and any(token in value for token in PRIVATE_TEXT):
        return "PRIVATE_COMPUTE_PATH_REMOVED"
    return value


def build(project_root: Path, output_dir: Path) -> dict[str, object]:
    sources = {
        "physical_response": project_root / PHYSICAL_SOURCE,
        "seed202_comparison": project_root / SEED202_CSV_SOURCE,
        "seed202_summary": project_root / SEED202_SUMMARY_SOURCE,
        "steady_model_comparison": project_root / STEADY_SOURCE,
        "transport_check": project_root / TRANSPORT_CHECK_SOURCE,
        "scope_summary": project_root / SCOPE_SUMMARY_SOURCE,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"processed figure sources are missing: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    physical_fields, physical_rows = read_csv(sources["physical_response"])
    seed_fields, seed_rows = read_csv(sources["seed202_comparison"])
    steady_fields, steady_rows = read_csv(sources["steady_model_comparison"])
    expected_counts = (len(physical_rows), len(seed_rows), len(steady_rows))
    if expected_counts != (60, 9, 25):
        raise ValueError(f"unexpected figure-data row counts: {expected_counts}")

    physical_out = output_dir / "physical_response_60.csv"
    seed_csv_out = output_dir / "seed202_integral_comparison_9.csv"
    seed_summary_out = output_dir / "seed202_integral_summary.json"
    steady_out = output_dir / "steady_model_comparison_5x5.csv"
    transport_out = output_dir / "openfoam13_direct_transport_build_public.json"
    scope_dir = output_dir / "scope_limits_public"
    write_csv(physical_out, physical_fields, physical_rows)
    write_csv(seed_csv_out, seed_fields, seed_rows)

    seed_summary = json.loads(sources["seed202_summary"].read_text(encoding="utf-8"))
    public_seed_summary = {
        "status": seed_summary["status"],
        "accepted_common_case_count": seed_summary["accepted_common_case_count"],
        "failed_seed202_case_count": seed_summary["failed_seed202_case_count"],
        "registered_case_count": seed_summary["registered_case_count"],
        "complete_nine_case_comparison": seed_summary["complete_nine_case_comparison"],
        "failed_seed202_cases": seed_summary.get("failed_seed202_cases", []),
        "metric_summary": seed_summary.get("metric_summary", {}),
        "new_physical_parameters": [],
        "source_paths_removed_for_public_release": True,
    }
    seed_summary_out.write_text(
        json.dumps(public_seed_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    public_steady_fields = [name for name in steady_fields if name != "summary_file"]
    public_steady_rows = [
        {name: row[name] for name in public_steady_fields} for row in steady_rows
    ]
    write_csv(steady_out, public_steady_fields, public_steady_rows)

    transport = json.loads(sources["transport_check"].read_text(encoding="utf-8"))
    public_transport = {
        key: value
        for key, value in transport.items()
        if key not in {"workstation", "library_path", "check_executable_path"}
    }
    public_transport["private_machine_paths_removed_for_public_release"] = True
    transport_out.write_text(
        json.dumps(public_transport, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    raw_scope_summary = json.loads(sources["scope_summary"].read_text(encoding="utf-8"))
    scope_source_dir = sources["scope_summary"].parent
    scope_dir.mkdir(parents=True, exist_ok=True)
    public_scope_rows = []
    scope_outputs: list[Path] = []
    for row in raw_scope_summary["records"]:
        name = str(row["filename"])
        source_path = scope_source_dir / name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        public_path = scope_dir / name
        public_payload = scrub_private_paths(
            json.loads(source_path.read_text(encoding="utf-8"))
        )
        public_path.write_text(
            json.dumps(public_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        public_row = scrub_private_paths(dict(row))
        public_row["size_bytes"] = public_path.stat().st_size
        public_row["sha256"] = sha256(public_path)
        public_scope_rows.append(public_row)
        scope_outputs.append(public_path)
    public_scope_summary = scrub_private_paths(dict(raw_scope_summary))
    public_scope_summary["destination"] = "scope_limits_public"
    public_scope_summary["records"] = public_scope_rows
    public_scope_summary["private_machine_paths_removed_for_public_release"] = True
    public_scope_summary_path = scope_dir / "scope_limits_summary.json"
    public_scope_summary_path.write_text(
        json.dumps(public_scope_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scope_outputs.append(public_scope_summary_path)

    readme = output_dir / "README.md"
    readme.write_text(
        """# Plot-ready processed data

These compact tables reproduce three quantitative figures without raw
OpenFOAM fields or machine-local paths. From the archive root, run:

```bash
python code/plot_hccb_p418_physical_response.py \
  --physical-csv results/hccb_p418_public_figure_data/physical_response_60.csv \
  --output-dir reproduced_figures
python code/plot_hccb_p418_seed202_integral_partial.py \
  --comparison-csv results/hccb_p418_public_figure_data/seed202_integral_comparison_9.csv \
  --summary-json results/hccb_p418_public_figure_data/seed202_integral_summary.json \
  --output-dir reproduced_figures
python code/plot_hccb_p418_steady_model_comparison.py \
  --comparison-csv results/hccb_p418_public_figure_data/steady_model_comparison_5x5.csv \
  --output-dir reproduced_figures
```

The domain rendering, three-dimensional cloud plots, and final transient
prediction panels require larger geometry or prediction arrays. They are kept
in the citable processed-data archive rather than this small source package.
""",
        encoding="utf-8",
    )

    public_files = (
        physical_out,
        seed_csv_out,
        seed_summary_out,
        steady_out,
        transport_out,
        *scope_outputs,
        readme,
    )
    for path in public_files:
        ensure_no_private_text(path)
    outputs = {
        path.relative_to(output_dir).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in public_files
    }
    payload: dict[str, object] = {
        "status": "completed_p418_public_figure_data",
        "figure_count_reproducible_from_small_package": 3,
        "row_counts": {
            "physical_response": len(physical_rows),
            "seed202_integral_comparison": len(seed_rows),
            "steady_model_comparison": len(steady_rows),
        },
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "outputs": outputs,
        "private_machine_paths_removed": True,
        "raw_openfoam_fields_included": False,
        "new_physical_parameters": [],
    }
    summary_out = output_dir / "summary.json"
    summary_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ensure_no_private_text(summary_out)
    return payload


def verify_existing(output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed_p418_public_figure_data":
        raise ValueError("public figure-data summary is incomplete")
    for name, record in payload.get("outputs", {}).items():
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"public figure-data size changed: {name}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"public figure-data SHA-256 changed: {name}")
        ensure_no_private_text(path)
    ensure_no_private_text(summary_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    payload = (
        verify_existing(output_dir)
        if args.verify_existing
        else build(args.project_root.resolve(), output_dir)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
