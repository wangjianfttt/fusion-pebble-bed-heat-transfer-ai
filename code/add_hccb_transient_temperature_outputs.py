#!/usr/bin/env python3
"""Add volume-average fluid and solid temperatures to a multi-region CHT case."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


BLOCK = '''fluidTemperatureVolumeAverage
{
    type volFieldValue;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    cellZone all;
    operation volAverage;
    log true;
    writeFields false;
    fields (T);
}
solidTemperatureVolumeAverage
{
    type volFieldValue;
    libs ("libfieldFunctionObjects.so");
    region solid;
    cellZone all;
    operation volAverage;
    log true;
    writeFields false;
    fields (T);
}
'''


def disable_wall_heat_flux_field_writes(text: str, path: Path) -> tuple[str, bool]:
    pattern = re.compile(
        r"(?ms)(^\s*fluidWallHeatFlux\s*\n\s*\{\s*\n)(.*?)(^\s*\}\s*$)"
    )
    match = pattern.search(text)
    if match is None:
        return text, False
    body = match.group(2)
    if re.search(r"(?m)^\s*writeFields\s+false\s*;", body):
        return text, False
    if re.search(r"(?m)^\s*writeFields\s+\w+\s*;", body):
        body = re.sub(
            r"(?m)^(\s*writeFields\s+)\w+(\s*;)",
            r"\g<1>false\g<2>",
            body,
            count=1,
        )
    else:
        body += "    writeFields false;\n"
    replacement = match.group(1) + body + match.group(3)
    return text[: match.start()] + replacement + text[match.end() :], True


def add_outputs(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    text, wall_heat_changed = disable_wall_heat_flux_field_writes(text, path)
    names = ("fluidTemperatureVolumeAverage", "solidTemperatureVolumeAverage")
    present = [bool(re.search(rf"(?m)^\s*{name}\s*$", text)) for name in names]
    if all(present):
        changed = wall_heat_changed
        for name in names:
            pattern = re.compile(
                rf"(?ms)(^\s*{name}\s*\n\s*\{{\s*\n)(.*?)(^\s*\}}\s*$)"
            )
            match = pattern.search(text)
            if match is None:
                raise ValueError(f"cannot parse {name} in {path}")
            if re.search(r"(?m)^\s*type\s+volFieldValue\s*;", match.group(2)):
                continue
            text = pattern.sub(
                rf"\g<1>    type volFieldValue;\n\g<2>\g<3>",
                text,
                count=1,
            )
            changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
        return changed
    if any(present):
        raise ValueError(f"only one transient temperature output is present in {path}")
    anchor = "solidTemperatureMaximum\n{"
    if anchor not in text:
        raise ValueError(f"cannot find insertion point in {path}")
    path.write_text(text.replace(anchor, BLOCK + anchor, 1), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    add_outputs(args.case.resolve() / "system/functions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
