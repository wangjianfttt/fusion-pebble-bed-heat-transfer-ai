#!/usr/bin/env python3
"""Record the software environment used by the formal P418 calculations."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit_learn": "sklearn",
    "torch": "torch",
    "matplotlib": "matplotlib",
    "numexpr": "numexpr",
    "bottleneck": "bottleneck",
}


def package_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def openfoam_version() -> str | None:
    executable = shutil.which("foamVersion")
    if executable is None:
        return os.environ.get("WM_PROJECT_VERSION")
    completed = subprocess.run(
        [executable], check=False, capture_output=True, text=True, timeout=30
    )
    value = completed.stdout.strip()
    return value or os.environ.get("WM_PROJECT_VERSION")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    payload = {
        "status": "hccb_p418_formal_runtime_environment_recorded",
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).absolute()),
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "platform": platform.platform(),
        "packages": {
            label: package_version(module_name)
            for label, module_name in PACKAGES.items()
        },
        "cuda_available": cuda_available,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "openfoam_version": openfoam_version(),
        "new_physical_parameters": [],
        "note": (
            "Software versions are numerical-environment information and are "
            "kept separate from literature-derived physical parameters."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
