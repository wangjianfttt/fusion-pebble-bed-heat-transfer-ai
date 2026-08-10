#!/usr/bin/env python3
"""Literature parameter groups used by the P418 steady CHT study."""

from __future__ import annotations


CASE_PHYSICS_PARAMETER_IDS = (
    "P070",
    "P071",
    "P092",
    "P388",
    "P389",
    "P403",
    "P406",
    "P418",
    "P424",
    "P425",
    "P426",
    "P427",
)

OPERATING_PARAMETER_IDS = (
    "P418",
    "P424",
    "P425",
    "P426",
    "P427",
)

# These rows reproduce the published packing sequence: construct the P050 bed,
# extract the P390 wall-adjacent region following P423, and reduce the particle
# diameter by P404 only for meshing. The later fine local crop and its realized
# triangulated porosity remain numerical geometry results reported separately.
MESH_GEOMETRY_SOURCE_PARAMETER_IDS = (
    "P048",
    "P049",
    "P050",
    "P390",
    "P404",
    "P423",
)

ALL_STEADY_PHYSICAL_PARAMETER_IDS = tuple(
    dict.fromkeys((*MESH_GEOMETRY_SOURCE_PARAMETER_IDS, *CASE_PHYSICS_PARAMETER_IDS))
)
