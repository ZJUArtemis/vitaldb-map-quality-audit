#!/usr/bin/env python3
"""Repository-relative path configuration for the analysis scripts."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def configured_path(variable: str, default: Path) -> Path:
    """Return an optional environment override or a repository-relative default."""
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


PROJECT_ROOT = configured_path("TOPIC10_PROJECT_ROOT", REPOSITORY_ROOT)
VITAL_DIR = configured_path("VITALDB_VITAL_DIR", PROJECT_ROOT / "vitaldb_data")

# Legacy scripts address data from the PhysioNet root. Users may set this
# explicitly; otherwise infer it from .../files/vitaldb/1.0.0/vital_files.
if "VITALDB_PHYSIONET_ROOT" in os.environ:
    PHYSIONET_ROOT = configured_path(
        "VITALDB_PHYSIONET_ROOT", PROJECT_ROOT / "physionet.org"
    )
elif VITAL_DIR.name == "vital_files" and len(VITAL_DIR.parents) >= 4:
    PHYSIONET_ROOT = VITAL_DIR.parents[3]
else:
    PHYSIONET_ROOT = PROJECT_ROOT / "physionet.org"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RI_OUTPUT_DIR = configured_path(
    "TOPIC10_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "ri_v14"
)
