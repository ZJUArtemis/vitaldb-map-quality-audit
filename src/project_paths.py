#!/usr/bin/env python3
"""Portable path configuration shared by the public analysis scripts."""

from __future__ import annotations

import os
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("TOPIC10_PROJECT_ROOT", ARCHIVE_ROOT)).expanduser().resolve()
VITAL_DIR = Path(
    os.environ.get("VITALDB_VITAL_DIR", PROJECT_ROOT / "vitaldb_data")
).expanduser().resolve()

# Legacy scripts address data from the PhysioNet root. Users may set this
# explicitly; otherwise infer it from .../files/vitaldb/1.0.0/vital_files.
if "VITALDB_PHYSIONET_ROOT" in os.environ:
    PHYSIONET_ROOT = Path(os.environ["VITALDB_PHYSIONET_ROOT"]).expanduser().resolve()
elif VITAL_DIR.name == "vital_files" and len(VITAL_DIR.parents) >= 4:
    PHYSIONET_ROOT = VITAL_DIR.parents[3]
else:
    PHYSIONET_ROOT = PROJECT_ROOT / "physionet.org"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
