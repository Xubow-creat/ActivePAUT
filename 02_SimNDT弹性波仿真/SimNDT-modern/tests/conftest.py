from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from simndt_modern.models import ProjectConfig


@pytest.fixture
def small_config() -> ProjectConfig:
    config = ProjectConfig()
    config.numerics.width_mm = 8.0
    config.numerics.height_mm = 8.0
    config.numerics.duration_us = 1.5
    config.numerics.points_per_wavelength = 8
    config.source.center_frequency_hz = 0.75e6
    config.source.center_x_mm = 4.0
    config.source.aperture_mm = 1.5
    config.defect.center_x_mm = 4.0
    config.defect.center_y_mm = 4.0
    config.defect.radius_mm = 0.7
    for side in (config.boundaries.top, config.boundaries.bottom,
                 config.boundaries.left, config.boundaries.right):
        side.thickness_mm = 1.0
    return config

