"""SimNDT Modern: a clean Python 3 implementation of a 2-D elastic NDT simulator."""

from .models import BackendKind, BoundaryKind, ProjectConfig, SourcePolarization
from .solver import (
    FMCResult, SimulationResult, available_backends, run_fmc, run_simulation,
    tfm_image,
)

__all__ = [
    "BackendKind",
    "BoundaryKind",
    "ProjectConfig",
    "SourcePolarization",
    "SimulationResult",
    "FMCResult",
    "available_backends",
    "run_simulation",
    "run_fmc",
    "tfm_image",
]

__version__ = "2.0.0"
