from __future__ import annotations

import numpy as np

from .base import BackendOutput, ProgressCallback, SnapshotCallback
from .vectorized import run_vectorized
from ..grid import PreparedSimulation


class NumbaBackend:
    """Compatibility path for the strict staggered-grid solver."""

    name = "Numba compatibility (strict NumPy core)"

    def run(
        self,
        prepared: PreparedSimulation,
        progress: ProgressCallback | None = None,
        snapshot_steps: frozenset[int] | None = None,
        snapshot: SnapshotCallback | None = None,
    ) -> BackendOutput:
        return run_vectorized(prepared, np, progress, snapshot_steps, snapshot)
