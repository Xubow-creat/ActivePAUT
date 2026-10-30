from __future__ import annotations

import numpy as np

from .base import BackendOutput, ProgressCallback, SnapshotCallback
from .vectorized import run_vectorized
from ..grid import PreparedSimulation


def available() -> bool:
    try:
        import pyopencl as cl

        return any(platform.get_devices() for platform in cl.get_platforms())
    except Exception:
        return False


class OpenCLBackend:
    """Compatibility path pending revalidation of a native CPML kernel."""

    name = "PyOpenCL compatibility (strict NumPy core)"

    def run(
        self,
        prepared: PreparedSimulation,
        progress: ProgressCallback | None = None,
        snapshot_steps: frozenset[int] | None = None,
        snapshot: SnapshotCallback | None = None,
    ) -> BackendOutput:
        return run_vectorized(prepared, np, progress, snapshot_steps, snapshot)
