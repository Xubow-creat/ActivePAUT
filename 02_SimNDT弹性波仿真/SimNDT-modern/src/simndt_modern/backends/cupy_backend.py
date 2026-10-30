from __future__ import annotations

from .base import BackendOutput, ProgressCallback, SnapshotCallback
from .vectorized import run_vectorized
from ..grid import PreparedSimulation


class CupyBackend:
    name = "CuPy CUDA GPU"

    def run(self, prepared: PreparedSimulation, progress: ProgressCallback | None = None,
            snapshot_steps: frozenset[int] | None = None,
            snapshot: SnapshotCallback | None = None) -> BackendOutput:
        import cupy as cp

        return run_vectorized(prepared, cp, progress, snapshot_steps, snapshot)


def available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
