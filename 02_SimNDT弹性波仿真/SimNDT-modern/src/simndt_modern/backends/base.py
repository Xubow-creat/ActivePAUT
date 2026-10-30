from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from ..grid import PreparedSimulation

ProgressCallback = Callable[[int, int], None]
SnapshotCallback = Callable[[int, NDArray[np.float32]], None]


@dataclass(slots=True)
class BackendOutput:
    receiver_pa: NDArray[np.float32]
    field_magnitude: NDArray[np.float32]
    receiver_channels_pa: NDArray[np.float32]


class Backend(Protocol):
    name: str

    def run(
        self,
        prepared: PreparedSimulation,
        progress: ProgressCallback | None = None,
        snapshot_steps: frozenset[int] | None = None,
        snapshot: SnapshotCallback | None = None,
    ) -> BackendOutput:
        ...
