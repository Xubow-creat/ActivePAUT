from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .backends.base import Backend, ProgressCallback
from .backends.numpy_backend import NumpyBackend
from .grid import PreparedSimulation, prepare_simulation
from .models import BackendKind, ProjectConfig


@dataclass(slots=True)
class SimulationResult:
    backend: str
    runtime_s: float
    nx: int
    ny: int
    dx_m: float
    dy_m: float
    dt_s: float
    time_s: NDArray[np.float32]
    receiver_pa: NDArray[np.float32]
    receiver_channels_pa: NDArray[np.float32]
    element_centers_x_m: NDArray[np.float64]
    transmit_delays_s: NDArray[np.float64]
    transmit_apodization: NDArray[np.float32]
    receive_apodization: NDArray[np.float32]
    field_magnitude: NDArray[np.float32]
    snapshot_steps: NDArray[np.int32]
    snapshot_time_s: NDArray[np.float32]
    snapshots_magnitude: NDArray[np.float32]


@dataclass(slots=True)
class FMCResult:
    backend: str
    runtime_s: float
    time_s: NDArray[np.float32]
    element_centers_x_m: NDArray[np.float64]
    data_pa: NDArray[np.float32]


def _load_backend(kind: BackendKind | str) -> Backend:
    kind = BackendKind(kind)
    if kind is BackendKind.NUMPY:
        return NumpyBackend()
    if kind is BackendKind.NUMBA:
        from .backends.numba_backend import NumbaBackend

        return NumbaBackend()
    if kind is BackendKind.CYTHON:
        from .backends.cython_backend import CythonBackend

        return CythonBackend()
    if kind is BackendKind.OPENCL:
        from .backends.opencl_backend import OpenCLBackend

        return OpenCLBackend()
    if kind is BackendKind.CUPY:
        from .backends.cupy_backend import CupyBackend

        return CupyBackend()
    raise ValueError(f"Unsupported backend: {kind}")


def backend_available(kind: BackendKind | str) -> bool:
    try:
        kind = BackendKind(kind)
        if kind is BackendKind.CUPY:
            from .backends.cupy_backend import available

            return available()
        if kind is BackendKind.OPENCL:
            from .backends.opencl_backend import available

            return available()
        if kind is BackendKind.CYTHON:
            from .backends.cython_backend import available

            return available()
        if kind is BackendKind.NUMBA:
            import numba  # noqa: F401
        return True
    except Exception:
        return False


def available_backends() -> list[BackendKind]:
    return [kind for kind in BackendKind if backend_available(kind)]


def run_prepared(
    prepared: PreparedSimulation,
    backend_kind: BackendKind | str,
    progress: ProgressCallback | None = None,
) -> SimulationResult:
    backend = _load_backend(backend_kind)
    requested_count = (
        prepared.config.numerics.snapshot_count
        if prepared.config.numerics.record_snapshots else 0
    )
    if requested_count:
        # Bound snapshot memory to 512 MiB. The selected steps remain evenly
        # distributed over the complete physical simulation duration.
        bytes_per_frame = prepared.nx * prepared.ny * np.dtype(np.float32).itemsize
        memory_limited_count = max(2, (512 * 1024 * 1024) // max(bytes_per_frame, 1))
        count = min(requested_count, prepared.time_s.size, memory_limited_count)
        selected = np.unique(
            np.linspace(0, prepared.time_s.size - 1, count, dtype=np.int32)
        )
        selected_set = frozenset(int(step) for step in selected)
    else:
        selected = np.empty(0, dtype=np.int32)
        selected_set = frozenset()
    captured_steps: list[int] = []
    captured_frames: list[NDArray[np.float32]] = []

    def capture(step: int, magnitude: NDArray[np.float32]) -> None:
        captured_steps.append(step)
        captured_frames.append(np.array(magnitude, dtype=np.float32, copy=True))

    started = perf_counter()
    output = backend.run(
        prepared,
        progress,
        selected_set,
        capture if selected_set else None,
    )
    runtime = perf_counter() - started
    snapshot_steps = np.asarray(captured_steps, dtype=np.int32)
    if captured_frames:
        snapshots = np.stack(captured_frames).astype(np.float32, copy=False)
        snapshot_times = prepared.time_s[snapshot_steps]
    else:
        snapshots = np.empty((0, prepared.ny, prepared.nx), dtype=np.float32)
        snapshot_times = np.empty(0, dtype=np.float32)
    return SimulationResult(
        backend=backend.name,
        runtime_s=runtime,
        nx=prepared.nx,
        ny=prepared.ny,
        dx_m=prepared.dx_m,
        dy_m=prepared.dy_m,
        dt_s=prepared.dt_s,
        time_s=prepared.time_s,
        receiver_pa=output.receiver_pa,
        receiver_channels_pa=output.receiver_channels_pa,
        element_centers_x_m=prepared.element_centers_x_m,
        transmit_delays_s=prepared.transmit_delays_s,
        transmit_apodization=prepared.transmit_apodization,
        receive_apodization=prepared.receive_apodization,
        field_magnitude=output.field_magnitude,
        snapshot_steps=snapshot_steps,
        snapshot_time_s=snapshot_times,
        snapshots_magnitude=snapshots,
    )


def run_simulation(
    config: ProjectConfig,
    progress: ProgressCallback | None = None,
) -> SimulationResult:
    prepared = prepare_simulation(config)
    return run_prepared(prepared, config.numerics.backend, progress)


def run_fmc(
    config: ProjectConfig,
    progress: ProgressCallback | None = None,
) -> FMCResult:
    """Acquire a full matrix capture by firing one physical element at a time."""
    config.validate()
    working = deepcopy(config)
    element_count = working.source.element_count
    working.numerics.record_snapshots = False
    working.source.transmit_apodization = "rectangular"
    working.source.receive_apodization = "rectangular"
    working.source.receive_active_elements = [True] * element_count
    matrix: NDArray[np.float32] | None = None
    centers: NDArray[np.float64] | None = None
    time_s: NDArray[np.float32] | None = None
    runtime = 0.0
    backend = ""
    for transmitter in range(element_count):
        working.source.transmit_active_elements = [
            index == transmitter for index in range(element_count)
        ]
        result = run_simulation(working)
        if matrix is None:
            matrix = np.empty(
                (element_count, element_count, result.time_s.size), dtype=np.float32
            )
            centers = result.element_centers_x_m.copy()
            time_s = result.time_s.copy()
        matrix[transmitter] = result.receiver_channels_pa
        runtime += result.runtime_s
        backend = result.backend
        if progress is not None:
            progress(transmitter + 1, element_count)
    assert matrix is not None and centers is not None and time_s is not None
    return FMCResult(backend, runtime, time_s, centers, matrix)


def tfm_image(
    fmc: FMCResult,
    x_m: NDArray[np.floating],
    y_m: NDArray[np.floating],
    wave_speed_m_s: float,
) -> NDArray[np.float32]:
    """Homogeneous-medium P-wave total focusing method reconstruction."""
    if wave_speed_m_s <= 0:
        raise ValueError("Wave speed must be positive.")
    x_axis = np.asarray(x_m, dtype=np.float64)
    y_axis = np.asarray(y_m, dtype=np.float64)
    image = np.zeros((y_axis.size, x_axis.size), dtype=np.float64)
    elements = fmc.element_centers_x_m
    for iy, y in enumerate(y_axis):
        for ix, x in enumerate(x_axis):
            distances = np.sqrt((elements - x) ** 2 + y**2)
            travel_times = (distances[:, None] + distances[None, :]) / wave_speed_m_s
            value = 0.0
            for tx in range(elements.size):
                for rx in range(elements.size):
                    value += np.interp(
                        travel_times[tx, rx], fmc.time_s, fmc.data_pa[tx, rx],
                        left=0.0, right=0.0,
                    )
            image[iy, ix] = value
    return image.astype(np.float32)
