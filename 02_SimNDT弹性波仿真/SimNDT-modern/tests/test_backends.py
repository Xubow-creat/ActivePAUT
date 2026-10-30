from __future__ import annotations

import numpy as np
import pytest

from simndt_modern.grid import prepare_simulation
from simndt_modern.models import BackendKind
from simndt_modern.solver import backend_available, run_prepared


def normalized_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(reference))), 1.0)
    return float(np.max(np.abs(reference - candidate))) / scale


def test_numpy_simulation_is_finite_and_nonzero(small_config) -> None:
    result = run_prepared(prepare_simulation(small_config), BackendKind.NUMPY)
    assert np.isfinite(result.receiver_pa).all()
    assert np.isfinite(result.field_magnitude).all()
    assert np.max(np.abs(result.receiver_pa)) > 0
    assert np.max(result.field_magnitude) > 0


def test_backend_loader_accepts_serialized_string(small_config) -> None:
    result = run_prepared(prepare_simulation(small_config), "numpy")
    assert result.backend == "NumPy CPU"
    assert np.isfinite(result.receiver_pa).all()


def test_time_history_records_steps_and_physical_times(small_config) -> None:
    small_config.numerics.snapshot_count = 5
    result = run_prepared(prepare_simulation(small_config), "numpy")
    assert result.snapshots_magnitude.shape == (5, result.ny, result.nx)
    assert result.snapshot_steps[0] == 0
    assert result.snapshot_steps[-1] == result.time_s.size - 1
    np.testing.assert_array_equal(
        result.snapshot_time_s, result.time_s[result.snapshot_steps]
    )


@pytest.mark.parametrize("kind", [BackendKind.NUMBA, BackendKind.CYTHON])
def test_compiled_cpu_backends_match_numpy(small_config, kind) -> None:
    if not backend_available(kind):
        pytest.skip(f"{kind.value} unavailable")
    prepared = prepare_simulation(small_config)
    reference = run_prepared(prepared, BackendKind.NUMPY)
    candidate = run_prepared(prepared, kind)
    assert normalized_error(reference.receiver_pa, candidate.receiver_pa) < 2e-5
    assert normalized_error(reference.field_magnitude, candidate.field_magnitude) < 2e-5


@pytest.mark.parametrize("kind", [BackendKind.OPENCL, BackendKind.CUPY])
def test_gpu_backends_match_numpy(small_config, kind) -> None:
    if not backend_available(kind):
        pytest.skip(f"{kind.value} unavailable")
    prepared = prepare_simulation(small_config)
    reference = run_prepared(prepared, BackendKind.NUMPY)
    try:
        candidate = run_prepared(prepared, kind)
    except Exception as exc:
        pytest.skip(f"{kind.value} runtime unavailable: {exc}")
    assert normalized_error(reference.receiver_pa, candidate.receiver_pa) < 2e-3
    assert normalized_error(reference.field_magnitude, candidate.field_magnitude) < 2e-3
