from __future__ import annotations

import numpy as np
import pytest

from simndt_modern.grid import prepare_simulation
from simndt_modern.models import BoundaryKind, ProjectConfig, SourcePolarization
from simndt_modern.signals import delay_and_sum
from simndt_modern.signals import focusing_delays_s
from simndt_modern.solver import FMCResult, run_fmc, run_simulation, tfm_image
from simndt_modern.validation import (
    array_quality_summary,
    far_field_array_factor,
    focus_arrival_spread_s,
)


def test_physical_element_mapping_conserves_width() -> None:
    prepared = prepare_simulation(ProjectConfig())
    summary = array_quality_summary(prepared)
    assert summary["maximum_element_width_error_m"] < 2.0e-10


def test_geometrical_focusing_has_machine_precision_arrival_match() -> None:
    config = ProjectConfig()
    config.source.steering_angle_deg = 18.0
    config.source.focus_depth_mm = 12.0
    prepared = prepare_simulation(config)
    assert focus_arrival_spread_s(prepared) < 1.0e-18


def test_focus_delays_match_kwave_linear_array_reference_formula() -> None:
    count, pitch, focus, speed = 15, 0.5e-3, 20.0e-3, 1500.0
    ids = np.arange(count) - (count - 1) / 2
    centers = ids * pitch
    # Formula used in k-Wave's Modelling A Linear Array Transducer example.
    kwave = -(np.sqrt((ids * pitch) ** 2 + focus**2) - focus) / speed
    kwave -= np.min(kwave)
    ours = focusing_delays_s(centers, 0.0, 0.0, focus, speed)
    np.testing.assert_allclose(ours, kwave, rtol=0.0, atol=1.0e-18)


def test_plane_wave_delay_law_steers_array_factor() -> None:
    config = ProjectConfig()
    config.source.focus_depth_mm = 0.0
    config.source.steering_angle_deg = 17.0
    config.source.transmit_apodization = "rectangular"
    prepared = prepare_simulation(config)
    angles = np.linspace(-45.0, 45.0, 3601)
    pattern = far_field_array_factor(prepared, angles)
    measured = float(angles[int(np.argmax(pattern))])
    assert measured == pytest.approx(17.0, abs=0.05)


def test_fractional_delay_and_sum_aligns_synthetic_echoes() -> None:
    time = np.linspace(0.0, 8.0e-6, 4001)
    delays = np.array([0.0, 0.137e-6, 0.311e-6])
    echo_time = 3.0e-6
    channels = np.stack([
        np.exp(-0.5 * ((time - (echo_time - delay)) / 0.08e-6) ** 2)
        for delay in delays
    ])
    focused = delay_and_sum(channels, time, delays, np.ones(3))
    assert time[int(np.argmax(focused))] == pytest.approx(echo_time, abs=time[1])
    assert float(np.max(focused)) > 0.999


@pytest.mark.parametrize(
    ("polarization", "expected_speed"),
    [
        (SourcePolarization.NORMAL, 5850.0),
        (SourcePolarization.TANGENTIAL, 3220.0),
    ],
)
def test_numerical_plane_wave_speed(polarization, expected_speed) -> None:
    config = ProjectConfig()
    config.defect.enabled = False
    config.source.element_count = 1
    config.source.aperture_mm = 18.0
    config.source.cycles = 2.0
    config.source.polarization = polarization
    config.boundaries.bottom.thickness_mm = 2.0
    config.numerics.duration_us = 4.0
    config.numerics.snapshot_count = 1000
    result = run_simulation(config)
    x = result.nx // 2
    y1 = round(4.0e-3 / result.dy_m)
    y2 = round(8.0e-3 / result.dy_m)
    first = result.snapshots_magnitude[:, y1, x]
    second = result.snapshots_magnitude[:, y2, x]
    def interpolated_crossing(trace: np.ndarray) -> float:
        # A high fractional crossing tracks the same dominant tone-burst lobe
        # at both depths and suppresses the finite-aperture precursor.
        threshold = 0.70 * float(np.max(trace))
        index = int(np.flatnonzero(trace >= threshold)[0])
        before, after = float(trace[index - 1]), float(trace[index])
        return index - 1 + (threshold - before) / (after - before)

    i1 = interpolated_crossing(first)
    i2 = interpolated_crossing(second)
    measured = (y2 - y1) * result.dy_m / ((i2 - i1) * result.dt_s)
    assert measured == pytest.approx(expected_speed, rel=0.005)


def test_contact_array_rejects_absorbing_top_boundary() -> None:
    config = ProjectConfig()
    config.boundaries.top.kind = BoundaryKind.ABSORBING
    with pytest.raises(ValueError, match="top Air/free-surface"):
        config.validate()


def test_cpml_plane_wave_reflection_is_below_minus_50_db() -> None:
    config = ProjectConfig()
    config.defect.enabled = False
    config.source.element_count = 1
    config.source.aperture_mm = 20.0
    config.source.cycles = 2.0
    config.boundaries.bottom.thickness_mm = 8.0
    config.numerics.duration_us = 9.0
    config.numerics.snapshot_count = 1000
    result = run_simulation(config)
    trace = result.snapshots_magnitude[:, round(10.0e-3 / result.dy_m), result.nx // 2]
    times = result.snapshot_time_s
    direct = float(np.max(trace[(times > 2.0e-6) & (times < 3.5e-6)]))
    reflected = float(np.max(trace[(times > 5.0e-6) & (times < 8.0e-6)]))
    reflection_db = 20.0 * np.log10(reflected / direct)
    assert reflection_db < -50.0


def test_three_resolution_p_wave_grid_convergence() -> None:
    errors = []
    for ppw in (8, 12, 16):
        config = ProjectConfig()
        config.defect.enabled = False
        config.source.element_count = 1
        config.source.aperture_mm = 18.0
        config.source.cycles = 2.0
        config.boundaries.bottom.thickness_mm = 2.0
        config.numerics.points_per_wavelength = ppw
        config.numerics.duration_us = 3.0
        config.numerics.snapshot_count = 1000
        result = run_simulation(config)
        x = result.nx // 2
        y1 = round(3.0e-3 / result.dy_m)
        y2 = round(6.0e-3 / result.dy_m)
        traces = [
            result.snapshots_magnitude[:, y1, x],
            result.snapshots_magnitude[:, y2, x],
        ]
        crossings = []
        for trace in traces:
            threshold = 0.70 * float(np.max(trace))
            index = int(np.flatnonzero(trace >= threshold)[0])
            before, after = float(trace[index - 1]), float(trace[index])
            crossings.append(index - 1 + (threshold - before) / (after - before))
        speed = (y2 - y1) * result.dy_m / (
            (crossings[1] - crossings[0]) * result.dt_s
        )
        errors.append(abs(speed / config.material.longitudinal_speed_m_s - 1.0))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1.0e-3


def test_result_contains_one_rf_trace_per_element() -> None:
    config = ProjectConfig()
    config.numerics.duration_us = 0.5
    config.numerics.record_snapshots = False
    config.source.element_count = 4
    result = run_simulation(config)
    assert result.receiver_channels_pa.shape == (4, result.time_s.size)
    assert result.element_centers_x_m.shape == (4,)
    assert result.transmit_delays_s.shape == (4,)


def test_fmc_fires_each_element_and_records_every_receiver() -> None:
    config = ProjectConfig()
    config.source.element_count = 3
    config.numerics.duration_us = 0.3
    result = run_fmc(config)
    assert result.data_pa.shape == (3, 3, result.time_s.size)
    assert np.isfinite(result.data_pa).all()


def test_tfm_reconstructs_synthetic_point_reflector() -> None:
    elements = np.array([-1.0e-3, 1.0e-3])
    time = np.linspace(0.0, 12.0e-6, 2401).astype(np.float32)
    target_x, target_y, speed = 0.0, 6.0e-3, 5850.0
    distances = np.sqrt((elements - target_x) ** 2 + target_y**2)
    data = np.empty((2, 2, time.size), dtype=np.float32)
    for tx in range(2):
        for rx in range(2):
            arrival = (distances[tx] + distances[rx]) / speed
            data[tx, rx] = np.exp(-0.5 * ((time - arrival) / 20.0e-9) ** 2)
    fmc = FMCResult("synthetic", 0.0, time, elements, data)
    x_axis = np.linspace(-2.0e-3, 2.0e-3, 17)
    y_axis = np.linspace(4.0e-3, 8.0e-3, 17)
    image = tfm_image(fmc, x_axis, y_axis, speed)
    iy, ix = np.unravel_index(int(np.argmax(image)), image.shape)
    assert x_axis[ix] == pytest.approx(target_x, abs=0.25e-3)
    assert y_axis[iy] == pytest.approx(target_y, abs=0.25e-3)
