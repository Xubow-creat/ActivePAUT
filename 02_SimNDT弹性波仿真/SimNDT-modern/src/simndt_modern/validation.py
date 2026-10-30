from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .grid import PreparedSimulation


def far_field_array_factor(
    prepared: PreparedSimulation,
    angles_deg: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Narrow-band far-field array factor for the prepared transmit law."""
    angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    frequency = prepared.config.source.center_frequency_hz
    speed = prepared.config.material.longitudinal_speed_m_s
    omega = 2.0 * np.pi * frequency
    wavenumber = omega / speed
    phase = (
        wavenumber * prepared.element_centers_x_m[:, None] * np.sin(angles)[None, :]
        - omega * prepared.transmit_delays_s[:, None]
    )
    field = np.sum(prepared.transmit_apodization[:, None] * np.exp(1j * phase), axis=0)
    normalizer = max(float(np.sum(np.abs(prepared.transmit_apodization))), 1.0e-30)
    return (np.abs(field) / normalizer).astype(np.float64)


def focus_arrival_spread_s(prepared: PreparedSimulation) -> float:
    """Geometrical spread in arrival time at the configured focal point."""
    source = prepared.config.source
    if source.focus_depth_mm <= 0:
        return 0.0
    depth = source.focus_depth_mm * 1.0e-3
    focus_x = source.center_x_mm * 1.0e-3 + depth * np.tan(
        np.deg2rad(source.steering_angle_deg)
    )
    distances = np.sqrt((prepared.element_centers_x_m - focus_x) ** 2 + depth**2)
    arrivals = prepared.transmit_delays_s + (
        distances / prepared.config.material.longitudinal_speed_m_s
    )
    return float(np.ptp(arrivals))


def array_quality_summary(prepared: PreparedSimulation) -> dict[str, float | bool | int]:
    source = prepared.config.source
    wavelength = (
        prepared.config.material.longitudinal_speed_m_s / source.center_frequency_hz
    )
    pitch = source.pitch_mm * 1.0e-3
    steering = np.deg2rad(abs(source.steering_angle_deg))
    no_grating_limit = wavelength / (1.0 + np.sin(steering))
    physical_width = (
        (source.aperture_mm if source.element_count == 1 else source.element_width_mm)
        * 1.0e-3
    )
    represented_widths = np.sum(prepared.element_weights, axis=1) * prepared.dx_m
    return {
        "element_count": int(source.element_count),
        "wavelength_m": float(wavelength),
        "pitch_m": float(pitch),
        "grating_lobe_free_at_configured_angle": bool(pitch <= no_grating_limit),
        "maximum_element_width_error_m": float(
            np.max(np.abs(represented_widths - physical_width))
        ),
        "focus_arrival_spread_s": focus_arrival_spread_s(prepared),
        "delay_span_s": float(np.ptp(prepared.transmit_delays_s)),
    }
