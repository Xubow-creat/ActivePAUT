from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, pi, sqrt

import numpy as np
from numpy.typing import NDArray

from .models import (
    ApodizationKind, BoundaryKind, BoundarySide, ProjectConfig,
    SourcePolarization,
)
from .signals import apodization_weights, delayed_tone_burst_bank, focusing_delays_s


@dataclass(slots=True)
class PreparedSimulation:
    config: ProjectConfig
    nx: int
    ny: int
    dx_m: float
    dy_m: float
    dt_s: float
    time_s: NDArray[np.float32]
    inv_rho: NDArray[np.float32]
    lambda_pa: NDArray[np.float32]
    mu_pa: NDArray[np.float32]
    cpml_a_x: NDArray[np.float32]
    cpml_b_x: NDArray[np.float32]
    cpml_inv_kappa_x: NDArray[np.float32]
    cpml_a_x_half: NDArray[np.float32]
    cpml_b_x_half: NDArray[np.float32]
    cpml_inv_kappa_x_half: NDArray[np.float32]
    cpml_a_y: NDArray[np.float32]
    cpml_b_y: NDArray[np.float32]
    cpml_inv_kappa_y: NDArray[np.float32]
    cpml_a_y_half: NDArray[np.float32]
    cpml_b_y_half: NDArray[np.float32]
    cpml_inv_kappa_y_half: NDArray[np.float32]
    free_top: bool
    free_bottom: bool
    free_left: bool
    free_right: bool
    element_centers_x_m: NDArray[np.float64]
    element_weights: NDArray[np.float32]
    receiver_element_weights: NDArray[np.float32]
    transmit_apodization: NDArray[np.float32]
    receive_apodization: NDArray[np.float32]
    transmit_delays_s: NDArray[np.float64]
    source_signals_pa: NDArray[np.float32]
    source_polarization: str
    source_row: int
    receiver_row: int
    # Legacy views retained for preview code and third-party callers.
    damping: NDArray[np.float32]
    constrained: NDArray[np.uint8]
    source_profile: NDArray[np.float32]
    receiver_indices: NDArray[np.int32]
    signal_pa: NDArray[np.float32]


def _edge_cells(thickness_mm: float, spacing_m: float, limit: int) -> int:
    if thickness_mm <= 0:
        return 0
    return max(1, min(limit, int(ceil(thickness_mm * 1.0e-3 / spacing_m))))


def _cpml_axis(
    count: int,
    spacing_m: float,
    low: BoundarySide,
    high: BoundarySide,
    wave_speed_m_s: float,
    center_frequency_hz: float,
    dt_s: float,
) -> tuple[NDArray[np.float32], ...]:
    """Create unsplit-CPML derivative recurrence coefficients."""
    polynomial_order = 3.0
    target_reflection = 1.0e-4
    alpha_max = pi * center_frequency_hz
    # Kappa is kept at unity for this second-order staggered scheme.  A larger
    # value benefits very-low-frequency waves but increases the measured
    # ultrasonic interface reflection on this grid.
    kappa_max = 1.0

    def coefficients(stagger: float) -> tuple[NDArray[np.float32], ...]:
        damping_rate = np.zeros(count, dtype=np.float64)
        alpha = np.zeros(count, dtype=np.float64)
        kappa = np.ones(count, dtype=np.float64)

        def add_side(side: BoundarySide, reverse: bool) -> None:
            if side.kind is not BoundaryKind.ABSORBING:
                return
            cells = _edge_cells(side.thickness_mm, spacing_m, max(1, count // 2))
            if not cells:
                return
            thickness = cells * spacing_m
            sigma_max = -(
                (polynomial_order + 1.0) * wave_speed_m_s * log(target_reflection)
                / (2.0 * thickness)
            )
            depth = np.maximum(
                0.0, (cells - np.arange(cells, dtype=np.float64) - stagger) / cells
            )
            indices = (
                np.arange(count - 1, count - cells - 1, -1)
                if reverse else np.arange(cells)
            )
            damping_rate[indices] = np.maximum(
                damping_rate[indices], sigma_max * depth**polynomial_order
            )
            alpha[indices] = np.maximum(
                alpha[indices], alpha_max * (1.0 - depth)
            )
            kappa[indices] = np.maximum(
                kappa[indices], 1.0 + (kappa_max - 1.0) * depth**polynomial_order
            )

        add_side(low, False)
        add_side(high, True)
        b = np.exp(-(damping_rate / kappa + alpha) * dt_s)
        denominator = kappa * (damping_rate + kappa * alpha)
        a = np.zeros_like(b)
        active = denominator > 0
        a[active] = damping_rate[active] * (b[active] - 1.0) / denominator[active]
        return (
            a.astype(np.float32), b.astype(np.float32),
            (1.0 / kappa).astype(np.float32),
        )

    return (*coefficients(0.0), *coefficients(0.5))


def _line_element_weights(
    x_m: NDArray[np.float64],
    spacing_m: float,
    centers_m: NDArray[np.float64],
    width_m: float,
) -> NDArray[np.float32]:
    """Conservative sub-cell coverage of physical line elements on x nodes."""
    node_left = np.maximum(x_m - 0.5 * spacing_m, 0.0)
    node_right = np.minimum(x_m + 0.5 * spacing_m, x_m[-1])
    rows = []
    for center in centers_m:
        left = center - 0.5 * width_m
        right = center + 0.5 * width_m
        overlap = np.maximum(
            0.0, np.minimum(node_right, right) - np.maximum(node_left, left)
        )
        weights = overlap / spacing_m
        if not np.any(weights > 0):
            weights[int(np.argmin(np.abs(x_m - center)))] = 1.0
        rows.append(weights.astype(np.float32))
    return np.stack(rows)


def prepare_simulation(config: ProjectConfig) -> PreparedSimulation:
    config.validate()
    n, m, s = config.numerics, config.material, config.source
    max_frequency = s.center_frequency_hz * n.max_frequency_factor
    controlling_speed = (
        m.shear_speed_m_s if m.shear_speed_m_s > 0 else m.longitudinal_speed_m_s
    )
    target_spacing = controlling_speed / (n.points_per_wavelength * max_frequency)

    nx = max(16, int(ceil(n.width_mm * 1.0e-3 / target_spacing)) + 1)
    ny = max(16, int(ceil(n.height_mm * 1.0e-3 / target_spacing)) + 1)
    dx = n.width_mm * 1.0e-3 / (nx - 1)
    dy = n.height_mm * 1.0e-3 / (ny - 1)
    dt = n.cfl * min(dx, dy) / (sqrt(2.0) * m.longitudinal_speed_m_s)
    steps = max(2, int(ceil(n.duration_us * 1.0e-6 / dt)) + 1)
    time_s = (np.arange(steps, dtype=np.float64) * dt).astype(np.float32)

    shape = (ny, nx)
    rho = np.full(shape, m.density_kg_m3, dtype=np.float64)
    lam = np.full(shape, m.lambda_pa, dtype=np.float64)
    mu = np.full(shape, m.mu_pa, dtype=np.float64)
    x_m = np.linspace(0.0, n.width_mm * 1.0e-3, nx, dtype=np.float64)
    y_m = np.linspace(0.0, n.height_mm * 1.0e-3, ny, dtype=np.float64)
    if config.defect.enabled:
        xx, yy = np.meshgrid(x_m, y_m)
        d = config.defect
        void = ((xx - d.center_x_mm * 1.0e-3) ** 2
                + (yy - d.center_y_mm * 1.0e-3) ** 2
                <= (d.radius_mm * 1.0e-3) ** 2)
        # A low-impedance acoustic fill regularises vacuum on the Cartesian
        # grid while retaining the steel-air reflection coefficient.
        air_rho, air_cp = 1.225, 343.0
        rho[void] = air_rho
        lam[void] = air_rho * air_cp**2
        mu[void] = 0.0

    (cpml_a_x, cpml_b_x, cpml_inv_kappa_x,
     cpml_a_x_half, cpml_b_x_half, cpml_inv_kappa_x_half) = _cpml_axis(
        nx, dx, config.boundaries.left, config.boundaries.right,
        m.longitudinal_speed_m_s, s.center_frequency_hz, dt,
    )
    (cpml_a_y, cpml_b_y, cpml_inv_kappa_y,
     cpml_a_y_half, cpml_b_y_half, cpml_inv_kappa_y_half) = _cpml_axis(
        ny, dy, config.boundaries.top, config.boundaries.bottom,
        m.longitudinal_speed_m_s, s.center_frequency_hz, dt,
    )

    element_count = s.element_count
    element_width_m = (
        s.aperture_mm if element_count == 1 else s.element_width_mm
    ) * 1.0e-3
    pitch_m = (s.element_width_mm + s.kerf_mm) * 1.0e-3
    ids = np.arange(element_count, dtype=np.float64) - 0.5 * (element_count - 1)
    element_centers = s.center_x_mm * 1.0e-3 + ids * pitch_m
    receiver_centers = element_centers + s.receiver_offset_mm * 1.0e-3
    if (receiver_centers[0] - element_width_m / 2 < 0
            or receiver_centers[-1] + element_width_m / 2 > x_m[-1]):
        raise ValueError("Receiver array offset moves an element outside the domain.")
    element_weights = _line_element_weights(x_m, dx, element_centers, element_width_m)
    receiver_weights = _line_element_weights(x_m, dx, receiver_centers, element_width_m)
    tx_apod = apodization_weights(
        element_count, ApodizationKind(s.transmit_apodization).value
    )
    rx_apod = apodization_weights(
        element_count, ApodizationKind(s.receive_apodization).value
    )
    if s.transmit_active_elements:
        tx_apod *= np.asarray(s.transmit_active_elements, dtype=np.float32)
    if s.receive_active_elements:
        rx_apod *= np.asarray(s.receive_active_elements, dtype=np.float32)
    delays = focusing_delays_s(
        element_centers, s.center_x_mm * 1.0e-3, s.steering_angle_deg,
        s.focus_depth_mm * 1.0e-3, m.longitudinal_speed_m_s,
    )
    source_signals = delayed_tone_burst_bank(
        time_s, s.center_frequency_hz, s.cycles, delays
    )
    source_signals *= np.float32(s.amplitude_pa)
    source_signals *= tx_apod[:, None]

    source_profile = np.sum(
        element_weights * tx_apod[:, None], axis=0
    ).astype(np.float32)
    receiver_union = np.flatnonzero(
        np.sum(receiver_weights, axis=0) > 0
    ).astype(np.int32)
    damping = np.outer(cpml_b_y, cpml_b_x).astype(np.float32)
    return PreparedSimulation(
        config=config, nx=nx, ny=ny, dx_m=dx, dy_m=dy, dt_s=dt,
        time_s=time_s, inv_rho=(1.0 / rho).astype(np.float32),
        lambda_pa=lam.astype(np.float32), mu_pa=mu.astype(np.float32),
        cpml_a_x=cpml_a_x, cpml_b_x=cpml_b_x,
        cpml_inv_kappa_x=cpml_inv_kappa_x,
        cpml_a_x_half=cpml_a_x_half, cpml_b_x_half=cpml_b_x_half,
        cpml_inv_kappa_x_half=cpml_inv_kappa_x_half,
        cpml_a_y=cpml_a_y, cpml_b_y=cpml_b_y,
        cpml_inv_kappa_y=cpml_inv_kappa_y,
        cpml_a_y_half=cpml_a_y_half, cpml_b_y_half=cpml_b_y_half,
        cpml_inv_kappa_y_half=cpml_inv_kappa_y_half,
        free_top=config.boundaries.top.kind is BoundaryKind.AIR,
        free_bottom=config.boundaries.bottom.kind is BoundaryKind.AIR,
        free_left=config.boundaries.left.kind is BoundaryKind.AIR,
        free_right=config.boundaries.right.kind is BoundaryKind.AIR,
        element_centers_x_m=element_centers,
        element_weights=element_weights,
        receiver_element_weights=receiver_weights,
        transmit_apodization=tx_apod, receive_apodization=rx_apod,
        transmit_delays_s=delays, source_signals_pa=source_signals,
        source_polarization=SourcePolarization(s.polarization).value,
        source_row=0, receiver_row=min(2, ny - 1), damping=damping,
        constrained=np.zeros(shape, dtype=np.uint8), source_profile=source_profile,
        receiver_indices=receiver_union,
        signal_pa=np.sum(source_signals, axis=0).astype(np.float32),
    )
