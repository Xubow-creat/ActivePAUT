from __future__ import annotations

from typing import Any

import numpy as np

from .base import BackendOutput, ProgressCallback, SnapshotCallback
from ..grid import PreparedSimulation
from ..signals import delay_and_sum


def _harmonic_pair(left: Any, right: Any, xp: Any) -> Any:
    denominator = left + right
    return xp.where(denominator > 0, 2.0 * left * right / denominator, 0.0)


def run_vectorized(
    prepared: PreparedSimulation,
    xp: Any,
    progress: ProgressCallback | None,
    snapshot_steps: frozenset[int] | None = None,
    snapshot: SnapshotCallback | None = None,
) -> BackendOutput:
    """Second-order staggered-grid velocity-stress P-SV solver with CPML.

    Normal stresses are located at cell centres, ``vx``/``vy`` on their
    corresponding faces, and ``sxy`` at cell corners.  Arrays share one shape
    for backend portability; unused terminal rows/columns remain zero.
    """
    shape = (prepared.ny, prepared.nx)
    vx = xp.zeros(shape, dtype=xp.float32)
    vy = xp.zeros(shape, dtype=xp.float32)
    sxx = xp.zeros(shape, dtype=xp.float32)
    syy = xp.zeros(shape, dtype=xp.float32)
    sxy = xp.zeros(shape, dtype=xp.float32)

    inv_rho = xp.asarray(prepared.inv_rho)
    lam = xp.asarray(prepared.lambda_pa)
    mu = xp.asarray(prepared.mu_pa)
    element_weights = xp.asarray(prepared.element_weights)
    receiver_weights = xp.asarray(prepared.receiver_element_weights)
    source_signals = xp.asarray(prepared.source_signals_pa)
    weight_sums = xp.maximum(xp.sum(receiver_weights, axis=1), xp.float32(1.0e-12))
    channels = xp.zeros(
        (prepared.element_weights.shape[0], prepared.time_s.size), dtype=xp.float32
    )

    # Density is averaged at velocity locations.  This expression computes
    # 1 / arithmetic_mean(rho) directly from inverse density.
    inv_rho_vx = inv_rho.copy()
    inv_rho_vx[:, :-1] = _harmonic_pair(inv_rho[:, :-1], inv_rho[:, 1:], xp)
    inv_rho_vy = inv_rho.copy()
    inv_rho_vy[:-1, :] = _harmonic_pair(inv_rho[:-1, :], inv_rho[1:, :], xp)

    # Harmonic shear modulus at cell corners enforces vanishing shear
    # traction at a solid-fluid (mu=0) interface.
    mu_sxy = xp.zeros(shape, dtype=xp.float32)
    m00, m10 = mu[:-1, :-1], mu[1:, :-1]
    m01, m11 = mu[:-1, 1:], mu[1:, 1:]
    all_solid = (m00 > 0) & (m10 > 0) & (m01 > 0) & (m11 > 0)
    reciprocal_sum = (
        1.0 / xp.maximum(m00, 1.0)
        + 1.0 / xp.maximum(m10, 1.0)
        + 1.0 / xp.maximum(m01, 1.0)
        + 1.0 / xp.maximum(m11, 1.0)
    )
    mu_sxy[:-1, :-1] = xp.where(all_solid, 4.0 / reciprocal_sum, 0.0)

    ax = xp.asarray(prepared.cpml_a_x)[None, :]
    bx = xp.asarray(prepared.cpml_b_x)[None, :]
    ikx = xp.asarray(prepared.cpml_inv_kappa_x)[None, :]
    axh = xp.asarray(prepared.cpml_a_x_half)[None, :]
    bxh = xp.asarray(prepared.cpml_b_x_half)[None, :]
    ikxh = xp.asarray(prepared.cpml_inv_kappa_x_half)[None, :]
    ay = xp.asarray(prepared.cpml_a_y)[:, None]
    by = xp.asarray(prepared.cpml_b_y)[:, None]
    iky = xp.asarray(prepared.cpml_inv_kappa_y)[:, None]
    ayh = xp.asarray(prepared.cpml_a_y_half)[:, None]
    byh = xp.asarray(prepared.cpml_b_y_half)[:, None]
    ikyh = xp.asarray(prepared.cpml_inv_kappa_y_half)[:, None]
    psi_sxx_x = xp.zeros(shape, dtype=xp.float32)
    psi_sxy_y = xp.zeros(shape, dtype=xp.float32)
    psi_sxy_x = xp.zeros(shape, dtype=xp.float32)
    psi_syy_y = xp.zeros(shape, dtype=xp.float32)
    psi_vx_x = xp.zeros(shape, dtype=xp.float32)
    psi_vy_y = xp.zeros(shape, dtype=xp.float32)
    psi_vx_y = xp.zeros(shape, dtype=xp.float32)
    psi_vy_x = xp.zeros(shape, dtype=xp.float32)

    raw_a = xp.zeros(shape, dtype=xp.float32)
    raw_b = xp.zeros(shape, dtype=xp.float32)
    dt = xp.float32(prepared.dt_s)
    inv_dx = xp.float32(1.0 / prepared.dx_m)
    inv_dy = xp.float32(1.0 / prepared.dy_m)
    report_stride = max(1, prepared.time_s.size // 100)

    for step in range(prepared.time_s.size):
        # Momentum equations at staggered velocity locations.
        raw_a.fill(0.0)
        raw_a[:, :-1] = (sxx[:, 1:] - sxx[:, :-1]) * inv_dx
        psi_sxx_x *= bxh
        psi_sxx_x += axh * raw_a
        dsxx_dx = ikxh * raw_a + psi_sxx_x

        raw_b.fill(0.0)
        raw_b[1:, :-1] = (sxy[1:, :-1] - sxy[:-1, :-1]) * inv_dy
        psi_sxy_y *= by
        psi_sxy_y += ay * raw_b
        dsxy_dy = iky * raw_b + psi_sxy_y
        vx += dt * inv_rho_vx * (dsxx_dx + dsxy_dy)

        raw_a.fill(0.0)
        raw_a[:-1, 1:] = (sxy[:-1, 1:] - sxy[:-1, :-1]) * inv_dx
        psi_sxy_x *= bx
        psi_sxy_x += ax * raw_a
        dsxy_dx = ikx * raw_a + psi_sxy_x

        raw_b.fill(0.0)
        raw_b[:-1, :] = (syy[1:, :] - syy[:-1, :]) * inv_dy
        psi_syy_y *= byh
        psi_syy_y += ayh * raw_b
        dsyy_dy = ikyh * raw_b + psi_syy_y
        vy += dt * inv_rho_vy * (dsxy_dx + dsyy_dy)

        # Constitutive equations at normal-stress cell centres.
        raw_a.fill(0.0)
        raw_a[:, 1:] = (vx[:, 1:] - vx[:, :-1]) * inv_dx
        psi_vx_x *= bx
        psi_vx_x += ax * raw_a
        dvx_dx = ikx * raw_a + psi_vx_x

        raw_b.fill(0.0)
        raw_b[1:, :] = (vy[1:, :] - vy[:-1, :]) * inv_dy
        psi_vy_y *= by
        psi_vy_y += ay * raw_b
        dvy_dy = iky * raw_b + psi_vy_y
        sxx += dt * ((lam + 2.0 * mu) * dvx_dx + lam * dvy_dy)
        syy += dt * (lam * dvx_dx + (lam + 2.0 * mu) * dvy_dy)

        raw_a.fill(0.0)
        raw_a[:-1, :-1] = (vx[1:, :-1] - vx[:-1, :-1]) * inv_dy
        psi_vx_y *= byh
        psi_vx_y += ayh * raw_a
        dvx_dy = ikyh * raw_a + psi_vx_y

        raw_b.fill(0.0)
        raw_b[:-1, :-1] = (vy[:-1, 1:] - vy[:-1, :-1]) * inv_dx
        psi_vy_x *= bxh
        psi_vy_x += axh * raw_b
        dvy_dx = ikxh * raw_b + psi_vy_x
        sxy += dt * mu_sxy * (dvx_dy + dvy_dx)

        # Exact zero-traction boundaries in the discrete stress variables.
        if prepared.free_top:
            syy[0, :] = 0.0
            sxy[0, :] = 0.0
        if prepared.free_bottom:
            syy[-1, :] = 0.0
            sxy[-2:, :] = 0.0
        if prepared.free_left:
            sxx[:, 0] = 0.0
            sxy[:, 0] = 0.0
        if prepared.free_right:
            sxx[:, -1] = 0.0
            sxy[:, -2:] = 0.0

        # Prescribed normal traction replaces the free-surface condition only
        # on the physical element footprints.  Units remain Pa.
        traction = element_weights.T @ source_signals[:, step]
        if prepared.source_polarization == "normal":
            syy[prepared.source_row, :] = -traction
            sensed_stress = syy[prepared.receiver_row, :]
        else:
            sxy[prepared.source_row, :] = -traction
            sensed_stress = sxy[prepared.receiver_row, :]

        channels[:, step] = -(
            receiver_weights @ sensed_stress
        ) / weight_sums

        if snapshot is not None and snapshot_steps is not None and step in snapshot_steps:
            magnitude = xp.sqrt(vx * vx + vy * vy)
            frame = magnitude if xp is np else xp.asnumpy(magnitude)
            snapshot(step, np.asarray(frame, dtype=np.float32))
        if progress is not None and (
            step % report_stride == 0 or step + 1 == prepared.time_s.size
        ):
            progress(step + 1, prepared.time_s.size)

    magnitude = xp.sqrt(vx * vx + vy * vy)
    channels_host = channels if xp is np else xp.asnumpy(channels)
    magnitude_host = magnitude if xp is np else xp.asnumpy(magnitude)
    beamformed = delay_and_sum(
        np.asarray(channels_host), prepared.time_s,
        prepared.transmit_delays_s, prepared.receive_apodization,
    )
    return BackendOutput(
        beamformed.astype(np.float32, copy=False),
        np.asarray(magnitude_host, dtype=np.float32),
        np.asarray(channels_host, dtype=np.float32),
    )
