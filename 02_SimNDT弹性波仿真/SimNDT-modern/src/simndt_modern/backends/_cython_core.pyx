# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t f32
ctypedef cnp.uint8_t u8
ctypedef cnp.int32_t i32


cpdef float advance(
    f32[:, ::1] vx,
    f32[:, ::1] vy,
    f32[:, ::1] sxx,
    f32[:, ::1] syy,
    f32[:, ::1] sxy,
    const f32[:, ::1] inv_rho,
    const f32[:, ::1] lam,
    const f32[:, ::1] mu,
    const f32[:, ::1] damping,
    const u8[:, ::1] constrained,
    const f32[::1] source_profile,
    int source_row,
    const i32[::1] receiver_indices,
    int receiver_row,
    float source_value,
    float dt,
    float dx,
    float dy,
):
    cdef int ny = vx.shape[0]
    cdef int nx = vx.shape[1]
    cdef int y, x, index
    cdef float two_dx = 2.0 * dx
    cdef float two_dy = 2.0 * dy
    cdef float dvx_dx, dvy_dy, dvx_dy, dvy_dx, damp, total = 0.0

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            vx[y, x] += dt * inv_rho[y, x] * (
                (sxx[y, x + 1] - sxx[y, x - 1]) / two_dx
                + (sxy[y + 1, x] - sxy[y - 1, x]) / two_dy)
            vy[y, x] += dt * inv_rho[y, x] * (
                (sxy[y, x + 1] - sxy[y, x - 1]) / two_dx
                + (syy[y + 1, x] - syy[y - 1, x]) / two_dy)

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            dvx_dx = (vx[y, x + 1] - vx[y, x - 1]) / two_dx
            dvy_dy = (vy[y + 1, x] - vy[y - 1, x]) / two_dy
            dvx_dy = (vx[y + 1, x] - vx[y - 1, x]) / two_dy
            dvy_dx = (vy[y, x + 1] - vy[y, x - 1]) / two_dx
            sxx[y, x] += dt * ((lam[y, x] + 2.0 * mu[y, x]) * dvx_dx + lam[y, x] * dvy_dy)
            syy[y, x] += dt * (lam[y, x] * dvx_dx + (lam[y, x] + 2.0 * mu[y, x]) * dvy_dy)
            sxy[y, x] += dt * mu[y, x] * (dvx_dy + dvy_dx)

    for x in range(nx):
        sxx[source_row, x] += source_value * source_profile[x]
        syy[source_row, x] += source_value * source_profile[x]

    for y in range(ny):
        for x in range(nx):
            if constrained[y, x] != 0:
                vx[y, x] = vy[y, x] = 0.0
                sxx[y, x] = syy[y, x] = sxy[y, x] = 0.0
            else:
                damp = damping[y, x]
                vx[y, x] *= damp
                vy[y, x] *= damp
                sxx[y, x] *= damp
                syy[y, x] *= damp
                sxy[y, x] *= damp

    for index in receiver_indices:
        total -= syy[receiver_row, index]
    return total / receiver_indices.shape[0]

