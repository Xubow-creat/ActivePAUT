# SimNDT Modern 2.0 validation

Validated on 2026-08-24 on Windows 11, Python 3.12.13, NumPy 2.5.2, CuPy 14.1.1/CUDA 13 and NVIDIA GeForce RTX 5060 Ti.

## Automated result

`pytest`: 28 passed, 0 failed, 0 skipped when run through the ASCII junction `D:\SimNDTModern`.

The suite covers:

- project JSON round trip and v1.1 single-aperture migration;
- conservative physical-element rasterisation;
- analytic fractional transmit delays and receive DAS;
- exact agreement with the delay formula in k-Wave's linear-array example;
- finite-focus arrival-time equality at machine precision;
- 17 degree plane-wave steering (array-factor maximum within 0.05 degree);
- measured steel P-wave speed error below 0.5%;
- measured steel SV-wave speed error below 0.5%;
- monotonic P-wave convergence over 8, 12, and 16 points per shortest wavelength;
- 8 mm CPML plane-wave reflection below -50 dB;
- per-element RF dimensions and finite wavefields;
- FMC transmit/receive channel cube;
- synthetic point-reflector TFM localisation;
- NumPy, compatibility backends and CuPy CUDA numerical agreement;
- physical time/snapshot consistency, GUI construction and exports.

## Common benchmark

Grid 76 x 76, 461 steps, 16 elements. The compatibility backends intentionally execute the same strict NumPy core until their former collocated kernels are replaced by revalidated staggered CPML kernels.

| Backend selection | Runtime (s) | Receiver peak (Pa) | Final field peak (m/s) |
|---|---:|---:|---:|
| NumPy | 0.0578 | 616653.38 | 0.00606344 |
| Numba compatibility | 0.0569 | 616653.38 | 0.00606344 |
| Cython compatibility | 0.0569 | 616653.38 | 0.00606344 |
| OpenCL compatibility | 0.0575 | 616653.38 | 0.00606344 |
| CuPy CUDA | 0.4637 | 616653.31 | 0.00606344 |

This grid is too small for a GPU performance advantage; the GPU launch count dominates. The purpose of this benchmark is numerical identity, not peak throughput.

## Physical interpretation

The accepted model is two-dimensional plane strain. The velocity validation measures propagation between two depths using interpolated crossings of the dominant pulse lobe. The CPML test compares the reflected plane-wave peak with the incident peak at an interior probe. See `PHASED_ARRAY_PHYSICS_zh.md` for equations and limitations.
