from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simndt_modern.grid import prepare_simulation
from simndt_modern.models import BackendKind, ProjectConfig
from simndt_modern.solver import available_backends, run_prepared


def main() -> int:
    config = ProjectConfig()
    config.numerics.width_mm = 12.0
    config.numerics.height_mm = 12.0
    config.numerics.duration_us = 4.0
    config.numerics.points_per_wavelength = 10
    config.numerics.record_snapshots = False
    config.source.center_x_mm = 6.0
    config.defect.center_x_mm = 6.0
    config.defect.center_y_mm = 6.0
    prepared = prepare_simulation(config)
    records = []
    for kind in available_backends():
        try:
            # Warm JIT compilers and device kernels before the measured run.
            run_prepared(prepared, kind)
            result = run_prepared(prepared, kind)
            records.append({
                "backend": kind.value,
                "runtime_s": result.runtime_s,
                "receiver_peak_pa": float(np.max(np.abs(result.receiver_pa))),
                "field_peak_m_s": float(np.max(result.field_magnitude)),
                "grid": [result.ny, result.nx],
                "steps": int(result.time_s.size),
            })
        except Exception as exc:
            records.append({"backend": kind.value, "error": repr(exc)})
    output = ROOT / "benchmark_results.json"
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
