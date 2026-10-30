from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import ProjectConfig
from .solver import FMCResult, SimulationResult


def save_project(config: ProjectConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_project(path: str | Path) -> ProjectConfig:
    return ProjectConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_result(result: SimulationResult, path: str | Path) -> None:
    np.savez_compressed(
        path,
        backend=result.backend,
        runtime_s=result.runtime_s,
        nx=result.nx,
        ny=result.ny,
        dx_m=result.dx_m,
        dy_m=result.dy_m,
        dt_s=result.dt_s,
        time_s=result.time_s,
        receiver_pa=result.receiver_pa,
        receiver_channels_pa=result.receiver_channels_pa,
        element_centers_x_m=result.element_centers_x_m,
        transmit_delays_s=result.transmit_delays_s,
        transmit_apodization=result.transmit_apodization,
        receive_apodization=result.receive_apodization,
        field_magnitude=result.field_magnitude,
        snapshot_steps=result.snapshot_steps,
        snapshot_time_s=result.snapshot_time_s,
        snapshots_magnitude=result.snapshots_magnitude,
    )


def save_fmc_result(result: FMCResult, path: str | Path) -> None:
    np.savez_compressed(
        path,
        backend=result.backend,
        runtime_s=result.runtime_s,
        time_s=result.time_s,
        element_centers_x_m=result.element_centers_x_m,
        fmc_data_pa=result.data_pa,
    )
