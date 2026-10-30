from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .export import export_animation, export_frame_images
from .io import load_project, save_fmc_result, save_result
from .models import BackendKind, ProjectConfig
from .solver import available_backends, run_fmc, run_simulation, tfm_image


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="SimNDT Modern command-line simulator")
    result.add_argument("--project", type=Path, help="Project JSON created by the GUI")
    result.add_argument("--backend", choices=[item.value for item in BackendKind])
    result.add_argument("--duration-us", type=float)
    result.add_argument("--output", type=Path, default=Path("simndt_result.npz"))
    result.add_argument("--snapshots", type=int, help="Record this many evenly spaced wavefield frames")
    result.add_argument("--animation", type=Path, help="Export recorded frames as .gif or .mp4")
    result.add_argument("--frames-dir", type=Path, help="Export annotated PNG frames and time manifest")
    result.add_argument("--fmc-output", type=Path, help="Run full matrix capture and save channel cube")
    result.add_argument("--tfm-output", type=Path, help="Save a homogeneous P-wave TFM image as NPZ")
    result.add_argument("--tfm-pixels", type=int, default=128, help="TFM pixels per x/y direction")
    result.add_argument("--list-backends", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.list_backends:
        print("\n".join(item.value for item in available_backends()))
        return 0
    config = load_project(args.project) if args.project else ProjectConfig()
    if args.backend:
        config.numerics.backend = BackendKind(args.backend)
    if args.duration_us:
        config.numerics.duration_us = args.duration_us
    if args.snapshots is not None:
        config.numerics.record_snapshots = args.snapshots > 0
        if args.snapshots > 0:
            config.numerics.snapshot_count = args.snapshots
    result = run_simulation(config)
    save_result(result, args.output)
    if args.animation:
        export_animation(result, args.animation)
    if args.frames_dir:
        export_frame_images(result, args.frames_dir)
    if args.fmc_output or args.tfm_output:
        fmc = run_fmc(config)
        if args.fmc_output:
            save_fmc_result(fmc, args.fmc_output)
        if args.tfm_output:
            x_m = np.linspace(0.0, config.numerics.width_mm * 1.0e-3, args.tfm_pixels)
            y_m = np.linspace(0.0, config.numerics.height_mm * 1.0e-3, args.tfm_pixels)
            image = tfm_image(
                fmc, x_m, y_m, config.material.longitudinal_speed_m_s
            )
            np.savez_compressed(args.tfm_output, x_m=x_m, y_m=y_m, tfm_pa=image)
    summary = {
        "backend": result.backend,
        "runtime_s": result.runtime_s,
        "grid": [result.ny, result.nx],
        "steps": int(result.time_s.size),
        "dt_s": result.dt_s,
        "receiver_peak_pa": float(np.max(np.abs(result.receiver_pa))),
        "receiver_channels": int(result.receiver_channels_pa.shape[0]),
        "field_peak_m_s": float(np.max(result.field_magnitude)),
        "recorded_frames": int(result.snapshots_magnitude.shape[0]),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
