from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .solver import SimulationResult


def _require_snapshots(result: SimulationResult) -> None:
    if result.snapshots_magnitude.shape[0] == 0:
        raise ValueError(
            "This result has no time-history frames. Enable 'Record time-history frames' "
            "and run the simulation again."
        )


def _extent_mm(result: SimulationResult) -> tuple[float, float, float, float]:
    return (0.0, (result.nx - 1) * result.dx_m * 1e3,
            (result.ny - 1) * result.dy_m * 1e3, 0.0)


def _global_peak(result: SimulationResult) -> float:
    return max(float(np.max(result.snapshots_magnitude)), 1e-30)


def _frame_db(frame: np.ndarray, peak: float) -> np.ndarray:
    return np.maximum(20.0 * np.log10(frame / peak + 1e-12), -60.0)


def _frame_title(result: SimulationResult, index: int) -> str:
    step = int(result.snapshot_steps[index])
    time_us = float(result.snapshot_time_s[index]) * 1e6
    return f"Wave propagation — step {step:,}, t = {time_us:.6f} µs"


def export_frame_images(
    result: SimulationResult,
    directory: str | Path,
    dpi: int = 140,
) -> Path:
    """Export one annotated PNG per recorded time and a machine-readable manifest."""
    _require_snapshots(result)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    peak = _global_peak(result)
    rows: list[dict[str, object]] = []

    figure = Figure(figsize=(8, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.imshow(
        _frame_db(result.snapshots_magnitude[0], peak),
        cmap="turbo", vmin=-60, vmax=0, origin="upper",
        extent=_extent_mm(result), interpolation="nearest",
    )
    axis.set(xlabel="X (mm)", ylabel="Y (mm)")
    figure.colorbar(image, ax=axis, label="Particle-velocity magnitude (dB re. history peak)")

    for index, frame in enumerate(result.snapshots_magnitude):
        step = int(result.snapshot_steps[index])
        time_s = float(result.snapshot_time_s[index])
        filename = f"frame_{index:04d}_step_{step:07d}_t_{time_s * 1e6:012.6f}us.png"
        image.set_data(_frame_db(frame, peak))
        axis.set_title(_frame_title(result, index))
        figure.savefig(target / filename, dpi=dpi)
        rows.append({
            "frame_index": index,
            "simulation_step": step,
            "time_s": time_s,
            "time_us": time_s * 1e6,
            "file": filename,
        })
    figure.clear()

    manifest_csv = target / "frames_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (target / "frames_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_csv


def export_animation(
    result: SimulationResult,
    path: str | Path,
    fps: int = 15,
    dpi: int = 120,
) -> Path:
    """Export the recorded wavefield as GIF or MP4 with step/time annotations."""
    _require_snapshots(result)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    peak = _global_peak(result)
    figure = Figure(figsize=(8, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.imshow(
        _frame_db(result.snapshots_magnitude[0], peak),
        cmap="turbo", vmin=-60, vmax=0, origin="upper",
        extent=_extent_mm(result), interpolation="nearest",
    )
    axis.set(xlabel="X (mm)", ylabel="Y (mm)")
    title = axis.set_title(_frame_title(result, 0))
    figure.colorbar(image, ax=axis, label="Particle-velocity magnitude (dB re. history peak)")

    def update(index: int):
        image.set_data(_frame_db(result.snapshots_magnitude[index], peak))
        title.set_text(_frame_title(result, index))
        return image, title

    animation = FuncAnimation(
        figure, update, frames=result.snapshots_magnitude.shape[0],
        interval=1000 / max(fps, 1), blit=False,
    )
    suffix = target.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=fps, metadata={"title": "SimNDT Modern wave propagation"})
    elif suffix == ".mp4":
        if not writers.is_available("ffmpeg"):
            raise RuntimeError("MP4 export requires FFmpeg. Export GIF instead, or install FFmpeg.")
        writer = FFMpegWriter(fps=fps, bitrate=2400, metadata={"title": "SimNDT Modern wave propagation"})
    else:
        raise ValueError("Animation file must use the .gif or .mp4 extension.")
    animation.save(target, writer=writer, dpi=dpi)
    figure.clear()
    return target
