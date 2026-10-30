from __future__ import annotations

import csv

from simndt_modern.export import export_animation, export_frame_images
from simndt_modern.grid import prepare_simulation
from simndt_modern.solver import run_prepared


def test_exports_annotated_frames_and_gif(small_config, tmp_path) -> None:
    small_config.numerics.snapshot_count = 3
    result = run_prepared(prepare_simulation(small_config), "numpy")

    frames_dir = tmp_path / "frames"
    manifest = export_frame_images(result, frames_dir, dpi=50)
    with manifest.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert rows[0]["simulation_step"] == "0"
    assert all((frames_dir / row["file"]).is_file() for row in rows)
    assert (frames_dir / "frames_manifest.json").is_file()

    gif = export_animation(result, tmp_path / "wavefield.gif", fps=5, dpi=50)
    assert gif.is_file()
    assert gif.stat().st_size > 0
