from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox,
    QSlider, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from .export import export_animation, export_frame_images
from .grid import prepare_simulation
from .io import load_project, save_project, save_result
from .models import (
    ApodizationKind, BackendKind, Boundaries, BoundaryKind, BoundarySide, CircularDefect,
    MATERIAL_PRESETS, Material, Numerics, ProjectConfig, Source,
    SourcePolarization,
)
from .solver import SimulationResult, available_backends, run_simulation
from .validation import array_quality_summary


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, rows: int = 1):
        self.figure = Figure(figsize=(7, 5), constrained_layout=True)
        self.axes = self.figure.subplots(rows, 1, squeeze=False).ravel().tolist()
        super().__init__(self.figure)


class SimulationWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(int)

    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            result = run_simulation(
                self.config,
                lambda step, total: self.progress.emit(int(round(100.0 * step / total))),
            )
            self.completed.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SimNDT Modern 2.0 — P–SV phased array")
        self.resize(1280, 820)
        self.result: SimulationResult | None = None
        self.thread: QThread | None = None
        self.worker: SimulationWorker | None = None
        self._build_ui()
        self._build_menu()
        initial = ProjectConfig()
        if BackendKind.OPENCL in available_backends():
            initial.numerics.backend = BackendKind.OPENCL
        self.apply_config(initial)
        self.preview()

    @staticmethod
    def _double(minimum: float, maximum: float, decimals: int = 3, suffix: str = "") -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSuffix(suffix)
        widget.setSingleStep(0.1)
        return widget

    def _build_ui(self) -> None:
        self.material = QComboBox(); self.material.addItems(MATERIAL_PRESETS)
        self.width = self._double(1, 1000, 2, " mm")
        self.height = self._double(1, 1000, 2, " mm")
        self.duration = self._double(0.1, 10000, 2, " µs")
        self.frequency = self._double(0.001, 1000, 3, " MHz")
        self.amplitude = self._double(1, 1.0e12, 1, " Pa")
        self.cycles = self._double(0.25, 100, 2)
        self.aperture = self._double(0.01, 1000, 2, " mm")
        self.source_x = self._double(0, 1000, 2, " mm")
        self.receiver_offset = self._double(-1000, 1000, 2, " mm")
        self.element_count = QSpinBox(); self.element_count.setRange(1, 1024)
        self.element_width = self._double(0.001, 100, 3, " mm")
        self.kerf = self._double(0, 100, 3, " mm")
        self.steering_angle = self._double(-89, 89, 2, "°")
        self.focus_depth = self._double(0, 10000, 2, " mm")
        self.tx_apodization = QComboBox(); self.rx_apodization = QComboBox()
        for kind in ApodizationKind:
            label = kind.value.title()
            self.tx_apodization.addItem(label, kind)
            self.rx_apodization.addItem(label, kind)
        self.polarization = QComboBox()
        self.polarization.addItem("Normal traction (P-dominant)", SourcePolarization.NORMAL)
        self.polarization.addItem("Tangential traction (SV-dominant)", SourcePolarization.TANGENTIAL)
        self.ppw = QSpinBox(); self.ppw.setRange(6, 50)
        self.cfl = self._double(0.05, 0.70, 3)
        self.record_snapshots = QCheckBox("Record time-history frames")
        self.snapshot_count = QSpinBox(); self.snapshot_count.setRange(2, 1000)
        self.backend = QComboBox()
        for kind in available_backends():
            self.backend.addItem(kind.value, kind)

        domain_box = QGroupBox("Domain and material")
        domain_form = QFormLayout(domain_box)
        for label, widget in (
            ("Material preset", self.material), ("Width", self.width),
            ("Height", self.height), ("Simulation time", self.duration),
        ):
            domain_form.addRow(label, widget)

        source_box = QGroupBox("P–SV phased-array traction source / receiver")
        source_form = QFormLayout(source_box)
        for label, widget in (
            ("Center frequency", self.frequency), ("Peak amplitude", self.amplitude),
            ("Tone-burst cycles", self.cycles), ("Array center X", self.source_x),
            ("Element count", self.element_count), ("Element width", self.element_width),
            ("Kerf", self.kerf), ("Legacy 1-element aperture", self.aperture),
            ("Steering angle", self.steering_angle), ("Focus depth (0 = plane wave)", self.focus_depth),
            ("Transmit apodization", self.tx_apodization),
            ("Receive apodization", self.rx_apodization),
            ("Traction polarization", self.polarization),
            ("Receiver offset", self.receiver_offset),
        ):
            source_form.addRow(label, widget)

        self.defect_enabled = QCheckBox("Enable circular void")
        self.defect_x = self._double(0, 1000, 2, " mm")
        self.defect_y = self._double(0, 1000, 2, " mm")
        self.defect_radius = self._double(0.01, 1000, 2, " mm")
        defect_box = QGroupBox("Defect")
        defect_form = QFormLayout(defect_box)
        defect_form.addRow(self.defect_enabled)
        defect_form.addRow("Center X", self.defect_x)
        defect_form.addRow("Center Y", self.defect_y)
        defect_form.addRow("Radius", self.defect_radius)

        boundary_box = QGroupBox("Boundary conditions")
        boundary_grid = QGridLayout(boundary_box)
        boundary_grid.addWidget(QLabel("Side"), 0, 0)
        boundary_grid.addWidget(QLabel("Type"), 0, 1)
        boundary_grid.addWidget(QLabel("Thickness"), 0, 2)
        self.boundary_widgets: dict[str, tuple[QComboBox, QDoubleSpinBox]] = {}
        for row, side in enumerate(("top", "bottom", "left", "right"), 1):
            combo = QComboBox()
            combo.addItem("Absorbing", BoundaryKind.ABSORBING)
            combo.addItem("Air / free surface", BoundaryKind.AIR)
            thickness = self._double(0, 1000, 2, " mm")
            boundary_grid.addWidget(QLabel(side.title()), row, 0)
            boundary_grid.addWidget(combo, row, 1)
            boundary_grid.addWidget(thickness, row, 2)
            self.boundary_widgets[side] = (combo, thickness)

        numerics_box = QGroupBox("Numerics")
        numerics_form = QFormLayout(numerics_box)
        numerics_form.addRow("Backend", self.backend)
        numerics_form.addRow("Points / shortest wavelength", self.ppw)
        numerics_form.addRow("CFL", self.cfl)
        numerics_form.addRow(self.record_snapshots)
        numerics_form.addRow("Frames to record", self.snapshot_count)

        self.preview_button = QPushButton("Preview grid")
        self.run_button = QPushButton("Run simulation")
        self.run_button.setDefault(True)
        self.progress = QProgressBar(); self.progress.setRange(0, 100)
        self.summary = QLabel(); self.summary.setWordWrap(True)
        self.preview_button.clicked.connect(self.preview)
        self.run_button.clicked.connect(self.start_run)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        for box in (domain_box, source_box, defect_box, boundary_box, numerics_box):
            controls_layout.addWidget(box)
        controls_layout.addWidget(self.preview_button)
        controls_layout.addWidget(self.run_button)
        controls_layout.addWidget(self.progress)
        controls_layout.addWidget(self.summary)
        controls_layout.addStretch(1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls)

        self.model_canvas = PlotCanvas(1)
        self.result_canvas = PlotCanvas(2)
        self.history_canvas = PlotCanvas(1)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_label = QLabel("No time-history frames")
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.update_history_frame)
        self.play_button.clicked.connect(self.toggle_history_playback)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(80)
        self.play_timer.timeout.connect(self.advance_history_frame)
        history_controls = QHBoxLayout()
        history_controls.addWidget(self.play_button)
        history_controls.addWidget(self.frame_slider, 1)
        history_controls.addWidget(self.frame_label)
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.addWidget(self.history_canvas, 1)
        history_layout.addLayout(history_controls)
        tabs = QTabWidget()
        tabs.addTab(self.model_canvas, "Model / grid")
        tabs.addTab(self.result_canvas, "Wavefield / A-scan")
        tabs.addTab(history_tab, "Time history")
        splitter = QSplitter(); splitter.addWidget(scroll); splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 890])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        save_action = QAction("Save project…", self)
        load_action = QAction("Open project…", self)
        export_action = QAction("Export result…", self)
        export_animation_action = QAction("Export animation (GIF/MP4)…", self)
        export_frames_action = QAction("Export time-step images…", self)
        quit_action = QAction("Quit", self)
        save_action.triggered.connect(self.save_project_dialog)
        load_action.triggered.connect(self.load_project_dialog)
        export_action.triggered.connect(self.export_result_dialog)
        export_animation_action.triggered.connect(self.export_animation_dialog)
        export_frames_action.triggered.connect(self.export_frames_dialog)
        quit_action.triggered.connect(self.close)
        file_menu.addActions([
            save_action, load_action, export_action,
            export_animation_action, export_frames_action,
        ])
        file_menu.addSeparator(); file_menu.addAction(quit_action)

    def collect_config(self) -> ProjectConfig:
        preset = MATERIAL_PRESETS[self.material.currentText()]
        boundaries = {}
        for side, (combo, thickness) in self.boundary_widgets.items():
            boundaries[side] = BoundarySide(
                BoundaryKind(combo.currentData()), thickness.value()
            )
        return ProjectConfig(
            material=Material(
                preset.name, preset.density_kg_m3,
                preset.longitudinal_speed_m_s, preset.shear_speed_m_s,
            ),
            boundaries=Boundaries(**boundaries),
            source=Source(
                center_frequency_hz=self.frequency.value() * 1.0e6,
                amplitude_pa=self.amplitude.value(), cycles=self.cycles.value(),
                aperture_mm=self.aperture.value(), center_x_mm=self.source_x.value(),
                receiver_offset_mm=self.receiver_offset.value(),
                element_count=self.element_count.value(),
                element_width_mm=self.element_width.value(), kerf_mm=self.kerf.value(),
                steering_angle_deg=self.steering_angle.value(),
                focus_depth_mm=self.focus_depth.value(),
                transmit_apodization=ApodizationKind(self.tx_apodization.currentData()),
                receive_apodization=ApodizationKind(self.rx_apodization.currentData()),
                polarization=SourcePolarization(self.polarization.currentData()),
            ),
            defect=CircularDefect(
                enabled=self.defect_enabled.isChecked(), center_x_mm=self.defect_x.value(),
                center_y_mm=self.defect_y.value(), radius_mm=self.defect_radius.value(),
            ),
            numerics=Numerics(
                width_mm=self.width.value(), height_mm=self.height.value(),
                duration_us=self.duration.value(), points_per_wavelength=self.ppw.value(),
                cfl=self.cfl.value(),
                backend=BackendKind(self.backend.currentData()),
                record_snapshots=self.record_snapshots.isChecked(),
                snapshot_count=self.snapshot_count.value(),
            ),
        )

    def apply_config(self, config: ProjectConfig) -> None:
        self.material.setCurrentText(config.material.name if config.material.name in MATERIAL_PRESETS else "Steel")
        n, s, d = config.numerics, config.source, config.defect
        self.width.setValue(n.width_mm); self.height.setValue(n.height_mm)
        self.duration.setValue(n.duration_us); self.ppw.setValue(n.points_per_wavelength)
        self.cfl.setValue(n.cfl)
        self.record_snapshots.setChecked(n.record_snapshots)
        self.snapshot_count.setValue(n.snapshot_count)
        index = self.backend.findData(n.backend)
        if index >= 0: self.backend.setCurrentIndex(index)
        self.frequency.setValue(s.center_frequency_hz * 1.0e-6)
        self.amplitude.setValue(s.amplitude_pa); self.cycles.setValue(s.cycles)
        self.aperture.setValue(s.aperture_mm); self.source_x.setValue(s.center_x_mm)
        self.receiver_offset.setValue(s.receiver_offset_mm)
        self.element_count.setValue(s.element_count)
        self.element_width.setValue(s.element_width_mm); self.kerf.setValue(s.kerf_mm)
        self.steering_angle.setValue(s.steering_angle_deg); self.focus_depth.setValue(s.focus_depth_mm)
        self.tx_apodization.setCurrentIndex(self.tx_apodization.findData(s.transmit_apodization))
        self.rx_apodization.setCurrentIndex(self.rx_apodization.findData(s.receive_apodization))
        self.polarization.setCurrentIndex(self.polarization.findData(s.polarization))
        self.defect_enabled.setChecked(d.enabled); self.defect_x.setValue(d.center_x_mm)
        self.defect_y.setValue(d.center_y_mm); self.defect_radius.setValue(d.radius_mm)
        for side in ("top", "bottom", "left", "right"):
            value = getattr(config.boundaries, side)
            combo, thickness = self.boundary_widgets[side]
            combo.setCurrentIndex(combo.findData(value.kind)); thickness.setValue(value.thickness_mm)

    @Slot()
    def preview(self) -> None:
        try:
            prepared = prepare_simulation(self.collect_config())
        except Exception as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc)); return
        ax = self.model_canvas.axes[0]; ax.clear()
        image = (1.0 - prepared.damping) + prepared.constrained.astype(float)
        ax.imshow(image, cmap="magma", origin="upper", extent=(0, prepared.config.numerics.width_mm,
                  prepared.config.numerics.height_mm, 0), interpolation="nearest")
        source_x = np.flatnonzero(prepared.source_profile > 0)
        if source_x.size:
            ax.plot(source_x * prepared.dx_m * 1e3,
                    np.full(source_x.size, prepared.source_row * prepared.dy_m * 1e3), "c-", lw=3)
        for index, center in enumerate(prepared.element_centers_x_m):
            ax.text(center * 1e3, 0.35, str(index + 1), color="white", fontsize=6,
                    ha="center", va="top", clip_on=True)
        focus_depth = prepared.config.source.focus_depth_mm
        if focus_depth > 0:
            focus_x = (prepared.config.source.center_x_mm
                       + focus_depth * np.tan(np.deg2rad(prepared.config.source.steering_angle_deg)))
            ax.plot([focus_x], [focus_depth], "wx", ms=8, mew=1.5)
        ax.set(title="CPML, defect and physical array elements", xlabel="X (mm)", ylabel="Y (mm)")
        self.model_canvas.draw_idle()
        quality = array_quality_summary(prepared)
        grating_status = (
            "no grating lobe predicted"
            if quality["grating_lobe_free_at_configured_angle"]
            else "WARNING: pitch permits a grating lobe"
        )
        self.summary.setText(
            f"Grid: {prepared.ny} × {prepared.nx}\n"
            f"dx={prepared.dx_m*1e3:.4f} mm, dy={prepared.dy_m*1e3:.4f} mm\n"
            f"dt={prepared.dt_s*1e9:.3f} ns, steps={prepared.time_s.size}\n"
            f"Elements={prepared.element_weights.shape[0]}, pitch={prepared.config.source.pitch_mm:.3f} mm, "
            f"delay span={np.ptp(prepared.transmit_delays_s)*1e9:.3f} ns\n{grating_status}"
        )

    @Slot()
    def start_run(self) -> None:
        try:
            config = self.collect_config(); config.validate()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc)); return
        self.run_button.setEnabled(False); self.preview_button.setEnabled(False)
        self.progress.setValue(0); self.statusBar().showMessage("Running simulation…")
        self.thread = QThread(self)
        self.worker = SimulationWorker(config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.completed.connect(self.finish_run)
        self.worker.failed.connect(self.fail_run)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    @Slot(object)
    def finish_run(self, result: SimulationResult) -> None:
        self.result = result
        field_ax, trace_ax = self.result_canvas.axes
        field_ax.clear(); trace_ax.clear()
        magnitude = result.field_magnitude
        db = 20.0 * np.log10(magnitude / max(float(np.max(magnitude)), 1e-30) + 1e-12)
        field_ax.imshow(db, cmap="turbo", vmin=-60, vmax=0, origin="upper",
                        extent=(0, self.width.value(), self.height.value(), 0), interpolation="nearest")
        field_ax.set(title=f"Final particle-velocity magnitude (dB), {result.backend}",
                     xlabel="X (mm)", ylabel="Y (mm)")
        for channel in result.receiver_channels_pa:
            trace_ax.plot(result.time_s * 1e6, channel * 1e-6,
                          color="#94a3b8", lw=0.45, alpha=0.25)
        trace_ax.plot(result.time_s * 1e6, result.receiver_pa * 1e-6,
                      color="#0b7285", lw=1.4, label="Focused DAS")
        trace_ax.set(title="Per-element RF and focused A-scan", xlabel="Time (µs)",
                     ylabel="Near-surface normal stress (MPa)")
        trace_ax.legend(loc="upper right")
        trace_ax.grid(True, alpha=0.25)
        self.result_canvas.draw_idle()
        frame_count = result.snapshots_magnitude.shape[0]
        self.frame_slider.setRange(0, max(0, frame_count - 1))
        self.frame_slider.setValue(0)
        self.play_button.setEnabled(frame_count > 1)
        self.update_history_frame(0)
        self.summary.setText(
            f"Completed with {result.backend}\nGrid: {result.ny} × {result.nx}, "
            f"steps: {result.time_s.size}, recorded frames: {frame_count}\n"
            f"Runtime: {result.runtime_s:.3f} s"
        )
        self.statusBar().showMessage("Simulation completed")

    @Slot(str)
    def fail_run(self, details: str) -> None:
        self.statusBar().showMessage("Simulation failed")
        QMessageBox.critical(self, "Simulation failed", details)

    @Slot()
    def _thread_finished(self) -> None:
        self.run_button.setEnabled(True); self.preview_button.setEnabled(True)
        self.thread = None; self.worker = None

    @Slot(int)
    def update_history_frame(self, index: int) -> None:
        if self.result is None or self.result.snapshots_magnitude.shape[0] == 0:
            self.frame_label.setText("No time-history frames")
            return
        index = min(index, self.result.snapshots_magnitude.shape[0] - 1)
        frame = self.result.snapshots_magnitude[index]
        peak = max(float(np.max(self.result.snapshots_magnitude)), 1e-30)
        db = np.maximum(20.0 * np.log10(frame / peak + 1e-12), -60.0)
        axis = self.history_canvas.axes[0]
        axis.clear()
        axis.imshow(
            db, cmap="turbo", vmin=-60, vmax=0, origin="upper",
            extent=(0, self.width.value(), self.height.value(), 0),
            interpolation="nearest",
        )
        step = int(self.result.snapshot_steps[index])
        time_us = float(self.result.snapshot_time_s[index]) * 1e6
        axis.set(
            title=f"Wave propagation — step {step:,}, t = {time_us:.6f} µs",
            xlabel="X (mm)", ylabel="Y (mm)",
        )
        self.frame_label.setText(
            f"Frame {index + 1}/{self.result.snapshots_magnitude.shape[0]} | "
            f"step {step} | {time_us:.6f} µs"
        )
        self.history_canvas.draw_idle()

    @Slot()
    def toggle_history_playback(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop(); self.play_button.setText("Play")
        else:
            if self.frame_slider.value() >= self.frame_slider.maximum():
                self.frame_slider.setValue(0)
            self.play_timer.start(); self.play_button.setText("Pause")

    @Slot()
    def advance_history_frame(self) -> None:
        if self.frame_slider.value() >= self.frame_slider.maximum():
            self.play_timer.stop(); self.play_button.setText("Play")
            return
        self.frame_slider.setValue(self.frame_slider.value() + 1)

    def save_project_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "simndt_project.json", "JSON (*.json)")
        if path: save_project(self.collect_config(), path)

    def load_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "JSON (*.json)")
        if path:
            try: self.apply_config(load_project(path)); self.preview()
            except Exception as exc: QMessageBox.critical(self, "Open failed", str(exc))

    def export_result_dialog(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "No result", "Run a simulation first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export result", "simndt_result.npz", "NumPy archive (*.npz)")
        if path: save_result(self.result, path)

    def export_animation_dialog(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "No result", "Run a simulation first."); return
        if self.result.snapshots_magnitude.shape[0] == 0:
            QMessageBox.information(
                self, "No time history",
                "Enable 'Record time-history frames' and run the simulation again.",
            ); return
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export wave-propagation animation", "simndt_wavefield.gif",
            "Animated GIF (*.gif);;MP4 video (*.mp4)",
        )
        if not path: return
        if not Path(path).suffix:
            path += ".mp4" if "MP4" in selected_filter else ".gif"
        try:
            export_animation(self.result, path)
            QMessageBox.information(self, "Export completed", f"Animation saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Animation export failed", str(exc))

    def export_frames_dialog(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "No result", "Run a simulation first."); return
        if self.result.snapshots_magnitude.shape[0] == 0:
            QMessageBox.information(
                self, "No time history",
                "Enable 'Record time-history frames' and run the simulation again.",
            ); return
        directory = QFileDialog.getExistingDirectory(self, "Choose folder for time-step images")
        if not directory: return
        try:
            manifest = export_frame_images(self.result, directory)
            QMessageBox.information(
                self, "Export completed",
                f"PNG frames and time manifest saved to:\n{manifest.parent}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Frame export failed", str(exc))


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SimNDT Modern")
    app.setOrganizationName("SimNDT Modern")
    app.setStyle("Fusion")
    app.setStyleSheet("QGroupBox { font-weight: 600; margin-top: 8px; } QGroupBox::title { left: 8px; }")
    window = MainWindow(); window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
