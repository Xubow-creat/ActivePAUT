from __future__ import annotations

from simndt_modern.app import MainWindow
from simndt_modern.models import BackendKind, BoundaryKind


def test_main_window_constructs_and_previews(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview()
    assert "Grid:" in window.summary.text()
    assert window.backend.count() >= 1


def test_gui_normalizes_qt_combo_data_to_enums(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    config = window.collect_config()
    assert isinstance(config.numerics.backend, BackendKind)
    assert isinstance(config.boundaries.top.kind, BoundaryKind)
