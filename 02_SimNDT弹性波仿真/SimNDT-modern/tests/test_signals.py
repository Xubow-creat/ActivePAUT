from __future__ import annotations

import numpy as np

from simndt_modern.signals import gaussian_tone_burst


def test_tone_burst_is_bounded_and_finite() -> None:
    time = np.linspace(0, 10e-6, 2000)
    signal = gaussian_tone_burst(time, 1e6, 3)
    assert signal.dtype == np.float32
    assert np.isfinite(signal).all()
    assert np.max(np.abs(signal)) == pytest.approx(1.0, rel=1e-6)
    assert np.all(signal[time > 3e-6] == 0)


import pytest

