from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gaussian_tone_burst(
    time_s: NDArray[np.floating],
    center_frequency_hz: float,
    cycles: float,
) -> NDArray[np.float32]:
    """Return a zero-mean Gaussian-windowed sinusoidal burst."""
    duration = cycles / center_frequency_hz
    center = 0.55 * duration
    sigma = duration / 6.0
    envelope = np.exp(-0.5 * ((time_s - center) / sigma) ** 2)
    signal = envelope * np.sin(2.0 * np.pi * center_frequency_hz * (time_s - center))
    signal[(time_s < 0.0) | (time_s > duration)] = 0.0
    peak = float(np.max(np.abs(signal)))
    if peak:
        signal /= peak
    return signal.astype(np.float32)


def apodization_weights(count: int, kind: str, tukey_alpha: float = 0.5) -> NDArray[np.float32]:
    """Return symmetric element weights with a unit peak."""
    if count < 1:
        raise ValueError("Element count must be positive.")
    if count == 1 or kind == "rectangular":
        return np.ones(count, dtype=np.float32)
    if kind == "hann":
        weights = np.hanning(count)
    elif kind == "hamming":
        weights = np.hamming(count)
    elif kind == "tukey":
        # Local implementation avoids making SciPy a requirement of the
        # numerical core.
        x = np.linspace(0.0, 1.0, count)
        weights = np.ones(count)
        edge = tukey_alpha / 2.0
        left = x < edge
        right = x > 1.0 - edge
        weights[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[left] / tukey_alpha - 1.0)))
        weights[right] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[right] / tukey_alpha - 2.0 / tukey_alpha + 1.0)))
    else:
        raise ValueError(f"Unsupported apodization: {kind}")
    peak = float(np.max(weights))
    if peak:
        weights /= peak
    return weights.astype(np.float32)


def focusing_delays_s(
    element_centers_m: NDArray[np.floating],
    array_center_m: float,
    steering_angle_deg: float,
    focus_depth_m: float,
    wave_speed_m_s: float,
) -> NDArray[np.float64]:
    """Geometrical transmit delays, shifted so the earliest delay is zero.

    Positive steering angles point towards increasing x.  A zero focus depth
    represents a plane wave; otherwise the focus lies at
    ``array_center + depth*tan(angle)``.
    """
    centers = np.asarray(element_centers_m, dtype=np.float64)
    theta = np.deg2rad(float(steering_angle_deg))
    if focus_depth_m > 0:
        focus_x = array_center_m + focus_depth_m * np.tan(theta)
        distances = np.sqrt((centers - focus_x) ** 2 + focus_depth_m**2)
        delays = (np.max(distances) - distances) / wave_speed_m_s
    else:
        projection = centers * np.sin(theta)
        delays = (projection - np.min(projection)) / wave_speed_m_s
    delays -= np.min(delays)
    return delays


def delayed_tone_burst_bank(
    time_s: NDArray[np.floating],
    center_frequency_hz: float,
    cycles: float,
    delays_s: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Evaluate the analytic burst at fractional time shifts per element."""
    rows = [gaussian_tone_burst(np.asarray(time_s) - delay, center_frequency_hz, cycles)
            for delay in np.asarray(delays_s, dtype=np.float64)]
    return np.stack(rows).astype(np.float32, copy=False)


def delay_and_sum(
    channels: NDArray[np.floating],
    time_s: NDArray[np.floating],
    delays_s: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Static receive focusing using fractional-delay linear interpolation."""
    channels = np.asarray(channels, dtype=np.float64)
    time = np.asarray(time_s, dtype=np.float64)
    delays = np.asarray(delays_s, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if channels.shape != (delays.size, time.size) or weights.size != delays.size:
        raise ValueError("Channel, delay, weight, and time dimensions are inconsistent.")
    result = np.zeros(time.size, dtype=np.float64)
    for channel, delay, weight in zip(channels, delays, weights, strict=True):
        # Echoes from the nearer elements arrive early.  Delaying those traces
        # by their geometrical transmit-delay complement aligns the focus.
        result += weight * np.interp(time - delay, time, channel, left=0.0, right=0.0)
    norm = float(np.sum(np.abs(weights)))
    if norm:
        result /= norm
    return result.astype(np.float32)


def aperture_profile(x_m: NDArray[np.floating], center_m: float, aperture_m: float) -> NDArray[np.float32]:
    half = max(aperture_m / 2.0, np.finfo(float).eps)
    relative = (x_m - center_m) / half
    profile = np.zeros_like(x_m, dtype=np.float32)
    inside = np.abs(relative) <= 1.0
    profile[inside] = (0.5 * (1.0 + np.cos(np.pi * relative[inside]))).astype(np.float32)
    return profile
