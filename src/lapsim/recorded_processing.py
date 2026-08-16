"""Signal processing used to prepare recorded laps for replay.

These functions only align and smooth measurements.  They do not identify or
change any vehicle-model parameter.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .recorded_lap import RecordedLap


GPS_SMOOTHING_WINDOW_S = 1.01
MAXIMUM_ABSOLUTE_CURVATURE_PER_M = 0.15


def smoothing_window_samples(time_s: np.ndarray) -> int:
    """Return an odd Savitzky-Golay window near the requested duration."""

    timestep_s = float(np.median(np.diff(time_s)))
    samples = max(5, int(round(GPS_SMOOTHING_WINDOW_S / timestep_s)))
    return samples + 1 if samples % 2 == 0 else samples


def smooth_measurements(lap: RecordedLap) -> dict[str, np.ndarray]:
    """Smooth GNSS motion and derive acceleration, heading, and curvature."""

    time_s = np.asarray(lap.time_s)
    window = smoothing_window_samples(time_s)
    timestep_s = float(np.median(np.diff(time_s)))
    x_m = np.asarray(lap.x_m)
    y_m = np.asarray(lap.y_m)
    speed_mps = np.asarray(lap.speed_mps)

    smoothed_x_m = savgol_filter(x_m, window, 3)
    smoothed_y_m = savgol_filter(y_m, window, 3)
    smoothed_speed_mps = savgol_filter(speed_mps, window, 3)
    acceleration_mps2 = savgol_filter(
        speed_mps, window, 3, deriv=1, delta=timestep_s
    )
    x_velocity_mps = savgol_filter(x_m, window, 3, deriv=1, delta=timestep_s)
    y_velocity_mps = savgol_filter(y_m, window, 3, deriv=1, delta=timestep_s)
    x_acceleration_mps2 = savgol_filter(
        x_m, window, 3, deriv=2, delta=timestep_s
    )
    y_acceleration_mps2 = savgol_filter(
        y_m, window, 3, deriv=2, delta=timestep_s
    )
    denominator = np.maximum(
        (x_velocity_mps**2 + y_velocity_mps**2) ** 1.5,
        1.0,
    )
    curvature_per_m = np.clip(
        (
            x_velocity_mps * y_acceleration_mps2
            - y_velocity_mps * x_acceleration_mps2
        )
        / denominator,
        -MAXIMUM_ABSOLUTE_CURVATURE_PER_M,
        MAXIMUM_ABSOLUTE_CURVATURE_PER_M,
    )
    return {
        "x_m": smoothed_x_m,
        "y_m": smoothed_y_m,
        "speed_mps": smoothed_speed_mps,
        "acceleration_mps2": acceleration_mps2,
        "curvature_per_m": curvature_per_m,
        "heading_rad": np.unwrap(np.arctan2(y_velocity_mps, x_velocity_mps)),
    }


def aligned(
    values: np.ndarray,
    time_s: np.ndarray,
    latency_s: float,
) -> np.ndarray:
    """Sample a measurement at ``time_s + latency_s``."""

    return np.interp(time_s + latency_s, time_s, values)
