r"""Downsample the 35 mph VCU coasts to 100 ms and fit acceleration vs speed.

Run from the repository root with::

    .venv\Scripts\python.exe analysis\testing\coastdown\plot_35mph_accel.py

Acceleration is the Savitzky-Golay derivative of the 100 ms median speed.
The empirical fit is a full quadratic in speed (mph), with acceleration in
m/s². Filter-edge samples are plotted but excluded from the regression.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from fit_35mph_drag import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WINDOWS,
    MPH_TO_MPS,
    read_control_summary,
    read_vcu_speed,
)


SAMPLE_PERIOD_S = 0.1
DERIVATIVE_WINDOW_S = 0.7


def downsample_coast(
    time_s: np.ndarray,
    speed_mph: np.ndarray,
    start_s: float,
    finish_s: float,
    derivative_window_s: float,
) -> dict[str, np.ndarray]:
    """Median-bin speed, then differentiate on the uniform 100 ms grid."""

    if not np.isfinite(derivative_window_s) or derivative_window_s <= 0:
        raise ValueError("Derivative window must be finite and positive")
    count = int(np.floor((finish_s - start_s) / SAMPLE_PERIOD_S + 1e-8))
    if count < 7:
        raise ValueError(f"Coast window {start_s:g}-{finish_s:g} s is too short")
    edges = start_s + np.arange(count + 1) * SAMPLE_PERIOD_S
    centers = edges[:-1] + SAMPLE_PERIOD_S / 2
    speed_100ms = np.empty(count, dtype=float)
    for i in range(count):
        lo = np.searchsorted(time_s, edges[i], side="left")
        hi = np.searchsorted(time_s, edges[i + 1], side="left")
        if hi == lo:
            raise ValueError(f"No VCU speed samples in {edges[i]:.2f}-{edges[i + 1]:.2f} s")
        speed_100ms[i] = float(np.median(speed_mph[lo:hi]))

    window_samples = max(5, int(round(derivative_window_s / SAMPLE_PERIOD_S)))
    if window_samples % 2 == 0:
        window_samples += 1
    if window_samples >= count:
        raise ValueError("Derivative window must be shorter than each coast window")
    raw_accel = np.gradient(speed_100ms * MPH_TO_MPS, SAMPLE_PERIOD_S)
    filtered_accel = savgol_filter(
        speed_100ms * MPH_TO_MPS,
        window_length=window_samples,
        polyorder=2,
        deriv=1,
        delta=SAMPLE_PERIOD_S,
        mode="interp",
    )
    fit_mask = np.zeros(count, dtype=bool)
    fit_mask[window_samples // 2 : -(window_samples // 2)] = True
    return {
        "time_s": centers,
        "speed_mph": speed_100ms,
        "raw_accel_mps2": raw_accel,
        "filtered_accel_mps2": filtered_accel,
        "fit_mask": fit_mask,
    }


def fit_quadratic(coasts: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, float, float]:
    """Fit a = c0 + c1*v + c2*v², using speed in mph."""

    speed = np.concatenate([run["speed_mph"][run["fit_mask"]] for run in coasts])
    accel = np.concatenate(
        [run["filtered_accel_mps2"][run["fit_mask"]] for run in coasts]
    )
    if len(speed) < 3 or np.ptp(speed) < 1.0:
        raise ValueError("Insufficient speed range for a quadratic fit")
    coefficients = np.polynomial.Polynomial.fit(speed, accel, 2).convert().coef
    prediction = np.polynomial.polynomial.polyval(speed, coefficients)
    residual = accel - prediction
    rmse = float(np.sqrt(np.mean(residual**2)))
    variance = float(np.sum((accel - np.mean(accel)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / variance if variance > 0 else float("nan")
    return coefficients, rmse, r_squared


def save_csv(coasts: list[dict[str, np.ndarray]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "run",
                "time_since_first_vcu_sample_s",
                "speed_mph_100ms_median",
                "accel_central_difference_mps2",
                "accel_savgol_mps2",
                "used_in_fit",
            )
        )
        for run_number, coast in enumerate(coasts, start=1):
            for time_s, speed, raw, filtered, included in zip(
                coast["time_s"],
                coast["speed_mph"],
                coast["raw_accel_mps2"],
                coast["filtered_accel_mps2"],
                coast["fit_mask"],
                strict=True,
            ):
                writer.writerow(
                    (run_number, f"{time_s:.3f}", f"{speed:.4f}", f"{raw:.6f}",
                     f"{filtered:.6f}", int(included))
                )


def plot_acceleration(
    coasts: list[dict[str, np.ndarray]],
    windows: tuple[tuple[float, float], ...],
    coefficients: np.ndarray,
    rmse: float,
    r_squared: float,
    derivative_window_s: float,
    path: Path,
) -> None:
    figure = plt.figure(figsize=(12, 8.5), layout="constrained")
    grid = figure.add_gridspec(2, len(coasts))
    colors = ("#1f77b4", "#ed7d31")
    for i, (coast, (start_s, finish_s)) in enumerate(zip(coasts, windows, strict=True)):
        axis = figure.add_subplot(grid[0, i])
        local_time = coast["time_s"] - start_s
        axis.plot(
            local_time,
            coast["raw_accel_mps2"],
            ".", color="#aaaaaa", markersize=5, label="100 ms finite difference",
        )
        axis.plot(
            local_time,
            np.where(coast["fit_mask"], coast["filtered_accel_mps2"], np.nan),
            "-o", color=colors[i % len(colors)], markersize=3,
            label=f"{derivative_window_s:g} s derivative (fit region)",
        )
        axis.plot(
            local_time,
            np.polynomial.polynomial.polyval(coast["speed_mph"], coefficients),
            "--", color="#c44e52", linewidth=1.8, label="Quadratic fit",
        )
        axis.set_title(f"Run {i + 1}: {start_s:g}-{finish_s:g} s")
        axis.set_xlabel("Time since coast-window start [s]")
        axis.set_ylabel("Acceleration [m/s²]")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

    axis = figure.add_subplot(grid[1, :])
    for i, coast in enumerate(coasts):
        selected = coast["fit_mask"]
        axis.scatter(
            coast["speed_mph"][selected],
            coast["filtered_accel_mps2"][selected],
            s=28, alpha=0.8, color=colors[i % len(colors)], label=f"Run {i + 1}",
        )
    speeds = np.concatenate([coast["speed_mph"][coast["fit_mask"]] for coast in coasts])
    fit_speed = np.linspace(float(np.min(speeds)), float(np.max(speeds)), 200)
    axis.plot(
        fit_speed,
        np.polynomial.polynomial.polyval(fit_speed, coefficients),
        color="#c44e52", linewidth=2.2, label="Quadratic fit",
    )
    axis.set_xlabel("100 ms median VCU speed [mph]")
    axis.set_ylabel("Acceleration [m/s²]")
    axis.set_title(
        "a = "
        f"{coefficients[0]:+.4f} {coefficients[1]:+.4f}v "
        f"{coefficients[2]:+.6f}v²   "
        f"(RMSE {rmse:.3f} m/s²; R² {r_squared:.2f})"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.suptitle("35 mph coastdowns: 100 ms VCU speed and empirical acceleration fit")
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--derivative-window-s", type=float, default=DERIVATIVE_WINDOW_S)
    args = parser.parse_args()
    windows = DEFAULT_WINDOWS
    time_s, speed_mph, _, _ = read_vcu_speed(args.input)
    read_control_summary(args.input, windows)
    coasts = [
        downsample_coast(time_s, speed_mph, start, finish, args.derivative_window_s)
        for start, finish in windows
    ]
    coefficients, rmse, r_squared = fit_quadratic(coasts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "vcu_speed_35mph_100ms_accel_quadratic.png"
    csv_path = args.output_dir / "vcu_speed_35mph_100ms.csv"
    report_path = args.output_dir / "vcu_speed_35mph_100ms_accel_quadratic.json"
    plot_acceleration(
        coasts, windows, coefficients, rmse, r_squared,
        args.derivative_window_s, plot_path,
    )
    save_csv(coasts, csv_path)
    report = {
        "input": str(args.input.resolve()),
        "windows_s": windows,
        "sample_period_s": SAMPLE_PERIOD_S,
        "downsampling": "Median raw VCU speed within each 100 ms bin.",
        "acceleration": f"Savitzky-Golay derivative, {args.derivative_window_s:g} s window, quadratic local polynomial; filter edges excluded from fit.",
        "fit_equation": "accel_mps2 = c0 + c1*speed_mph + c2*speed_mph^2",
        "coefficients": {name: float(value) for name, value in zip(("c0", "c1", "c2"), coefficients, strict=True)},
        "fit_samples": int(sum(np.sum(run["fit_mask"]) for run in coasts)),
        "rmse_mps2": rmse,
        "r_squared": r_squared,
        "plot": str(plot_path.resolve()),
        "samples_csv": str(csv_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "Empirical quadratic (v in mph, a in m/s²): "
        f"a = {coefficients[0]:+.6f} {coefficients[1]:+.6f}v "
        f"{coefficients[2]:+.8f}v²"
    )
    print(f"Fit: {report['fit_samples']} points, RMSE {rmse:.4f} m/s², R² {r_squared:.3f}")
    print(f"Plot: {plot_path.resolve()}")
    print(f"100 ms samples: {csv_path.resolve()}")
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
