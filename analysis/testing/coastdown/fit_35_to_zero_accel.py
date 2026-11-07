r"""Smooth the composite 35-to-0 mph coast and fit acceleration versus speed.

Run from the repository root after stitch_35_to_zero.py::

    .venv\Scripts\python.exe analysis\testing\coastdown\fit_35_to_zero_accel.py

Each measured segment is median-binned at 100 ms, then smoothed with a
Savitzky-Golay local quadratic. Its derivative gives acceleration. Filtering
is done separately at the two joins; edge bins are omitted from the fit.
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

from fit_35mph_drag import DEFAULT_OUTPUT_DIR, MPH_TO_MPS


SAMPLE_PERIOD_S = 0.1
DEFAULT_WINDOW_S = 1.1


def read_composite(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {
                "time_s": float(row["composite_time_s"]),
                "speed_mph": float(row["speed_mph"]),
                "segment": int(row["segment"]),
            }
            for row in csv.DictReader(stream)
        ]
    if len(rows) < 10 or not np.all(np.diff([row["time_s"] for row in rows]) > 0):
        raise ValueError("Composite CSV needs at least 10 samples and increasing time")
    return rows


def smooth_segment(
    rows: list[dict[str, float | int]],
    start_s: float,
    finish_s: float,
    window_s: float,
) -> dict[str, np.ndarray]:
    time = np.asarray([row["time_s"] for row in rows], dtype=float)
    speed = np.asarray([row["speed_mph"] for row in rows], dtype=float)
    count = int(np.floor((finish_s - start_s) / SAMPLE_PERIOD_S))
    window_samples = max(5, int(round(window_s / SAMPLE_PERIOD_S)))
    if window_samples % 2 == 0:
        window_samples += 1
    if count <= window_samples:
        raise ValueError("Smoothing window must be shorter than every segment")

    edges = start_s + np.arange(count + 1) * SAMPLE_PERIOD_S
    centers = edges[:-1] + SAMPLE_PERIOD_S / 2
    median_speed = np.empty(count)
    for i in range(count):
        lo = np.searchsorted(time, edges[i], side="left")
        hi = np.searchsorted(time, edges[i + 1], side="left")
        if hi <= lo:
            raise ValueError(f"No speed samples in {edges[i]:.2f}-{edges[i + 1]:.2f} s")
        median_speed[i] = np.median(speed[lo:hi])

    smooth_speed = savgol_filter(
        median_speed, window_length=window_samples, polyorder=2, mode="interp"
    )
    acceleration = savgol_filter(
        median_speed * MPH_TO_MPS,
        window_length=window_samples,
        polyorder=2,
        deriv=1,
        delta=SAMPLE_PERIOD_S,
        mode="interp",
    )
    included = np.zeros(count, dtype=bool)
    half_window = window_samples // 2
    included[half_window:-half_window] = True
    return {
        "time_s": centers,
        "median_speed_mph": median_speed,
        "smooth_speed_mph": smooth_speed,
        "accel_mps2": acceleration,
        "used_in_fit": included,
    }


def fit_quadratic(segments: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, float, float, int, tuple[float, float]]:
    velocity = np.concatenate([segment["smooth_speed_mph"][segment["used_in_fit"]] for segment in segments])
    acceleration = np.concatenate([segment["accel_mps2"][segment["used_in_fit"]] for segment in segments])
    if len(velocity) < 3 or np.ptp(velocity) < 1:
        raise ValueError("Insufficient velocity range for quadratic fit")
    coefficients = np.polynomial.Polynomial.fit(velocity, acceleration, 2).convert().coef
    residual = acceleration - np.polynomial.polynomial.polyval(velocity, coefficients)
    rmse = float(np.sqrt(np.mean(residual**2)))
    variation = float(np.sum((acceleration - np.mean(acceleration)) ** 2))
    r_squared = 1 - float(np.sum(residual**2)) / variation if variation > 0 else float("nan")
    return coefficients, rmse, r_squared, len(velocity), (float(np.min(velocity)), float(np.max(velocity)))


def save_outputs(
    output_dir: Path,
    input_csv: Path,
    raw_rows: list[dict[str, float | int]],
    segments: list[dict[str, np.ndarray]],
    joins_s: list[float],
    window_s: float,
) -> None:
    coefficients, rmse, r_squared, fit_count, fit_range = fit_quadratic(segments)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "vcu_coastdown_35_to_0_smoothed_accel.csv"
    trace_path = output_dir / "vcu_coastdown_35_to_0_smoothed_trace.png"
    fit_path = output_dir / "vcu_coastdown_35_to_0_accel_quadratic.png"
    report_path = output_dir / "vcu_coastdown_35_to_0_accel_quadratic.json"

    with samples_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("segment", "composite_time_s", "speed_mph_100ms_median", "speed_mph_smoothed", "acceleration_mps2", "used_in_fit", "quadratic_acceleration_mps2"))
        for number, segment in enumerate(segments, start=1):
            predicted = np.polynomial.polynomial.polyval(segment["smooth_speed_mph"], coefficients)
            for t, median, smooth, accel, fit, estimate in zip(
                segment["time_s"], segment["median_speed_mph"], segment["smooth_speed_mph"],
                segment["accel_mps2"], segment["used_in_fit"], predicted, strict=True,
            ):
                writer.writerow((number, f"{t:.5f}", f"{median:.5f}", f"{smooth:.5f}", f"{accel:.6f}", int(fit), f"{estimate:.6f}"))

    colors = ("#1f77b4", "#e6862d", "#2a9d8f")
    figure, (speed_axis, accel_axis) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained")
    speed_axis.plot([row["time_s"] for row in raw_rows], [row["speed_mph"] for row in raw_rows], color="0.78", lw=0.8, label="Raw VCU speed")
    for number, segment in enumerate(segments, start=1):
        color = colors[(number - 1) % len(colors)]
        speed_axis.plot(segment["time_s"], segment["smooth_speed_mph"], color=color, lw=2, label=f"Smoothed segment {number}")
        accel_axis.plot(
            segment["time_s"],
            np.where(segment["used_in_fit"], segment["accel_mps2"], np.nan),
            color=color, lw=1.8, label=f"Derived segment {number}",
        )
        accel_axis.plot(segment["time_s"], np.polynomial.polynomial.polyval(segment["smooth_speed_mph"], coefficients), color=color, lw=1, linestyle="--", alpha=0.8)
    for axis in (speed_axis, accel_axis):
        for join in joins_s:
            axis.axvline(join, color="0.5", linestyle=":", linewidth=1)
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    speed_axis.set(ylabel="VCU speed [mph]", title="Smoothed composite coastdown")
    accel_axis.set(xlabel="Composite elapsed time [s]", ylabel="Acceleration [m/s²]", title="Derived acceleration (join/filter edges omitted; dashed: quadratic prediction)")
    speed_axis.legend(frameon=False, fontsize=8)
    accel_axis.legend(frameon=False, fontsize=8)
    figure.savefig(trace_path, dpi=200, facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6), layout="constrained")
    for number, segment in enumerate(segments, start=1):
        fit = segment["used_in_fit"]
        axis.scatter(segment["smooth_speed_mph"][fit], segment["accel_mps2"][fit], s=20, alpha=0.7, color=colors[(number - 1) % len(colors)], label=f"Segment {number}")
    velocity_line = np.linspace(fit_range[0], fit_range[1], 300)
    axis.plot(velocity_line, np.polynomial.polynomial.polyval(velocity_line, coefficients), color="#c44e52", lw=2.5, label="Quadratic fit")
    axis.set(xlabel="Smoothed VCU speed [mph]", ylabel="Acceleration [m/s²]", title=f"a = {coefficients[0]:+.4f} {coefficients[1]:+.4f}v {coefficients[2]:+.6f}v²\nRMSE {rmse:.3f} m/s²; R² {r_squared:.3f} (v in mph)")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.savefig(fit_path, dpi=200, facecolor="white")
    plt.close(figure)

    report = {
        "input_csv": str(input_csv.resolve()),
        "description": "Composite measured coast; filter and derivative applied separately to each segment.",
        "sample_period_s": SAMPLE_PERIOD_S,
        "smoothing": f"100 ms median bins followed by {window_s:g} s Savitzky-Golay local quadratic",
        "acceleration": "Savitzky-Golay first derivative of median speed converted to m/s",
        "fit_equation": "acceleration_mps2 = c0 + c1*speed_mph + c2*speed_mph^2",
        "coefficients": {name: float(value) for name, value in zip(("c0", "c1", "c2"), coefficients, strict=True)},
        "fit_samples": fit_count,
        "fit_velocity_range_mph": fit_range,
        "excluded": "First and last half-window of each segment, including both joins",
        "rmse_mps2": rmse,
        "r_squared": r_squared,
        "samples_csv": str(samples_path.resolve()),
        "trace_plot": str(trace_path.resolve()),
        "quadratic_plot": str(fit_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"a [m/s²] = {coefficients[0]:+.6f} {coefficients[1]:+.6f}v {coefficients[2]:+.8f}v² (v in mph)")
    print(f"Fit: {fit_count} points, RMSE {rmse:.4f} m/s², R² {r_squared:.3f}")
    print(f"Trace: {trace_path.resolve()}")
    print(f"Acceleration vs velocity: {fit_path.resolve()}")
    print(f"Samples: {samples_path.resolve()}")
    print(f"Report: {report_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_OUTPUT_DIR / "vcu_coastdown_35_to_0_composite.csv")
    parser.add_argument("--input-report", type=Path, default=DEFAULT_OUTPUT_DIR / "vcu_coastdown_35_to_0_composite.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    args = parser.parse_args()
    if not np.isfinite(args.window_s) or args.window_s <= 0:
        raise ValueError("--window-s must be finite and positive")
    raw_rows = read_composite(args.input_csv)
    report = json.loads(args.input_report.read_text(encoding="utf-8"))
    source_rows = report["sources"]
    segments = []
    for source in source_rows:
        number = source["segment"]
        selected = [row for row in raw_rows if row["segment"] == number]
        start, finish = source["composite_window_s"]
        segments.append(smooth_segment(selected, start, finish, args.window_s))
    joins_s = [float(source["composite_window_s"][0]) for source in source_rows[1:]]
    save_outputs(args.output_dir, args.input_csv, raw_rows, segments, joins_s, args.window_s)


if __name__ == "__main__":
    main()
