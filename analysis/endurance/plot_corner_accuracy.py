"""Create corner-by-corner accuracy plots from the distance-domain lap replay."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis/endurance/output/first_lap_all_data.csv"
DEFAULT_TRACK = ROOT / "analysis/data/track/gnss_imu_endurance_track.csv"
DEFAULT_OUTPUT = ROOT / "analysis/endurance/output/corner_accuracy"

MEASURED_COLOR = "#2678B2"
SIMULATED_COLOR = "#E67E22"
APEX_COLOR = "#6F4C9B"
NEUTRAL_COLOR = "#737373"
CORNER_COLORS = (
    "#2678B2",
    "#E67E22",
    "#2C9F67",
    "#8A5FB5",
    "#D64F70",
    "#A1761D",
)


@dataclass(frozen=True, slots=True)
class CornerWindow:
    number: int
    start_m: float
    end_m: float
    apex_m: float
    peak_absolute_curvature_per_m: float
    approximate_apex_radius_m: float
    curvature_sign: str
    sample_count: int
    speed_rmse_mps: float
    speed_bias_mps: float
    longitudinal_acceleration_rmse_mps2: float
    longitudinal_acceleration_bias_mps2: float
    lateral_acceleration_rmse_mps2: float
    lateral_acceleration_bias_mps2: float

    @property
    def label(self) -> str:
        return f"C{self.number}  {self.start_m:.0f}-{self.end_m:.0f} m"


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def read_track(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"Track CSV needs at least two points: {path}")
    return {
        "distance_m": np.asarray([float(row["distance_m"]) for row in rows]),
        "x_m": np.asarray([float(row["x_m"]) for row in rows]),
        "y_m": np.asarray([float(row["y_m"]) for row in rows]),
        "curvature_per_m": np.asarray(
            [float(row["curvature_per_m"]) for row in rows[:-1]]
        ),
    }


def rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def detect_corner_regions(
    track: dict[str, np.ndarray],
    *,
    corner_count: int,
    curvature_threshold_per_m: float,
    smoothing_distance_m: float,
    maximum_bridge_distance_m: float,
    minimum_region_length_m: float,
    lap_start_exclusion_m: float,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    station_m = track["distance_m"][:-1]
    curvature_per_m = track["curvature_per_m"]
    spacing_m = float(np.median(np.diff(station_m)))
    sigma_samples = max(smoothing_distance_m / spacing_m, 1.0)
    smooth_absolute_curvature = gaussian_filter1d(
        np.abs(curvature_per_m), sigma_samples, mode="wrap"
    )
    corner_mask = smooth_absolute_curvature >= curvature_threshold_per_m

    corner_indices = np.flatnonzero(corner_mask)
    maximum_gap_samples = max(int(round(maximum_bridge_distance_m / spacing_m)), 1)
    for lower_index, upper_index in zip(
        corner_indices[:-1], corner_indices[1:], strict=False
    ):
        if upper_index - lower_index <= maximum_gap_samples:
            corner_mask[lower_index : upper_index + 1] = True

    transitions = np.diff(np.r_[False, corner_mask, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    candidates: list[tuple[float, int, int]] = []
    for start_index, end_index in zip(starts, ends, strict=True):
        start_m = station_m[start_index]
        end_m = station_m[end_index] + spacing_m
        if start_m < lap_start_exclusion_m:
            continue
        if end_m - start_m < minimum_region_length_m:
            continue
        region = slice(start_index, end_index + 1)
        score = float(
            np.trapezoid(smooth_absolute_curvature[region], station_m[region])
        )
        candidates.append((score, start_index, end_index))

    if len(candidates) < corner_count:
        raise ValueError(
            f"Only {len(candidates)} corner regions met the selection criteria"
        )
    selected = sorted(candidates, reverse=True)[:corner_count]
    return (
        sorted(
            [(start_index, end_index) for _, start_index, end_index in selected],
            key=lambda item: item[0],
        ),
        smooth_absolute_curvature,
    )


def build_corner_windows(
    data: dict[str, np.ndarray],
    track: dict[str, np.ndarray],
    regions: list[tuple[int, int]],
    smooth_absolute_curvature: np.ndarray,
    *,
    padding_m: float,
    maximum_window_length_m: float,
) -> list[CornerWindow]:
    required = (
        "measured_distance_m",
        "analysis_shifted_gnss_speed_mps",
        "aligned_imu_longitudinal_accel_mps2",
        "aligned_imu_lateral_accel_mps2",
        "sim_vehicle_speed_mps",
        "sim_vehicle_longitudinal_acceleration_mps2",
        "sim_vehicle_lateral_acceleration_mps2",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("Full-lap CSV is missing: " + ", ".join(missing))

    track_station_m = track["distance_m"][:-1]
    track_curvature_per_m = track["curvature_per_m"]
    spacing_m = float(np.median(np.diff(track_station_m)))
    lap_length_m = float(track["distance_m"][-1])
    measured_station_m = data["measured_distance_m"]
    windows: list[CornerWindow] = []
    for number, (start_index, end_index) in enumerate(regions, start=1):
        region_indices = np.arange(start_index, end_index + 1)
        apex_index = int(
            region_indices[
                np.argmax(smooth_absolute_curvature[start_index : end_index + 1])
            ]
        )
        apex_m = float(track_station_m[apex_index])
        start_m = max(float(track_station_m[start_index]) - padding_m, 0.0)
        end_m = min(
            float(track_station_m[end_index]) + spacing_m + padding_m,
            lap_length_m,
        )
        if end_m - start_m > maximum_window_length_m:
            start_m = max(apex_m - 0.5 * maximum_window_length_m, 0.0)
            end_m = min(start_m + maximum_window_length_m, lap_length_m)
            start_m = max(end_m - maximum_window_length_m, 0.0)

        sample_mask = (measured_station_m >= start_m) & (measured_station_m <= end_m)
        sample_count = int(np.count_nonzero(sample_mask))
        if sample_count < 4:
            raise ValueError(
                f"Corner {number} has only {sample_count} recorded samples"
            )
        speed_error = (
            data["sim_vehicle_speed_mps"][sample_mask]
            - data["analysis_shifted_gnss_speed_mps"][sample_mask]
        )
        longitudinal_error = (
            data["sim_vehicle_longitudinal_acceleration_mps2"][sample_mask]
            - data["aligned_imu_longitudinal_accel_mps2"][sample_mask]
        )
        lateral_error = (
            data["sim_vehicle_lateral_acceleration_mps2"][sample_mask]
            - data["aligned_imu_lateral_accel_mps2"][sample_mask]
        )
        peak_curvature_per_m = float(smooth_absolute_curvature[apex_index])
        signed_curvature_per_m = float(track_curvature_per_m[apex_index])
        windows.append(
            CornerWindow(
                number=number,
                start_m=start_m,
                end_m=end_m,
                apex_m=apex_m,
                peak_absolute_curvature_per_m=peak_curvature_per_m,
                approximate_apex_radius_m=1.0 / peak_curvature_per_m,
                curvature_sign=(
                    "positive" if signed_curvature_per_m >= 0 else "negative"
                ),
                sample_count=sample_count,
                speed_rmse_mps=rms(speed_error),
                speed_bias_mps=mean(speed_error),
                longitudinal_acceleration_rmse_mps2=rms(longitudinal_error),
                longitudinal_acceleration_bias_mps2=mean(longitudinal_error),
                lateral_acceleration_rmse_mps2=rms(lateral_error),
                lateral_acceleration_bias_mps2=mean(lateral_error),
            )
        )
    return windows


def plot_summary(
    track: dict[str, np.ndarray],
    windows: list[CornerWindow],
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(16, 11), layout="constrained", facecolor="#F7F7F5")
    grid = figure.add_gridspec(2, 3, height_ratios=(1.35, 1.0))
    map_axis = figure.add_subplot(grid[0, :])
    map_axis.set_facecolor("#F7F7F5")
    map_axis.plot(track["x_m"], track["y_m"], color="#C8C8C4", lw=3.0)
    station_m = track["distance_m"]
    for corner, color in zip(windows, CORNER_COLORS, strict=True):
        mask = (station_m >= corner.start_m) & (station_m <= corner.end_m)
        map_axis.plot(
            track["x_m"][mask],
            track["y_m"][mask],
            color=color,
            lw=5.0,
            solid_capstyle="round",
        )
        apex_index = int(np.argmin(np.abs(station_m - corner.apex_m)))
        map_axis.annotate(
            f"C{corner.number}",
            (track["x_m"][apex_index], track["y_m"][apex_index]),
            xytext=(5, 5),
            textcoords="offset points",
            color="#202020",
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": color,
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )
    map_axis.set_title("Selected fused-track corner windows", loc="left", fontsize=14)
    map_axis.set_aspect("equal", adjustable="datalim")
    map_axis.axis("off")

    labels = [corner.label for corner in windows]
    metric_specs = (
        (
            [corner.speed_rmse_mps for corner in windows],
            "Speed RMSE",
            "m/s",
            MEASURED_COLOR,
        ),
        (
            [corner.longitudinal_acceleration_rmse_mps2 for corner in windows],
            "Longitudinal acceleration RMSE",
            "m/s²",
            SIMULATED_COLOR,
        ),
        (
            [corner.lateral_acceleration_rmse_mps2 for corner in windows],
            "Lateral acceleration RMSE",
            "m/s²",
            APEX_COLOR,
        ),
    )
    for axis, (values, title, unit, color) in zip(
        [figure.add_subplot(grid[1, index]) for index in range(3)],
        metric_specs,
        strict=True,
    ):
        positions = np.arange(len(windows))
        axis.barh(positions, values, color=color, alpha=0.88, height=0.62)
        axis.set_yticks(
            positions, labels if axis.get_subplotspec().colspan.start == 0 else []
        )
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_xlabel(unit)
        maximum_value = max(values)
        axis.set_xlim(0.0, maximum_value * 1.23 if maximum_value > 0 else 1.0)
        for position, value in zip(positions, values, strict=True):
            axis.text(
                value + 0.025 * maximum_value,
                position,
                f"{value:.2f}",
                va="center",
                fontsize=9,
                color="#202020",
            )
        axis.grid(axis="x", color="#D8D8D4", lw=0.7, alpha=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_facecolor("#F7F7F5")

    figure.suptitle(
        "Endurance corner accuracy",
        x=0.02,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=210, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_traces(
    data: dict[str, np.ndarray],
    windows: list[CornerWindow],
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(18, 18), layout="constrained", facecolor="#F7F7F5")
    outer = figure.add_gridspec(3, 2)
    distance_m = data["measured_distance_m"]
    for index, corner in enumerate(windows):
        row, column = divmod(index, 2)
        inner = outer[row, column].subgridspec(3, 1, hspace=0.04)
        axes = [figure.add_subplot(inner[axis_index, 0]) for axis_index in range(3)]
        mask = (distance_m >= corner.start_m) & (distance_m <= corner.end_m)
        relative_distance_m = distance_m[mask] - corner.start_m
        apex_relative_m = corner.apex_m - corner.start_m
        panels = (
            (
                data["analysis_shifted_gnss_speed_mps"][mask],
                data["sim_vehicle_speed_mps"][mask],
                "Speed [m/s]",
            ),
            (
                data["aligned_imu_longitudinal_accel_mps2"][mask],
                data["sim_vehicle_longitudinal_acceleration_mps2"][mask],
                "Longitudinal a [m/s²]",
            ),
            (
                data["aligned_imu_lateral_accel_mps2"][mask],
                data["sim_vehicle_lateral_acceleration_mps2"][mask],
                "Lateral a [m/s²]",
            ),
        )
        for axis, (measured, simulated, ylabel) in zip(axes, panels, strict=True):
            axis.set_facecolor("#F7F7F5")
            axis.plot(
                relative_distance_m,
                measured,
                color=MEASURED_COLOR,
                lw=1.7,
                marker="o",
                markersize=2.8,
            )
            axis.plot(
                relative_distance_m,
                simulated,
                color=SIMULATED_COLOR,
                lw=1.9,
            )
            axis.fill_between(
                relative_distance_m,
                measured,
                simulated,
                color=NEUTRAL_COLOR,
                alpha=0.09,
                linewidth=0,
            )
            axis.axvline(apex_relative_m, color=APEX_COLOR, lw=1.0, ls="--")
            axis.axhline(0.0, color="#B7B7B3", lw=0.7)
            axis.set_ylabel(ylabel, fontsize=9)
            axis.grid(color="#D8D8D4", lw=0.65, alpha=0.75)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_title(
            f"{corner.label}  ·  apex R≈{corner.approximate_apex_radius_m:.1f} m\n"
            f"RMSE: speed {corner.speed_rmse_mps:.2f} m/s  |  "
            f"long {corner.longitudinal_acceleration_rmse_mps2:.2f}  |  "
            f"lat {corner.lateral_acceleration_rmse_mps2:.2f} m/s²",
            loc="left",
            fontsize=11,
        )
        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(labelbottom=False)
        axes[2].set_xlabel("Distance through window [m]")
        for axis in axes:
            axis.set_xlim(0.0, corner.end_m - corner.start_m)

    legend_handles = (
        Line2D(
            [], [], color=MEASURED_COLOR, marker="o", markersize=4, label="Measured"
        ),
        Line2D([], [], color=SIMULATED_COLOR, lw=2, label="Simulation"),
        Line2D([], [], color=APEX_COLOR, ls="--", label="Detected apex"),
    )
    figure.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(0.985, 0.995),
    )
    figure.suptitle(
        "Corner traces: measured versus simulation",
        x=0.02,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def write_metrics(
    windows: list[CornerWindow],
    output_dir: Path,
    *,
    input_csv: Path,
    track_csv: Path,
    selection: dict[str, float | int],
) -> None:
    rows = [asdict(corner) for corner in windows]
    with (output_dir / "corner_accuracy_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "independent_coordinate": "recorded lap distance_m",
        "input_csv": str(input_csv.resolve()),
        "track_csv": str(track_csv.resolve()),
        "corner_selection": selection,
        "corners": rows,
    }
    (output_dir / "corner_accuracy_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--track-csv", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corner-count", type=int, default=6)
    parser.add_argument("--curvature-threshold-per-m", type=float, default=0.02)
    parser.add_argument("--smoothing-distance-m", type=float, default=2.0)
    parser.add_argument("--maximum-bridge-distance-m", type=float, default=10.0)
    parser.add_argument("--minimum-region-length-m", type=float, default=12.0)
    parser.add_argument("--lap-start-exclusion-m", type=float, default=50.0)
    parser.add_argument("--padding-m", type=float, default=6.0)
    parser.add_argument("--maximum-window-length-m", type=float, default=120.0)
    args = parser.parse_args()
    if args.corner_count <= 0:
        parser.error("corner-count must be positive")
    if (
        any(
            value <= 0.0
            for value in (
                args.curvature_threshold_per_m,
                args.smoothing_distance_m,
                args.maximum_bridge_distance_m,
                args.minimum_region_length_m,
                args.padding_m,
                args.maximum_window_length_m,
            )
        )
        or args.lap_start_exclusion_m < 0.0
    ):
        parser.error("corner-selection distances and thresholds must be positive")

    data = read_numeric_csv(args.input_csv)
    track = read_track(args.track_csv)
    regions, smooth_absolute_curvature = detect_corner_regions(
        track,
        corner_count=args.corner_count,
        curvature_threshold_per_m=args.curvature_threshold_per_m,
        smoothing_distance_m=args.smoothing_distance_m,
        maximum_bridge_distance_m=args.maximum_bridge_distance_m,
        minimum_region_length_m=args.minimum_region_length_m,
        lap_start_exclusion_m=args.lap_start_exclusion_m,
    )
    windows = build_corner_windows(
        data,
        track,
        regions,
        smooth_absolute_curvature,
        padding_m=args.padding_m,
        maximum_window_length_m=args.maximum_window_length_m,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_summary(track, windows, args.output_dir / "corner_accuracy_summary.png")
    plot_traces(data, windows, args.output_dir / "corner_accuracy_traces.png")
    selection = {
        "corner_count": args.corner_count,
        "curvature_threshold_per_m": args.curvature_threshold_per_m,
        "smoothing_distance_m": args.smoothing_distance_m,
        "maximum_bridge_distance_m": args.maximum_bridge_distance_m,
        "minimum_region_length_m": args.minimum_region_length_m,
        "lap_start_exclusion_m": args.lap_start_exclusion_m,
        "padding_m": args.padding_m,
        "maximum_window_length_m": args.maximum_window_length_m,
    }
    write_metrics(
        windows,
        args.output_dir,
        input_csv=args.input_csv,
        track_csv=args.track_csv,
        selection=selection,
    )
    print(f"Wrote {len(windows)} corner comparisons to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
