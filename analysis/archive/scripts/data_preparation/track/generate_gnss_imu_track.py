"""Reconstruct an endurance-lap path from GNSS position and corrected IMU Y."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ANALYSIS_DIR / "corrected_imu" / "first_lap_corrected_imu.csv"
DEFAULT_MAP = ANALYSIS_DIR / "first_endurance_lap" / "official_course_map.png"
DEFAULT_ALIGNMENT = ANALYSIS_DIR / "first_endurance_lap" / "manual_map_alignment.json"
DEFAULT_REFERENCE_TRACK = ANALYSIS_DIR / "first_endurance_lap" / "map_derived_track.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def read_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
        if all(row[name] not in (None, "") for row in rows)
    }


def odd_window(distance_m: float, spacing_m: float, sample_count: int) -> int:
    window = max(5, int(round(distance_m / spacing_m)))
    window += 1 - window % 2
    maximum = sample_count if sample_count % 2 else sample_count - 1
    return min(window, maximum)


def cumulative_trapezoid(values: np.ndarray, spacing: float) -> np.ndarray:
    return np.r_[0.0, np.cumsum(0.5 * (values[:-1] + values[1:]) * spacing)]


def reconstruct_track(
    data: dict[str, np.ndarray],
    *,
    reference_track: dict[str, np.ndarray] | None,
    reference_geometry_weight: float,
    reference_station_search_m: float,
    reference_station_smoothing_m: float,
    solver_curvature_correction_m: float,
    spacing_m: float,
    imu_smoothing_m: float,
    gnss_heading_window_m: float,
    heading_correction_m: float,
    position_correction_m: float,
    minimum_curvature_speed_mps: float,
    gnss_speed_lag_s: float,
    curvature_gain: float | None,
) -> dict[str, np.ndarray]:
    """Fuse low-frequency GNSS geometry with high-frequency IMU curvature."""

    required = (
        "gps_distance_trip_m",
        "gps_x_filtered_m",
        "gps_y_filtered_m",
        "gps_speed_mps",
        "corrected_lateral_accel_mps2",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(
            f"Missing required corrected-IMU columns: {', '.join(missing)}"
        )

    source_distance = data["gps_distance_trip_m"] - data["gps_distance_trip_m"][0]
    finite = np.logical_and.reduce(
        [np.isfinite(data[name]) for name in required]
    ) & np.isfinite(source_distance)
    source_distance = source_distance[finite]
    keep = np.r_[True, np.diff(source_distance) > 1e-6]
    source_distance = source_distance[keep]
    if source_distance.size < 10 or source_distance[-1] <= 0.0:
        raise ValueError("The lap needs at least ten increasing distance samples")

    distance_m = np.arange(0.0, source_distance[-1] + 0.5 * spacing_m, spacing_m)

    def spatial(name: str) -> np.ndarray:
        return np.interp(distance_m, source_distance, data[name][finite][keep])

    gnss_x_m = spatial("gps_x_filtered_m")
    gnss_y_m = spatial("gps_y_filtered_m")
    if "time_s" not in data:
        raise ValueError("Missing time_s needed to correct the GNSS speed lag")
    time_s = data["time_s"]
    shifted_source_speed = np.interp(
        time_s + gnss_speed_lag_s, time_s, data["gps_speed_mps"]
    )
    speed_mps = np.interp(
        distance_m, source_distance, shifted_source_speed[finite][keep]
    )
    corrected_imu_y_mps2 = spatial("corrected_lateral_accel_mps2")

    imu_sigma_samples = max(imu_smoothing_m / spacing_m, 0.0)
    smoothed_imu_y = gaussian_filter1d(
        corrected_imu_y_mps2, imu_sigma_samples, mode="nearest"
    )
    valid_speed = speed_mps >= minimum_curvature_speed_mps
    if np.count_nonzero(valid_speed) < 2:
        raise ValueError("Not enough samples exceed the minimum curvature speed")
    raw_imu_curvature = np.full_like(speed_mps, np.nan)
    raw_imu_curvature[valid_speed] = (
        smoothed_imu_y[valid_speed] / speed_mps[valid_speed] ** 2
    )
    raw_imu_curvature = np.interp(
        distance_m,
        distance_m[valid_speed],
        raw_imu_curvature[valid_speed],
    )
    raw_heading_change = float(cumulative_trapezoid(raw_imu_curvature, spacing_m)[-1])
    if curvature_gain is None:
        if abs(raw_heading_change) < 1e-6:
            raise ValueError("Cannot calibrate a near-zero full-lap IMU heading change")
        # A simple, closed endurance course has winding number +/-1. Correcting
        # the full-lap scale preserves every local IMU feature while preventing
        # all corners from being systematically too shallow.
        curvature_gain = (
            np.copysign(2.0 * np.pi, raw_heading_change) / raw_heading_change
        )
    imu_curvature = raw_imu_curvature * curvature_gain

    heading_window = odd_window(gnss_heading_window_m, spacing_m, len(distance_m))
    dx_ds = savgol_filter(
        gnss_x_m, heading_window, 3, deriv=1, delta=spacing_m, mode="interp"
    )
    dy_ds = savgol_filter(
        gnss_y_m, heading_window, 3, deriv=1, delta=spacing_m, mode="interp"
    )
    gnss_heading_rad = np.unwrap(np.arctan2(dy_ds, dx_ds))

    imu_heading_rad = gnss_heading_rad[0] + cumulative_trapezoid(
        imu_curvature, spacing_m
    )
    heading_error = np.unwrap(gnss_heading_rad - imu_heading_rad)
    heading_sigma = max(heading_correction_m / spacing_m, 1.0)
    low_frequency_heading_error = gaussian_filter1d(
        heading_error, heading_sigma, mode="nearest"
    )
    fused_heading_rad = imu_heading_rad + low_frequency_heading_error

    dead_reckoned_x = gnss_x_m[0] + cumulative_trapezoid(
        np.cos(fused_heading_rad), spacing_m
    )
    dead_reckoned_y = gnss_y_m[0] + cumulative_trapezoid(
        np.sin(fused_heading_rad), spacing_m
    )
    position_sigma = max(position_correction_m / spacing_m, 1.0)
    correction_x = gaussian_filter1d(
        gnss_x_m - dead_reckoned_x, position_sigma, mode="nearest"
    )
    correction_y = gaussian_filter1d(
        gnss_y_m - dead_reckoned_y, position_sigma, mode="nearest"
    )

    # Preserve the measured start and finish positions exactly. The interior
    # correction remains low-frequency, so local corner shape still comes from IMU Y.
    progress = distance_m / distance_m[-1]
    correction_x += (1.0 - progress) * (-correction_x[0])
    correction_x += progress * (gnss_x_m[-1] - dead_reckoned_x[-1] - correction_x[-1])
    correction_y += (1.0 - progress) * (-correction_y[0])
    correction_y += progress * (gnss_y_m[-1] - dead_reckoned_y[-1] - correction_y[-1])
    fused_x_m = dead_reckoned_x + correction_x
    fused_y_m = dead_reckoned_y + correction_y

    if reference_track is not None and reference_geometry_weight > 0.0:
        reference_required = ("distance_m", "x_m", "y_m")
        reference_missing = [
            name for name in reference_required if name not in reference_track
        ]
        if reference_missing:
            raise ValueError(
                "Missing reference-track columns: " + ", ".join(reference_missing)
            )
        reference_distance = reference_track["distance_m"]
        proportional_reference_station = (
            distance_m / distance_m[-1] * float(reference_distance[-1])
        )

        # The map trace and telemetry lap have slightly different local
        # stationing even when their geometry agrees.  A direct percentage-of-
        # lap blend put corresponding corners several metres apart and could
        # cancel a real curve (notably near telemetry station 500 m).  Project
        # the telemetry reconstruction onto a local window of the official
        # centerline, then smooth only the resulting station offset.  This
        # registers like-with-like before applying the geometric constraint.
        nearest_reference_station = np.empty_like(distance_m)
        for index, nominal_station in enumerate(proportional_reference_station):
            candidates = np.flatnonzero(
                np.abs(reference_distance - nominal_station)
                <= reference_station_search_m
            )
            if candidates.size == 0:
                candidates = np.asarray(
                    [int(np.argmin(np.abs(reference_distance - nominal_station)))]
                )
            squared_distance = (
                reference_track["x_m"][candidates] - fused_x_m[index]
            ) ** 2 + (
                reference_track["y_m"][candidates] - fused_y_m[index]
            ) ** 2
            nearest_reference_station[index] = reference_distance[
                candidates[int(np.argmin(squared_distance))]
            ]

        station_offset = nearest_reference_station - proportional_reference_station
        station_sigma = max(reference_station_smoothing_m / spacing_m, 1.0)
        station_offset = gaussian_filter1d(
            station_offset, station_sigma, mode="nearest"
        )
        # Keep start and finish tied to the official start/finish and enforce a
        # physically monotonic station map after smoothing.
        progress = distance_m / distance_m[-1]
        station_offset -= (1.0 - progress) * station_offset[0]
        station_offset -= progress * station_offset[-1]
        reference_station = np.maximum.accumulate(
            proportional_reference_station + station_offset
        )
        reference_station = np.clip(
            reference_station, reference_distance[0], reference_distance[-1]
        )
        reference_x_m = np.interp(
            reference_station, reference_distance, reference_track["x_m"]
        )
        reference_y_m = np.interp(
            reference_station, reference_distance, reference_track["y_m"]
        )
        telemetry_weight = 1.0 - reference_geometry_weight
        fused_x_m = (
            reference_geometry_weight * reference_x_m + telemetry_weight * fused_x_m
        )
        fused_y_m = (
            reference_geometry_weight * reference_y_m + telemetry_weight * fused_y_m
        )
    else:
        reference_x_m = np.full_like(distance_m, np.nan)
        reference_y_m = np.full_like(distance_m, np.nan)
        reference_station = np.full_like(distance_m, np.nan)

    fused_heading_rad = np.unwrap(
        np.arctan2(np.gradient(fused_y_m), np.gradient(fused_x_m))
    )
    fused_dx = np.gradient(fused_x_m, spacing_m, edge_order=2)
    fused_dy = np.gradient(fused_y_m, spacing_m, edge_order=2)
    fused_d2x = np.gradient(fused_dx, spacing_m, edge_order=2)
    fused_d2y = np.gradient(fused_dy, spacing_m, edge_order=2)
    fused_geometric_curvature = (
        fused_dx * fused_d2y - fused_dy * fused_d2x
    ) / np.maximum(fused_dx**2 + fused_dy**2, 1e-12) ** 1.5

    # Physics needs the measured local corner shape, while the official
    # centerline is most trustworthy at course scale.  Retain the raw,
    # speed-normalized IMU curvature at short wavelengths and use only the
    # low-frequency part of the map/IMU difference to correct drift.  Using
    # the pointwise blended-XY derivative directly created false sharp bends
    # wherever the two sources had slightly different local stationing.
    solver_curvature_sigma = max(solver_curvature_correction_m / spacing_m, 1.0)
    solver_curvature = raw_imu_curvature + gaussian_filter1d(
        fused_geometric_curvature - raw_imu_curvature,
        solver_curvature_sigma,
        mode="wrap",
    )

    return {
        "distance_m": distance_m,
        # Canonical SpatialTrack columns make the generated artifact directly
        # consumable by the endurance and lap-time solvers.  Keep the detailed
        # fusion channels below for diagnostics and provenance.
        "x_m": fused_x_m,
        "y_m": fused_y_m,
        "curvature_per_m": solver_curvature,
        "gnss_x_filtered_m": gnss_x_m,
        "gnss_y_filtered_m": gnss_y_m,
        "fused_x_m": fused_x_m,
        "fused_y_m": fused_y_m,
        "official_reference_x_m": reference_x_m,
        "official_reference_y_m": reference_y_m,
        "official_reference_station_m": reference_station,
        "gps_speed_mps": speed_mps,
        "corrected_imu_y_mps2": corrected_imu_y_mps2,
        "smoothed_corrected_imu_y_mps2": smoothed_imu_y,
        "raw_imu_curvature_per_m": raw_imu_curvature,
        "imu_curvature_per_m": imu_curvature,
        "fused_geometric_curvature_per_m": fused_geometric_curvature,
        "solver_complementary_curvature_per_m": solver_curvature,
        "applied_curvature_gain": np.full_like(distance_m, curvature_gain),
        "gnss_heading_rad": gnss_heading_rad,
        "fused_heading_rad": fused_heading_rad,
        "low_frequency_position_correction_x_m": correction_x,
        "low_frequency_position_correction_y_m": correction_y,
    }


def map_pixels(
    x_m: np.ndarray, y_m: np.ndarray, alignment: dict[str, object]
) -> np.ndarray:
    transform = alignment["effective_affine_transform"]
    matrix = np.asarray(transform["matrix_px_per_m"], dtype=float)
    offset = np.asarray(transform["offset_px"], dtype=float)
    return np.column_stack((x_m, y_m)) @ matrix.T + offset


def save_overlay(
    path: Path,
    course_map_path: Path,
    alignment: dict[str, object],
    result: dict[str, np.ndarray],
) -> None:
    image = plt.imread(course_map_path)
    gnss_px = map_pixels(
        result["gnss_x_filtered_m"], result["gnss_y_filtered_m"], alignment
    )
    fused_px = map_pixels(result["fused_x_m"], result["fused_y_m"], alignment)
    segments = np.stack((fused_px[:-1], fused_px[1:]), axis=1)
    segment_curvature = 0.5 * (
        result["imu_curvature_per_m"][:-1] + result["imu_curvature_per_m"][1:]
    )
    color_limit = float(np.percentile(np.abs(segment_curvature), 98.0))
    normalization = Normalize(-color_limit, color_limit, clip=True)

    figure = plt.figure(figsize=(16, 7), layout="constrained")
    grid = figure.add_gridspec(2, 1, height_ratios=(2.1, 1.0))
    map_axis = figure.add_subplot(grid[0])
    signal_axis = figure.add_subplot(grid[1])
    map_axis.imshow(image, origin="upper")
    map_axis.plot(
        gnss_px[:, 0],
        gnss_px[:, 1],
        color="#00A6A6",
        lw=1.4,
        ls="--",
        alpha=0.9,
        label="Filtered GNSS position",
    )
    outline = LineCollection(segments, colors="black", linewidth=5.0, alpha=0.85)
    fused_line = LineCollection(
        segments, cmap="coolwarm", norm=normalization, linewidth=3.2
    )
    fused_line.set_array(segment_curvature)
    map_axis.add_collection(outline)
    map_axis.add_collection(fused_line)
    map_axis.scatter(
        fused_px[0, 0], fused_px[0, 1], c="#F2CF5B", edgecolors="black", s=45, zorder=5
    )
    map_axis.set_xlim(0, image.shape[1])
    map_axis.set_ylim(image.shape[0], 0)
    map_axis.set_aspect("equal")
    map_axis.axis("off")
    map_axis.legend(
        handles=(
            Line2D([], [], color="#00A6A6", lw=1.4, ls="--", label="Filtered GNSS"),
            Line2D([], [], color="black", lw=4.0, label="Curve-complete fused track"),
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markerfacecolor="#F2CF5B",
                markeredgecolor="black",
                label="Start / finish",
            ),
        ),
        loc="lower right",
        framealpha=0.9,
        ncol=3,
    )
    colorbar = figure.colorbar(fused_line, ax=map_axis, fraction=0.018, pad=0.012)
    colorbar.set_label("Corrected-IMU curvature [1/m] (98th-percentile clipping)")

    distance = result["distance_m"]
    signal_axis.plot(
        distance,
        result["corrected_imu_y_mps2"],
        color="#4C78A8",
        lw=1.1,
        label="Corrected IMU Y",
    )
    curvature_axis = signal_axis.twinx()
    curvature_axis.plot(
        distance,
        result["imu_curvature_per_m"],
        color="#E45756",
        lw=1.1,
        label="IMU Y / speed²",
    )
    signal_axis.axhline(0.0, color="0.4", lw=0.7)
    signal_axis.set_xlabel("Lap distance [m]")
    signal_axis.set_ylabel("Corrected IMU Y [m/s²]", color="#4C78A8")
    curvature_axis.set_ylabel("Signed curvature [1/m]", color="#E45756")
    signal_axis.tick_params(axis="y", colors="#4C78A8")
    curvature_axis.tick_params(axis="y", colors="#E45756")
    signal_axis.grid(alpha=0.25)
    signal_axis.spines["top"].set_visible(False)
    curvature_axis.spines["top"].set_visible(False)
    signal_axis.legend(
        handles=(signal_axis.lines[0], curvature_axis.lines[0]),
        loc="upper right",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Curve-complete endurance track: GNSS + corrected IMU Y + official centerline constraint"
    )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_csv(path: Path, result: dict[str, np.ndarray]) -> None:
    names = tuple(result)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for index in range(len(result["distance_m"])):
            # SpatialTrack defines curvature per cell, so its canonical
            # curvature column is blank at the final endpoint. Diagnostic
            # point-curvature channels retain their final sampled value.
            writer.writerow(
                ""
                if name == "curvature_per_m" and index == len(result["distance_m"]) - 1
                else result[name][index]
                for name in names
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--course-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--reference-track", type=Path, default=DEFAULT_REFERENCE_TRACK)
    parser.add_argument(
        "--reference-geometry-weight",
        type=float,
        default=0.85,
        help="Official-centerline constraint weight from 0 (telemetry only) to 1",
    )
    parser.add_argument(
        "--reference-station-search-m",
        type=float,
        default=60.0,
        help="Local official-centerline station search half-width",
    )
    parser.add_argument(
        "--reference-station-smoothing-m",
        type=float,
        default=10.0,
        help="Smoothing length for telemetry-to-map station registration",
    )
    parser.add_argument(
        "--solver-curvature-correction-m",
        type=float,
        default=60.0,
        help="Length scale below which solver curvature follows corrected IMU Y",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spacing-m", type=float, default=0.5)
    parser.add_argument("--imu-smoothing-m", type=float, default=1.5)
    parser.add_argument("--gnss-heading-window-m", type=float, default=35.0)
    parser.add_argument("--heading-correction-m", type=float, default=55.0)
    parser.add_argument("--position-correction-m", type=float, default=45.0)
    parser.add_argument("--minimum-curvature-speed-mps", type=float, default=4.0)
    parser.add_argument("--gnss-speed-lag-s", type=float, default=0.3072)
    parser.add_argument(
        "--curvature-gain",
        type=float,
        default=None,
        help="IMU-curvature scale; default calibrates the closed lap to one full turn",
    )
    args = parser.parse_args()
    positive = (
        args.spacing_m,
        args.gnss_heading_window_m,
        args.heading_correction_m,
        args.position_correction_m,
        args.minimum_curvature_speed_mps,
        args.reference_station_search_m,
        args.reference_station_smoothing_m,
        args.solver_curvature_correction_m,
    )
    if (
        any(value <= 0 for value in positive)
        or args.imu_smoothing_m < 0
        or args.gnss_speed_lag_s < 0
        or (args.curvature_gain is not None and args.curvature_gain <= 0)
        or not 0.0 <= args.reference_geometry_weight <= 1.0
    ):
        parser.error("Distances and minimum speed must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_columns(args.input)
    reference_track = (
        read_columns(args.reference_track)
        if args.reference_geometry_weight > 0.0
        else None
    )
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    result = reconstruct_track(
        data,
        reference_track=reference_track,
        reference_geometry_weight=args.reference_geometry_weight,
        reference_station_search_m=args.reference_station_search_m,
        reference_station_smoothing_m=args.reference_station_smoothing_m,
        solver_curvature_correction_m=args.solver_curvature_correction_m,
        spacing_m=args.spacing_m,
        imu_smoothing_m=args.imu_smoothing_m,
        gnss_heading_window_m=args.gnss_heading_window_m,
        heading_correction_m=args.heading_correction_m,
        position_correction_m=args.position_correction_m,
        minimum_curvature_speed_mps=args.minimum_curvature_speed_mps,
        gnss_speed_lag_s=args.gnss_speed_lag_s,
        curvature_gain=args.curvature_gain,
    )
    csv_path = args.output_dir / "gnss_imu_endurance_track.csv"
    plot_path = args.output_dir / "gnss_imu_track_on_official_map.png"
    metadata_path = args.output_dir / "gnss_imu_endurance_track.json"
    write_csv(csv_path, result)
    save_overlay(plot_path, args.course_map, alignment, result)

    displacement = np.hypot(
        result["fused_x_m"] - result["gnss_x_filtered_m"],
        result["fused_y_m"] - result["gnss_y_filtered_m"],
    )
    closure = float(
        np.hypot(
            result["fused_x_m"][-1] - result["fused_x_m"][0],
            result["fused_y_m"][-1] - result["fused_y_m"][0],
        )
    )
    metadata = {
        "method": "spatial complementary GNSS-position / corrected-IMU-Y-curvature fusion with geometry-registered official-centerline constraint",
        "input_csv": str(args.input.resolve()),
        "corrected_imu_channel": "corrected_lateral_accel_mps2",
        "curvature_equation": "kappa_solver = kappa_imu_raw + lowpass(kappa_official_geometry - kappa_imu_raw)",
        "curvature_scale": (
            "automatic closed-lap winding calibration"
            if args.curvature_gain is None
            else "explicit --curvature-gain"
        ),
        "course_map": str(args.course_map.resolve()),
        "map_alignment": str(args.alignment.resolve()),
        "official_reference_track": str(args.reference_track.resolve()),
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "input",
                "course_map",
                "alignment",
                "reference_track",
                "output_dir",
            }
        },
        "outputs": {
            "track_csv": str(csv_path.resolve()),
            "map_overlay": str(plot_path.resolve()),
        },
        "metrics": {
            "track_length_m": float(result["distance_m"][-1]),
            "sample_count": len(result["distance_m"]),
            "fused_closure_error_m": closure,
            "rms_fused_to_filtered_gnss_m": float(np.sqrt(np.mean(displacement**2))),
            "maximum_fused_to_filtered_gnss_m": float(np.max(displacement)),
            "maximum_absolute_imu_curvature_per_m": float(
                np.max(np.abs(result["imu_curvature_per_m"]))
            ),
            "applied_curvature_gain": float(result["applied_curvature_gain"][0]),
            "raw_imu_integrated_heading_change_rad": float(
                cumulative_trapezoid(result["raw_imu_curvature_per_m"], args.spacing_m)[
                    -1
                ]
            ),
            "scaled_imu_integrated_heading_change_rad": float(
                cumulative_trapezoid(result["imu_curvature_per_m"], args.spacing_m)[-1]
            ),
        },
        "limitations": [
            "The official map is schematic; its saved affine alignment is display-only.",
            "The default output uses the extracted official centerline as an 85% geometric constraint after local station registration, so corresponding corners are blended together.",
            "Solver curvature follows corrected IMU Y locally and uses official geometry only for low-frequency drift correction.",
            "The fixed IMU correction removes stationary mounting tilt, not dynamic roll.",
            "IMU curvature below the minimum speed is interpolated from valid samples.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
