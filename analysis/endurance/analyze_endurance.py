"""Replay the first endurance lap and compare model telemetry with the log."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.common import (  # noqa: E402
    DEFAULT_GNSS_LAG_S,
    TIME_COLUMNS,
    UnmodeledRecordedControlError,
    corrected_imu_at_lap_times,
    first_present,
    interpolate_channel,
    periodic_curvature_at_station,
    project_to_track_distance,
    read_numeric_csv,
)
from lapsim import Controls, SpatialCoordinate, SpatialTrack  # noqa: E402
from vehicle_model import Vehicle  # noqa: E402


def _rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def _gnss_path_curvature(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    sample_spacing_m: float = 0.5,
    smoothing_distance_m: float = 11.0,
) -> np.ndarray:
    """Return signed curvature of the filtered GNSS path at its input samples."""

    finite = np.isfinite(x_m) & np.isfinite(y_m)
    if np.count_nonzero(finite) < 7:
        raise ValueError("GNSS curvature needs at least seven finite path samples")
    source_index = np.flatnonzero(finite)
    x_finite = np.asarray(x_m[finite], dtype=float)
    y_finite = np.asarray(y_m[finite], dtype=float)
    chord_m = np.hypot(np.diff(x_finite), np.diff(y_finite))
    path_distance_m = np.r_[0.0, np.cumsum(chord_m)]
    unique = np.r_[True, np.diff(path_distance_m) > 1e-6]
    path_distance_m = path_distance_m[unique]
    x_unique = x_finite[unique]
    y_unique = y_finite[unique]
    if path_distance_m.size < 7 or path_distance_m[-1] <= smoothing_distance_m:
        raise ValueError("GNSS path is too short to derive smoothed curvature")

    # The selected data covers one closed lap. Distribute the small GNSS gate
    # closure error over the lap before applying a periodic spatial filter.
    progress = path_distance_m / path_distance_m[-1]
    x_unique = x_unique - progress * (x_unique[-1] - x_unique[0])
    y_unique = y_unique - progress * (y_unique[-1] - y_unique[0])
    uniform_distance_m = np.arange(0.0, path_distance_m[-1], sample_spacing_m)
    uniform_x_m = np.interp(uniform_distance_m, path_distance_m, x_unique)
    uniform_y_m = np.interp(uniform_distance_m, path_distance_m, y_unique)

    if smoothing_distance_m <= 0.0:
        dx_ds = np.gradient(uniform_x_m, sample_spacing_m, edge_order=2)
        dy_ds = np.gradient(uniform_y_m, sample_spacing_m, edge_order=2)
        d2x_ds2 = np.gradient(dx_ds, sample_spacing_m, edge_order=2)
        d2y_ds2 = np.gradient(dy_ds, sample_spacing_m, edge_order=2)
    else:
        window_samples = max(7, int(round(smoothing_distance_m / sample_spacing_m)))
        window_samples += 1 - window_samples % 2
        maximum_window = len(uniform_distance_m) - (1 - len(uniform_distance_m) % 2)
        window_samples = min(window_samples, maximum_window)
        dx_ds = savgol_filter(
            uniform_x_m,
            window_samples,
            3,
            deriv=1,
            delta=sample_spacing_m,
            mode="wrap",
        )
        dy_ds = savgol_filter(
            uniform_y_m,
            window_samples,
            3,
            deriv=1,
            delta=sample_spacing_m,
            mode="wrap",
        )
        d2x_ds2 = savgol_filter(
            uniform_x_m,
            window_samples,
            3,
            deriv=2,
            delta=sample_spacing_m,
            mode="wrap",
        )
        d2y_ds2 = savgol_filter(
            uniform_y_m,
            window_samples,
            3,
            deriv=2,
            delta=sample_spacing_m,
            mode="wrap",
        )
    denominator = np.maximum(dx_ds**2 + dy_ds**2, 1e-12) ** 1.5
    uniform_curvature = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / denominator

    extended_distance_m = np.r_[
        uniform_distance_m[-1] - path_distance_m[-1],
        uniform_distance_m,
        uniform_distance_m[0] + path_distance_m[-1],
    ]
    extended_curvature = np.r_[
        uniform_curvature[-1], uniform_curvature, uniform_curvature[0]
    ]
    curvature_finite = np.interp(
        path_distance_m,
        extended_distance_m,
        extended_curvature,
    )
    result = np.full(len(x_m), np.nan)
    result[source_index[unique]] = curvature_finite
    missing = ~np.isfinite(result)
    result[missing] = np.interp(
        np.flatnonzero(missing),
        np.flatnonzero(~missing),
        result[~missing],
    )
    return result


def _save_imu_curvature_plot(
    path: Path,
    distance_m: np.ndarray,
    imu_y_mps2: np.ndarray,
    curvature_per_m: np.ndarray,
    *,
    curvature_label: str,
    title: str,
) -> None:
    """Save a clean dual-axis comparison of lateral IMU and path curvature."""

    figure, imu_axis = plt.subplots(figsize=(14, 6), layout="constrained")
    curvature_axis = imu_axis.twinx()
    imu_line = imu_axis.plot(
        distance_m,
        imu_y_mps2,
        color="#4C78A8",
        lw=1.35,
        label="Corrected IMU Y (lateral acceleration)",
    )[0]
    curvature_line = curvature_axis.plot(
        distance_m,
        curvature_per_m,
        color="#F58518",
        lw=1.35,
        label=curvature_label,
    )[0]
    imu_axis.axhline(0.0, color="0.35", lw=0.8, alpha=0.7)
    imu_axis.set_xlabel("Recorded lap station [m]")
    imu_axis.set_ylabel("Corrected IMU Y [m/s²]", color=imu_line.get_color())
    curvature_axis.set_ylabel(
        "Signed curvature [1/m]", color=curvature_line.get_color()
    )
    imu_axis.tick_params(axis="y", colors=imu_line.get_color())
    curvature_axis.tick_params(axis="y", colors=curvature_line.get_color())
    imu_axis.grid(alpha=0.25)
    imu_axis.spines["top"].set_visible(False)
    curvature_axis.spines["top"].set_visible(False)
    imu_axis.legend(
        [imu_line, curvature_line],
        [imu_line.get_label(), curvature_line.get_label()],
        loc="upper right",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(title)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _save_imu_curvature_acceleration_plot(
    path: Path,
    distance_m: np.ndarray,
    imu_y_mps2: np.ndarray,
    curvature_lateral_accel_mps2: np.ndarray,
    *,
    curvature_label: str,
    title: str,
) -> None:
    """Compare IMU Y with the speed-scaled curvature on one physical axis."""

    figure, axis = plt.subplots(figsize=(14, 6), layout="constrained")
    axis.plot(
        distance_m,
        imu_y_mps2,
        color="#4C78A8",
        lw=1.35,
        label="Corrected IMU Y",
    )
    axis.plot(
        distance_m,
        curvature_lateral_accel_mps2,
        color="#F58518",
        lw=1.35,
        label=curvature_label,
    )
    axis.axhline(0.0, color="0.35", lw=0.8, alpha=0.7)
    axis.set_xlabel("Recorded lap station [m]")
    axis.set_ylabel("Lateral acceleration [m/s²]")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", ncol=2, frameon=False)
    figure.suptitle(title)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _normalize_to_reference(
    values: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match a signal's mean and standard deviation to a reference signal."""

    finite_values = values[np.isfinite(values)]
    finite_reference = reference[np.isfinite(reference)]
    if finite_values.size < 2 or finite_reference.size < 2:
        raise ValueError("Normalization requires at least two finite samples")
    scale = float(np.std(finite_values))
    if scale <= 1e-12:
        raise ValueError("Cannot normalize a constant signal")
    return (values - float(np.mean(finite_values))) * (
        float(np.std(finite_reference)) / scale
    ) + float(np.mean(finite_reference))


def _save_normalized_curvature_comparison(
    path: Path,
    distance_m: np.ndarray,
    imu_y_mps2: np.ndarray,
    normalized_map_mps2: np.ndarray,
    normalized_gnss_mps2: np.ndarray,
) -> None:
    """Save aligned map and GNSS curvature comparisons on the IMU scale."""

    figure, axes = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, sharey=True, layout="constrained"
    )
    panels = (
        (normalized_map_mps2, "Map v²κ normalized to IMU Y", "Map-derived"),
        (normalized_gnss_mps2, "GNSS v²κ normalized to IMU Y", "GNSS-derived"),
    )
    for axis, (normalized, label, panel_title) in zip(axes, panels, strict=True):
        axis.plot(
            distance_m, imu_y_mps2, color="#4C78A8", lw=1.3, label="Corrected IMU Y"
        )
        axis.plot(distance_m, normalized, color="#F58518", lw=1.3, label=label)
        axis.axhline(0.0, color="0.35", lw=0.8, alpha=0.7)
        axis.set_ylabel("Lateral acceleration [m/s²]")
        axis.set_title(panel_title, loc="left")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xlabel("Recorded lap station [m]")
    figure.suptitle(
        "First endurance lap: post-offset v²κ normalized to IMU Y mean and standard deviation"
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _save_imu_gnss_smoothing_overlay(
    path: Path,
    distance_m: np.ndarray,
    imu_y_mps2: np.ndarray,
    normalized_gnss_by_window_m: dict[float, np.ndarray],
) -> None:
    """Overlay IMU Y with several normalized GNSS-curvature windows."""

    figure, axis = plt.subplots(figsize=(14, 7), layout="constrained")
    axis.plot(
        distance_m,
        imu_y_mps2,
        color="black",
        lw=1.7,
        label="Corrected IMU Y",
        zorder=5,
    )
    colors = ("#E45756", "#54A24B", "#F2CF5B", "#4C78A8")
    for color, (window_m, normalized) in zip(
        colors, normalized_gnss_by_window_m.items(), strict=True
    ):
        label = "GNSS v²κ: no additional smoothing"
        if window_m > 0.0:
            label = f"GNSS v²κ: {window_m:g} m window"
        axis.plot(distance_m, normalized, color=color, lw=1.15, label=label)
    axis.axhline(0.0, color="0.35", lw=0.8, alpha=0.7)
    axis.set_xlabel("Recorded lap station [m]")
    axis.set_ylabel("Lateral acceleration [m/s²]")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", ncol=2, frameon=False)
    figure.suptitle(
        "First endurance lap: IMU Y vs normalized post-offset GNSS v²κ smoothing"
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _save_gnss_curvature_map_comparison(
    path: Path,
    course_map_path: Path,
    alignment_path: Path,
    gps_x_m: np.ndarray,
    gps_y_m: np.ndarray,
    curvature_by_window_m: dict[float, np.ndarray],
    imu_equivalent_curvature_per_m: np.ndarray,
    combined_gnss_curvature_per_m: np.ndarray,
) -> None:
    """Overlay GNSS smoothing and an IMU/GNSS comparison on the course map."""

    map_image = plt.imread(course_map_path)
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    transform = alignment["effective_affine_transform"]
    matrix = np.asarray(transform["matrix_px_per_m"], dtype=float)
    offset = np.asarray(transform["offset_px"], dtype=float)
    gps_pixels = np.column_stack((gps_x_m, gps_y_m)) @ matrix.T + offset
    segments = np.stack((gps_pixels[:-1], gps_pixels[1:]), axis=1)

    # Use one robust, symmetric scale so unsmoothed differentiation noise is
    # visible without preventing the 5 m and 10 m panels from using the colorbar.
    finite_absolute_curvature = np.concatenate(
        [
            *[
                np.abs(values[np.isfinite(values)])
                for values in curvature_by_window_m.values()
            ],
            np.abs(
                imu_equivalent_curvature_per_m[
                    np.isfinite(imu_equivalent_curvature_per_m)
                ]
            ),
            np.abs(
                combined_gnss_curvature_per_m[
                    np.isfinite(combined_gnss_curvature_per_m)
                ]
            ),
        ]
    )
    color_limit = float(np.percentile(finite_absolute_curvature, 98.0))
    normalization = Normalize(vmin=-color_limit, vmax=color_limit, clip=True)

    figure, axes = plt.subplots(
        len(curvature_by_window_m) + 1,
        1,
        figsize=(16, 2.25 * (len(curvature_by_window_m) + 1) + 1.0),
        layout="constrained",
    )
    line_collection = None
    for axis, (window_m, curvature) in zip(
        axes[:-1], curvature_by_window_m.items(), strict=True
    ):
        segment_curvature = 0.5 * (curvature[:-1] + curvature[1:])
        trace_outline = LineCollection(
            segments,
            colors="black",
            linewidth=4.8,
            alpha=0.75,
        )
        line_collection = LineCollection(
            segments,
            cmap="PRGn",
            norm=normalization,
            linewidth=3.2,
            alpha=1.0,
        )
        line_collection.set_array(segment_curvature)
        axis.imshow(map_image, origin="upper")
        axis.add_collection(trace_outline)
        axis.add_collection(line_collection)
        axis.scatter(
            gps_pixels[0, 0],
            gps_pixels[0, 1],
            s=24,
            c="black",
            marker="o",
            zorder=4,
        )
        label = (
            "No additional smoothing" if window_m == 0.0 else f"{window_m:g} m window"
        )
        axis.set_title(label, loc="left")
        axis.set_xlim(0, map_image.shape[1])
        axis.set_ylim(map_image.shape[0], 0)
        axis.set_aspect("equal")
        axis.axis("off")

    combined_axis = axes[-1]
    imu_segments = 0.5 * (
        imu_equivalent_curvature_per_m[:-1] + imu_equivalent_curvature_per_m[1:]
    )
    gnss_segments = 0.5 * (
        combined_gnss_curvature_per_m[:-1] + combined_gnss_curvature_per_m[1:]
    )
    combined_outline = LineCollection(
        segments, colors="black", linewidth=8.4, alpha=0.8
    )
    imu_outer = LineCollection(
        segments, cmap="PRGn", norm=normalization, linewidth=6.8, alpha=1.0
    )
    imu_outer.set_array(imu_segments)
    inner_separator = LineCollection(segments, colors="black", linewidth=4.2, alpha=0.9)
    gnss_inner = LineCollection(
        segments, cmap="PRGn", norm=normalization, linewidth=2.8, alpha=1.0
    )
    gnss_inner.set_array(gnss_segments)
    combined_axis.imshow(map_image, origin="upper")
    for collection in (combined_outline, imu_outer, inner_separator, gnss_inner):
        combined_axis.add_collection(collection)
    combined_axis.scatter(
        gps_pixels[0, 0],
        gps_pixels[0, 1],
        s=24,
        c="black",
        marker="o",
        zorder=5,
    )
    combined_axis.set_title(
        "IMU-equivalent curvature (outer) + GNSS 11 m curvature (inner)",
        loc="left",
    )
    combined_axis.set_xlim(0, map_image.shape[1])
    combined_axis.set_ylim(map_image.shape[0], 0)
    combined_axis.set_aspect("equal")
    combined_axis.axis("off")
    if line_collection is not None:
        colorbar = figure.colorbar(
            line_collection,
            ax=axes,
            orientation="vertical",
            fraction=0.018,
            pad=0.012,
        )
        colorbar.set_label(
            "Signed GNSS curvature [1/m] (98th-percentile color clipping)"
        )
    figure.suptitle("GNSS-derived curvature smoothing on official endurance map")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def replay_lap_distance(
    distance_m: np.ndarray,
    recorded: dict[str, np.ndarray],
    curvature_per_m: np.ndarray,
    output_distance_m: np.ndarray,
    *,
    initial_speed_mps: float,
    initial_soc_percent: float,
    drag_coefficient: float | None = None,
    motor_to_wheel_efficiency: float | None = None,
    front_brake_torque_per_psi_lbfin: float | None = None,
    rear_brake_torque_per_psi_lbfin: float | None = None,
    brake_gain_count_per_axle: float = 1.0,
    brake_pressure_model: str = "linear-hardware-gains",
    brake_deadband_psi: float = 0.0,
    maximum_brake_force_request_n: float | None = None,
    negative_torque_policy: str = "error",
    longitudinal_slip_relaxation_length_m: float = 0.0,
    constant_tire_mu: float | None = None,
    cornering_drag_coefficient: float = 0.0,
) -> dict[str, np.ndarray]:
    """Replay recorded controls by station; ``Vehicle`` derives elapsed time."""

    required = (
        "torque_feedback_nm",
        "front_brake_pressure_psi",
        "rear_brake_pressure_psi",
    )
    missing = [name for name in required if name not in recorded]
    if missing:
        raise ValueError(
            "Distance replay is missing high-rate channel(s): " + ", ".join(missing)
        )
    distance_m = np.asarray(distance_m, dtype=float)
    curvature_per_m = np.asarray(curvature_per_m, dtype=float)
    output_distance_m = np.asarray(output_distance_m, dtype=float)
    if len(distance_m) < 2 or np.any(np.diff(distance_m) <= 0.0):
        raise ValueError("Distance replay station must be strictly increasing")
    if len(curvature_per_m) != len(distance_m):
        raise ValueError("Curvature must contain one value per distance sample")
    if any(len(np.asarray(recorded[name])) != len(distance_m) for name in required):
        raise ValueError("Recorded controls must match the distance coordinate")

    replay_distance_m = distance_m - distance_m[0]
    target_distance_m = output_distance_m - output_distance_m[0]
    trailing_gap_m = target_distance_m[-1] - replay_distance_m[-1]
    if trailing_gap_m > 1.0:
        raise ValueError("Output station lies outside the replay distance domain")
    if trailing_gap_m > 1e-9:
        replay_distance_m = np.r_[replay_distance_m, target_distance_m[-1]]
        curvature_per_m = np.r_[curvature_per_m, curvature_per_m[-1]]
        recorded = {
            name: np.r_[np.asarray(values), np.asarray(values)[-1]]
            for name, values in recorded.items()
        }

    vehicle = Vehicle(initial_speed_mps=max(initial_speed_mps, 0.0))
    if drag_coefficient is not None:
        vehicle.aero.drag_coefficient = drag_coefficient
    if motor_to_wheel_efficiency is not None:
        vehicle.drivetrain.chain_drive.efficiency = motor_to_wheel_efficiency
    if front_brake_torque_per_psi_lbfin is not None:
        vehicle.brakes.front_torque_per_pressure_lbfin_per_psi = (
            brake_gain_count_per_axle * front_brake_torque_per_psi_lbfin
        )
    else:
        vehicle.brakes.front_torque_per_pressure_lbfin_per_psi *= (
            brake_gain_count_per_axle
        )
    if rear_brake_torque_per_psi_lbfin is not None:
        vehicle.brakes.rear_torque_per_pressure_lbfin_per_psi = (
            brake_gain_count_per_axle * rear_brake_torque_per_psi_lbfin
        )
    else:
        vehicle.brakes.rear_torque_per_pressure_lbfin_per_psi *= (
            brake_gain_count_per_axle
        )
    vehicle.brakes.pressure_force_model = brake_pressure_model
    vehicle.brakes.pressure_deadband_psi = brake_deadband_psi
    vehicle.brakes.maximum_force_request_n = maximum_brake_force_request_n
    vehicle.tire.longitudinal_slip_relaxation_length_m = (
        longitudinal_slip_relaxation_length_m
    )
    vehicle.tire.constant_friction_coefficient = constant_tire_mu
    vehicle.cornering_drag_coefficient = cornering_drag_coefficient
    vehicle.battery.initial_state_of_charge = min(
        max(initial_soc_percent / 100.0, 0.0), 1.0
    )
    vehicle.validate()
    vehicle.reset_state()

    rolling_radius_m = vehicle.drivetrain.rolling_radius_m
    output: dict[str, list[float]] = {
        "time_s": [],
        "distance_m": [],
        "speed_mps": [],
        "longitudinal_accel_mps2": [],
        "motor_rpm": [],
        "pack_power_kw": [],
        "battery_soc_percent": [],
        "requested_curvature_per_m": [],
        "achieved_curvature_per_m": [],
        "path_speed_ceiling_mps": [],
        "path_torque_limited": [],
        "path_brake_force_added_n": [],
    }
    telemetry_names: tuple[str, ...] | None = None

    def append_state(requested_curvature: float) -> None:
        nonlocal telemetry_names
        telemetry = vehicle.telemetry_snapshot()
        if telemetry_names is None:
            telemetry_names = tuple(telemetry)
            for name in telemetry_names:
                output[f"telemetry.{name}"] = []
        output["time_s"].append(vehicle.time_s)
        output["distance_m"].append(vehicle.distance_m)
        output["speed_mps"].append(vehicle.speed_mps)
        output["longitudinal_accel_mps2"].append(vehicle.longitudinal_acceleration_mps2)
        output["motor_rpm"].append(telemetry["motor.speed_rpm"])
        output["pack_power_kw"].append(telemetry["battery.power_w"] / 1000.0)
        output["battery_soc_percent"].append(
            telemetry["battery.state_of_charge"] * 100.0
        )
        output["requested_curvature_per_m"].append(requested_curvature)
        output["achieved_curvature_per_m"].append(vehicle.curvature_per_m)
        output["path_speed_ceiling_mps"].append(float("nan"))
        output["path_torque_limited"].append(0.0)
        output["path_brake_force_added_n"].append(0.0)
        for name in telemetry_names:
            output[f"telemetry.{name}"].append(float(telemetry[name]))

    append_state(float(curvature_per_m[0]))
    for index, distance_step_m in enumerate(np.diff(replay_distance_m)):
        torque_nm = float(recorded["torque_feedback_nm"][index])
        if torque_nm < -0.5 and negative_torque_policy == "error":
            raise UnmodeledRecordedControlError(
                "Negative motor torque is active; select rear-brake or clip"
            )
        front_pressure_psi = max(
            float(recorded["front_brake_pressure_psi"][index]), 0.0
        )
        rear_pressure_psi = max(float(recorded["rear_brake_pressure_psi"][index]), 0.0)
        rear_motor_brake_force_n = 0.0
        if torque_nm < -0.5 and negative_torque_policy == "rear-brake":
            rear_motor_brake_force_n = (
                -torque_nm
                * vehicle.drivetrain.chain_drive.ratio
                / vehicle.drivetrain.chain_drive.efficiency
                / rolling_radius_m
            )
        requested_curvature = float(curvature_per_m[index])
        controls = Controls(
            motor_torque_request_nm=max(torque_nm, 0.0),
            front_brake_pressure_psi=front_pressure_psi,
            rear_brake_pressure_psi=rear_pressure_psi,
            rear_regenerative_brake_force_request_n=rear_motor_brake_force_n,
            steering_angle_rad=np.arctan(
                requested_curvature * vehicle.chassis.wheelbase_m
            ),
        )
        vehicle.update_state(controls, float(distance_step_m))
        append_state(requested_curvature)

    native = {name: np.asarray(values) for name, values in output.items()}
    return {
        name: np.interp(target_distance_m, native["distance_m"], values)
        for name, values in native.items()
    }


def _save_plot(
    path: Path,
    x_measured: np.ndarray,
    x_sim: np.ndarray,
    x_label: str,
    measured: dict[str, np.ndarray],
    simulated: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(13, 17), sharex=True, layout="constrained")
    panels = (
        (
            "gnss_speed_mps",
            "speed_mps",
            "Speed [m/s]",
            "Measured GNSS (shifted back)",
            "Simulation",
        ),
        (
            "imu_longitudinal_accel_mps2",
            "longitudinal_accel_mps2",
            "Longitudinal accel. [m/s²]",
            "Measured corrected IMU",
            "Simulation",
        ),
        (
            "motor_rpm",
            "motor_rpm",
            "Motor speed [RPM]",
            "Measured inverter feedback",
            "Simulation",
        ),
        (
            "battery_power_kw",
            "pack_power_kw",
            "Pack power [kW]",
            "Measured HVC",
            "Simulation",
        ),
    )
    for axis, (measured_key, sim_key, ylabel, measured_label, sim_label) in zip(
        axes, panels, strict=False
    ):
        axis.plot(x_measured, measured[measured_key], lw=1.3, label=measured_label)
        axis.plot(x_sim, simulated[sim_key], lw=1.3, label=sim_label)
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right", ncol=2)
    if "path_speed_ceiling_mps" in simulated and np.any(
        np.isfinite(simulated["path_speed_ceiling_mps"])
    ):
        axes[0].plot(
            x_sim,
            simulated["path_speed_ceiling_mps"],
            color="black",
            lw=1.0,
            ls="--",
            alpha=0.75,
            label="Hard path ceiling",
        )
        axes[0].legend(loc="upper right", ncol=3)
    axes[1].plot(
        x_measured,
        measured["gnss_kinematic_acceleration_mps2"],
        lw=1.4,
        color="#54A24B",
        label="GNSS kinematic (21 m smooth)",
    )
    axes[1].legend(loc="upper right", ncol=3)
    axes[4].plot(
        x_measured, measured["torque_command_nm"], lw=1.3, label="Torque command"
    )
    axes[4].plot(
        x_measured, measured["torque_feedback_nm"], lw=1.3, label="Torque feedback"
    )
    axes[4].axhline(0.0, color="0.4", linewidth=0.8)
    axes[4].set_ylabel("Motor torque [Nm]")
    axes[4].legend(loc="upper right", ncol=2)
    axes[5].plot(
        x_measured, measured["brake_pressure_psi"], lw=1.3, label="Brake pressure"
    )
    axes[5].axhline(0.0, color="0.4", linewidth=0.8)
    axes[5].set(xlabel=x_label, ylabel="Brake pressure [psi]")
    axes[5].legend(loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("First endurance lap: measured vs current simulation")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _save_soc_plot(
    path: Path,
    distance_m: np.ndarray,
    measured_soc_percent: np.ndarray,
    simulated_soc_percent: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        layout="constrained",
    )
    axes[0].plot(distance_m, measured_soc_percent, lw=2.0, label="Recorded HVC SOC")
    axes[0].plot(distance_m, simulated_soc_percent, lw=2.0, label="Simulation")
    axes[0].set_ylabel("Pack SOC [%]")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(
        distance_m,
        measured_soc_percent - measured_soc_percent[0],
        lw=2.0,
        label="Recorded change",
    )
    axes[1].plot(
        distance_m,
        simulated_soc_percent - simulated_soc_percent[0],
        lw=2.0,
        label="Simulated change",
    )
    axes[1].set_ylabel("SOC change [percentage points]")
    axes[1].set_xlabel("Recorded lap station [m]")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("First endurance lap: recorded vs simulated pack SOC")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _save_brake_validation_plot(
    path: Path,
    recorded_distance_m: np.ndarray,
    measured: dict[str, np.ndarray],
    simulated: dict[str, np.ndarray],
) -> None:
    """Plot the native-time brake replay on the recorded distance coordinate."""

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(14, 13),
        sharex=True,
        layout="constrained",
    )
    axes[0].plot(
        recorded_distance_m,
        measured["gnss_speed_mps"],
        lw=1.6,
        label="Measured GNSS",
    )
    axes[0].plot(
        recorded_distance_m,
        simulated["speed_mps"],
        lw=1.6,
        label="Simulation",
    )
    axes[0].set_ylabel("Speed [m/s]")

    axes[1].plot(
        recorded_distance_m,
        measured["imu_longitudinal_accel_mps2"],
        lw=1.1,
        alpha=0.8,
        label="Corrected IMU X",
    )
    axes[1].plot(
        recorded_distance_m,
        measured["gnss_kinematic_acceleration_mps2"],
        lw=1.5,
        label="GNSS kinematic",
    )
    axes[1].plot(
        recorded_distance_m,
        simulated["longitudinal_accel_mps2"],
        lw=1.5,
        label="Simulation",
    )
    axes[1].set_ylabel("Longitudinal accel. [m/s²]")

    axes[2].plot(
        recorded_distance_m,
        measured.get("front_brake_pressure_psi", measured["brake_pressure_psi"]),
        lw=1.5,
        label="Front BSE pressure",
    )
    if "rear_brake_pressure_psi" in measured:
        axes[2].plot(
            recorded_distance_m,
            measured["rear_brake_pressure_psi"],
            lw=1.5,
            label="Rear MOBO pressure",
        )
    axes[2].set_ylabel("Pressure [psi]")

    axes[3].plot(
        recorded_distance_m,
        simulated["telemetry.brakes.front_friction_force_n"],
        lw=1.4,
        label="Front applied friction",
    )
    axes[3].plot(
        recorded_distance_m,
        simulated["telemetry.brakes.rear_friction_force_n"],
        lw=1.4,
        label="Rear applied total",
    )
    axes[3].plot(
        recorded_distance_m,
        simulated["telemetry.controls.rear_regenerative_brake_force_request_n"],
        lw=1.2,
        ls="--",
        label="Rear motor/backdrive request",
    )
    axes[3].set(xlabel="Recorded lap station [m]", ylabel="Brake force [N]")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=3)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Distance-native independent-axle brake replay")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lap-csv",
        type=Path,
        default=ROOT / "analysis/data/lap/first_lap.csv",
    )
    parser.add_argument(
        "--high-rate-controls-csv",
        type=Path,
        default=ROOT / "analysis/data/brakes/first_lap_100hz.csv",
        help="High-rate controls used to preserve impulse between GNSS samples",
    )
    parser.add_argument(
        "--track-csv",
        type=Path,
        default=ROOT / "analysis/data/track/gnss_imu_endurance_track.csv",
    )
    parser.add_argument(
        "--corrected-imu-csv",
        type=Path,
        default=ROOT / "analysis/data/imu/first_lap_corrected_imu.csv",
    )
    parser.add_argument(
        "--course-map",
        type=Path,
        default=ROOT / "analysis/data/lap/official_course_map.png",
    )
    parser.add_argument(
        "--map-alignment",
        type=Path,
        default=ROOT / "analysis/data/lap/manual_map_alignment.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "analysis/endurance/output"
    )
    parser.add_argument("--gnss-lag-s", type=float, default=DEFAULT_GNSS_LAG_S)
    parser.add_argument(
        "--track-station-offset-m",
        type=float,
        default=2.5,
        help=(
            "Distance added when sampling every track-derived physics channel; "
            "positive values move track features earlier relative to telemetry"
        ),
    )
    parser.add_argument(
        "--brake-pressure-station-offset-m",
        type=float,
        default=0.0,
        help="Distance added when sampling recorded brake pressure",
    )
    parser.add_argument("--brake-deadband-psi", type=float, default=0.0)
    parser.add_argument(
        "--front-brake-torque-per-psi-lbfin",
        type=float,
        default=10.12849472,
    )
    parser.add_argument(
        "--rear-brake-torque-per-psi-lbfin",
        type=float,
        default=5.390972994,
    )
    parser.add_argument(
        "--brake-gain-count-per-axle",
        type=float,
        default=1.0,
        help=(
            "Number of wheel/caliper assemblies represented by each supplied "
            "front and rear torque-per-pressure gain"
        ),
    )
    parser.add_argument(
        "--brake-pressure-model",
        choices=("linear-hardware-gains", "firmware-force-map"),
        default="linear-hardware-gains",
        help=(
            "Use the supplied linear torque gains or the opt-in nonlinear "
            "front/rear force map found in TREV4-Controls"
        ),
    )
    parser.add_argument(
        "--longitudinal-slip-relaxation-length-m",
        type=float,
        default=0.0,
        help="Zero disables transient tire-slip buildup; positive values enable it",
    )
    parser.add_argument(
        "--constant-tire-mu",
        type=float,
        default=None,
        help="Use one load-independent longitudinal/lateral tire coefficient",
    )
    parser.add_argument(
        "--cornering-drag-coefficient",
        type=float,
        default=0.0,
        help="Tire scrub loss coefficient in F_drag = coefficient * Fy^2 / Fz",
    )
    parser.add_argument(
        "--maximum-brake-force-request-n",
        type=float,
        default=None,
        help="Optional saturation cap for the pressure-derived brake-force request",
    )
    parser.add_argument(
        "--drag-coefficient",
        type=float,
        help="Optional scenario override for aerodynamic Cd.",
    )
    parser.add_argument(
        "--motor-to-wheel-efficiency",
        type=float,
        help="Optional scenario override for chain-drive motor-shaft-to-wheel efficiency.",
    )
    parser.add_argument(
        "--negative-torque-policy",
        choices=("error", "clip", "rear-brake"),
        default="error",
        help=(
            "Reject, clip, or reflect negative motor torque to a rear-axle "
            "braking force"
        ),
    )
    args = parser.parse_args()
    if (
        args.gnss_lag_s < 0
        or args.brake_deadband_psi < 0
        or args.front_brake_torque_per_psi_lbfin < 0
        or args.rear_brake_torque_per_psi_lbfin < 0
        or args.brake_gain_count_per_axle <= 0
        or args.longitudinal_slip_relaxation_length_m < 0
        or (args.constant_tire_mu is not None and args.constant_tire_mu <= 0)
        or args.cornering_drag_coefficient < 0
        or (
            args.maximum_brake_force_request_n is not None
            and args.maximum_brake_force_request_n <= 0
        )
        or (args.drag_coefficient is not None and args.drag_coefficient < 0)
        or (
            args.motor_to_wheel_efficiency is not None
            and not 0 < args.motor_to_wheel_efficiency <= 1
        )
    ):
        parser.error("GNSS lag and brake parameters cannot be negative")

    lap = read_numeric_csv(args.lap_csv)
    corrected = read_numeric_csv(args.corrected_imu_csv)
    track = SpatialTrack.from_csv(args.track_csv)
    required = (
        "gps_x_m",
        "gps_y_m",
        "gps_x_filtered_m",
        "gps_y_filtered_m",
        "gps_speed_mps",
        "motor_rpm",
        "battery_power_kw",
        "battery_soc_percent",
        "torque_command_nm",
        "torque_feedback_nm",
        "brake_pressure_psi",
    )
    missing = [name for name in required if name not in lap]
    if missing:
        raise ValueError(f"First-lap CSV is missing: {', '.join(missing)}")
    gnss_curvature_map_variants = {
        window_m: _gnss_path_curvature(
            lap["gps_x_filtered_m"],
            lap["gps_y_filtered_m"],
            smoothing_distance_m=window_m,
        )
        for window_m in (0.0, 5.0, 10.0)
    }
    time_key = first_present(lap, TIME_COLUMNS, "first-lap timestamp")
    time_s = lap[time_key]
    if not args.high_rate_controls_csv.is_file():
        parser.error("Distance replay requires --high-rate-controls-csv")
    high_rate = read_numeric_csv(args.high_rate_controls_csv)
    high_rate_time_key = first_present(
        high_rate, TIME_COLUMNS, "high-rate control timestamp"
    )
    high_rate_time_s = high_rate[high_rate_time_key]
    imu, imu_columns = corrected_imu_at_lap_times(lap, corrected)
    shifted_gnss_speed = interpolate_channel(
        time_s,
        lap["gps_speed_mps"],
        time_s + args.gnss_lag_s,
    )
    station_m = project_to_track_distance(track, lap["gps_x_m"], lap["gps_y_m"])
    spatial = SpatialCoordinate.from_samples(station_m)
    compact_time_s = spatial.values(time_s)
    measured = {name: spatial.values(lap[name]) for name in required}
    measured.update({name: spatial.values(values) for name, values in imu.items()})
    measured["gnss_speed_mps"] = spatial.values(shifted_gnss_speed)
    gnss_curvature_at_lap_times = _gnss_path_curvature(
        lap["gps_x_filtered_m"], lap["gps_y_filtered_m"]
    )
    measured["gnss_curvature_per_m"] = spatial.values(gnss_curvature_at_lap_times)
    measured_distance_m = spatial.distance_m
    high_rate_required = (
        "torque_feedback_nm",
        "front_brake_pressure_psi",
        "rear_brake_pressure_psi",
    )
    missing_high_rate = [name for name in high_rate_required if name not in high_rate]
    if missing_high_rate:
        raise ValueError(
            "High-rate controls CSV is missing: " + ", ".join(missing_high_rate)
        )
    high_rate_station_m = np.interp(high_rate_time_s, time_s, station_m)
    high_rate_spatial = SpatialCoordinate.from_samples(high_rate_station_m)
    high_rate_distance_m = high_rate_spatial.distance_m
    spatial_controls = {
        name: high_rate_spatial.values(high_rate[name]) for name in high_rate_required
    }
    unshifted_front_pressure_psi = spatial_controls["front_brake_pressure_psi"].copy()
    unshifted_rear_pressure_psi = spatial_controls["rear_brake_pressure_psi"].copy()
    if abs(args.brake_pressure_station_offset_m) > 1e-12:
        for channel_name, unshifted_pressure_psi in (
            ("front_brake_pressure_psi", unshifted_front_pressure_psi),
            ("rear_brake_pressure_psi", unshifted_rear_pressure_psi),
        ):
            spatial_controls[channel_name] = np.interp(
                high_rate_distance_m + args.brake_pressure_station_offset_m,
                high_rate_distance_m,
                unshifted_pressure_psi,
                left=0.0,
                right=0.0,
            )
    measured["front_brake_pressure_psi"] = np.interp(
        measured_distance_m,
        high_rate_distance_m,
        spatial_controls["front_brake_pressure_psi"],
    )
    measured["rear_brake_pressure_psi"] = np.interp(
        measured_distance_m,
        high_rate_distance_m,
        spatial_controls["rear_brake_pressure_psi"],
    )
    unshifted_brake_pressure_psi = np.interp(
        measured_distance_m,
        high_rate_distance_m,
        unshifted_front_pressure_psi,
    )
    measured["brake_pressure_psi"] = measured["front_brake_pressure_psi"]
    physics_station_m = measured_distance_m + args.track_station_offset_m
    curvature = periodic_curvature_at_station(track, physics_station_m)
    # Use the post-offset GNSS speed already shifted backward by gnss_lag_s.
    # This keeps v²*kappa on the same corrected time/station basis as the
    # other measured comparison channels.
    post_offset_speed_squared_m2ps2 = measured["gnss_speed_mps"] ** 2
    map_curvature_lateral_accel_mps2 = post_offset_speed_squared_m2ps2 * curvature
    gnss_curvature_lateral_accel_mps2 = (
        post_offset_speed_squared_m2ps2 * measured["gnss_curvature_per_m"]
    )
    normalized_map_curvature_lateral_accel_mps2 = _normalize_to_reference(
        map_curvature_lateral_accel_mps2,
        measured["imu_lateral_accel_mps2"],
    )
    normalized_gnss_curvature_lateral_accel_mps2 = _normalize_to_reference(
        gnss_curvature_lateral_accel_mps2,
        measured["imu_lateral_accel_mps2"],
    )
    gnss_curvature_by_window_at_station = {
        window_m: spatial.values(values)
        for window_m, values in gnss_curvature_map_variants.items()
    }
    gnss_curvature_by_window_at_station[11.0] = measured["gnss_curvature_per_m"]
    normalized_gnss_v2_curvature_by_window_m = {
        window_m: _normalize_to_reference(
            post_offset_speed_squared_m2ps2 * window_curvature,
            measured["imu_lateral_accel_mps2"],
        )
        for window_m, window_curvature in gnss_curvature_by_window_at_station.items()
    }
    uniform_distance_m = np.arange(
        measured_distance_m[0], measured_distance_m[-1] + 0.25, 0.5
    )
    uniform_speed_mps = np.interp(
        uniform_distance_m,
        measured_distance_m,
        measured["gnss_speed_mps"],
    )
    # A 21 m cubic Savitzky-Golay window suppresses GNSS quantization while
    # retaining each major acceleration and braking event.
    gnss_window_samples = 43
    uniform_gnss_acceleration_mps2 = 0.5 * savgol_filter(
        uniform_speed_mps**2,
        gnss_window_samples,
        3,
        deriv=1,
        delta=0.5,
    )
    gnss_kinematic_acceleration_mps2 = np.interp(
        measured_distance_m,
        uniform_distance_m,
        uniform_gnss_acceleration_mps2,
    )
    measured["gnss_kinematic_acceleration_mps2"] = gnss_kinematic_acceleration_mps2
    high_rate_curvature_per_m = periodic_curvature_at_station(
        track,
        high_rate_distance_m + args.track_station_offset_m,
    )
    try:
        simulated = replay_lap_distance(
            high_rate_distance_m,
            spatial_controls,
            high_rate_curvature_per_m,
            measured_distance_m,
            initial_speed_mps=float(measured["gnss_speed_mps"][0]),
            initial_soc_percent=float(measured["battery_soc_percent"][0]),
            drag_coefficient=args.drag_coefficient,
            motor_to_wheel_efficiency=args.motor_to_wheel_efficiency,
            front_brake_torque_per_psi_lbfin=(args.front_brake_torque_per_psi_lbfin),
            rear_brake_torque_per_psi_lbfin=args.rear_brake_torque_per_psi_lbfin,
            brake_gain_count_per_axle=args.brake_gain_count_per_axle,
            brake_pressure_model=args.brake_pressure_model,
            brake_deadband_psi=args.brake_deadband_psi,
            maximum_brake_force_request_n=args.maximum_brake_force_request_n,
            negative_torque_policy=args.negative_torque_policy,
            longitudinal_slip_relaxation_length_m=(
                args.longitudinal_slip_relaxation_length_m
            ),
            constant_tire_mu=args.constant_tire_mu,
            cornering_drag_coefficient=args.cornering_drag_coefficient,
        )
    except UnmodeledRecordedControlError as error:
        parser.error(str(error))

    elapsed_s = compact_time_s - compact_time_s[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_gnss_curvature_map_comparison(
        args.output_dir / "gnss_curvature_smoothing_on_official_map.png",
        args.course_map,
        args.map_alignment,
        lap["gps_x_filtered_m"],
        lap["gps_y_filtered_m"],
        gnss_curvature_map_variants,
        imu["imu_lateral_accel_mps2"] / np.maximum(shifted_gnss_speed**2, 1.0),
        gnss_curvature_at_lap_times,
    )
    _save_soc_plot(
        args.output_dir / "first_lap_soc_distance.png",
        measured_distance_m,
        measured["battery_soc_percent"],
        simulated["battery_soc_percent"],
    )
    _save_imu_curvature_plot(
        args.output_dir / "imu_y_vs_map_curvature_distance.png",
        measured_distance_m,
        measured["imu_lateral_accel_mps2"],
        curvature,
        curvature_label="Map-derived track curvature",
        title="First endurance lap: IMU Y vs map-derived curvature",
    )
    _save_imu_curvature_plot(
        args.output_dir / "imu_y_vs_gnss_curvature_distance.png",
        measured_distance_m,
        measured["imu_lateral_accel_mps2"],
        measured["gnss_curvature_per_m"],
        curvature_label="GNSS-derived path curvature (11 m smooth)",
        title="First endurance lap: IMU Y vs GNSS-derived curvature",
    )
    _save_imu_curvature_acceleration_plot(
        args.output_dir / "imu_y_vs_map_v2_curvature_distance.png",
        measured_distance_m,
        measured["imu_lateral_accel_mps2"],
        map_curvature_lateral_accel_mps2,
        curvature_label="Map v²κ (post-offset GNSS speed)",
        title="First endurance lap: IMU Y vs map-derived v²κ",
    )
    _save_imu_curvature_acceleration_plot(
        args.output_dir / "imu_y_vs_gnss_v2_curvature_distance.png",
        measured_distance_m,
        measured["imu_lateral_accel_mps2"],
        gnss_curvature_lateral_accel_mps2,
        curvature_label="GNSS v²κ (post-offset GNSS speed)",
        title="First endurance lap: IMU Y vs GNSS-derived v²κ",
    )
    _save_normalized_curvature_comparison(
        args.output_dir / "imu_y_vs_normalized_v2_curvature_distance.png",
        measured_distance_m,
        measured["imu_lateral_accel_mps2"],
        normalized_map_curvature_lateral_accel_mps2,
        normalized_gnss_curvature_lateral_accel_mps2,
    )
    _save_imu_gnss_smoothing_overlay(
        args.output_dir / "imu_y_vs_gnss_smoothing_windows_distance.png",
        measured_distance_m,
        measured["imu_lateral_accel_mps2"],
        normalized_gnss_v2_curvature_by_window_m,
    )
    _save_plot(
        args.output_dir / "first_lap_comparison_distance.png",
        measured_distance_m,
        measured_distance_m,
        "Recorded lap station [m]",
        measured,
        simulated,
    )
    _save_brake_validation_plot(
        args.output_dir / "braking_accuracy_distance.png",
        measured_distance_m,
        measured,
        simulated,
    )

    rows = []
    for index in range(len(compact_time_s)):
        rows.append(
            {
                "elapsed_time_s": float(elapsed_s[index]),
                "measured_distance_m": float(measured_distance_m[index]),
                "sim_distance_m": float(simulated["distance_m"][index]),
                "measured_front_brake_pressure_psi": float(
                    measured.get(
                        "front_brake_pressure_psi", measured["brake_pressure_psi"]
                    )[index]
                ),
                "measured_rear_brake_pressure_psi": float(
                    measured.get(
                        "rear_brake_pressure_psi", measured["brake_pressure_psi"]
                    )[index]
                ),
                **{
                    f"measured_{name}": float(measured[name][index])
                    for name in (
                        "gnss_speed_mps",
                        "imu_longitudinal_accel_mps2",
                        "motor_rpm",
                        "battery_power_kw",
                        "battery_soc_percent",
                        "torque_command_nm",
                        "torque_feedback_nm",
                        "brake_pressure_psi",
                    )
                },
                "sim_speed_mps": float(simulated["speed_mps"][index]),
                "sim_longitudinal_accel_mps2": float(
                    simulated["longitudinal_accel_mps2"][index]
                ),
                "sim_motor_rpm": float(simulated["motor_rpm"][index]),
                "sim_pack_power_kw": float(simulated["pack_power_kw"][index]),
                "sim_battery_soc_percent": float(
                    simulated["battery_soc_percent"][index]
                ),
                "map_curvature_per_m": float(curvature[index]),
                "gnss_curvature_per_m": float(measured["gnss_curvature_per_m"][index]),
                "map_v2_curvature_lateral_accel_mps2": float(
                    map_curvature_lateral_accel_mps2[index]
                ),
                "gnss_v2_curvature_lateral_accel_mps2": float(
                    gnss_curvature_lateral_accel_mps2[index]
                ),
                "normalized_map_v2_curvature_lateral_accel_mps2": float(
                    normalized_map_curvature_lateral_accel_mps2[index]
                ),
                "normalized_gnss_v2_curvature_lateral_accel_mps2": float(
                    normalized_gnss_curvature_lateral_accel_mps2[index]
                ),
                "sim_achieved_curvature_per_m": float(
                    simulated["achieved_curvature_per_m"][index]
                ),
                "sim_path_speed_ceiling_mps": float(
                    simulated["path_speed_ceiling_mps"][index]
                ),
                "sim_path_torque_limited": float(
                    simulated["path_torque_limited"][index]
                ),
                "sim_path_brake_force_added_n": float(
                    simulated["path_brake_force_added_n"][index]
                ),
            }
        )
    with (args.output_dir / "first_lap_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    soc_rows = [
        {
            "recorded_lap_station_m": float(measured_distance_m[index]),
            "recorded_elapsed_time_s": float(elapsed_s[index]),
            "simulated_elapsed_time_s": float(simulated["time_s"][index]),
            "recorded_battery_soc_percent": float(
                measured["battery_soc_percent"][index]
            ),
            "simulated_battery_soc_percent": float(
                simulated["battery_soc_percent"][index]
            ),
            "recorded_battery_power_kw": float(measured["battery_power_kw"][index]),
            "simulated_battery_power_kw": float(simulated["pack_power_kw"][index]),
        }
        for index in range(len(compact_time_s))
    ]
    with (args.output_dir / "first_lap_soc.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(soc_rows[0]))
        writer.writeheader()
        writer.writerows(soc_rows)

    # Lossless analysis table: retain every numeric recorded channel, aligned
    # corrected-IMU axes, derived track channels, and every model telemetry
    # channel. The smaller comparison CSV above remains stable for existing
    # consumers.
    all_rows = []
    telemetry_names = sorted(
        name.removeprefix("telemetry.")
        for name in simulated
        if name.startswith("telemetry.")
    )
    for index in range(len(compact_time_s)):
        all_rows.append(
            {
                "elapsed_time_s": float(elapsed_s[index]),
                "measured_distance_m": float(measured_distance_m[index]),
                "analysis_shifted_gnss_speed_mps": float(
                    measured["gnss_speed_mps"][index]
                ),
                "analysis_map_curvature_per_m": float(curvature[index]),
                "analysis_gnss_curvature_per_m": float(
                    measured["gnss_curvature_per_m"][index]
                ),
                "analysis_map_v2_curvature_lateral_accel_mps2": float(
                    map_curvature_lateral_accel_mps2[index]
                ),
                "analysis_gnss_v2_curvature_lateral_accel_mps2": float(
                    gnss_curvature_lateral_accel_mps2[index]
                ),
                "analysis_normalized_map_v2_curvature_lateral_accel_mps2": float(
                    normalized_map_curvature_lateral_accel_mps2[index]
                ),
                "analysis_normalized_gnss_v2_curvature_lateral_accel_mps2": float(
                    normalized_gnss_curvature_lateral_accel_mps2[index]
                ),
                "analysis_gnss_kinematic_acceleration_mps2": float(
                    gnss_kinematic_acceleration_mps2[index]
                ),
                "sim_path_speed_ceiling_mps": float(
                    simulated["path_speed_ceiling_mps"][index]
                ),
                "sim_path_torque_limited": float(
                    simulated["path_torque_limited"][index]
                ),
                "sim_path_brake_force_added_n": float(
                    simulated["path_brake_force_added_n"][index]
                ),
                **{
                    f"recorded_{name}": float(spatial.values(values)[index])
                    for name, values in lap.items()
                },
                "recorded_raw_5hz_brake_pressure_psi": float(
                    spatial.values(lap["brake_pressure_psi"])[index]
                ),
                "recorded_unshifted_spatial_front_brake_pressure_psi": float(
                    unshifted_brake_pressure_psi[index]
                ),
                **{
                    f"aligned_{name}": float(values[index])
                    for name, values in imu.items()
                },
                **{
                    f"sim_{name.replace('.', '_')}": float(
                        simulated[f"telemetry.{name}"][index]
                    )
                    for name in telemetry_names
                },
            }
        )
    with (args.output_dir / "first_lap_all_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    acceleration_error = (
        simulated["longitudinal_accel_mps2"] - measured["imu_longitudinal_accel_mps2"]
    )
    gnss_acceleration_error = (
        simulated["longitudinal_accel_mps2"] - gnss_kinematic_acceleration_mps2
    )
    unique_sim_distance_m, unique_sim_indices = np.unique(
        simulated["distance_m"],
        return_index=True,
    )
    distance_overlap = (measured_distance_m >= unique_sim_distance_m[0]) & (
        measured_distance_m <= unique_sim_distance_m[-1]
    )
    simulated_speed_at_measured_distance_mps = np.interp(
        measured_distance_m[distance_overlap],
        unique_sim_distance_m,
        simulated["speed_mps"][unique_sim_indices],
    )
    torque_feedback_nm = measured["torque_feedback_nm"]
    brake_pressure_psi = measured["brake_pressure_psi"]
    regime_masks = {
        "drive": (torque_feedback_nm >= 10.0) & (brake_pressure_psi < 5.0),
        "braking": brake_pressure_psi >= 5.0,
        "negative_motor_torque": (
            (torque_feedback_nm < -0.5) & (brake_pressure_psi < 5.0)
        ),
        "coast": (
            (torque_feedback_nm >= -0.5)
            & (torque_feedback_nm < 10.0)
            & (brake_pressure_psi < 5.0)
        ),
    }

    def regime_metrics(mask: np.ndarray) -> dict[str, float | int]:
        imu_error = acceleration_error[mask]
        gnss_error = gnss_acceleration_error[mask]
        return {
            "sample_count": int(np.count_nonzero(mask)),
            "imu_error_bias_mps2": float(np.mean(imu_error)),
            "imu_error_rmse_mps2": _rms(imu_error),
            "gnss_kinematic_error_bias_mps2": float(np.mean(gnss_error)),
            "gnss_kinematic_error_rmse_mps2": _rms(gnss_error),
        }

    metrics = {
        "speed_rmse_mps": _rms(simulated["speed_mps"] - measured["gnss_speed_mps"]),
        "speed_rmse_distance_aligned_mps": _rms(
            simulated_speed_at_measured_distance_mps
            - measured["gnss_speed_mps"][distance_overlap]
        ),
        "acceleration_rmse_mps2": _rms(acceleration_error),
        "acceleration_rmse_vs_gnss_kinematic_mps2": _rms(gnss_acceleration_error),
        "corrected_imu_vs_gnss_kinematic_rmse_mps2": _rms(
            measured["imu_longitudinal_accel_mps2"] - gnss_kinematic_acceleration_mps2
        ),
        "motor_rpm_rmse": _rms(simulated["motor_rpm"] - measured["motor_rpm"]),
        "pack_power_rmse_kw": _rms(
            simulated["pack_power_kw"] - measured["battery_power_kw"]
        ),
        "recorded_initial_soc_percent": float(measured["battery_soc_percent"][0]),
        "recorded_final_soc_percent": float(measured["battery_soc_percent"][-1]),
        "recorded_soc_drop_percentage_points": float(
            measured["battery_soc_percent"][0] - measured["battery_soc_percent"][-1]
        ),
        "simulated_initial_soc_percent": float(simulated["battery_soc_percent"][0]),
        "simulated_final_soc_percent": float(simulated["battery_soc_percent"][-1]),
        "simulated_soc_drop_percentage_points": float(
            simulated["battery_soc_percent"][0] - simulated["battery_soc_percent"][-1]
        ),
        "recorded_duration_s": float(elapsed_s[-1]),
        "simulated_duration_s": float(simulated["time_s"][-1]),
        "recorded_distance_m": float(measured_distance_m[-1]),
        "simulated_distance_m": float(simulated["distance_m"][-1]),
        "gnss_speed_lag_correction_s": args.gnss_lag_s,
        "corrected_imu_columns": imu_columns,
        "simulation_inputs": {
            "independent_coordinate": "projected recorded lap station",
            "control_replay_domain": "distance",
            "speed_integration": "v_next^2 = v^2 + 2*a*distance_step",
            "component_timestep": (
                "derived internally by Vehicle from each spatial cell"
            ),
            "torque": "motor torque feedback",
            "curvature": "selected SpatialTrack curvature (GNSS/IMU fused track for this run)",
            "track_station_offset_m": args.track_station_offset_m,
            "brake_pressure_station_offset_m": (args.brake_pressure_station_offset_m),
            "brake_deadband_psi": args.brake_deadband_psi,
            "high_rate_controls_csv": str(args.high_rate_controls_csv.resolve()),
            "high_rate_control_station_projection": (
                "timestamps are used only to project samples onto recorded GNSS station"
            ),
            "front_brake_torque_per_psi_lbfin": (args.front_brake_torque_per_psi_lbfin),
            "rear_brake_torque_per_psi_lbfin": (args.rear_brake_torque_per_psi_lbfin),
            "brake_gain_count_per_axle": args.brake_gain_count_per_axle,
            "brake_pressure_model": args.brake_pressure_model,
            "brake_pressure_channels": (
                "independent front VCU BSE and rear MOBO BSE indexed by station"
            ),
            "longitudinal_slip_relaxation_length_m": (
                args.longitudinal_slip_relaxation_length_m
            ),
            "constant_tire_mu": args.constant_tire_mu,
            "cornering_drag_coefficient": args.cornering_drag_coefficient,
            "maximum_brake_force_request_n": args.maximum_brake_force_request_n,
            "negative_torque_policy": args.negative_torque_policy,
            "drag_coefficient": args.drag_coefficient,
            "motor_to_wheel_efficiency": args.motor_to_wheel_efficiency,
            "path_constraint": "disabled: raw recorded-control replay",
        },
        "path_controller": {
            "enabled": False,
            "torque_limited_sample_count": 0,
            "brake_added_sample_count": 0,
            "maximum_added_brake_force_n": 0.0,
            "maximum_speed_above_ceiling_mps": None,
        },
        "acceleration_error_by_regime": {
            name: regime_metrics(mask) for name, mask in regime_masks.items()
        },
        "active_limit_sample_fraction": {
            name: float(np.mean(simulated[f"telemetry.limits.{name}"]))
            for name in (
                "motor_envelope_active",
                "traction_active",
                "lateral_saturated",
                "brake_grip_active",
                "speed_active",
            )
        },
        "limitations": [
            "Road grade is not modeled; tire scrub uses a single fitted cornering-drag coefficient.",
            "The rear MOBO pressure channel is sampled at about 5 Hz and interpolated before station projection.",
            (
                "The firmware force-map run assumes the brake-balance force terms are newtons; a brake dyno or high-rate rear-pressure log is needed to confirm that interpretation."
                if args.brake_pressure_model == "firmware-force-map"
                else "The linear pressure model retains the supplied front/rear hardware gains exactly."
            ),
            "Track curvature is converted to an equivalent bicycle steering angle.",
            "Recorded controls are applied at their projected track stations; elapsed time is generated only inside Vehicle.",
        ],
    }
    (args.output_dir / "first_lap_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote full-lap comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
