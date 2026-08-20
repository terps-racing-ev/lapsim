"""Compare recorded and simulated signals by distance on fused-track straights.

This intentionally is an analysis script rather than a new solver.  It joins the
first-lap log to a corrected-IMU CSV by timestamp, projects each logged GNSS
position onto the combined GNSS/IMU ``SpatialTrack``, identifies low-curvature
runs, and replays the logged controls once for each accepted straight.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.common import (  # noqa: E402
    DEFAULT_GNSS_LAG_S,
    TIME_COLUMNS,
    RecordedControlAdapter,
    UnmodeledRecordedControlError,
    corrected_imu_at_lap_times,
    first_present,
    interpolate_channel,
    project_to_track_distance,
    read_numeric_csv,
)
from lapsim import SpatialTrack  # noqa: E402
from vehicle_model import RCTheveninBattery, Vehicle  # noqa: E402

DEFAULT_CURVATURE_THRESHOLD_PER_M = 0.015
DEFAULT_MINIMUM_STRAIGHT_LENGTH_M = 18.0
DEFAULT_HVC_POWER_LAG_S = 0.09
DEFAULT_COURSE_MAP = ROOT / "analysis/data/lap/official_course_map.png"
DEFAULT_MAP_ALIGNMENT = ROOT / "analysis/data/lap/manual_map_alignment.json"


def vehicle_model_brake_force_per_psi_n() -> float:
    """Return combined front-plus-rear brake force from the vehicle model."""

    vehicle = Vehicle()
    return vehicle.brakes.equivalent_vehicle_force_per_pressure_n_per_psi(
        vehicle.tire.rolling_radius_m
    )


@dataclass(frozen=True)
class Straight:
    """One contiguous map cell range classified as a straight."""

    number: int
    start_m: float
    end_m: float
    cells: slice

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


def identify_straights(
    track: SpatialTrack,
    *,
    curvature_threshold_per_m: float,
    minimum_length_m: float,
) -> list[Straight]:
    """Find contiguous cells whose absolute map curvature is below threshold."""

    if curvature_threshold_per_m <= 0 or minimum_length_m <= 0:
        raise ValueError("straight thresholds must be positive")
    mask = np.abs(np.asarray(track.curvature_per_m)) <= curvature_threshold_per_m
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    result: list[Straight] = []
    for start_cell, end_cell in zip(starts, ends, strict=True):
        start_m, end_m = track.distance_m[start_cell], track.distance_m[end_cell]
        if end_m - start_m >= minimum_length_m:
            result.append(
                Straight(len(result) + 1, start_m, end_m, slice(start_cell, end_cell))
            )
    return result


def _station_in_straight(
    station_m: np.ndarray, straight: Straight, lap_length_m: float
) -> np.ndarray:
    # Stations have been unwrapped, while map straight limits live in [0, L).
    wrapped = np.mod(station_m, lap_length_m)
    return (wrapped >= straight.start_m) & (wrapped <= straight.end_m)


def _rms(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def simulate_straight(
    straight: Straight,
    sample: dict[str, np.ndarray],
    *,
    torque_source: str,
    control_adapter: RecordedControlAdapter,
    spatial_step_m: float,
    motor_to_wheel_efficiency: float | None = None,
    drag_coefficient: float | None = None,
) -> dict[str, np.ndarray]:
    """Replay controls as functions of distance through ``Vehicle``.

    Torque and brake pressure are interpolated at the vehicle's current
    simulated distance. ``Vehicle`` derives elapsed time internally from each
    requested distance step, so recorded timestamps never select the controls.
    """

    if spatial_step_m <= 0:
        raise ValueError("spatial_step_m must be positive")
    measured_distance = np.asarray(sample["distance_from_straight_start_m"])
    initial_soc = float(np.clip(sample["battery_soc_percent"][0] / 100.0, 0.0, 1.0))
    vehicle = Vehicle(
        initial_speed_mps=max(float(sample["gnss_speed_mps"][0]), 0.0),
        battery=RCTheveninBattery(initial_state_of_charge=initial_soc),
    )
    if motor_to_wheel_efficiency is not None:
        vehicle.drivetrain.chain_drive.efficiency = motor_to_wheel_efficiency
    if drag_coefficient is not None:
        vehicle.aero.drag_coefficient = drag_coefficient
    control_adapter.configure_vehicle(vehicle)
    vehicle.validate()
    vehicle.reset_state()
    initial_acceleration_mps2 = float(sample["imu_longitudinal_accel_mps2"][0])
    vehicle.longitudinal_acceleration_mps2 = initial_acceleration_mps2
    initial_lateral_acceleration_mps2 = float(sample["imu_lateral_accel_mps2"][0])
    vehicle.lateral_acceleration_mps2 = initial_lateral_acceleration_mps2
    torque_key = (
        "torque_feedback_nm" if torque_source == "feedback" else "torque_command_nm"
    )
    control_adapter.validate_trace(sample[torque_key], sample["brake_pressure_psi"])

    finite = np.isfinite(measured_distance)
    finite &= np.isfinite(sample[torque_key])
    finite &= np.isfinite(sample["brake_pressure_psi"])
    if finite.sum() < 2:
        raise ValueError("A spatial replay needs at least two finite control samples")
    order = np.argsort(measured_distance[finite])
    profile_distance = measured_distance[finite][order]
    profile_torque = sample[torque_key][finite][order]
    profile_brake = sample["brake_pressure_psi"][finite][order]
    profile_distance, unique_indices = np.unique(profile_distance, return_index=True)
    profile_torque = profile_torque[unique_indices]
    profile_brake = profile_brake[unique_indices]
    straight_end_m = float(profile_distance[-1])
    if straight_end_m <= 0:
        raise ValueError("Spatial replay distance must increase beyond zero")

    output: dict[str, list[float]] = {"sim_time_s": []}
    fields: dict[str, list[float]] = {
        "sim_distance_m": [],
        "sim_speed_mps": [],
        "sim_accel_mps2": [],
        "sim_lateral_accel_mps2": [],
        "sim_motor_rpm": [],
        "sim_motor_torque_nm": [],
        "sim_pack_power_kw": [],
        "sim_pack_current_a": [],
        "sim_pack_voltage_v": [],
        "sim_pack_soc_percent": [],
        "sim_drive_force_n": [],
        "sim_brake_force_n": [],
    }
    maximum_steps = 1_000_000
    while (
        vehicle.distance_m < straight_end_m
        and len(fields["sim_distance_m"]) < maximum_steps
    ):
        start_distance_m = vehicle.distance_m
        start_speed_mps = vehicle.speed_mps
        motor_torque_nm = float(
            np.interp(start_distance_m, profile_distance, profile_torque)
        )
        brake_pressure_psi = float(
            np.interp(start_distance_m, profile_distance, profile_brake)
        )
        controls = control_adapter.controls(
            motor_torque_nm=motor_torque_nm,
            brake_pressure_psi=brake_pressure_psi,
            curvature_per_m=0.0,
            wheelbase_m=vehicle.chassis.wheelbase_m,
        )
        distance_step_m = min(spatial_step_m, straight_end_m - start_distance_m)
        start_time_s = vehicle.time_s
        vehicle.update_state(controls, float(distance_step_m))
        telemetry = vehicle.telemetry_snapshot()
        output["sim_time_s"].append(start_time_s)
        fields["sim_distance_m"].append(start_distance_m)
        fields["sim_speed_mps"].append(start_speed_mps)
        fields["sim_accel_mps2"].append(
            initial_acceleration_mps2
            if len(fields["sim_accel_mps2"]) == 0
            else vehicle.longitudinal_acceleration_mps2
        )
        fields["sim_lateral_accel_mps2"].append(
            initial_lateral_acceleration_mps2
            if len(fields["sim_lateral_accel_mps2"]) == 0
            else vehicle.lateral_acceleration_mps2
        )
        fields["sim_motor_rpm"].append(telemetry["motor.speed_rpm"])
        fields["sim_motor_torque_nm"].append(telemetry["motor.torque_nm"])
        fields["sim_pack_power_kw"].append(telemetry["battery.power_w"] / 1000.0)
        fields["sim_pack_current_a"].append(telemetry["battery.current_a"])
        fields["sim_pack_voltage_v"].append(telemetry["battery.terminal_voltage_v"])
        fields["sim_pack_soc_percent"].append(
            100.0 * telemetry["battery.state_of_charge"]
        )
        fields["sim_drive_force_n"].append(vehicle.current_drive_force_n)
        fields["sim_brake_force_n"].append(vehicle.current_friction_braking_force_n)
        if vehicle.distance_m <= start_distance_m + 1e-9:
            break
    if len(fields["sim_distance_m"]) >= maximum_steps:
        raise RuntimeError("Spatial replay exceeded its integration-step limit")
    if not fields["sim_distance_m"]:
        raise RuntimeError("Spatial replay did not advance")

    # Add the final state so interpolation covers the measured straight end.
    telemetry = vehicle.telemetry_snapshot()
    output["sim_time_s"].append(vehicle.time_s)
    fields["sim_distance_m"].append(vehicle.distance_m)
    fields["sim_speed_mps"].append(vehicle.speed_mps)
    fields["sim_accel_mps2"].append(vehicle.longitudinal_acceleration_mps2)
    fields["sim_lateral_accel_mps2"].append(vehicle.lateral_acceleration_mps2)
    fields["sim_motor_rpm"].append(telemetry["motor.speed_rpm"])
    fields["sim_motor_torque_nm"].append(telemetry["motor.torque_nm"])
    fields["sim_pack_power_kw"].append(telemetry["battery.power_w"] / 1000.0)
    fields["sim_pack_current_a"].append(telemetry["battery.current_a"])
    fields["sim_pack_voltage_v"].append(telemetry["battery.terminal_voltage_v"])
    fields["sim_pack_soc_percent"].append(100.0 * telemetry["battery.state_of_charge"])
    fields["sim_drive_force_n"].append(vehicle.current_drive_force_n)
    fields["sim_brake_force_n"].append(vehicle.current_friction_braking_force_n)
    return {
        name: np.asarray(values, dtype=float)
        for name, values in {**output, **fields}.items()
    }


def _save_requested_comparison_plot(
    output_dir: Path,
    straight: Straight,
    data: dict[str, np.ndarray],
    simulated: dict[str, np.ndarray],
) -> None:
    """Plot measured/model quantities only against distance for one straight."""

    distance = data["distance_from_straight_start_m"]
    sim_distance = simulated["sim_distance_m"]
    fig, axes = plt.subplots(6, 1, figsize=(11, 17), sharex=True, layout="constrained")
    panels = (
        (
            data["gnss_speed_mps"],
            simulated["sim_speed_mps"],
            "Speed [m/s]",
            "Measured GNSS",
            "Simulation",
        ),
        (
            data["imu_longitudinal_accel_mps2"],
            simulated["sim_accel_mps2"],
            "Longitudinal accel. [m/s^2]",
            "Measured corrected IMU",
            "Simulation",
        ),
        (
            data["imu_lateral_accel_mps2"],
            simulated["sim_lateral_accel_mps2"],
            "Lateral accel. [m/s^2]",
            "Measured corrected IMU",
            "Simulation (straight)",
        ),
    )
    for axis, (measured, model, ylabel, measured_label, model_label) in zip(
        axes, panels, strict=False
    ):
        axis.plot(distance, measured, "o-", ms=3, label=measured_label)
        axis.plot(sim_distance, model, "o-", ms=3, label=model_label)
        axis.set_ylabel(ylabel)
    axes[0].plot(
        distance,
        data["motor_rpm_vehicle_speed_mps"],
        "o--",
        ms=3,
        label="Measured motor RPM -> speed",
    )
    axes[1].plot(
        distance,
        data["motor_rpm_longitudinal_accel_mps2"],
        "o--",
        ms=3,
        label="Measured motor RPM -> longitudinal accel.",
    )
    torque_axis = axes[3]
    torque_axis.plot(
        distance, data["torque_command_nm"], "o-", ms=3, label="Measured command"
    )
    torque_axis.plot(
        distance, data["torque_feedback_nm"], "o-", ms=3, label="Measured feedback"
    )
    torque_axis.plot(
        sim_distance,
        simulated["sim_motor_torque_nm"],
        "o-",
        ms=3,
        label="Sim delivered",
    )
    torque_axis.plot(
        distance,
        data["measured_equivalent_motor_brake_torque_nm"],
        "o--",
        ms=3,
        label="Measured friction brake (motor equiv.)",
    )
    torque_axis.plot(
        sim_distance,
        simulated["sim_equivalent_motor_brake_torque_nm"],
        "o--",
        ms=3,
        label="Sim friction brake (motor equiv.)",
    )
    torque_axis.set_ylabel("Motor torque [Nm]")
    brake_axis = axes[4]
    brake_axis.plot(
        distance,
        data["measured_friction_braking_force_n"],
        "o-",
        ms=3,
        label="Measured pressure -> force",
    )
    brake_axis.plot(
        sim_distance,
        simulated["sim_brake_force_n"],
        "o-",
        ms=3,
        label="Simulation applied",
    )
    brake_axis.set_ylabel("Friction braking force [N]")
    power_axis = axes[5]
    power_axis.plot(
        distance,
        data["battery_power_kw"],
        "o-",
        ms=3,
        label="Measured HVC (time-aligned)",
    )
    power_axis.plot(
        sim_distance, simulated["sim_pack_power_kw"], "o-", ms=3, label="Simulation"
    )
    power_axis.set_ylabel("Pack power [kW]")
    power_axis.set_xlabel("Distance from straight start [m]")
    for axis in axes:
        axis.axhline(0.0, color="0.4", linewidth=0.7)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(
        f"Straight {straight.number}: measured vs simulation ({straight.length_m:.1f} m)"
    )
    fig.savefig(output_dir / f"straight_{straight.number:02d}_comparison.png", dpi=190)
    plt.close(fig)


def _save_drivetrain_comparison_plot(
    output_dir: Path,
    straight: Straight,
    data: dict[str, np.ndarray],
    simulated: dict[str, np.ndarray],
) -> None:
    """Plot recorded and simulated drivetrain channels against straight distance."""

    measured_distance_m = data["distance_from_straight_start_m"]
    simulated_distance_m = simulated["sim_distance_m"]
    figure, axes = plt.subplots(
        3, 2, figsize=(15, 12), sharex=True, layout="constrained"
    )
    panels = (
        (
            data["gnss_speed_mps"],
            simulated["sim_speed_mps"],
            "Vehicle speed [m/s]",
            "Recorded GNSS",
            "Simulation",
        ),
        (
            data["motor_rpm"],
            simulated["sim_motor_rpm"],
            "Motor speed [rpm]",
            "Recorded inverter",
            "Simulation",
        ),
        (
            data["battery_power_kw"],
            simulated["sim_pack_power_kw"],
            "Pack power [kW]",
            "Recorded HVC",
            "Simulation",
        ),
        (
            data["battery_current_a"],
            simulated["sim_pack_current_a"],
            "Pack current [A]",
            "Recorded HVC",
            "Simulation",
        ),
        (
            data["battery_voltage_v"],
            simulated["sim_pack_voltage_v"],
            "Pack voltage [V]",
            "Recorded HVC",
            "Simulation",
        ),
    )
    for axis, (recorded, model, ylabel, recorded_label, model_label) in zip(
        axes.ravel(), panels, strict=False
    ):
        axis.plot(
            measured_distance_m,
            recorded,
            "o-",
            lw=1.25,
            ms=3.5,
            label=recorded_label,
        )
        axis.plot(
            simulated_distance_m,
            model,
            lw=1.5,
            label=model_label,
        )
        axis.set_ylabel(ylabel)

    torque_axis = axes.ravel()[5]
    torque_axis.plot(
        measured_distance_m,
        data["torque_command_nm"],
        "o-",
        lw=1.0,
        ms=3.5,
        label="Recorded command",
    )
    torque_axis.plot(
        measured_distance_m,
        data["torque_feedback_nm"],
        "o-",
        lw=1.25,
        ms=3.5,
        label="Recorded feedback",
    )
    torque_axis.plot(
        simulated_distance_m,
        simulated["sim_motor_torque_nm"],
        lw=1.5,
        label="Sim delivered",
    )
    torque_axis.set_ylabel("Motor torque [N m]")

    for axis in axes.ravel():
        axis.axhline(0.0, color="0.45", lw=0.7)
        axis.grid(alpha=0.23)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", frameon=False)
    axes[-1, 0].set_xlabel("Distance from straight start [m]")
    axes[-1, 1].set_xlabel("Distance from straight start [m]")
    figure.suptitle(
        f"Straight {straight.number}: recorded vs simulated drivetrain "
        f"({straight.length_m:.1f} m)"
    )
    figure.savefig(
        output_dir / f"straight_{straight.number:02d}_drivetrain_vs_distance.png",
        dpi=190,
    )
    plt.close(figure)


def _save_drivetrain_summary_plot(
    output_dir: Path,
    all_data: list[tuple[Straight, dict[str, np.ndarray], dict[str, np.ndarray]]],
) -> None:
    """Plot drivetrain signals over only the accepted straight track stations."""

    figure, axes = plt.subplots(
        3, 2, figsize=(16, 13), sharex=True, layout="constrained"
    )
    axes = axes.ravel()
    for straight_index, (straight, data, simulated) in enumerate(all_data):
        measured_station_m = (
            straight.start_m + data["distance_from_straight_start_m"]
        )
        simulated_station_m = straight.start_m + simulated["sim_distance_m"]
        label_recorded = "Recorded" if straight_index == 0 else None
        label_simulated = "Simulation" if straight_index == 0 else None
        panels = (
            (data["gnss_speed_mps"], simulated["sim_speed_mps"]),
            (data["motor_rpm"], simulated["sim_motor_rpm"]),
            (data["battery_power_kw"], simulated["sim_pack_power_kw"]),
            (data["battery_current_a"], simulated["sim_pack_current_a"]),
            (data["battery_voltage_v"], simulated["sim_pack_voltage_v"]),
        )
        for axis, (recorded, model) in zip(axes, panels, strict=False):
            axis.plot(
                measured_station_m,
                recorded,
                "o-",
                color="#1f77b4",
                lw=1.15,
                ms=3.0,
                label=label_recorded,
            )
            axis.plot(
                simulated_station_m,
                model,
                color="#ff7f0e",
                lw=1.4,
                label=label_simulated,
            )
        torque_axis = axes[5]
        torque_axis.plot(
            measured_station_m,
            data["torque_feedback_nm"],
            "o-",
            color="#1f77b4",
            lw=1.15,
            ms=3.0,
            label=("Recorded feedback" if straight_index == 0 else None),
        )
        torque_axis.plot(
            simulated_station_m,
            simulated["sim_motor_torque_nm"],
            color="#ff7f0e",
            lw=1.4,
            label=("Sim delivered" if straight_index == 0 else None),
        )

    for axis, ylabel in zip(
        axes,
        (
            "Vehicle speed [m/s]",
            "Motor speed [rpm]",
            "Pack power [kW]",
            "Pack current [A]",
            "Pack voltage [V]",
            "Motor torque [N m]",
        ),
        strict=True,
    ):
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.23)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", frameon=False)
    axes[-2].set_xlabel("Lap distance [m] (accepted straights only)")
    axes[-1].set_xlabel("Lap distance [m] (accepted straights only)")
    figure.suptitle("Strict straights: recorded vs simulated drivetrain signals")
    figure.savefig(output_dir / "drivetrain_signals_vs_distance.png", dpi=190)
    plt.close(figure)


def _save_summary_plots(
    output_dir: Path,
    track: SpatialTrack,
    straights: list[Straight],
    all_data: list[tuple[Straight, dict[str, np.ndarray], dict[str, np.ndarray]]],
    *,
    course_map_path: Path,
    map_alignment_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    ax.plot(track.x_m, track.y_m, color="0.75", lw=1.2, label="Combined GNSS/IMU track")
    for straight in straights:
        cell = straight.cells
        ax.plot(
            np.asarray(track.x_m)[cell],
            np.asarray(track.y_m)[cell],
            lw=3,
            label=f"Straight {straight.number}",
        )
    ax.set(
        aspect="equal",
        xlabel="East [m]",
        ylabel="North [m]",
        title="Straight sections isolated from the combined GNSS/IMU track",
    )
    ax.legend(fontsize="small", ncol=2)
    fig.savefig(output_dir / "straights_on_fused_track.png", dpi=170)
    # Retain the original filename for callers that already consume it.
    fig.savefig(output_dir / "straights_on_track.png", dpi=170)
    plt.close(fig)

    course_map = plt.imread(course_map_path)
    with map_alignment_path.open(encoding="utf-8") as stream:
        alignment = json.load(stream)
    transform = alignment["effective_affine_transform"]
    matrix = np.asarray(transform["matrix_px_per_m"], dtype=float)
    offset = np.asarray(transform["offset_px"], dtype=float)

    def to_pixels(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        return np.column_stack((x_m, y_m)) @ matrix.T + offset

    official_fig, official_axis = plt.subplots(figsize=(14, 4), layout="constrained")
    official_axis.imshow(course_map, origin="upper")
    official_axis.set_title("Official endurance course map")
    official_axis.axis("off")
    official_fig.savefig(output_dir / "official_course_map.png", dpi=190)
    plt.close(official_fig)

    track_pixels = to_pixels(np.asarray(track.x_m), np.asarray(track.y_m))
    overlay_fig, overlay_axis = plt.subplots(figsize=(14, 4), layout="constrained")
    overlay_axis.imshow(course_map, origin="upper")
    overlay_axis.plot(
        track_pixels[:, 0],
        track_pixels[:, 1],
        color="white",
        lw=3.4,
        alpha=0.9,
    )
    overlay_axis.plot(
        track_pixels[:, 0],
        track_pixels[:, 1],
        color="0.25",
        lw=1.4,
        label="Combined GNSS/IMU track",
    )
    for straight in straights:
        indices = np.arange(straight.cells.start, straight.cells.stop + 1)
        pixels = track_pixels[indices]
        overlay_axis.plot(
            pixels[:, 0],
            pixels[:, 1],
            lw=4.0,
            label=f"Straight {straight.number}",
        )
    overlay_axis.set(
        xlim=(0, course_map.shape[1]),
        ylim=(course_map.shape[0], 0),
        title="Official course map + combined GNSS/IMU straights overlay",
    )
    overlay_axis.set_aspect("equal")
    overlay_axis.axis("off")
    overlay_axis.legend(loc="lower right", fontsize="small", ncol=3)
    overlay_fig.savefig(output_dir / "straights_official_overlay.png", dpi=190)
    plt.close(overlay_fig)

    # Deliver the three requested map views as one PNG. The tall fused-track
    # panel spans both rows so the very wide official drawing remains legible.
    combined_fig = plt.figure(figsize=(18, 10), layout="constrained")
    grid = combined_fig.add_gridspec(2, 2, width_ratios=(0.72, 2.4))
    fused_axis = combined_fig.add_subplot(grid[:, 0])
    official_axis = combined_fig.add_subplot(grid[0, 1])
    combined_overlay_axis = combined_fig.add_subplot(grid[1, 1])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fused_axis.plot(track.x_m, track.y_m, color="0.75", lw=1.2)
    for index, straight in enumerate(straights):
        point_indices = np.arange(straight.cells.start, straight.cells.stop + 1)
        fused_axis.plot(
            np.asarray(track.x_m)[point_indices],
            np.asarray(track.y_m)[point_indices],
            color=colors[index % len(colors)],
            lw=4.0,
            label=f"Straight {straight.number}",
        )
    fused_axis.set(
        aspect="equal",
        xlabel="East [m]",
        ylabel="North [m]",
        title="Combined GNSS/IMU track",
    )
    fused_axis.grid(alpha=0.18)

    official_axis.imshow(course_map, origin="upper")
    official_axis.set_title("Official endurance course map")
    official_axis.axis("off")

    combined_overlay_axis.imshow(course_map, origin="upper")
    combined_overlay_axis.plot(
        track_pixels[:, 0], track_pixels[:, 1], color="white", lw=3.4, alpha=0.9
    )
    combined_overlay_axis.plot(
        track_pixels[:, 0],
        track_pixels[:, 1],
        color="0.25",
        lw=1.4,
        label="Combined GNSS/IMU track",
    )
    for index, straight in enumerate(straights):
        point_indices = np.arange(straight.cells.start, straight.cells.stop + 1)
        pixels = track_pixels[point_indices]
        combined_overlay_axis.plot(
            pixels[:, 0],
            pixels[:, 1],
            color=colors[index % len(colors)],
            lw=4.0,
            label=f"Straight {straight.number}",
        )
    combined_overlay_axis.set(
        xlim=(0, course_map.shape[1]),
        ylim=(course_map.shape[0], 0),
        title="Combined overlay",
    )
    combined_overlay_axis.set_aspect("equal")
    combined_overlay_axis.axis("off")
    combined_overlay_axis.legend(loc="lower right", fontsize="small", ncol=3)
    combined_fig.suptitle("Straight-section map comparison", fontsize=18)
    combined_fig.savefig(output_dir / "straights_map_comparison.png", dpi=190)
    plt.close(combined_fig)

    if not all_data:
        return
    columns = min(3, len(all_data))
    rows = int(np.ceil(len(all_data) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.5 * rows),
        squeeze=False,
        layout="constrained",
    )
    for axis, (straight, data, simulated) in zip(axes.flat, all_data, strict=False):
        axis.plot(
            data["distance_from_straight_start_m"],
            data["imu_longitudinal_accel_mps2"],
            "o-",
            ms=2.5,
            label="IMU",
        )
        axis.plot(
            simulated["sim_distance_m"],
            simulated["sim_accel_mps2"],
            "o-",
            ms=2.5,
            label="Model",
        )
        axis.set(
            title=f"Straight {straight.number} ({straight.length_m:.0f} m)",
            xlabel="Distance [m]",
            ylabel="Accel. [m/s²]",
        )
        axis.legend(fontsize="small")
    for axis in axes.flat[len(all_data) :]:
        axis.remove()
    fig.savefig(output_dir / "summary_acceleration_by_distance.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lap-csv",
        type=Path,
        default=ROOT / "analysis/data/lap/first_lap.csv",
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
    parser.add_argument("--course-map", type=Path, default=DEFAULT_COURSE_MAP)
    parser.add_argument("--map-alignment", type=Path, default=DEFAULT_MAP_ALIGNMENT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/accel/output",
    )
    parser.add_argument(
        "--gnss-lag-s",
        type=float,
        default=DEFAULT_GNSS_LAG_S,
        help="GNSS delay; speed and position are shifted backward by this amount.",
    )
    parser.add_argument(
        "--hvc-power-lag-s",
        type=float,
        default=DEFAULT_HVC_POWER_LAG_S,
        help="Desired total HVC power delay correction; metadata prevents double shifting.",
    )
    parser.add_argument(
        "--curvature-threshold-per-m",
        type=float,
        default=DEFAULT_CURVATURE_THRESHOLD_PER_M,
    )
    parser.add_argument(
        "--minimum-straight-length-m",
        type=float,
        default=DEFAULT_MINIMUM_STRAIGHT_LENGTH_M,
    )
    parser.add_argument(
        "--torque-source", choices=("feedback", "command"), default="command"
    )
    parser.add_argument(
        "--simulation-spatial-step-m",
        type=float,
        default=0.25,
        help="Maximum distance advanced before spatial controls are sampled again.",
    )
    parser.add_argument(
        "--motor-to-wheel-efficiency",
        type=float,
        help="Override the vehicle chain-drive efficiency for this replay.",
    )
    parser.add_argument(
        "--drag-coefficient",
        type=float,
        help="Override the vehicle aerodynamic drag coefficient for this replay.",
    )
    parser.add_argument(
        "--brake-force-per-psi-n",
        type=float,
        default=vehicle_model_brake_force_per_psi_n(),
        help=(
            "Brake-force calibration. Defaults to the vehicle model's combined "
            "front-plus-rear brake torque divided by tire rolling radius."
        ),
    )
    parser.add_argument("--brake-deadband-psi", type=float, default=0.0)
    parser.add_argument(
        "--negative-torque-policy", choices=("error", "clip"), default="error"
    )
    parser.add_argument(
        "--straight-numbers",
        type=int,
        nargs="+",
        help="Analyze only these detected straight numbers.",
    )
    args = parser.parse_args()
    if (
        args.gnss_lag_s < 0
        or args.hvc_power_lag_s < 0
        or (args.brake_force_per_psi_n is not None and args.brake_force_per_psi_n < 0)
        or args.brake_deadband_psi < 0
        or args.simulation_spatial_step_m <= 0
        or (
            args.motor_to_wheel_efficiency is not None
            and not 0 < args.motor_to_wheel_efficiency <= 1
        )
        or (args.drag_coefficient is not None and args.drag_coefficient < 0)
    ):
        parser.error(
            "lag and brake parameters cannot be negative; spatial integration "
            "parameters must be positive"
        )

    track = SpatialTrack.from_csv(args.track_csv)
    lap = read_numeric_csv(args.lap_csv)
    corrected = read_numeric_csv(args.corrected_imu_csv)
    required_lap = (
        "gps_x_m",
        "gps_y_m",
        "gps_speed_mps",
        "motor_rpm",
        "battery_power_kw",
        "battery_voltage_v",
        "battery_current_a",
        "battery_soc_percent",
        "torque_command_nm",
        "torque_feedback_nm",
        "brake_pressure_psi",
    )
    missing = [name for name in required_lap if name not in lap]
    if missing:
        raise ValueError(f"First-lap CSV is missing: {', '.join(missing)}")
    lap_time_name = first_present(lap, TIME_COLUMNS, "first-lap timestamp")
    imu_data, imu_columns = corrected_imu_at_lap_times(lap, corrected)
    time_s = lap[lap_time_name]
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("First-lap timestamps must strictly increase")

    # A backward shift means the GNSS value recorded at t + lag is compared at
    # physical time t. Apply it consistently to both speed and the position that
    # defines the distance axis.
    shifted_gnss_speed = interpolate_channel(
        time_s, lap["gps_speed_mps"], time_s + args.gnss_lag_s
    )
    shifted_gnss_x = interpolate_channel(
        time_s, lap["gps_x_m"], time_s + args.gnss_lag_s
    )
    shifted_gnss_y = interpolate_channel(
        time_s, lap["gps_y_m"], time_s + args.gnss_lag_s
    )
    lap_metadata_path = args.lap_csv.with_suffix(".json")
    lap_metadata = (
        json.loads(lap_metadata_path.read_text(encoding="utf-8"))
        if lap_metadata_path.is_file()
        else {}
    )
    existing_hvc_power_shift_s = float(
        lap_metadata.get("channels", {})
        .get("battery_power_kw", {})
        .get("backward_time_shift_s", 0.0)
    )
    additional_hvc_power_shift_s = max(
        args.hvc_power_lag_s - existing_hvc_power_shift_s, 0.0
    )
    shifted_battery_power_kw = interpolate_channel(
        time_s,
        lap["battery_power_kw"],
        time_s + additional_hvc_power_shift_s,
    )
    station_m = project_to_track_distance(track, shifted_gnss_x, shifted_gnss_y)
    straights = identify_straights(
        track,
        curvature_threshold_per_m=args.curvature_threshold_per_m,
        minimum_length_m=args.minimum_straight_length_m,
    )
    if not straights:
        raise ValueError("No fused-track straight meets the supplied thresholds")
    all_straights = straights
    if args.straight_numbers:
        requested = set(args.straight_numbers)
        available = {straight.number for straight in straights}
        missing_numbers = sorted(requested - available)
        if missing_numbers:
            parser.error(
                f"straight numbers not detected: {missing_numbers}; "
                f"available: {sorted(available)}"
            )
        straights = [straight for straight in straights if straight.number in requested]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_rows: list[dict[str, float | int]] = []
    for straight in straights:
        for cell_index in range(straight.cells.start, straight.cells.stop):
            geometry_rows.append(
                {
                    "straight_number": straight.number,
                    "track_distance_m": float(track.distance_m[cell_index]),
                    "distance_from_straight_start_m": float(
                        track.distance_m[cell_index] - straight.start_m
                    ),
                    "x_m": float(track.x_m[cell_index]),
                    "y_m": float(track.y_m[cell_index]),
                    "curvature_per_m": float(track.curvature_per_m[cell_index]),
                }
            )
    with (args.output_dir / "straight_geometry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(geometry_rows[0]))
        writer.writeheader()
        writer.writerows(geometry_rows)

    metrics: list[dict[str, float | int]] = []
    combined_rows: list[dict[str, float | int]] = []
    plotted: list[tuple[Straight, dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    base = {name: lap[name] for name in required_lap if name in lap}
    base.update(imu_data)
    base["time_s"] = time_s
    base["gnss_speed_mps"] = shifted_gnss_speed
    base["battery_power_kw"] = shifted_battery_power_kw
    kinematic_vehicle = Vehicle()
    motor_speed_rad_s = lap["motor_rpm"] * (2.0 * np.pi / 60.0)
    base["motor_rpm_vehicle_speed_mps"] = (
        motor_speed_rad_s
        / kinematic_vehicle.drivetrain.chain_drive.ratio
        * kinematic_vehicle.tire.rolling_radius_m
    )
    base["motor_rpm_longitudinal_accel_mps2"] = np.gradient(
        base["motor_rpm_vehicle_speed_mps"], time_s, edge_order=2
    )
    effective_brake_pressure_psi = np.maximum(
        lap["brake_pressure_psi"] - args.brake_deadband_psi, 0.0
    )
    base["measured_friction_braking_force_n"] = (
        args.brake_force_per_psi_n or 0.0
    ) * effective_brake_pressure_psi
    effective_motor_to_wheel_efficiency = (
        args.motor_to_wheel_efficiency
        if args.motor_to_wheel_efficiency is not None
        else kinematic_vehicle.drivetrain.chain_drive.efficiency
    )
    motor_equivalent_brake_scale_nm_per_n = kinematic_vehicle.tire.rolling_radius_m / (
        kinematic_vehicle.drivetrain.chain_drive.ratio
        * effective_motor_to_wheel_efficiency
    )
    base["measured_equivalent_motor_brake_torque_nm"] = (
        -base["measured_friction_braking_force_n"]
        * motor_equivalent_brake_scale_nm_per_n
    )
    base["shifted_gnss_x_m"] = shifted_gnss_x
    base["shifted_gnss_y_m"] = shifted_gnss_y
    base["track_station_m"] = station_m
    control_adapter = RecordedControlAdapter(
        brake_force_per_psi_n=args.brake_force_per_psi_n,
        brake_deadband_psi=args.brake_deadband_psi,
        negative_torque_policy=args.negative_torque_policy,
    )
    for straight in straights:
        mask = _station_in_straight(station_m, straight, track.length_m)
        if mask.sum() < 3:
            continue
        data = {name: values[mask] for name, values in base.items()}
        # Track is closed. After chronological ordering, put both traces at
        # zero distance at the first recorded sample so their start states are
        # compared at the same spatial origin.
        data["distance_from_straight_start_m"] = np.mod(
            data["track_station_m"] - straight.start_m, track.length_m
        )
        order = np.argsort(data["time_s"])
        data = {name: values[order] for name, values in data.items()}
        data["distance_from_straight_start_m"] -= data[
            "distance_from_straight_start_m"
        ][0]
        try:
            simulated = simulate_straight(
                straight,
                data,
                torque_source=args.torque_source,
                control_adapter=control_adapter,
                spatial_step_m=args.simulation_spatial_step_m,
                motor_to_wheel_efficiency=args.motor_to_wheel_efficiency,
                drag_coefficient=args.drag_coefficient,
            )
            simulated["sim_equivalent_motor_brake_torque_nm"] = (
                -simulated["sim_brake_force_n"] * motor_equivalent_brake_scale_nm_per_n
            )
        except UnmodeledRecordedControlError as error:
            parser.error(f"straight {straight.number}: {error}")
        measured_distance = data["distance_from_straight_start_m"]
        aligned = {
            name: np.interp(
                measured_distance,
                simulated["sim_distance_m"],
                simulated[source],
                left=np.nan,
                right=np.nan,
            )
            for name, source in {
                "accel": "sim_accel_mps2",
                "lateral_accel": "sim_lateral_accel_mps2",
                "speed": "sim_speed_mps",
                "rpm": "sim_motor_rpm",
                "motor_torque": "sim_motor_torque_nm",
                "pack_power": "sim_pack_power_kw",
                "pack_current": "sim_pack_current_a",
                "pack_voltage": "sim_pack_voltage_v",
                "pack_soc": "sim_pack_soc_percent",
                "brake_force": "sim_brake_force_n",
                "equivalent_motor_brake_torque": (
                    "sim_equivalent_motor_brake_torque_nm"
                ),
            }.items()
        }
        simulation_at_measured_distance = aligned["accel"]
        speed_at_measured_distance = aligned["speed"]
        rpm_at_measured_distance = aligned["rpm"]
        pack_power_at_measured_distance = aligned["pack_power"]
        accel_error = (
            simulation_at_measured_distance - data["imu_longitudinal_accel_mps2"]
        )
        metrics.append(
            {
                "straight_number": straight.number,
                "track_start_m": straight.start_m,
                "track_end_m": straight.end_m,
                "track_length_m": straight.length_m,
                "sample_count": int(len(data["time_s"])),
                "duration_s": float(data["time_s"][-1] - data["time_s"][0]),
                "start_speed_mps": float(data["gnss_speed_mps"][0]),
                "start_imu_accel_mps2": float(data["imu_longitudinal_accel_mps2"][0]),
                "start_imu_lateral_accel_mps2": float(
                    data["imu_lateral_accel_mps2"][0]
                ),
                "start_battery_soc_percent": float(data["battery_soc_percent"][0]),
                "measured_mean_speed_mps": float(np.mean(data["gnss_speed_mps"])),
                "sim_mean_speed_mps": float(np.nanmean(speed_at_measured_distance)),
                "imu_accel_mean_mps2": float(
                    np.mean(data["imu_longitudinal_accel_mps2"])
                ),
                "sim_accel_mean_mps2": float(
                    np.nanmean(simulation_at_measured_distance)
                ),
                "accel_bias_mps2": float(np.nanmean(accel_error)),
                "accel_rmse_mps2": _rms(accel_error),
                "lateral_accel_rmse_mps2": _rms(
                    aligned["lateral_accel"] - data["imu_lateral_accel_mps2"]
                ),
                "speed_rmse_mps": _rms(
                    speed_at_measured_distance - data["gnss_speed_mps"]
                ),
                "motor_rpm_speed_vs_gnss_rmse_mps": _rms(
                    data["motor_rpm_vehicle_speed_mps"] - data["gnss_speed_mps"]
                ),
                "motor_rpm_accel_vs_imu_rmse_mps2": _rms(
                    data["motor_rpm_longitudinal_accel_mps2"]
                    - data["imu_longitudinal_accel_mps2"]
                ),
                "motor_rpm_rmse": _rms(rpm_at_measured_distance - data["motor_rpm"]),
                "pack_power_rmse_kw": _rms(
                    pack_power_at_measured_distance - data["battery_power_kw"]
                ),
                "pack_current_rmse_a": _rms(
                    aligned["pack_current"] - data["battery_current_a"]
                ),
                "pack_voltage_rmse_v": _rms(
                    aligned["pack_voltage"] - data["battery_voltage_v"]
                ),
                "motor_torque_feedback_rmse_nm": _rms(
                    aligned["motor_torque"] - data["torque_feedback_nm"]
                ),
                "braking_force_rmse_n": _rms(
                    aligned["brake_force"] - data["measured_friction_braking_force_n"]
                ),
                "equivalent_motor_brake_torque_rmse_nm": _rms(
                    aligned["equivalent_motor_brake_torque"]
                    - data["measured_equivalent_motor_brake_torque_nm"]
                ),
                "measured_end_speed_mps": float(data["gnss_speed_mps"][-1]),
                "sim_end_speed_mps": float(speed_at_measured_distance[-1]),
                "measured_mean_motor_rpm": float(np.mean(data["motor_rpm"])),
                "sim_mean_motor_rpm": float(np.nanmean(rpm_at_measured_distance)),
                "measured_mean_pack_power_kw": float(np.mean(data["battery_power_kw"])),
                "sim_mean_pack_power_kw": float(
                    np.nanmean(pack_power_at_measured_distance)
                ),
            }
        )
        for index in range(len(data["time_s"])):
            combined_rows.append(
                {
                    "straight_number": straight.number,
                    **{name: float(values[index]) for name, values in data.items()},
                    **{
                        f"sim_{name}_at_measured_distance": float(values[index])
                        for name, values in aligned.items()
                    },
                }
            )
        _save_requested_comparison_plot(args.output_dir, straight, data, simulated)
        _save_drivetrain_comparison_plot(args.output_dir, straight, data, simulated)
        plotted.append((straight, data, simulated))

    _save_drivetrain_summary_plot(args.output_dir, plotted)
    _save_summary_plots(
        args.output_dir,
        track,
        straights,
        plotted,
        course_map_path=args.course_map,
        map_alignment_path=args.map_alignment,
    )
    if not metrics:
        raise ValueError(
            "No first-lap samples project inside the qualifying fused-track straights"
        )
    with (args.output_dir / "straight_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (args.output_dir / "straight_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)
    summary = {
        "inputs": {
            "lap_csv": str(args.lap_csv.resolve()),
            "track_csv": str(args.track_csv.resolve()),
            "corrected_imu_csv": str(args.corrected_imu_csv.resolve()),
            "course_map": str(args.course_map.resolve()),
            "map_alignment": str(args.map_alignment.resolve()),
        },
        "corrected_imu_columns": imu_columns,
        "gnss_shift": {
            "direction": "backward",
            "lag_s": args.gnss_lag_s,
            "shifted_channels": ["gps_speed_mps", "gps_x_m", "gps_y_m"],
            "basis": "RPM/GNSS timing alignment; vehicle-speed conversion uses the configured ratio and tire rolling radius",
        },
        "hvc_power_shift": {
            "direction": "backward",
            "desired_total_lag_s": args.hvc_power_lag_s,
            "already_applied_in_input_s": existing_hvc_power_shift_s,
            "additional_applied_by_analysis_s": additional_hvc_power_shift_s,
            "effective_total_shift_s": existing_hvc_power_shift_s
            + additional_hvc_power_shift_s,
            "basis": "HVC pack power aligned to torque feedback * motor speed; best transition estimate approximately 0.09 s",
        },
        "motor_rpm_to_vehicle_speed": {
            "equation": "vehicle_speed_mps = motor_rpm * 2*pi/60 / ratio * rolling_radius_m",
            "ratio": kinematic_vehicle.drivetrain.chain_drive.ratio,
            "rolling_radius_m": kinematic_vehicle.tire.rolling_radius_m,
            "assumption": "kinematic conversion without driven-wheel slip",
        },
        "motor_rpm_to_longitudinal_acceleration": {
            "equation": "longitudinal_accel_mps2 = d(motor_rpm_vehicle_speed_mps)/dt",
            "derivative": "second-order numpy gradient on the full lap before straight selection",
            "assumption": "all RPM-derived wheel acceleration is longitudinal; no driven-wheel slip",
        },
        "straight_definition": {
            "absolute_curvature_threshold_per_m": args.curvature_threshold_per_m,
            "minimum_contiguous_length_m": args.minimum_straight_length_m,
            "qualifying_fused_track_straights": len(all_straights),
            "requested_straight_numbers": args.straight_numbers,
            "analyzed_straights": len(metrics),
        },
        "simulation": {
            "torque_source": args.torque_source,
            "control_domain": "recorded controls interpolated by measured distance and queried at current simulated distance",
            "spatial_control_step_m": args.simulation_spatial_step_m,
            "time_integration": "derived internally by Vehicle from each distance step",
            "motor_to_wheel_efficiency": (
                args.motor_to_wheel_efficiency
                if args.motor_to_wheel_efficiency is not None
                else Vehicle().drivetrain.chain_drive.efficiency
            ),
            "motor_to_wheel_efficiency_source": (
                "CLI override"
                if args.motor_to_wheel_efficiency is not None
                else "vehicle default"
            ),
            "drag_coefficient": (
                args.drag_coefficient
                if args.drag_coefficient is not None
                else Vehicle().aero.drag_coefficient
            ),
            "drag_coefficient_source": (
                "CLI override"
                if args.drag_coefficient is not None
                else "vehicle default"
            ),
            "brake_force_per_psi_n": args.brake_force_per_psi_n,
            "brake_deadband_psi": args.brake_deadband_psi,
            "friction_brake_torque_plot": {
                "domain": "negative motor-shaft-equivalent torque",
                "equation": "-F_brake * tire_radius / (ratio * motor_to_wheel_efficiency)",
                "tire_radius_m": kinematic_vehicle.tire.rolling_radius_m,
                "ratio": kinematic_vehicle.drivetrain.chain_drive.ratio,
            },
            "negative_torque_policy": args.negative_torque_policy,
            "curvature_per_m": 0.0,
            "comparison_domain": "distance from each straight entry in meters",
            "interval_signal_alignment": "simulated torque, pack power/current/voltage, and forces are evaluated at the current simulated distance at the beginning of each integration interval",
            "initial_speed": "shifted GNSS speed at each straight entry",
            "initial_acceleration": "corrected IMU longitudinal acceleration at each straight entry; seeds the first load-transfer iteration and is then recomputed from force balance",
            "initial_lateral_acceleration": "corrected IMU lateral acceleration at each straight entry; the zero-curvature replay recomputes it as zero on the next step",
            "initial_battery_soc": "recorded HVC SOC at each straight entry",
        },
        "limitations": [
            "GNSS positions are projected to the combined GNSS/IMU centerline; fusion and projection error can move straight membership.",
            "Vehicle is a no-slip point mass and has no transient acceleration, tire, yaw, or wheel-speed state beyond the scalar initial acceleration seed.",
            "Brake pressure requires an explicit pressure-to-force calibration; zero is accepted only when explicitly supplied.",
            "Chronological Controls does not support regen; negative torque can only be rejected or explicitly clipped.",
        ],
        "metrics": metrics,
    }
    (args.output_dir / "straight_acceleration_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(metrics)} straight comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
