"""Plot the recorded and optimized first endurance laps as PNG comparisons."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

from lapsim import (
    EnduranceRunConfig,
    EnduranceSimulator,
    PathConstraintSolver,
    SpatialTrack,
    UniformPeriodicTorqueParameterization,
)
from vehicle_model import Vehicle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_LAP = ROOT / "analysis/data/lap/first_lap.csv"
DEFAULT_TRACK = ROOT / "analysis/data/track/gnss_imu_endurance_track.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/endurance_optimization/real_track_comparison_2m_braking_limited"
)
REAL_COLOR = "#0072B2"
SIM_COLOR = "#D55E00"
SIM_ALT_COLOR = "#009E73"
REFERENCE_COLOR = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-lap", type=Path, default=DEFAULT_REAL_LAP)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cell-length-m", type=float, default=2.0)
    parser.add_argument("--torque-fraction", type=float, default=0.26)
    parser.add_argument("--maximum-brake-pressure-psi", type=float, default=300.0)
    return parser.parse_args()


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    data: dict[str, np.ndarray] = {}
    for name in rows[0]:
        try:
            data[name] = np.asarray([float(row[name]) for row in rows], dtype=float)
        except ValueError:
            continue
    return data


def coarsen_track(source: SpatialTrack, target_cell_length_m: float) -> SpatialTrack:
    lengths: list[float] = []
    curvatures: list[float] = []
    group_length_m = 0.0
    group_turn_rad = 0.0
    for length_m, curvature_per_m in zip(
        source.cell_length_m, source.curvature_per_m, strict=True
    ):
        group_length_m += length_m
        group_turn_rad += curvature_per_m * length_m
        if group_length_m >= target_cell_length_m - 1e-12:
            lengths.append(group_length_m)
            curvatures.append(group_turn_rad / group_length_m)
            group_length_m = 0.0
            group_turn_rad = 0.0
    if group_length_m > 0.0:
        lengths.append(group_length_m)
        curvatures.append(group_turn_rad / group_length_m)
    return SpatialTrack.from_cells(
        cell_length_m=lengths,
        curvature_per_m=curvatures,
    )


def vehicle_factory() -> Vehicle:
    vehicle = Vehicle()
    vehicle.aero.drag_coefficient = 2.5
    vehicle.drivetrain.chain_drive.efficiency = 0.86
    vehicle.tire.constant_friction_coefficient = 1.8
    vehicle.cornering_drag_coefficient = 0.036
    vehicle.battery.initial_state_of_charge = 0.9815
    vehicle.validate()
    return vehicle


def add_speed_map(
    axis: plt.Axes,
    x_m: np.ndarray,
    y_m: np.ndarray,
    speed_mps: np.ndarray,
    normalization: Normalize,
    title: str,
) -> LineCollection:
    points = np.column_stack((x_m, y_m)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    speed_segments = 0.5 * (speed_mps[:-1] + speed_mps[1:])
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=normalization,
        linewidth=4.0,
        capstyle="round",
    )
    collection.set_array(speed_segments)
    axis.add_collection(collection)
    axis.scatter(
        [x_m[0]],
        [y_m[0]],
        marker="o",
        s=38,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
        label="Start/finish",
    )
    axis.autoscale()
    axis.margins(0.04)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("Local east–west position (m)")
    axis.set_ylabel("Local north–south position (m)")
    axis.grid(False)
    axis.legend(loc="best", frameon=False)
    return collection


def style_signal_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.set_xlim(0.0, 989.0)
    axis.grid(True, alpha=0.25)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.torque_fraction <= 1.0:
        raise ValueError("torque-fraction must lie in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real = read_numeric_csv(args.real_lap)
    source_track = SpatialTrack.from_csv(args.track)
    track = coarsen_track(source_track, args.cell_length_m)
    real_start_speed_mps = float(real["gps_speed_mps"][0])

    constraints = PathConstraintSolver(
        maximum_brake_pressure_psi=args.maximum_brake_pressure_psi
    ).solve(track, vehicle_factory())
    profile = UniformPeriodicTorqueParameterization(8).build(
        (args.torque_fraction,) * 8,
        track,
    )
    result = EnduranceSimulator().run(
        vehicle_factory(),
        constraints,
        profile,
        EnduranceRunConfig(
            laps=1,
            starting_speed_mps=real_start_speed_mps,
            maximum_brake_pressure_psi=args.maximum_brake_pressure_psi,
        ),
        record_telemetry=True,
    )
    if not result.completed or result.telemetry is None:
        raise RuntimeError(result.failure_reason or "simulation did not complete")
    sim = {
        name: np.asarray(values, dtype=float)
        for name, values in result.telemetry.items()
    }

    real_distance_m = real["gps_distance_trip_m"]
    sim_distance_m = sim["endurance.lap_distance_m"]
    sim_map_x_m = np.interp(
        sim_distance_m,
        np.asarray(source_track.distance_m),
        np.asarray(source_track.x_m),
    )
    sim_map_y_m = np.interp(
        sim_distance_m,
        np.asarray(source_track.distance_m),
        np.asarray(source_track.y_m),
    )

    real_energy_kwh = float(
        np.trapezoid(real["battery_power_kw"], real["time_s"]) / 3_600.0
    )
    speed_normalization = Normalize(
        vmin=0.0,
        vmax=float(max(np.max(real["gps_speed_mps"]), np.max(sim["vehicle.speed_mps"]))),
    )
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), constrained_layout=True)
    map_collection = add_speed_map(
        axes[0],
        real["gps_x_filtered_m"],
        real["gps_y_filtered_m"],
        real["gps_speed_mps"],
        speed_normalization,
        (
            f"Recorded competition lap — {real['time_s'][-1]:.2f} s, "
            f"{real_energy_kwh:.3f} kWh"
        ),
    )
    add_speed_map(
        axes[1],
        sim_map_x_m,
        sim_map_y_m,
        sim["vehicle.speed_mps"],
        speed_normalization,
        (
            f"Simulation, {100.0 * args.torque_fraction:.0f}% torque — "
            f"{result.lap_times_s[0]:.2f} s, {result.pack_energy_kwh:.3f} kWh"
        ),
    )
    figure.colorbar(
        map_collection,
        ax=axes,
        location="bottom",
        shrink=0.72,
        pad=0.08,
        label="Speed (m/s)",
    )
    figure.suptitle(
        "First endurance lap: speed around the competition course",
        fontsize=16,
    )
    map_path = args.output_dir / "first_lap_speed_maps.png"
    figure.savefig(map_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    effective_sim_demand_percent = 100.0 * np.divide(
        sim["controls.motor_torque_request_nm"],
        sim["motor.peak_torque_limit_nm"],
        out=np.zeros_like(sim["controls.motor_torque_request_nm"]),
        where=sim["motor.peak_torque_limit_nm"] > 0.0,
    )
    figure, axes = plt.subplots(3, 2, figsize=(15.5, 12.5), sharex=True)
    axes = axes.ravel()
    axes[0].plot(real_distance_m, real["gps_speed_mps"], color=REAL_COLOR, label="Recorded")
    axes[0].plot(sim_distance_m, sim["vehicle.speed_mps"], color=SIM_COLOR, label="Simulation")
    style_signal_axis(axes[0], "Speed (m/s)")

    axes[1].plot(real_distance_m, real["motor_rpm"], color=REAL_COLOR, label="Recorded")
    axes[1].plot(sim_distance_m, sim["motor.speed_rpm"], color=SIM_COLOR, label="Simulation")
    style_signal_axis(axes[1], "Motor speed (rpm)")

    axes[2].plot(real_distance_m, real["torque_command_nm"], color=REAL_COLOR, label="Recorded")
    axes[2].plot(
        sim_distance_m,
        sim["controls.motor_torque_request_nm"],
        color=SIM_COLOR,
        label="Simulation",
    )
    style_signal_axis(axes[2], "Torque command (Nm)")

    axes[3].plot(real_distance_m, real["torque_feedback_nm"], color=REAL_COLOR, label="Recorded")
    axes[3].plot(sim_distance_m, sim["motor.torque_nm"], color=SIM_COLOR, label="Simulation")
    axes[3].axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8)
    style_signal_axis(axes[3], "Motor torque (Nm)")

    axes[4].plot(real_distance_m, real["apps_percent"], color=REAL_COLOR, label="Recorded APPS")
    axes[4].plot(
        sim_distance_m,
        effective_sim_demand_percent,
        color=SIM_COLOR,
        label="Sim effective request",
    )
    axes[4].axhline(
        100.0 * args.torque_fraction,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.0,
        label="Sim base request",
    )
    style_signal_axis(axes[4], "Driver demand (%)")

    axes[5].plot(real_distance_m, real["brake_pressure_psi"], color=REAL_COLOR, label="Recorded")
    axes[5].plot(
        sim_distance_m,
        sim["controls.front_brake_pressure_psi"],
        color=SIM_COLOR,
        label="Sim front",
    )
    axes[5].plot(
        sim_distance_m,
        sim["controls.rear_brake_pressure_psi"],
        color=SIM_ALT_COLOR,
        linestyle="--",
        label="Sim rear",
    )
    axes[5].axhline(
        args.maximum_brake_pressure_psi,
        color=REFERENCE_COLOR,
        linestyle=":",
        linewidth=1.0,
        label="300 psi cap",
    )
    style_signal_axis(axes[5], "Brake pressure (psi)")

    for axis in axes:
        axis.legend(loc="best", frameon=False, ncols=2)
    axes[-2].set_xlabel("Lap distance (m)")
    axes[-1].set_xlabel("Lap distance (m)")
    figure.suptitle(
        "First endurance lap: mechanical drivetrain and driver controls",
        fontsize=16,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    mechanical_path = args.output_dir / "first_lap_mechanical_comparison.png"
    figure.savefig(mechanical_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 9.0), sharex=True)
    axes = axes.ravel()
    axes[0].plot(real_distance_m, real["battery_power_kw"], color=REAL_COLOR, label="Recorded")
    axes[0].plot(sim_distance_m, sim["battery.power_w"] / 1_000.0, color=SIM_COLOR, label="Simulation")
    axes[0].axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8)
    style_signal_axis(axes[0], "Pack power (kW)")

    axes[1].plot(real_distance_m, real["battery_voltage_v"], color=REAL_COLOR, label="Recorded")
    axes[1].plot(sim_distance_m, sim["battery.terminal_voltage_v"], color=SIM_COLOR, label="Simulation")
    style_signal_axis(axes[1], "Pack voltage (V)")

    axes[2].plot(real_distance_m, real["battery_current_a"], color=REAL_COLOR, label="Recorded")
    axes[2].plot(sim_distance_m, sim["battery.current_a"], color=SIM_COLOR, label="Simulation")
    axes[2].axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8)
    style_signal_axis(axes[2], "Pack current (A)")

    axes[3].plot(real_distance_m, real["battery_soc_percent"], color=REAL_COLOR, label="Recorded")
    axes[3].plot(
        sim_distance_m,
        100.0 * sim["battery.state_of_charge"],
        color=SIM_COLOR,
        label="Simulation",
    )
    style_signal_axis(axes[3], "State of charge (%)")

    for axis in axes:
        axis.legend(loc="best", frameon=False)
    axes[-2].set_xlabel("Lap distance (m)")
    axes[-1].set_xlabel("Lap distance (m)")
    figure.suptitle(
        "First endurance lap: electrical drivetrain signals",
        fontsize=16,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    electrical_path = args.output_dir / "first_lap_electrical_comparison.png"
    figure.savefig(electrical_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"recorded_lap_time_s={real['time_s'][-1]:.9f}")
    print(f"recorded_energy_kwh={real_energy_kwh:.9f}")
    print(f"simulated_lap_time_s={result.lap_times_s[0]:.9f}")
    print(f"simulated_energy_kwh={result.pack_energy_kwh:.9f}")
    print(map_path.resolve())
    print(mechanical_path.resolve())
    print(electrical_path.resolve())


if __name__ == "__main__":
    main()
