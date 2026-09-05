"""Plot drivetrain telemetry from the standard simulated acceleration event."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/events/acceleration/acceleration_telemetry.csv"
DEFAULT_SUMMARY = ROOT / "outputs/events/acceleration/acceleration_summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/events/acceleration"


def read_telemetry(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No telemetry rows found in {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def style_axes(axes: np.ndarray | list[plt.Axes]) -> None:
    for axis in np.asarray(axes).ravel():
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best")


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(figure)


def plot_vehicle_response(
    data: dict[str, np.ndarray], output_dir: Path, acceleration_time_s: float
) -> Path:
    distance = data["vehicle.distance_m"]
    speed_kph = 3.6 * data["vehicle.speed_mps"]
    acceleration_g = data["vehicle.longitudinal_acceleration_mps2"] / 9.80665

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained")
    axes[0].plot(distance, speed_kph, lw=2.2, label="Vehicle speed")
    axes[0].set_ylabel("Speed [km/h]")
    axes[1].plot(distance, acceleration_g, lw=2.2, color="tab:orange", label="Longitudinal acceleration")
    axes[1].set_ylabel("Acceleration [g]")
    axes[1].set_xlabel("Distance [m]")
    style_axes(axes)
    figure.suptitle(
        "Simulated FSAE Acceleration — Vehicle Response "
        f"({acceleration_time_s:.3f} s)"
    )

    path = output_dir / "simulated_acceleration_vehicle_response.png"
    save_figure(figure, path)
    return path


def plot_drivetrain(data: dict[str, np.ndarray], output_dir: Path) -> Path:
    distance = data["vehicle.distance_m"]
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, layout="constrained")

    axes[0].plot(distance, data["controls.motor_torque_request_nm"], lw=1.8, ls="--", label="Requested")
    axes[0].plot(distance, data["motor.torque_nm"], lw=2.2, label="Delivered")
    axes[0].plot(distance, data["motor.peak_torque_limit_nm"], lw=1.7, ls=":", label="Peak limit")
    axes[0].plot(distance, data["motor.continuous_torque_limit_nm"], lw=1.7, ls="-.", label="Continuous limit")
    axes[0].set_ylabel("Motor torque [N·m]")

    axes[1].plot(distance, data["vehicle.requested_drive_force_n"], lw=1.8, ls="--", label="Requested drive force")
    axes[1].plot(distance, data["vehicle.drive_force_n"], lw=2.2, label="Delivered drive force")
    axes[1].plot(distance, data["vehicle.rear_drive_capacity_n"], lw=1.7, ls=":", label="Rear traction capacity")
    axes[1].set_ylabel("Force [N]")
    axes[1].set_xlabel("Distance [m]")
    style_axes(axes)
    figure.suptitle("Simulated FSAE Acceleration — Drivetrain")

    path = output_dir / "simulated_acceleration_drivetrain.png"
    save_figure(figure, path)
    return path


def plot_electrical(data: dict[str, np.ndarray], output_dir: Path) -> Path:
    distance = data["vehicle.distance_m"]
    figure, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True, layout="constrained")

    axes[0].plot(distance, data["battery.power_w"] / 1000.0, lw=2.2, label="Battery DC input")
    axes[0].plot(distance, data["inverter.motor_output_power_w"] / 1000.0, lw=1.9, label="Inverter motor output")
    axes[0].plot(distance, data["motor.mechanical_power_w"] / 1000.0, lw=1.9, label="Motor mechanical output")
    axes[0].set_ylabel("Power [kW]")

    axes[1].plot(distance, data["battery.current_a"], lw=2.2, color="tab:red", label="Pack current")
    axes[1].set_ylabel("Current [A]")

    axes[2].plot(distance, data["battery.terminal_voltage_v"], lw=2.2, color="tab:green", label="Terminal voltage")
    axes[2].set_ylabel("Voltage [V]")
    axes[2].set_xlabel("Distance [m]")
    style_axes(axes)
    figure.suptitle("Simulated FSAE Acceleration — Electrical Powertrain")

    path = output_dir / "simulated_acceleration_electrical.png"
    save_figure(figure, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    telemetry = read_telemetry(args.input)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    acceleration_time_s = float(summary["scoring_time_s"])
    paths = (
        plot_vehicle_response(telemetry, args.output_dir, acceleration_time_s),
        plot_drivetrain(telemetry, args.output_dir),
        plot_electrical(telemetry, args.output_dir),
    )
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
