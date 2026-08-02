"""Generate the full-throttle 75 m acceleration telemetry figure."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lapsim.controls import Controls  # noqa: E402
from lapsim.utils.units import meters_per_second_to_miles_per_hour  # noqa: E402
from lapsim.vehicle import Vehicle  # noqa: E402


ROLLOUT_M = 0.3
TIMED_DISTANCE_M = 75.0
TIMESTEP_S = 0.001
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "acceleration_75m_ryder_torque_100mph_rollout_telemetry.png"
)


def crossing_time_s(
    distance_m: np.ndarray,
    time_s: np.ndarray,
    target_distance_m: float,
) -> float:
    """Linearly interpolate the time at a requested distance."""

    after_index = int(np.searchsorted(distance_m, target_distance_m))
    before_index = after_index - 1
    distance_fraction = (
        (target_distance_m - distance_m[before_index])
        / (distance_m[after_index] - distance_m[before_index])
    )
    return float(
        time_s[before_index]
        + distance_fraction * (time_s[after_index] - time_s[before_index])
    )


def main() -> None:
    vehicle = Vehicle()
    controls = Controls(
        motor_torque_request_nm=vehicle.drivetrain.max_motor_torque_nm
    )

    samples: list[tuple[float, float, float, float, float]] = [
        (vehicle.time_s, vehicle.distance_m, vehicle.speed_mps, 0.0, 0.0)
    ]
    finish_distance_m = ROLLOUT_M + TIMED_DISTANCE_M
    while vehicle.distance_m < finish_distance_m:
        vehicle.update_state(controls, TIMESTEP_S)
        samples.append(
            (
                vehicle.time_s,
                vehicle.distance_m,
                vehicle.speed_mps,
                vehicle.drivetrain.current_motor_torque_nm,
                vehicle.battery.current_power_w,
            )
        )

    sample_array = np.asarray(samples)
    time_s = sample_array[:, 0]
    launch_distance_m = sample_array[:, 1]
    timed_distance_m = launch_distance_m - ROLLOUT_M
    speed_mph = np.asarray(
        [meters_per_second_to_miles_per_hour(value) for value in sample_array[:, 2]]
    )
    motor_torque_nm = sample_array[:, 3]
    battery_power_kw = sample_array[:, 4] / 1_000.0
    acceleration_mps2 = np.gradient(sample_array[:, 2], time_s)
    energy_wh = np.cumsum(sample_array[:, 4] * TIMESTEP_S) / 3_600.0

    timer_start_s = crossing_time_s(launch_distance_m, time_s, ROLLOUT_M)
    finish_s = crossing_time_s(launch_distance_m, time_s, finish_distance_m)
    timed_time_s = time_s - timer_start_s
    official_time_s = finish_s - timer_start_s

    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(14, 11), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)
    distance_axis = figure.add_subplot(grid[0, 0])
    time_axis = figure.add_subplot(grid[0, 1])
    torque_axis = figure.add_subplot(grid[1, 0])
    acceleration_axis = figure.add_subplot(grid[1, 1])
    power_axis = figure.add_subplot(grid[2, :])

    speed_color = "#0072B2"
    torque_color = "#009E73"
    acceleration_color = "#D55E00"
    power_color = "#CC79A7"
    energy_color = "#E69F00"
    rollout_color = "#6F4E9C"

    distance_axis.plot(timed_distance_m, speed_mph, color=speed_color, linewidth=2.2)
    distance_axis.axvspan(-ROLLOUT_M, 0.0, color=rollout_color, alpha=0.13)
    distance_axis.axvline(0.0, color=rollout_color, linewidth=1.2)
    distance_axis.set(
        title="Vehicle velocity vs distance",
        xlabel="Timed-course distance (m)",
        ylabel="Velocity (mph)",
        xlim=(-ROLLOUT_M, TIMED_DISTANCE_M),
    )

    time_axis.plot(timed_time_s, speed_mph, color=speed_color, linewidth=2.2)
    time_axis.axvspan(timed_time_s[0], 0.0, color=rollout_color, alpha=0.13)
    time_axis.axvline(0.0, color=rollout_color, linewidth=1.2)
    time_axis.set(
        title="Vehicle velocity vs time",
        xlabel="Time from timing beam (s)",
        ylabel="Velocity (mph)",
        xlim=(timed_time_s[0], official_time_s),
    )
    time_axis.annotate(
        f"75 m: {official_time_s:.3f} s",
        xy=(official_time_s, speed_mph[-1]),
        xytext=(-82, -26),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "0.35"},
    )

    torque_axis.plot(timed_distance_m, motor_torque_nm, color=torque_color, linewidth=2.0)
    torque_axis.axvspan(-ROLLOUT_M, 0.0, color=rollout_color, alpha=0.13)
    torque_axis.axvline(0.0, color=rollout_color, linewidth=1.2)
    torque_axis.set(
        title="Delivered motor torque",
        xlabel="Timed-course distance (m)",
        ylabel="Torque (N m)",
        xlim=(-ROLLOUT_M, TIMED_DISTANCE_M),
    )

    acceleration_axis.plot(
        timed_distance_m,
        acceleration_mps2,
        color=acceleration_color,
        linewidth=2.0,
    )
    acceleration_axis.axvspan(-ROLLOUT_M, 0.0, color=rollout_color, alpha=0.13)
    acceleration_axis.axvline(0.0, color=rollout_color, linewidth=1.2)
    acceleration_axis.set(
        title="Longitudinal acceleration",
        xlabel="Timed-course distance (m)",
        ylabel="Acceleration (m/s²)",
        xlim=(-ROLLOUT_M, TIMED_DISTANCE_M),
    )

    power_axis.plot(
        timed_distance_m,
        battery_power_kw,
        color=power_color,
        linewidth=2.0,
        label="Battery power",
    )
    power_axis.axvspan(-ROLLOUT_M, 0.0, color=rollout_color, alpha=0.13)
    power_axis.axvline(0.0, color=rollout_color, linewidth=1.2)
    power_axis.set(
        title="Electrical power and cumulative energy",
        xlabel="Timed-course distance (m)",
        ylabel="Battery power (kW)",
        xlim=(-ROLLOUT_M, TIMED_DISTANCE_M),
    )
    energy_axis = power_axis.twinx()
    energy_axis.plot(
        timed_distance_m,
        energy_wh,
        color=energy_color,
        linewidth=2.0,
        label="Cumulative energy",
    )
    energy_axis.set_ylabel("Energy from launch (Wh)")
    handles_left, labels_left = power_axis.get_legend_handles_labels()
    handles_right, labels_right = energy_axis.get_legend_handles_labels()
    power_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="center right",
        frameon=True,
    )

    time_axis.text(
        timed_time_s[0] / 2.0,
        0.97,
        "rollout",
        transform=time_axis.get_xaxis_transform(),
        ha="center",
        va="top",
        color=rollout_color,
        fontsize=9,
    )

    figure.suptitle(
        "75 m Full-Throttle Acceleration — Ryder Torque Curve",
        fontsize=16,
        fontweight="bold",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=180)
    plt.close(figure)

    print(f"Official 75 m time: {official_time_s:.6f} s")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
