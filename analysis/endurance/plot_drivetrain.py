"""Create a distance-domain drivetrain comparison from full-lap telemetry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def cumulative_energy_kwh(
    distance_m: np.ndarray, power_kw: np.ndarray, speed_mps: np.ndarray
) -> np.ndarray:
    ds_m = np.diff(distance_m)
    cell_power_kw = 0.5 * (power_kw[:-1] + power_kw[1:])
    cell_speed_mps = np.maximum(0.5 * (speed_mps[:-1] + speed_mps[1:]), 0.1)
    cell_energy_kwh = cell_power_kw * ds_m / cell_speed_mps / 3600.0
    return np.r_[0.0, np.cumsum(cell_energy_kwh)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = read_csv(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distance_m = data["measured_distance_m"]
    recorded_soc_percent = data["recorded_battery_soc_percent"]
    simulated_soc_percent = 100.0 * data["sim_battery_state_of_charge"]
    recorded_energy_kwh = cumulative_energy_kwh(
        distance_m,
        data["recorded_battery_power_kw"],
        data["analysis_shifted_gnss_speed_mps"],
    )
    simulated_energy_kwh = cumulative_energy_kwh(
        distance_m,
        data["sim_battery_power_w"] / 1000.0,
        data["sim_vehicle_speed_mps"],
    )

    figure, axes = plt.subplots(
        4, 2, figsize=(16, 15), sharex=True, layout="constrained"
    )
    axes = axes.ravel()
    panels = (
        (
            data["analysis_shifted_gnss_speed_mps"],
            data["sim_vehicle_speed_mps"],
            "Vehicle speed [m/s]",
            "Recorded GNSS",
            "Simulation",
        ),
        (
            data["recorded_motor_rpm"],
            data["sim_motor_speed_rpm"],
            "Motor speed [rpm]",
            "Recorded inverter",
            "Simulation",
        ),
        (
            data["recorded_battery_power_kw"],
            data["sim_battery_power_w"] / 1000.0,
            "Pack power [kW]",
            "Recorded HVC",
            "Simulation",
        ),
        (
            data["recorded_battery_current_a"],
            data["sim_battery_current_a"],
            "Pack current [A]",
            "Recorded HVC",
            "Simulation",
        ),
        (
            data["recorded_battery_voltage_v"],
            data["sim_battery_terminal_voltage_v"],
            "Pack voltage [V]",
            "Recorded HVC",
            "Simulation",
        ),
        (
            recorded_soc_percent,
            simulated_soc_percent,
            "Pack SOC [%]",
            "Recorded HVC",
            "Simulation",
        ),
    )
    for axis, (recorded, simulated, ylabel, recorded_label, simulated_label) in zip(
        axes, panels, strict=False
    ):
        axis.plot(distance_m, recorded, lw=1.25, label=recorded_label)
        axis.plot(distance_m, simulated, lw=1.25, label=simulated_label)
        axis.set_ylabel(ylabel)
        axis.legend(loc="best", frameon=False, ncol=2)
    if "sim_path_speed_ceiling_mps" in data and np.any(
        np.isfinite(data["sim_path_speed_ceiling_mps"])
    ):
        axes[0].plot(
            distance_m,
            data["sim_path_speed_ceiling_mps"],
            color="black",
            lw=1.0,
            ls="--",
            alpha=0.75,
            label="Hard path ceiling",
        )
        axes[0].legend(loc="best", frameon=False, ncol=3)

    torque_axis = axes[6]
    torque_axis.plot(
        distance_m,
        data["recorded_torque_command_nm"],
        lw=1.0,
        label="Recorded command",
    )
    torque_axis.plot(
        distance_m,
        data["recorded_torque_feedback_nm"],
        lw=1.15,
        label="Recorded feedback",
    )
    torque_axis.plot(
        distance_m,
        data["sim_motor_torque_nm"],
        lw=1.15,
        label="Sim delivered",
    )
    torque_axis.axhline(0.0, color="0.35", lw=0.7)
    torque_axis.set_ylabel("Motor torque [N m]")
    torque_axis.legend(loc="best", frameon=False, ncol=3)

    limit_axis = axes[7]
    limit_names = (
        ("sim_limits_traction_active", "Traction", 0.0),
        ("sim_limits_lateral_saturated", "Lateral saturation", 1.2),
        ("sim_limits_motor_envelope_active", "Motor envelope", 2.4),
        ("sim_limits_brake_grip_active", "Brake grip", 3.6),
    )
    for name, label, offset in limit_names:
        limit_axis.fill_between(
            distance_m,
            offset,
            offset + 0.8 * (data[name] > 0.5),
            step="post",
            alpha=0.75,
            label=label,
        )
    if "sim_path_torque_limited" in data:
        offset = 4.8
        limit_axis.fill_between(
            distance_m,
            offset,
            offset + 0.8 * (data["sim_path_torque_limited"] > 0.5),
            step="post",
            alpha=0.75,
            label="Path controller",
        )
    limit_axis.set_yticks([])
    limit_axis.set_ylabel("Active limits")
    limit_axis.legend(loc="best", frameon=False, ncol=2)

    for axis in axes:
        axis.grid(alpha=0.23)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(float(distance_m[0]), float(distance_m[-1]))
    axes[-2].set_xlabel("Lap distance [m]")
    axes[-1].set_xlabel("Lap distance [m]")
    figure.suptitle("Endurance full lap: recorded vs simulated drivetrain signals")
    figure.savefig(args.output_dir / "drivetrain_signals_vs_distance.png", dpi=190)
    plt.close(figure)

    if all(
        name in data
        for name in (
            "sim_brakes_front_force_request_n",
            "sim_brakes_rear_force_request_n",
            "sim_brakes_front_friction_force_n",
            "sim_brakes_rear_friction_force_n",
            "sim_suspension_front_axle_normal_load_n",
            "sim_suspension_rear_axle_normal_load_n",
        )
    ):
        has_braking_slip = all(
            name in data
            for name in (
                "sim_tire_front_left_slip_percent",
                "sim_tire_front_right_slip_percent",
                "sim_tire_rear_left_slip_percent",
                "sim_tire_rear_right_slip_percent",
            )
        )
        brake_figure, brake_axes = plt.subplots(
            4 if has_braking_slip else 3,
            1,
            figsize=(14, 12 if has_braking_slip else 10),
            sharex=True,
            layout="constrained",
        )
        brake_axes[0].plot(
            distance_m,
            data["recorded_brake_pressure_psi"],
            color="#4C78A8",
            lw=1.2,
            label="Recorded line pressure",
        )
        brake_axes[0].set_ylabel("Pressure [psi]")
        brake_axes[1].plot(
            distance_m,
            data["sim_brakes_front_force_request_n"],
            color="#E45756",
            lw=1.1,
            ls="--",
            label="Front requested",
        )
        brake_axes[1].plot(
            distance_m,
            data["sim_brakes_front_friction_force_n"],
            color="#E45756",
            lw=1.5,
            label="Front applied",
        )
        brake_axes[1].plot(
            distance_m,
            data["sim_brakes_rear_force_request_n"],
            color="#54A24B",
            lw=1.1,
            ls="--",
            label="Rear requested",
        )
        brake_axes[1].plot(
            distance_m,
            data["sim_brakes_rear_friction_force_n"],
            color="#54A24B",
            lw=1.5,
            label="Rear applied",
        )
        brake_axes[1].set_ylabel("Brake force [N]")
        load_axis = brake_axes[-1]
        if has_braking_slip:
            brake_axes[2].plot(
                distance_m,
                -0.5
                * (
                    data["sim_tire_front_left_slip_percent"]
                    + data["sim_tire_front_right_slip_percent"]
                ),
                color="#E45756",
                lw=1.3,
                label="Front braking slip",
            )
            brake_axes[2].plot(
                distance_m,
                -0.5
                * (
                    data["sim_tire_rear_left_slip_percent"]
                    + data["sim_tire_rear_right_slip_percent"]
                ),
                color="#54A24B",
                lw=1.3,
                label="Rear braking slip",
            )
            brake_axes[2].set_ylabel("Braking slip [%]")
        load_axis.plot(
            distance_m,
            data["sim_suspension_front_axle_normal_load_n"],
            color="#E45756",
            lw=1.3,
            label="Front axle normal load",
        )
        load_axis.plot(
            distance_m,
            data["sim_suspension_rear_axle_normal_load_n"],
            color="#54A24B",
            lw=1.3,
            label="Rear axle normal load",
        )
        load_axis.set_ylabel("Normal load [N]")
        load_axis.set_xlabel("Lap distance [m]")
        for axis in brake_axes:
            axis.grid(alpha=0.23)
            axis.legend(loc="best", frameon=False, ncol=2)
            axis.spines[["top", "right"]].set_visible(False)
        brake_figure.suptitle(
            "Independent front/rear braking: requests, applied forces, and load transfer"
        )
        brake_figure.savefig(
            args.output_dir / "front_rear_braking_vs_distance.png", dpi=190
        )
        plt.close(brake_figure)

    energy_figure, energy_axis = plt.subplots(figsize=(14, 5.5), layout="constrained")
    energy_axis.plot(
        distance_m, recorded_energy_kwh, lw=2.0, label="Recorded cumulative energy"
    )
    energy_axis.plot(
        distance_m, simulated_energy_kwh, lw=2.0, label="Sim cumulative energy"
    )
    energy_axis.set(xlabel="Lap distance [m]", ylabel="Cumulative pack energy [kWh]")
    energy_axis.grid(alpha=0.25)
    energy_axis.spines[["top", "right"]].set_visible(False)
    energy_axis.legend(frameon=False)
    energy_figure.suptitle("Endurance full lap: pack energy versus distance")
    energy_figure.savefig(args.output_dir / "pack_energy_vs_distance.png", dpi=190)
    plt.close(energy_figure)

    longitudinal_acceleration_error_mps2 = (
        data["sim_vehicle_longitudinal_acceleration_mps2"]
        - data["aligned_imu_longitudinal_accel_mps2"]
    )
    lateral_acceleration_error_mps2 = (
        data["sim_vehicle_lateral_acceleration_mps2"]
        - data["aligned_imu_lateral_accel_mps2"]
    )
    acceleration_figure, acceleration_axes = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, layout="constrained"
    )
    acceleration_panels = (
        (
            data["aligned_imu_longitudinal_accel_mps2"],
            data["sim_vehicle_longitudinal_acceleration_mps2"],
            "X / longitudinal acceleration [m/s²]",
        ),
        (
            data["aligned_imu_lateral_accel_mps2"],
            data["sim_vehicle_lateral_acceleration_mps2"],
            "Y / lateral acceleration [m/s²]",
        ),
    )
    for axis, (imu_acceleration, simulated_acceleration, ylabel) in zip(
        acceleration_axes, acceleration_panels, strict=True
    ):
        axis.plot(distance_m, imu_acceleration, lw=1.25, label="Corrected IMU")
        axis.plot(
            distance_m,
            simulated_acceleration,
            lw=1.25,
            label="Simulation expected",
        )
        axis.axhline(0.0, color="0.35", lw=0.7)
        axis.set_ylabel(ylabel)
        axis.legend(loc="best", frameon=False, ncol=2)
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(float(distance_m[0]), float(distance_m[-1]))
    acceleration_axes[-1].set_xlabel("Lap distance [m]")
    acceleration_figure.suptitle(
        "Endurance full lap: corrected IMU vs expected acceleration"
    )
    acceleration_figure.savefig(
        args.output_dir / "acceleration_xy_vs_distance.png", dpi=190
    )
    plt.close(acceleration_figure)

    slip_figure, slip_axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, layout="constrained"
    )
    slip_axes[0].plot(
        distance_m,
        data["sim_tire_driven_slip_percent"],
        color="#E45756",
        lw=1.25,
        label="Simulated driven-wheel slip",
    )
    slip_axes[0].set_ylabel("Longitudinal slip [%]")
    slip_axes[0].legend(loc="best", frameon=False)
    slip_axes[1].plot(
        distance_m,
        data["sim_tire_vehicle_speed_mps"],
        lw=1.25,
        label="Vehicle speed",
    )
    slip_axes[1].plot(
        distance_m,
        data["sim_tire_driven_wheel_surface_speed_mps"],
        lw=1.25,
        label="Driven-wheel surface speed",
    )
    slip_axes[1].set(
        xlabel="Lap distance [m]",
        ylabel="Speed [m/s]",
    )
    slip_axes[1].legend(loc="best", frameon=False, ncol=2)
    for axis in slip_axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(float(distance_m[0]), float(distance_m[-1]))
    slip_figure.suptitle("Endurance full lap: per-tire longitudinal slip model")
    slip_figure.savefig(args.output_dir / "wheel_slip_vs_distance.png", dpi=190)
    plt.close(slip_figure)

    selected = {
        "distance_m": distance_m,
        "recorded_speed_mps": data["analysis_shifted_gnss_speed_mps"],
        "simulated_speed_mps": data["sim_vehicle_speed_mps"],
        "corrected_imu_longitudinal_acceleration_mps2": data[
            "aligned_imu_longitudinal_accel_mps2"
        ],
        "simulated_longitudinal_acceleration_mps2": data[
            "sim_vehicle_longitudinal_acceleration_mps2"
        ],
        "sim_minus_imu_longitudinal_acceleration_mps2": longitudinal_acceleration_error_mps2,
        "corrected_imu_lateral_acceleration_mps2": data[
            "aligned_imu_lateral_accel_mps2"
        ],
        "simulated_lateral_acceleration_mps2": data[
            "sim_vehicle_lateral_acceleration_mps2"
        ],
        "sim_minus_imu_lateral_acceleration_mps2": lateral_acceleration_error_mps2,
        "recorded_motor_rpm": data["recorded_motor_rpm"],
        "simulated_motor_rpm": data["sim_motor_speed_rpm"],
        "recorded_torque_command_nm": data["recorded_torque_command_nm"],
        "recorded_torque_feedback_nm": data["recorded_torque_feedback_nm"],
        "simulated_motor_torque_nm": data["sim_motor_torque_nm"],
        "recorded_pack_power_kw": data["recorded_battery_power_kw"],
        "simulated_pack_power_kw": data["sim_battery_power_w"] / 1000.0,
        "recorded_pack_voltage_v": data["recorded_battery_voltage_v"],
        "simulated_pack_voltage_v": data["sim_battery_terminal_voltage_v"],
        "recorded_pack_current_a": data["recorded_battery_current_a"],
        "simulated_pack_current_a": data["sim_battery_current_a"],
        "recorded_soc_percent": recorded_soc_percent,
        "simulated_soc_percent": simulated_soc_percent,
        "recorded_cumulative_energy_kwh": recorded_energy_kwh,
        "simulated_cumulative_energy_kwh": simulated_energy_kwh,
        "simulated_wheel_slip_percent": data["sim_tire_driven_slip_percent"],
        "simulated_wheel_slip_speed_mps": data["sim_tire_driven_slip_speed_mps"],
        "simulated_wheel_surface_speed_mps": data[
            "sim_tire_driven_wheel_surface_speed_mps"
        ],
        **{name: data[name] for name, _, _ in limit_names},
    }
    with (args.output_dir / "drivetrain_signals_vs_distance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(selected)
        writer.writerows(zip(*selected.values(), strict=True))

    metrics = {
        "independent_coordinate": "lap distance_m",
        "lap_distance_m": float(distance_m[-1] - distance_m[0]),
        "scenario": {
            "drag_coefficient": float(data["sim_aero_drag_coefficient"][0]),
            "motor_to_wheel_efficiency": float(data["sim_chain_drive_efficiency"][0]),
        },
        "rmse": {
            "speed_mps": rms(
                data["sim_vehicle_speed_mps"] - data["analysis_shifted_gnss_speed_mps"]
            ),
            "longitudinal_acceleration_vs_corrected_imu_mps2": rms(
                longitudinal_acceleration_error_mps2
            ),
            "lateral_acceleration_vs_corrected_imu_mps2": rms(
                lateral_acceleration_error_mps2
            ),
            "motor_speed_rpm": rms(
                data["sim_motor_speed_rpm"] - data["recorded_motor_rpm"]
            ),
            "motor_torque_vs_feedback_nm": rms(
                data["sim_motor_torque_nm"] - data["recorded_torque_feedback_nm"]
            ),
            "pack_power_kw": rms(
                data["sim_battery_power_w"] / 1000.0 - data["recorded_battery_power_kw"]
            ),
            "pack_voltage_v": rms(
                data["sim_battery_terminal_voltage_v"]
                - data["recorded_battery_voltage_v"]
            ),
            "pack_current_a": rms(
                data["sim_battery_current_a"] - data["recorded_battery_current_a"]
            ),
        },
        "recorded": {
            "peak_motor_rpm": float(np.max(data["recorded_motor_rpm"])),
            "peak_torque_feedback_nm": float(
                np.max(data["recorded_torque_feedback_nm"])
            ),
            "peak_pack_power_kw": float(np.max(data["recorded_battery_power_kw"])),
            "peak_pack_current_a": float(np.max(data["recorded_battery_current_a"])),
            "minimum_pack_voltage_v": float(np.min(data["recorded_battery_voltage_v"])),
            "soc_drop_percentage_points": float(
                recorded_soc_percent[0] - recorded_soc_percent[-1]
            ),
            "net_pack_energy_kwh": float(recorded_energy_kwh[-1]),
        },
        "simulated": {
            "peak_motor_rpm": float(np.max(data["sim_motor_speed_rpm"])),
            "peak_motor_torque_nm": float(np.max(data["sim_motor_torque_nm"])),
            "peak_pack_power_kw": float(np.max(data["sim_battery_power_w"]) / 1000.0),
            "peak_pack_current_a": float(np.max(data["sim_battery_current_a"])),
            "minimum_pack_voltage_v": float(
                np.min(data["sim_battery_terminal_voltage_v"])
            ),
            "soc_drop_percentage_points": float(
                simulated_soc_percent[0] - simulated_soc_percent[-1]
            ),
            "net_pack_energy_kwh": float(simulated_energy_kwh[-1]),
        },
        "active_limit_distance_fraction": {
            label: float(
                np.sum(np.diff(distance_m) * (data[name][:-1] > 0.5))
                / (distance_m[-1] - distance_m[0])
            )
            for name, label, _ in limit_names
        },
        "wheel_slip": {
            "model": "per-tire combined-force utilization model",
            "peak_configured_percent": 10.0,
            "maximum_simulated_percent": float(
                np.max(np.abs(data["sim_tire_driven_slip_percent"]))
            ),
            "mean_simulated_percent": float(
                np.mean(np.abs(data["sim_tire_driven_slip_percent"]))
            ),
            "distance_fraction_above_1_percent": float(
                np.sum(
                    np.diff(distance_m)
                    * (np.abs(data["sim_tire_driven_slip_percent"][:-1]) > 1.0)
                )
                / (distance_m[-1] - distance_m[0])
            ),
        },
    }
    (args.output_dir / "drivetrain_distance_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
