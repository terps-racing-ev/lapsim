"""Infer an equivalent brake-pressure-to-total-wheel-torque relationship."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
ANALYSIS = ROOT / "analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.common import corrected_imu_at_lap_times, read_numeric_csv
from vehicle_model.vehicle import Vehicle


def _fit_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / np.dot(x, x))


def _bootstrap_interval(
    pressure_psi: np.ndarray,
    torque_nm: np.ndarray,
    *,
    samples: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(20260810)
    estimates = np.empty(samples)
    for index in range(samples):
        selected = rng.integers(0, len(pressure_psi), len(pressure_psi))
        estimates[index] = _fit_through_origin(
            pressure_psi[selected], torque_nm[selected]
        )
    return tuple(float(value) for value in np.percentile(estimates, (2.5, 97.5)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lap-csv",
        type=Path,
        default=ROOT / "analysis/data/lap/first_lap.csv",
    )
    parser.add_argument(
        "--corrected-imu-csv",
        type=Path,
        default=ROOT / "analysis/data/imu/first_lap_corrected_imu.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/accel/output/brake_fit",
    )
    parser.add_argument("--minimum-pressure-psi", type=float, default=5.0)
    parser.add_argument(
        "--maximum-absolute-lateral-accel-mps2", type=float, default=2.0
    )
    parser.add_argument("--minimum-speed-mps", type=float, default=3.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lap = read_numeric_csv(args.lap_csv)
    corrected = read_numeric_csv(args.corrected_imu_csv)
    imu, imu_columns = corrected_imu_at_lap_times(lap, corrected)
    required = ("gps_speed_mps", "torque_feedback_nm", "brake_pressure_psi")
    missing = [name for name in required if name not in lap]
    if missing:
        raise ValueError(f"Lap CSV is missing: {', '.join(missing)}")

    vehicle = Vehicle()
    speed_mps = lap["gps_speed_mps"]
    motor_torque_nm = lap["torque_feedback_nm"]
    pressure_psi = lap["brake_pressure_psi"]
    acceleration_mps2 = imu["imu_longitudinal_accel_mps2"]
    lateral_acceleration_mps2 = imu["imu_lateral_accel_mps2"]

    dynamic_pressure_pa = 0.5 * vehicle.air_density_kgpm3 * speed_mps**2
    drag_force_n = dynamic_pressure_pa * vehicle.aero.drag_area_m2
    downforce_n = dynamic_pressure_pa * vehicle.aero.downforce_area_m2
    rolling_force_n = vehicle.rolling_resistance_coefficient * (
        vehicle.mass_kg * vehicle.gravity_mps2 + downforce_n
    )
    # Positive motor torque is retained as propulsion. Negative torque is not
    # assigned to friction braking because regen/backdrive efficiency is not
    # identified by the current model.
    positive_drive_force_n = (
        np.maximum(motor_torque_nm, 0.0)
        * vehicle.drivetrain.chain_drive.ratio
        * vehicle.drivetrain.chain_drive.efficiency
        / vehicle.tire.rolling_radius_m
    )
    inferred_brake_force_n = (
        positive_drive_force_n
        - drag_force_n
        - rolling_force_n
        - vehicle.effective_longitudinal_mass_kg * acceleration_mps2
    )
    inferred_total_wheel_brake_torque_nm = (
        inferred_brake_force_n * vehicle.tire.rolling_radius_m
    )
    selected = (
        (pressure_psi >= args.minimum_pressure_psi)
        & (
            np.abs(lateral_acceleration_mps2)
            <= args.maximum_absolute_lateral_accel_mps2
        )
        & (speed_mps >= args.minimum_speed_mps)
        & (inferred_total_wheel_brake_torque_nm > 0.0)
    )
    if np.count_nonzero(selected) < 3:
        raise ValueError("Fewer than three samples satisfy the brake-fit filters")

    fit_pressure = pressure_psi[selected]
    fit_torque = inferred_total_wheel_brake_torque_nm[selected]
    torque_per_psi_nm = _fit_through_origin(fit_pressure, fit_torque)
    force_per_psi_n = torque_per_psi_nm / vehicle.tire.rolling_radius_m
    predicted_torque_nm = torque_per_psi_nm * fit_pressure
    residual_torque_nm = predicted_torque_nm - fit_torque
    rmse_torque_nm = float(np.sqrt(np.mean(residual_torque_nm**2)))
    correlation = float(np.corrcoef(fit_pressure, fit_torque)[0, 1])
    ci_lower, ci_upper = _bootstrap_interval(fit_pressure, fit_torque)

    figure, axis = plt.subplots(figsize=(10.5, 6.5), layout="constrained")
    points = axis.scatter(
        fit_pressure,
        fit_torque,
        c=speed_mps[selected],
        cmap="viridis",
        s=72,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
        label="Selected low-lateral samples",
    )
    pressure_grid = np.linspace(0.0, float(np.max(fit_pressure)) * 1.06, 200)
    axis.plot(
        pressure_grid,
        torque_per_psi_nm * pressure_grid,
        color="#E45756",
        lw=2.5,
        label=f"Origin fit: {torque_per_psi_nm:.3f} N·m/psi",
    )
    axis.set(
        title="Brake pressure versus inferred total wheel-brake torque",
        xlabel="Recorded brake pressure [psi]",
        ylabel="Equivalent total wheel-brake torque [N·m]",
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    colorbar = figure.colorbar(points, ax=axis, pad=0.02)
    colorbar.set_label("GNSS speed [m/s]")
    figure.savefig(args.output_dir / "brake_pressure_to_wheel_torque.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    axes[0].scatter(
        fit_pressure,
        fit_torque / fit_pressure,
        c=np.abs(lateral_acceleration_mps2[selected]),
        cmap="plasma",
        s=65,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].axhline(torque_per_psi_nm, color="#E45756", lw=2.3, label="Aggregate fit")
    axes[0].set(
        xlabel="Brake pressure [psi]",
        ylabel="Pointwise torque/pressure [N·m/psi]",
        title="Pointwise conversion spread",
    )
    axes[0].legend(frameon=False)
    axes[1].scatter(
        fit_pressure,
        residual_torque_nm,
        c=speed_mps[selected],
        cmap="viridis",
        s=65,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[1].axhline(0.0, color="#222222", lw=1.0)
    axes[1].set(
        xlabel="Brake pressure [psi]",
        ylabel="Predicted − inferred torque [N·m]",
        title="Linear-fit residual",
    )
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle("Brake pressure calibration diagnostics", fontsize=16)
    figure.savefig(args.output_dir / "brake_pressure_fit_diagnostics.png", dpi=220)
    plt.close(figure)

    report = {
        "inputs": {
            "lap_csv": str(args.lap_csv.resolve()),
            "corrected_imu_csv": str(args.corrected_imu_csv.resolve()),
            "corrected_imu_columns": imu_columns,
        },
        "selection": {
            "minimum_pressure_psi": args.minimum_pressure_psi,
            "maximum_absolute_lateral_acceleration_mps2": args.maximum_absolute_lateral_accel_mps2,
            "minimum_speed_mps": args.minimum_speed_mps,
            "selected_samples": int(np.count_nonzero(selected)),
            "available_pressure_active_samples": int(
                np.count_nonzero(pressure_psi >= args.minimum_pressure_psi)
            ),
        },
        "force_balance": "F_brake = F_positive_drive - F_drag - F_rolling - m_effective*a_IMU; T_brake_total = F_brake*tire_radius",
        "result": {
            "total_wheel_brake_torque_per_psi_nm": torque_per_psi_nm,
            "equivalent_vehicle_brake_force_per_psi_n": force_per_psi_n,
            "bootstrap_95pct_torque_per_psi_nm": [ci_lower, ci_upper],
            "torque_rmse_nm": rmse_torque_nm,
            "pressure_torque_correlation": correlation,
        },
        "vehicle_parameters": {
            "motor_to_wheel_efficiency": vehicle.drivetrain.chain_drive.efficiency,
            "tire_radius_m": vehicle.tire.rolling_radius_m,
            "effective_longitudinal_mass_kg": vehicle.effective_longitudinal_mass_kg,
            "drag_coefficient": vehicle.aero.drag_coefficient,
            "lift_coefficient": vehicle.aero.lift_coefficient,
        },
        "limitations": [
            "The logged pressure channel is treated as a vehicle-level brake command even though available documentation identifies it as rear-brake pressure.",
            "Negative motor torque is ignored; the current model cannot separate regenerative/backdrive braking from friction braking.",
            "No pressure or IMU latency is fitted, and only 16 low-lateral samples are retained.",
            "Road grade, wind, tire-radius error, IMU bias, and rotating-inertia error are absorbed into the inferred relationship.",
            "The result is an equivalent total-wheel calibration for replay, not a caliper or axle hardware constant.",
        ],
    }
    (args.output_dir / "brake_pressure_fit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Fitted {torque_per_psi_nm:.4f} N·m/psi total wheel torque "
        f"({force_per_psi_n:.4f} N/psi), 95% bootstrap "
        f"{ci_lower:.4f}–{ci_upper:.4f} N·m/psi, "
        f"from {np.count_nonzero(selected)} samples"
    )


if __name__ == "__main__":
    main()
