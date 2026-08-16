"""Infer motor-shaft-to-wheel efficiency from straight-line telemetry.

This consumes ``straight_samples.csv`` written by
``analyze_acceleration.py``. It is deliberately separate from the
distance-domain replay: the calibration uses measured acceleration in a direct
force balance rather than fitting the replay's speed trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vehicle_model.vehicle import Vehicle

DEFAULT_MINIMUM_TORQUE_NM = 30.0
DEFAULT_MINIMUM_ACCELERATION_MPS2 = 0.2
DEFAULT_MAXIMUM_BRAKE_PRESSURE_PSI = 1.0


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def _through_origin_fit(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / np.dot(x, x))


def _bootstrap_interval(
    x: np.ndarray,
    y: np.ndarray,
    straight: np.ndarray,
    *,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Stratified sample bootstrap retaining every represented straight."""

    rng = np.random.default_rng(20260810)
    groups = [np.flatnonzero(straight == value) for value in np.unique(straight)]
    estimates = np.empty(samples)
    for sample_index in range(samples):
        selected = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        estimates[sample_index] = _through_origin_fit(x[selected], y[selected])
    lower, upper = np.percentile(estimates, (2.5, 97.5))
    return float(lower), float(upper)


def _save_acceleration_plot(
    output_dir: Path,
    torque_nm: np.ndarray,
    acceleration_mps2: np.ndarray,
    speed_mps: np.ndarray,
    vehicle: Vehicle,
    fitted_efficiency: float,
) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 6.5), layout="constrained")
    points = axis.scatter(
        torque_nm,
        acceleration_mps2,
        c=speed_mps,
        cmap="viridis",
        s=62,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.92,
        label="Selected telemetry",
        zorder=3,
    )
    torque_grid = np.linspace(0.0, max(105.0, float(np.max(torque_nm)) * 1.08), 250)
    reference_speeds = np.quantile(speed_mps, (0.1, 0.5, 0.9))
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for speed, color in zip(reference_speeds, colors, strict=True):
        aero = vehicle.aero.forces_n(float(speed), vehicle.air_density_kgpm3)
        resistance = aero.drag_n + vehicle.rolling_resistance_coefficient * (
            vehicle.mass_kg * vehicle.gravity_mps2 + aero.downforce_n
        )
        wheel_force = (
            torque_grid
            * vehicle.drivetrain.chain_drive.ratio
            * fitted_efficiency
            / vehicle.tire.rolling_radius_m
        )
        predicted_acceleration = (
            wheel_force - resistance
        ) / vehicle.effective_longitudinal_mass_kg
        axis.plot(
            torque_grid,
            predicted_acceleration,
            color=color,
            lw=2.2,
            label=f"Fitted model at {speed:.1f} m/s",
        )
    axis.axhline(0.0, color="#666666", lw=0.8)
    axis.set(
        title="Measured straight-line acceleration versus motor torque",
        xlabel="Motor torque feedback [N·m]",
        ylabel="Corrected longitudinal acceleration [m/s²]",
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncols=2)
    colorbar = fig.colorbar(points, ax=axis, pad=0.02)
    colorbar.set_label("GNSS speed [m/s]")
    fig.savefig(output_dir / "acceleration_vs_motor_torque.png", dpi=220)
    plt.close(fig)


def _save_diagnostics_plot(
    output_dir: Path,
    *,
    ideal_wheel_force_n: np.ndarray,
    required_wheel_force_n: np.ndarray,
    point_efficiency: np.ndarray,
    speed_mps: np.ndarray,
    torque_nm: np.ndarray,
    measured_acceleration_mps2: np.ndarray,
    fitted_acceleration_mps2: np.ndarray,
    default_acceleration_mps2: np.ndarray,
    straight: np.ndarray,
    fitted_efficiency: float,
    default_efficiency: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    palette = plt.get_cmap("tab10")
    for value in np.unique(straight):
        mask = straight == value
        color = palette((int(value) - 1) % 10)
        axes[0, 0].scatter(
            ideal_wheel_force_n[mask],
            required_wheel_force_n[mask],
            s=48,
            color=color,
            alpha=0.82,
            label=f"Straight {int(value)}",
        )
        axes[0, 1].scatter(
            speed_mps[mask], point_efficiency[mask], s=48, color=color, alpha=0.82
        )
    force_grid = np.linspace(0.0, float(np.max(ideal_wheel_force_n)) * 1.07, 100)
    axes[0, 0].plot(
        force_grid,
        fitted_efficiency * force_grid,
        color="#E45756",
        lw=2.4,
        label=f"Fit: {fitted_efficiency:.3f}",
    )
    axes[0, 0].plot(
        force_grid,
        default_efficiency * force_grid,
        "--",
        color="#777777",
        lw=2.0,
        label=f"Current: {default_efficiency:.3f}",
    )
    axes[0, 0].set(
        title="Force-balance fit",
        xlabel="Ideal wheel force at 100% efficiency [N]",
        ylabel="Wheel force required by measured acceleration [N]",
    )
    axes[0, 0].legend(frameon=False, ncols=2)

    axes[0, 1].axhline(
        fitted_efficiency, color="#E45756", lw=2.3, label="Aggregate fit"
    )
    axes[0, 1].axhline(
        default_efficiency, color="#777777", lw=2.0, ls="--", label="Current default"
    )
    axes[0, 1].axhline(1.0, color="#222222", lw=0.9, ls=":")
    axes[0, 1].set(
        title="Pointwise inferred efficiency",
        xlabel="GNSS speed [m/s]",
        ylabel="Motor-to-wheel efficiency",
    )
    axes[0, 1].legend(frameon=False)

    lower = (
        min(
            float(np.min(measured_acceleration_mps2)),
            float(np.min(fitted_acceleration_mps2)),
        )
        - 0.3
    )
    upper = (
        max(
            float(np.max(measured_acceleration_mps2)),
            float(np.max(default_acceleration_mps2)),
        )
        + 0.3
    )
    axes[1, 0].scatter(
        measured_acceleration_mps2,
        default_acceleration_mps2,
        s=52,
        color="#888888",
        alpha=0.72,
        label=f"Current {default_efficiency:.3f}",
    )
    axes[1, 0].scatter(
        measured_acceleration_mps2,
        fitted_acceleration_mps2,
        s=52,
        color="#E45756",
        alpha=0.78,
        label=f"Fitted {fitted_efficiency:.3f}",
    )
    axes[1, 0].plot(
        (lower, upper), (lower, upper), color="#222222", lw=1.2, ls=":", label="1:1"
    )
    axes[1, 0].set(
        xlim=(lower, upper),
        ylim=(lower, upper),
        title="Acceleration prediction",
        xlabel="Measured acceleration [m/s²]",
        ylabel="Predicted acceleration [m/s²]",
    )
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(
        torque_nm,
        fitted_acceleration_mps2 - measured_acceleration_mps2,
        c=speed_mps,
        cmap="viridis",
        s=55,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1, 1].axhline(0.0, color="#222222", lw=1.0)
    axes[1, 1].set(
        title="Fitted-model residual",
        xlabel="Motor torque feedback [N·m]",
        ylabel="Predicted − measured acceleration [m/s²]",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    fig.suptitle("Motor-to-wheel efficiency calibration diagnostics", fontsize=16)
    fig.savefig(output_dir / "motor_to_wheel_efficiency_diagnostics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=ROOT / "analysis/accel/output/straight_samples.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--torque-column", default="torque_feedback_nm")
    parser.add_argument(
        "--minimum-torque-nm", type=float, default=DEFAULT_MINIMUM_TORQUE_NM
    )
    parser.add_argument(
        "--minimum-acceleration-mps2",
        type=float,
        default=DEFAULT_MINIMUM_ACCELERATION_MPS2,
    )
    parser.add_argument(
        "--maximum-brake-pressure-psi",
        type=float,
        default=DEFAULT_MAXIMUM_BRAKE_PRESSURE_PSI,
    )
    args = parser.parse_args()
    if (
        args.minimum_torque_nm <= 0
        or args.minimum_acceleration_mps2 < 0
        or args.maximum_brake_pressure_psi < 0
    ):
        parser.error(
            "selection thresholds must be nonnegative and torque must be positive"
        )

    output_dir = args.output_dir or args.samples_csv.parent / "efficiency_fit"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _read_csv(args.samples_csv)
    required = (
        "straight_number",
        "gnss_speed_mps",
        args.torque_column,
        "imu_longitudinal_accel_mps2",
        "brake_pressure_psi",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Samples CSV is missing: {', '.join(missing)}")

    vehicle = Vehicle()
    speed_mps = data["gnss_speed_mps"]
    torque_nm = data[args.torque_column]
    acceleration_mps2 = data["imu_longitudinal_accel_mps2"]
    brake_pressure_psi = data["brake_pressure_psi"]
    straight = data["straight_number"].astype(int)
    aero = [
        vehicle.aero.forces_n(float(speed), vehicle.air_density_kgpm3)
        for speed in speed_mps
    ]
    drag_force_n = np.asarray([forces.drag_n for forces in aero])
    downforce_n = np.asarray([forces.downforce_n for forces in aero])
    rolling_force_n = vehicle.rolling_resistance_coefficient * (
        vehicle.mass_kg * vehicle.gravity_mps2 + downforce_n
    )
    required_wheel_force_n = (
        vehicle.effective_longitudinal_mass_kg * acceleration_mps2
        + drag_force_n
        + rolling_force_n
    )
    ideal_wheel_force_n = (
        torque_nm * vehicle.drivetrain.chain_drive.ratio / vehicle.tire.rolling_radius_m
    )
    selected = (
        (torque_nm >= args.minimum_torque_nm)
        & (acceleration_mps2 >= args.minimum_acceleration_mps2)
        & (brake_pressure_psi <= args.maximum_brake_pressure_psi)
        & (required_wheel_force_n > 0.0)
    )
    if np.count_nonzero(selected) < 3:
        raise ValueError("Fewer than three samples satisfy the calibration filters")

    x = ideal_wheel_force_n[selected]
    y = required_wheel_force_n[selected]
    fitted_efficiency = _through_origin_fit(x, y)
    if not 0.0 < fitted_efficiency <= 1.0:
        raise ValueError(f"Unphysical fitted efficiency: {fitted_efficiency:.4f}")
    ci_lower, ci_upper = _bootstrap_interval(x, y, straight[selected])
    default_efficiency = vehicle.drivetrain.chain_drive.efficiency
    point_efficiency = y / x
    resistance_n = drag_force_n[selected] + rolling_force_n[selected]
    fitted_acceleration = (
        fitted_efficiency * x - resistance_n
    ) / vehicle.effective_longitudinal_mass_kg
    default_acceleration = (
        default_efficiency * x - resistance_n
    ) / vehicle.effective_longitudinal_mass_kg
    measured_acceleration = acceleration_mps2[selected]
    fitted_rmse = float(
        np.sqrt(np.mean((fitted_acceleration - measured_acceleration) ** 2))
    )
    default_rmse = float(
        np.sqrt(np.mean((default_acceleration - measured_acceleration) ** 2))
    )

    per_straight = []
    for value in np.unique(straight[selected]):
        mask = selected & (straight == value)
        local_x, local_y = ideal_wheel_force_n[mask], required_wheel_force_n[mask]
        per_straight.append(
            {
                "straight_number": int(value),
                "sample_count": int(np.count_nonzero(mask)),
                "through_origin_efficiency": _through_origin_fit(local_x, local_y),
                "median_point_efficiency": float(np.median(local_y / local_x)),
            }
        )

    _save_acceleration_plot(
        output_dir,
        torque_nm[selected],
        measured_acceleration,
        speed_mps[selected],
        vehicle,
        fitted_efficiency,
    )
    _save_diagnostics_plot(
        output_dir,
        ideal_wheel_force_n=x,
        required_wheel_force_n=y,
        point_efficiency=point_efficiency,
        speed_mps=speed_mps[selected],
        torque_nm=torque_nm[selected],
        measured_acceleration_mps2=measured_acceleration,
        fitted_acceleration_mps2=fitted_acceleration,
        default_acceleration_mps2=default_acceleration,
        straight=straight[selected],
        fitted_efficiency=fitted_efficiency,
        default_efficiency=default_efficiency,
    )

    selected_indices = np.flatnonzero(selected)
    with (output_dir / "efficiency_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [
            "source_row",
            "straight_number",
            "speed_mps",
            "motor_torque_nm",
            "measured_acceleration_mps2",
            "drag_force_n",
            "rolling_force_n",
            "required_wheel_force_n",
            "ideal_wheel_force_n",
            "point_motor_to_wheel_efficiency",
            "fitted_acceleration_mps2",
            "default_acceleration_mps2",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for local_index, source_index in enumerate(selected_indices):
            writer.writerow(
                {
                    "source_row": int(source_index),
                    "straight_number": int(straight[source_index]),
                    "speed_mps": speed_mps[source_index],
                    "motor_torque_nm": torque_nm[source_index],
                    "measured_acceleration_mps2": acceleration_mps2[source_index],
                    "drag_force_n": drag_force_n[source_index],
                    "rolling_force_n": rolling_force_n[source_index],
                    "required_wheel_force_n": required_wheel_force_n[source_index],
                    "ideal_wheel_force_n": ideal_wheel_force_n[source_index],
                    "point_motor_to_wheel_efficiency": point_efficiency[local_index],
                    "fitted_acceleration_mps2": fitted_acceleration[local_index],
                    "default_acceleration_mps2": default_acceleration[local_index],
                }
            )

    report = {
        "input": str(args.samples_csv.resolve()),
        "selection": {
            "torque_column": args.torque_column,
            "minimum_torque_nm": args.minimum_torque_nm,
            "minimum_acceleration_mps2": args.minimum_acceleration_mps2,
            "maximum_brake_pressure_psi": args.maximum_brake_pressure_psi,
            "source_samples": int(len(speed_mps)),
            "selected_samples": int(np.count_nonzero(selected)),
            "represented_straights": [
                int(value) for value in np.unique(straight[selected])
            ],
        },
        "aero": {
            "reference_area_m2": vehicle.aero.frontal_area_m2,
            "drag_coefficient": vehicle.aero.drag_coefficient,
            "lift_coefficient": vehicle.aero.lift_coefficient,
            "front_downforce_fraction": vehicle.aero.front_downforce_fraction,
        },
        "force_balance": "F_wheel = m_effective*a_IMU + F_drag + F_rolling; efficiency = F_wheel*r_tire/(T_motor*ratio)",
        "result": {
            "fitted_motor_to_wheel_efficiency": fitted_efficiency,
            "stratified_bootstrap_95pct_interval": [ci_lower, ci_upper],
            "current_default_efficiency": default_efficiency,
            "fitted_acceleration_rmse_mps2": fitted_rmse,
            "current_default_acceleration_rmse_mps2": default_rmse,
            "point_efficiency_median": float(np.median(point_efficiency)),
            "point_efficiency_10th_90th_percentile": [
                float(np.percentile(point_efficiency, 10)),
                float(np.percentile(point_efficiency, 90)),
            ],
        },
        "per_straight": per_straight,
        "limitations": [
            "This is an in-sample fit to the same first endurance lap previously used for drivetrain calibration.",
            "The corrected IMU, torque feedback, and GNSS speed are used at their recorded timestamps; no torque-to-acceleration latency is fitted.",
            "The point-mass force balance attributes unmodeled acceleration bias, road grade, wind, tire-radius error, and inertial-model error to efficiency.",
            "The bootstrap captures sample scatter within represented straights, not systematic model uncertainty.",
        ],
    }
    (output_dir / "efficiency_fit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Fitted motor-to-wheel efficiency {fitted_efficiency:.4f} "
        f"(95% bootstrap {ci_lower:.4f}–{ci_upper:.4f}) from "
        f"{np.count_nonzero(selected)} samples"
    )


if __name__ == "__main__":
    main()
