r"""Fit an effective aerodynamic drag coefficient to the two 35 mph coasts.

Run from the repository root with::

    .venv\Scripts\python.exe analysis\testing\coastdown\fit_35mph_drag.py

Times are seconds after the first VCU speed sample in the decoded MF4. The
default windows exclude powered acceleration and the later brake application.
The fit assumes level ground, no wind, no drivetrain drag, and the Vehicle
model's rolling resistance (including aero downforce); violations appear in
the fitted effective Cd.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from asammdf import MDF
from scipy.optimize import least_squares
from vehicle_model.vehicle import Vehicle

from plot_vcu_speed import DEFAULT_INPUT, ROOT, VCU_SPEED_CHANNEL, read_vcu_speed


DEFAULT_WINDOWS = ((488.0, 490.2), (534.8, 538.8))
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "testing" / "coastdown"
MPH_TO_MPS = 0.44704
CONTROL_CHANNELS = {
    "commanded_torque_nm": "CAN1.VCU_INV_Command.VCU_INV_Torque_Cmd",
    "front_brake_psi": "CAN1.VCU_Regen_Debug.VCU_Regen_Front_PSI",
    "rear_brake_psi": "CAN1.VCU_Regen_Debug.VCU_Regen_Rear_PSI",
    "regen_active": "CAN1.VCU_Regen_Debug.VCU_Regen_Active",
}


def coast_speed_mps(
    elapsed_s: np.ndarray,
    initial_speed_mps: float,
    drag_coefficient: float,
    vehicle: Vehicle,
) -> np.ndarray:
    """Solve the coast ODE with aero drag and downforce-dependent tire drag."""

    rolling_acceleration = (
        vehicle.rolling_resistance_coefficient
        * vehicle.mass_kg
        * vehicle.gravity_mps2
        / vehicle.effective_longitudinal_mass_kg
    )
    quadratic = (
        0.5
        * vehicle.air_density_kgpm3
        * vehicle.aero.frontal_area_m2
        * (drag_coefficient - vehicle.rolling_resistance_coefficient * vehicle.aero.lift_coefficient)
        / vehicle.effective_longitudinal_mass_kg
    )
    if quadratic == 0.0:
        return initial_speed_mps - rolling_acceleration * elapsed_s
    ratio = np.sqrt(quadratic / rolling_acceleration)
    return np.sqrt(rolling_acceleration / quadratic) * np.tan(
        np.arctan(initial_speed_mps * ratio)
        - np.sqrt(rolling_acceleration * quadratic) * elapsed_s
    )


def fit_coasts(
    time_s: np.ndarray,
    speed_mph: np.ndarray,
    windows: tuple[tuple[float, float], ...],
    vehicle: Vehicle,
) -> tuple[float, list[dict[str, float | int | tuple[float, float]]]]:
    """Fit one shared Cd and separate starting speeds to the raw VCU traces."""

    segments = []
    for start_s, finish_s in windows:
        if not (np.isfinite(start_s) and np.isfinite(finish_s) and start_s < finish_s):
            raise ValueError("Each coast window needs finite start < finish")
        selected = (time_s >= start_s) & (time_s <= finish_s)
        if selected.sum() < 10:
            raise ValueError(f"Too few VCU speed samples in {start_s:g}-{finish_s:g} s")
        segments.append((time_s[selected] - start_s, speed_mph[selected] * MPH_TO_MPS))

    def residual(parameters: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                coast_speed_mps(elapsed, parameters[i + 1], parameters[0], vehicle)
                - measured
                for i, (elapsed, measured) in enumerate(segments)
            ]
        )

    starting_speeds = [float(measured[0]) for _, measured in segments]
    result = least_squares(
        residual,
        [vehicle.aero.drag_coefficient, *starting_speeds],
        bounds=([0.0, *([0.0] * len(segments))], [10.0, *([50.0] * len(segments))]),
    )
    if not result.success:
        raise RuntimeError(f"Shared coast fit failed: {result.message}")
    shared_cd = float(result.x[0])

    rows = []
    for i, ((start_s, finish_s), (elapsed, measured)) in enumerate(
        zip(windows, segments, strict=True)
    ):
        fitted = coast_speed_mps(elapsed, result.x[i + 1], shared_cd, vehicle)
        independent = least_squares(
            lambda parameters: coast_speed_mps(
                elapsed, parameters[1], parameters[0], vehicle
            )
            - measured,
            [shared_cd, result.x[i + 1]],
            bounds=([0.0, 0.0], [10.0, 50.0]),
        )
        rows.append(
            {
                "window_s": (start_s, finish_s),
                "samples": int(len(measured)),
                "starting_speed_mph": float(result.x[i + 1] / MPH_TO_MPS),
                "ending_speed_mph": float(measured[-1] / MPH_TO_MPS),
                "individual_cd": float(independent.x[0]),
                "shared_fit_rmse_mph": float(
                    np.sqrt(np.mean((fitted - measured) ** 2)) / MPH_TO_MPS
                ),
            }
        )
    return shared_cd, rows


def read_control_summary(
    path: Path, windows: tuple[tuple[float, float], ...]
) -> list[dict[str, float | bool]]:
    """Check the CAN torque and brake channels in VCU-speed-relative time."""

    mdf = MDF(path)
    try:
        speed_group, speed_index = mdf.channels_db[VCU_SPEED_CHANNEL][0]
        speed_signal = mdf.get(
            VCU_SPEED_CHANNEL, group=speed_group, index=speed_index
        )
        finite = np.isfinite(speed_signal.timestamps) & np.isfinite(
            speed_signal.samples
        )
        speed_origin_s = float(speed_signal.timestamps[finite][0])

        channels = {}
        for label, name in CONTROL_CHANNELS.items():
            group, index = mdf.channels_db[name][0]
            signal = mdf.get(name, group=group, index=index)
            channels[label] = (np.asarray(signal.timestamps) - speed_origin_s, signal.samples)
    finally:
        mdf.close()

    summaries = []
    for start_s, finish_s in windows:
        values = {}
        for label, (timestamps, samples) in channels.items():
            selected = (timestamps >= start_s) & (timestamps <= finish_s)
            if not np.any(selected):
                raise ValueError(f"No {label} samples in {start_s:g}-{finish_s:g} s")
            values[label] = samples[selected]
        summary = {
            "max_abs_commanded_torque_nm": float(
                np.max(np.abs(values["commanded_torque_nm"]))
            ),
            "max_front_brake_psi": float(np.max(values["front_brake_psi"])),
            "max_rear_brake_psi": float(np.max(values["rear_brake_psi"])),
            "any_regen_active": bool(
                np.any(np.char.upper(values["regen_active"].astype("U")) == "TRUE")
            ),
        }
        if (
            summary["max_abs_commanded_torque_nm"] > 2.0
            or summary["max_front_brake_psi"] > 8.0
            or summary["max_rear_brake_psi"] > 1.0
            or summary["any_regen_active"]
        ):
            raise ValueError(
                f"{start_s:g}-{finish_s:g} s is not a clean coast: {summary}"
            )
        summaries.append(summary)
    return summaries


def plot_fit(
    time_s: np.ndarray,
    speed_mph: np.ndarray,
    windows: tuple[tuple[float, float], ...],
    rows: list[dict[str, float | int | tuple[float, float]]],
    cd: float,
    vehicle: Vehicle,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(windows), figsize=(12, 4.8), layout="constrained")
    for i, (axis, (start_s, finish_s), row) in enumerate(
        zip(np.atleast_1d(axes), windows, rows, strict=True), start=1
    ):
        selected = (time_s >= start_s) & (time_s <= finish_s)
        elapsed = time_s[selected] - start_s
        observed = speed_mph[selected]
        model = coast_speed_mps(
            elapsed, float(row["starting_speed_mph"]) * MPH_TO_MPS, cd, vehicle
        ) / MPH_TO_MPS
        axis.plot(elapsed, observed, color="#9aa0a6", lw=1.2, label="VCU speed")
        axis.plot(elapsed, model, color="#c44e52", lw=2.0, label="Coast model")
        axis.set_title(
            f"35 mph run {i}: {start_s:g}-{finish_s:g} s\n"
            f"Independent Cd {row['individual_cd']:.2f}; "
            f"RMSE {row['shared_fit_rmse_mph']:.2f} mph"
        )
        axis.set_xlabel("Time since coast-window start [s]")
        axis.set_ylabel("VCU speed [mph]")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.suptitle(
        f"35 mph coastdown: shared effective Cd = {cd:.2f} "
        f"(CdA = {cd * vehicle.aero.frontal_area_m2:.2f} m²)"
    )
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--window", nargs=2, type=float, action="append", metavar=("START", "FINISH"),
        help="Coast window in seconds after first VCU sample; repeat for multiple runs.",
    )
    args = parser.parse_args()
    windows = tuple(map(tuple, args.window)) if args.window else DEFAULT_WINDOWS
    time_s, speed_mph, _, _ = read_vcu_speed(args.input)
    vehicle = Vehicle()
    controls = read_control_summary(args.input, windows)
    cd, rows = fit_coasts(time_s, speed_mph, windows, vehicle)
    for row, control in zip(rows, controls, strict=True):
        row.update(control)

    drag_area_m2 = cd * vehicle.aero.frontal_area_m2
    speed_35_mps = 35.0 * MPH_TO_MPS
    report = {
        "input": str(args.input.resolve()),
        "speed_channel": VCU_SPEED_CHANNEL,
        "method": "Least-squares speed-trace fit of dv/dt = -[Crr*m*g + rho*A*(Cd - Crr*Cl)*v^2/2]/m_effective; a separate initial speed is fitted per run. Cl is the vehicle model's signed lift coefficient.",
        "assumptions": "Level ground, still air, zero drivetrain and brake drag beyond modeled rolling resistance; Cd is an effective value if these assumptions fail.",
        "vehicle": {
            "mass_kg": vehicle.mass_kg,
            "effective_longitudinal_mass_kg": vehicle.effective_longitudinal_mass_kg,
            "rolling_resistance_coefficient": vehicle.rolling_resistance_coefficient,
            "air_density_kgpm3": vehicle.air_density_kgpm3,
            "frontal_area_m2": vehicle.aero.frontal_area_m2,
            "lift_coefficient": vehicle.aero.lift_coefficient,
        },
        "shared_cd": cd,
        "drag_area_m2": drag_area_m2,
        "aero_drag_at_35mph_n": 0.5 * vehicle.air_density_kgpm3 * drag_area_m2 * speed_35_mps**2,
        "runs": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "vcu_speed_35mph_drag_fit.png"
    report_path = args.output_dir / "vcu_speed_35mph_drag_fit.json"
    plot_fit(time_s, speed_mph, windows, rows, cd, vehicle, plot_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Shared effective Cd: {cd:.3f} (CdA {drag_area_m2:.3f} m²)")
    for i, row in enumerate(rows, start=1):
        print(
            f"Run {i} {row['window_s'][0]:g}-{row['window_s'][1]:g} s: "
            f"individual Cd {row['individual_cd']:.3f}, "
            f"fit RMSE {row['shared_fit_rmse_mph']:.3f} mph"
        )
    print(f"Plot: {plot_path.resolve()}")
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
