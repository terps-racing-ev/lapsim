r"""Level recorded IMU accelerations and remove stationary gravity.

Run from the ``python_lapsim`` root::

    .\.venv\Scripts\python.exe analysis\corrected_imu\generate_corrected_imu.py

The calibration comes from the stationary 450--465 s plateau in the complete
selected recording.  The correction is applied to the first-lap CSV, whose
samples begin after the car is moving and are therefore unsuitable for a
gravity calibration.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "analysis" / "first_endurance_lap" / "first_lap.csv"
DEFAULT_CALIBRATION_INPUT = (
    PROJECT_ROOT / "analysis" / "mf4_to_csv" / "endurance_selected.csv"
)
DEFAULT_OUTPUT = ANALYSIS_DIR / "first_lap_corrected_imu.csv"
DEFAULT_METADATA = ANALYSIS_DIR / "first_lap_corrected_imu.json"
DEFAULT_AXES_PLOT = ANALYSIS_DIR / "first_lap_raw_vs_corrected_axes.png"
DEFAULT_STATIONARY_PLOT = ANALYSIS_DIR / "stationary_gravity_validation.png"


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv", nargs="?", type=Path, default=DEFAULT_INPUT,
        help="Recording to correct (default: first endurance lap).",
    )
    parser.add_argument(
        "--calibration-input", type=Path, default=DEFAULT_CALIBRATION_INPUT,
        help="Complete recording used to find stationary gravity.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--axes-plot", type=Path, default=DEFAULT_AXES_PLOT)
    parser.add_argument(
        "--stationary-plot", type=Path, default=DEFAULT_STATIONARY_PLOT
    )
    parser.add_argument(
        "--calibration-start-mf4-time-s", type=float, default=450.0,
        help="Beginning of the known pre-lap stationary plateau (MF4 time).",
    )
    parser.add_argument(
        "--calibration-end-mf4-time-s", type=float, default=465.0,
        help="End of the known pre-lap stationary plateau (MF4 time).",
    )
    parser.add_argument(
        "--stationary-max-speed-mps", type=float, default=0.30,
        help="Reject calibration samples above this GNSS speed.",
    )
    parser.add_argument(
        "--stationary-max-angular-rate-degps", type=float, default=5.0,
        help="Reject calibration samples whose 3-axis angular-rate magnitude exceeds this.",
    )
    parser.add_argument(
        "--pitch-deg", type=float,
        help="Override the derived sensor-to-level pitch correction in degrees.",
    )
    parser.add_argument(
        "--roll-deg", type=float,
        help="Override the derived sensor-to-level roll correction in degrees.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def require_columns(data: dict[str, np.ndarray], names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def rotation_from_pitch_roll(pitch_rad: float, roll_rad: float) -> np.ndarray:
    """Return the active sensor-to-level rotation, roll first then pitch.

    Axes use right-handed vehicle convention: +X forward, +Y left, +Z up.
    ``roll_rad`` rotates about sensor +X and ``pitch_rad`` subsequently rotates
    about the rolled +Y.  At rest this maps the measured gravity vector onto
    +Z.  It intentionally makes no yaw correction.
    """
    cos_roll, sin_roll = np.cos(roll_rad), np.sin(roll_rad)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    roll = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_roll, -sin_roll], [0.0, sin_roll, cos_roll]]
    )
    pitch = np.array(
        [[cos_pitch, 0.0, sin_pitch], [0.0, 1.0, 0.0], [-sin_pitch, 0.0, cos_pitch]]
    )
    return pitch @ roll


def derive_pitch_roll_deg(gravity_sensor_mps2: np.ndarray) -> tuple[float, float]:
    """Find the sequential correction that maps static gravity to +Z."""
    x, y, z = gravity_sensor_mps2
    if not np.all(np.isfinite(gravity_sensor_mps2)) or z <= 0.0:
        raise ValueError("Stationary gravity must be finite with a positive sensor Z value")
    roll_rad = np.arctan2(y, z)
    gravity_after_roll_z = np.hypot(y, z)
    pitch_rad = -np.arctan2(x, gravity_after_roll_z)
    return float(np.degrees(pitch_rad)), float(np.degrees(roll_rad))


def select_stationary_gravity(
    calibration: dict[str, np.ndarray], args: Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = (
        "mf4_time_s", "gps_speed_mps", "accel_x_mps2", "accel_y_mps2",
        "accel_z_mps2", "angular_rate_x_degps", "angular_rate_y_degps",
        "angular_rate_z_degps",
    )
    require_columns(calibration, required, "Calibration CSV")
    if args.calibration_end_mf4_time_s <= args.calibration_start_mf4_time_s:
        raise ValueError("Calibration end time must be after the start time")
    if args.stationary_max_speed_mps < 0 or args.stationary_max_angular_rate_degps < 0:
        raise ValueError("Stationary thresholds must be non-negative")
    acceleration = np.column_stack(
        (calibration["accel_x_mps2"], calibration["accel_y_mps2"], calibration["accel_z_mps2"])
    )
    angular_rate = np.column_stack(
        (
            calibration["angular_rate_x_degps"],
            calibration["angular_rate_y_degps"],
            calibration["angular_rate_z_degps"],
        )
    )
    in_window = (
        (calibration["mf4_time_s"] >= args.calibration_start_mf4_time_s)
        & (calibration["mf4_time_s"] <= args.calibration_end_mf4_time_s)
    )
    stationary = (
        in_window
        & np.isfinite(calibration["gps_speed_mps"])
        & (calibration["gps_speed_mps"] <= args.stationary_max_speed_mps)
        & np.isfinite(angular_rate).all(axis=1)
        & (np.linalg.norm(angular_rate, axis=1) <= args.stationary_max_angular_rate_degps)
        & np.isfinite(acceleration).all(axis=1)
    )
    samples = acceleration[stationary]
    if len(samples) < 3:
        raise ValueError(
            "Fewer than three stationary calibration samples remain; adjust the "
            "calibration window or stationary thresholds."
        )
    # The component-wise median is robust to an occasional sample at the edge
    # of the plateau and is the documented estimate of the static gravity vector.
    return np.median(samples, axis=0), stationary, acceleration


def write_csv(columns: dict[str, np.ndarray], output: Path) -> Path:
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("Output columns do not have a common length")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        names = list(columns)
        writer.writerow(names)
        writer.writerows(zip(*(columns[name] for name in names), strict=True))
    return output.resolve()


def plot_raw_vs_corrected(
    time_s: np.ndarray,
    raw: np.ndarray,
    leveled: np.ndarray,
    linear: np.ndarray,
    output: Path,
) -> Path:
    labels = ("Longitudinal X", "Lateral Y", "Vertical Z")
    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout="constrained")
    for index, axis in enumerate(axes):
        axis.plot(time_s, raw[:, index], color="#9ca3af", linewidth=1.0, label="Raw sensor")
        axis.plot(time_s, leveled[:, index], color="#2563eb", linewidth=1.15, label="Leveled")
        axis.plot(time_s, linear[:, index], color="#dc2626", linewidth=1.15, label="Leveled, gravity removed")
        axis.axhline(0.0, color="0.2", linewidth=0.7)
        axis.set_ylabel(f"{labels[index]}\n[m/s²]")
        axis.grid(True, alpha=0.28)
        if index == 0:
            axis.legend(ncols=3, fontsize=9, loc="upper right")
    axes[-1].set_xlabel("First-lap elapsed time [s]")
    figure.suptitle("First-lap IMU: raw axes, leveled axes, and gravity-removed acceleration")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output.resolve()


def plot_stationary_validation(
    calibration_time_s: np.ndarray,
    calibration_window: np.ndarray,
    stationary: np.ndarray,
    raw: np.ndarray,
    leveled: np.ndarray,
    linear: np.ndarray,
    output: Path,
) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, layout="constrained")
    displayed_time_s = calibration_time_s[calibration_window]
    colors = ("#dc2626", "#2563eb", "#16a34a")
    for index, (name, color) in enumerate(zip(("X", "Y", "Z"), colors, strict=True)):
        axes[0].plot(
            displayed_time_s,
            raw[calibration_window, index],
            color=color,
            alpha=0.35,
            label=f"Raw {name}",
        )
        axes[0].plot(calibration_time_s[stationary], leveled[stationary, index], color=color, linewidth=1.2, label=f"Leveled {name}")
        axes[1].plot(calibration_time_s[stationary], linear[stationary, index], color=color, linewidth=1.2, label=f"Gravity-removed {name}")
    axes[0].set_title("Stationary calibration: gravity moves from tilted sensor axes to +Z")
    axes[0].set_ylabel("Leveled / raw acceleration [m/s²]")
    axes[1].set_title("Stationary validation: corrected axes should be near zero")
    axes[1].set_ylabel("Gravity-removed acceleration [m/s²]")
    axes[1].set_xlabel("MF4 recording time [s]")
    axes[1].set_ylim(-0.2, 0.2)
    axes[1].axhspan(-0.1, 0.1, color="#dcfce7", alpha=0.55, zorder=0, label="±0.1 m/s²")
    for axis in axes:
        axis.axhline(0.0, color="0.2", linewidth=0.7)
        axis.grid(True, alpha=0.28)
        axis.legend(ncols=3, fontsize=8, loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output.resolve()


def main() -> None:
    args = parse_args()
    corrected_input = read_csv(args.input_csv)
    require_columns(
        corrected_input,
        ("time_s", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2"),
        "Input CSV",
    )
    calibration_input = read_csv(args.calibration_input)
    gravity_sensor, stationary, calibration_raw = select_stationary_gravity(
        calibration_input, args
    )
    derived_pitch_deg, derived_roll_deg = derive_pitch_roll_deg(gravity_sensor)
    pitch_deg = args.pitch_deg if args.pitch_deg is not None else derived_pitch_deg
    roll_deg = args.roll_deg if args.roll_deg is not None else derived_roll_deg
    rotation = rotation_from_pitch_roll(np.radians(pitch_deg), np.radians(roll_deg))
    gravity_magnitude_mps2 = float(np.linalg.norm(gravity_sensor))
    gravity_level = np.array((0.0, 0.0, gravity_magnitude_mps2))

    raw = np.column_stack(
        (
            corrected_input["accel_x_mps2"],
            corrected_input["accel_y_mps2"],
            corrected_input["accel_z_mps2"],
        )
    )
    leveled = raw @ rotation.T
    linear = leveled - gravity_level
    calibration_leveled = calibration_raw @ rotation.T
    calibration_linear = calibration_leveled - gravity_level
    calibration_window = (
        (calibration_input["mf4_time_s"] >= args.calibration_start_mf4_time_s)
        & (calibration_input["mf4_time_s"] <= args.calibration_end_mf4_time_s)
    )

    output_columns = {
        **corrected_input,
        "accel_x_leveled_mps2": leveled[:, 0],
        "accel_y_leveled_mps2": leveled[:, 1],
        "accel_z_leveled_mps2": leveled[:, 2],
        "accel_x_gravity_removed_mps2": linear[:, 0],
        "accel_y_gravity_removed_mps2": linear[:, 1],
        "accel_z_gravity_removed_mps2": linear[:, 2],
        "corrected_longitudinal_accel_mps2": linear[:, 0],
        "corrected_lateral_accel_mps2": linear[:, 1],
        "corrected_vertical_accel_mps2": linear[:, 2],
    }
    csv_output = write_csv(output_columns, args.output)
    axes_plot = plot_raw_vs_corrected(
        corrected_input["time_s"], raw, leveled, linear, args.axes_plot
    )
    stationary_plot = plot_stationary_validation(
        calibration_input["mf4_time_s"],
        calibration_window,
        stationary,
        calibration_raw,
        calibration_leveled,
        calibration_linear,
        args.stationary_plot,
    )

    stationary_linear = calibration_linear[stationary]
    metadata = {
        "input_csv": str(args.input_csv.resolve()),
        "calibration_input_csv": str(args.calibration_input.resolve()),
        "output_csv": str(csv_output),
        "plots": {
            "raw_vs_corrected_axes": str(axes_plot),
            "stationary_gravity_validation": str(stationary_plot),
        },
        "axis_convention": {
            "sensor_and_output": "+X longitudinal/forward, +Y lateral/left, +Z up; right-handed",
            "accelerometer_interpretation": "At rest the recorded accelerometer reads +gravity in +Z after leveling.",
            "yaw": "No yaw rotation is applied; X/Y names retain the assumed vehicle mounting axes.",
        },
        "calibration": {
            "time_basis": "mf4_time_s from calibration_input_csv",
            "requested_window_mf4_time_s": [args.calibration_start_mf4_time_s, args.calibration_end_mf4_time_s],
            "stationary_selection": {
                "max_gnss_speed_mps": args.stationary_max_speed_mps,
                "max_3axis_angular_rate_degps": args.stationary_max_angular_rate_degps,
                "selected_samples": int(np.count_nonzero(stationary)),
            },
            "estimator": "component-wise median of selected raw accelerometer samples",
            "stationary_gravity_sensor_mps2": gravity_sensor.tolist(),
            "stationary_gravity_magnitude_mps2": gravity_magnitude_mps2,
            "derived_sensor_to_level_pitch_deg": derived_pitch_deg,
            "derived_sensor_to_level_roll_deg": derived_roll_deg,
            "applied_sensor_to_level_pitch_deg": pitch_deg,
            "applied_sensor_to_level_roll_deg": roll_deg,
            "override_used": bool(args.pitch_deg is not None or args.roll_deg is not None),
            "rotation_order": "active Rx(roll), then active Ry(pitch): level_vector = Ry(pitch) @ Rx(roll) @ sensor_vector",
            "rotation_matrix_sensor_to_level": rotation.tolist(),
            "stationary_corrected_mean_mps2": np.mean(stationary_linear, axis=0).tolist(),
            "stationary_corrected_rms_mps2": np.sqrt(np.mean(stationary_linear**2, axis=0)).tolist(),
        },
        "output_columns_added": {
            "accel_[xyz]_leveled_mps2": "Accelerometer axes after the fixed sensor-to-level rotation; includes +gravity in Z.",
            "accel_[xyz]_gravity_removed_mps2": "Leveled acceleration minus stationary gravity [0, 0, |g_stationary|]; use X/Y for longitudinal/lateral comparison.",
            "corrected_[longitudinal|lateral|vertical]_accel_mps2": "Semantic aliases for the gravity-removed X/Y/Z channels consumed by downstream analyses.",
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote corrected IMU CSV: {csv_output}")
    print(f"Wrote metadata: {args.metadata.resolve()}")
    print(f"Applied pitch/roll correction: {pitch_deg:.3f} / {roll_deg:.3f} deg")


if __name__ == "__main__":
    main()
