"""Plot VCU vehicle speed from a decoded MF4 recording.

Run from the repository root with::

    .venv\\Scripts\\python.exe analysis\\testing\\coastdown\\plot_vcu_speed.py

The decoded MF4 contains VCU speed in ``CAN1.VCU_Summary.VCU_Speed_MPH``.
The plot uses elapsed recording time as its x-axis and preserves the VCU's
native miles-per-hour units.

For a road-load fit, add ``--include-acceleration --fit-road-load``. The fit
uses ``a = b0 + b1*v + b2*v²`` with speed in m/s. Its inferred Cd is only an
apparent value until grade, wind, and other speed-dependent losses are checked.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from vehicle_model.vehicle import Vehicle

try:
    from asammdf import MDF
except ImportError as exc:  # pragma: no cover - depends on the optional MF4 extra
    raise ImportError("asammdf is required: pip install -e '.[mf4]'") from exc


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "logs" / "2026-09-07_01-00-07_10m.mf4"
DEFAULT_OUTPUT = ROOT / "outputs" / "testing" / "coastdown" / "vcu_speed.png"
VCU_SPEED_CHANNEL = "CAN1.VCU_Summary.VCU_Speed_MPH"
EASTERN = ZoneInfo("America/New_York")


def read_vcu_speed(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, str, datetime]:
    """Return speed data and the UTC timestamp of its first sample."""

    if not path.is_file():
        raise FileNotFoundError(f"MF4 not found: {path}")

    mdf = MDF(path)
    try:
        references = mdf.channels_db.get(VCU_SPEED_CHANNEL)
        if not references:
            raise KeyError(
                f"Decoded channel not found: {VCU_SPEED_CHANNEL}. "
                "This script expects an MF4 decoded with the VCU DBC."
            )

        # Select the first occurrence explicitly. The unqualified channel
        # name can occur on both CAN1 and CAN2 and is ambiguous to asammdf.
        group, index = references[0]
        signal = mdf.get(VCU_SPEED_CHANNEL, group=group, index=index)
        time_s = np.asarray(signal.timestamps, dtype=float)
        speed_mph = np.asarray(signal.samples, dtype=float)
        file_start = mdf.start_time
    finally:
        mdf.close()

    finite = np.isfinite(time_s) & np.isfinite(speed_mph)
    if finite.sum() < 2:
        raise ValueError(f"VCU speed channel has fewer than two finite samples: {path}")

    time_s = time_s[finite]
    speed_mph = speed_mph[finite]
    order = np.argsort(time_s)
    time_s = time_s[order]
    speed_mph = speed_mph[order]
    if file_start.tzinfo is None:
        file_start = file_start.replace(tzinfo=timezone.utc)
    signal_start_utc = file_start.astimezone(timezone.utc) + timedelta(
        seconds=float(time_s[0])
    )
    return time_s - time_s[0], speed_mph, VCU_SPEED_CHANNEL, signal_start_utc


def parse_eastern_datetime(value: str) -> datetime:
    """Parse a local Eastern datetime and return it as an aware UTC datetime."""

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=EASTERN)
    return parsed.astimezone(timezone.utc)


def rolling_average(
    time_s: np.ndarray,
    values: np.ndarray,
    window_s: float,
) -> tuple[np.ndarray, int]:
    """Apply a centered rolling average using the signal's median sample rate."""

    if not np.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("The rolling-average window must be finite and positive")
    intervals_s = np.diff(time_s)
    intervals_s = intervals_s[np.isfinite(intervals_s) & (intervals_s > 0.0)]
    if len(intervals_s) == 0:
        raise ValueError("Cannot determine the VCU speed sample period")

    window_samples = max(1, int(round(window_s / float(np.median(intervals_s)))))
    if window_samples % 2 == 0:
        window_samples += 1
    half_window = window_samples // 2
    padded = np.pad(values, (half_window, half_window), mode="edge")
    kernel = np.ones(window_samples, dtype=float) / window_samples
    return np.convolve(padded, kernel, mode="valid"), window_samples


def savgol_derivative(
    time_s: np.ndarray,
    values: np.ndarray,
    window_s: float,
) -> tuple[np.ndarray, int]:
    """Estimate a smooth first derivative with a local quadratic fit."""

    if not np.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("The derivative window must be finite and positive")
    intervals_s = np.diff(time_s)
    intervals_s = intervals_s[np.isfinite(intervals_s) & (intervals_s > 0.0)]
    if len(intervals_s) == 0:
        raise ValueError("Cannot determine the VCU speed sample period")

    window_samples = max(3, int(round(window_s / float(np.median(intervals_s)))))
    if window_samples % 2 == 0:
        window_samples += 1
    if window_samples >= len(values):
        window_samples = len(values) if len(values) % 2 else len(values) - 1
    if window_samples < 3:
        raise ValueError("The selected acceleration window contains too few samples")

    derivative = savgol_filter(
        values,
        window_length=window_samples,
        polyorder=2,
        deriv=1,
        delta=float(np.median(intervals_s)),
        mode="interp",
    )
    return derivative, window_samples


def fit_road_load_acceleration(
    speed_mps: np.ndarray,
    acceleration_mps2: np.ndarray,
    *,
    fixed_intercept_mps2: float | None = None,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float, float]:
    """Fit constant, speed, and speed-squared acceleration terms.

    If a rolling-resistance acceleration is supplied, it is separated from a
    fitted constant offset for reporting. Because that offset is free, the
    supplied rolling value does not constrain any fitted coefficient.
    """

    speed_squared = speed_mps**2
    finite = np.isfinite(speed_squared) & np.isfinite(acceleration_mps2)
    if fit_mask is not None:
        if fit_mask.shape != finite.shape:
            raise ValueError("The speed-squared fit mask must match the signal shape")
        finite &= fit_mask
    if finite.sum() < 3:
        raise ValueError("Not enough finite samples for a road-load fit")

    design = np.column_stack(
        (np.ones(finite.sum(), dtype=float), speed_mps[finite], speed_squared[finite])
    )
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("The selected speed has insufficient variation for a road-load fit")

    target = acceleration_mps2[finite]
    if fixed_intercept_mps2 is not None:
        target = target - float(fixed_intercept_mps2)
    extra_intercept, linear_coefficient, quadratic_coefficient = np.linalg.lstsq(
        design, target, rcond=None
    )[0]
    intercept = float(extra_intercept)
    if fixed_intercept_mps2 is not None:
        intercept += float(fixed_intercept_mps2)
    linear_coefficient = float(linear_coefficient)
    quadratic_coefficient = float(quadratic_coefficient)

    fitted_acceleration_mps2 = (
        intercept + linear_coefficient * speed_mps + quadratic_coefficient * speed_squared
    )
    residual = acceleration_mps2[finite] - fitted_acceleration_mps2[finite]
    total_variation = acceleration_mps2[finite] - np.mean(acceleration_mps2[finite])
    r_squared = 1.0 - float(np.dot(residual, residual)) / float(
        np.dot(total_variation, total_variation)
    )
    return (
        fitted_acceleration_mps2,
        intercept,
        linear_coefficient,
        quadratic_coefficient,
        r_squared,
    )


def apparent_drag_from_quadratic_coefficient(
    quadratic_coefficient: float,
) -> tuple[float, float]:
    """Convert the v² slope to CdA and Cd using the current vehicle model.

    This subtracts rolling resistance caused by modeled aerodynamic downforce.
    Grade, wind, curvature, and unmodeled speed-squared losses remain confounders.
    """

    vehicle = Vehicle()
    combined_area_m2 = (
        -2.0
        * vehicle.effective_longitudinal_mass_kg
        * quadratic_coefficient
        / vehicle.air_density_kgpm3
    )
    drag_area_m2 = (
        combined_area_m2
        + vehicle.rolling_resistance_coefficient
        * vehicle.aero.lift_coefficient
        * vehicle.aero.frontal_area_m2
    )
    return drag_area_m2, drag_area_m2 / vehicle.aero.frontal_area_m2


def model_rolling_resistance_acceleration() -> tuple[float, float, float]:
    """Return model Crr, rolling force, and fixed rolling acceleration."""

    vehicle = Vehicle()
    rolling_force_n = (
        vehicle.rolling_resistance_coefficient
        * vehicle.mass_kg
        * vehicle.gravity_mps2
    )
    rolling_acceleration_mps2 = (
        -rolling_force_n / vehicle.effective_longitudinal_mass_kg
    )
    return (
        vehicle.rolling_resistance_coefficient,
        rolling_force_n,
        rolling_acceleration_mps2,
    )


def plot_vcu_speed(
    time_s: np.ndarray,
    speed_mph: np.ndarray,
    *,
    input_path: Path,
    output_path: Path,
    start_s: float | None = None,
    finish_s: float | None = None,
    window_label: str | None = None,
    rolling_average_s: float | None = None,
    include_acceleration: bool = False,
    fit_road_load: bool = False,
    fixed_rolling_acceleration_mps2: float | None = None,
) -> Path:
    """Plot and save the selected VCU speed trace."""

    if start_s is None:
        start_s = float(time_s[0])
    if finish_s is None:
        finish_s = float(time_s[-1])
    if not np.isfinite(start_s) or not np.isfinite(finish_s) or finish_s <= start_s:
        raise ValueError("The plot window must have finite start < finish")

    selected = (time_s >= start_s) & (time_s <= finish_s)
    if selected.sum() < 2:
        raise ValueError("The selected plot window contains fewer than two samples")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if include_acceleration:
        figure, axes = plt.subplots(
            2, 1, figsize=(12, 8.5), sharex=True, layout="constrained"
        )
        speed_axis, acceleration_axis = axes
    else:
        figure, speed_axis = plt.subplots(figsize=(12, 5.5), layout="constrained")
        acceleration_axis = None

    selected_time_s = time_s[selected] - start_s
    selected_speed_mph = speed_mph[selected]
    if rolling_average_s is None:
        speed_for_acceleration_mph = selected_speed_mph
        speed_axis.plot(
            selected_time_s,
            selected_speed_mph,
            color="#1f77b4",
            linewidth=1.2,
            label="VCU speed",
        )
    else:
        speed_for_acceleration_mph, _ = rolling_average(
            selected_time_s, selected_speed_mph, rolling_average_s
        )
        speed_axis.plot(
            selected_time_s,
            selected_speed_mph,
            color="#9aa0a6",
            linewidth=0.8,
            alpha=0.55,
            label="Raw VCU speed",
        )
        speed_axis.plot(
            selected_time_s,
            speed_for_acceleration_mph,
            color="#1f77b4",
            linewidth=1.8,
            label=f"{rolling_average_s:g} s rolling average",
        )
    title = f"VCU speed - {input_path.name}"
    if window_label:
        title += f"\n{window_label}"
    speed_axis.set_title(title)
    speed_axis.set_ylabel("VCU speed [mph]")
    speed_axis.grid(alpha=0.25)
    speed_axis.spines[["top", "right"]].set_visible(False)
    speed_axis.legend(frameon=False, loc="best")

    if acceleration_axis is None:
        speed_axis.set_xlabel("Elapsed time in selected window [s]")
    else:
        acceleration_window_s = rolling_average_s or 0.75
        acceleration_mps2, _ = savgol_derivative(
            selected_time_s,
            speed_for_acceleration_mph * 0.44704,
            acceleration_window_s,
        )
        acceleration_axis.plot(
            selected_time_s,
            acceleration_mps2,
            color="#e45756",
            linewidth=1.3,
            label="Derived acceleration",
        )
        if fit_road_load:
            # Exclude the derivative filter edges from the fit. The plotted
            # acceleration is still retained over the complete selected window.
            fit_edge_s = min(
                acceleration_window_s / 2.0,
                (selected_time_s[-1] - selected_time_s[0]) / 4.0,
            )
            fit_mask = (selected_time_s >= selected_time_s[0] + fit_edge_s) & (
                selected_time_s <= selected_time_s[-1] - fit_edge_s
            )
            fitted_acceleration_mps2, intercept, linear, quadratic, r_squared = (
                fit_road_load_acceleration(
                    speed_for_acceleration_mph * 0.44704,
                    acceleration_mps2,
                    fixed_intercept_mps2=fixed_rolling_acceleration_mps2,
                    fit_mask=fit_mask,
                )
            )
            fitted_acceleration_mps2 = np.where(
                fit_mask, fitted_acceleration_mps2, np.nan
            )
            acceleration_axis.plot(
                selected_time_s,
                fitted_acceleration_mps2,
                color="#4c78a8",
                linewidth=1.7,
                linestyle="--",
                label=(
                    f"RR + fitted offset + v + v²: a = {intercept:.3f} "
                    f"{linear:+.4f} v {quadratic:+.4f} v² (R²={r_squared:.2f}; "
                    f"RR={fixed_rolling_acceleration_mps2:.3f})"
                ),
            )
        acceleration_axis.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.7)
        acceleration_axis.set_ylabel("Acceleration [m/s²]")
        acceleration_axis.set_xlabel("Elapsed time in selected window [s]")
        acceleration_axis.set_title(
            "Derived acceleration from smoothed VCU speed (Savitzky-Golay)"
        )
        acceleration_axis.grid(alpha=0.25)
        acceleration_axis.spines[["top", "right"]].set_visible(False)
        acceleration_axis.legend(frameon=False, loc="best")

    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--start-s",
        type=float,
        default=None,
        help="Start of the elapsed-time plot window in seconds.",
    )
    parser.add_argument(
        "--finish-s",
        type=float,
        default=None,
        help="End of the elapsed-time plot window in seconds.",
    )
    parser.add_argument(
        "--start-eastern",
        default=None,
        help="Eastern wall-clock start, e.g. 2026-09-06T23:55:00.",
    )
    parser.add_argument(
        "--finish-eastern",
        default=None,
        help="Eastern wall-clock finish, e.g. 2026-09-07T00:05:00.",
    )
    parser.add_argument(
        "--rolling-average-s",
        type=float,
        default=None,
        help="Centered rolling-average window in seconds.",
    )
    parser.add_argument(
        "--include-acceleration",
        action="store_true",
        help="Add acceleration derived from the plotted VCU speed trace.",
    )
    parser.add_argument(
        "--fit-road-load",
        action="store_true",
        help="Fit constant, speed, and speed-squared road-load terms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fit_road_load and not args.include_acceleration:
        raise ValueError("--fit-road-load requires --include-acceleration")
    if (args.start_eastern is None) != (args.finish_eastern is None):
        raise ValueError("--start-eastern and --finish-eastern must be provided together")

    time_s, speed_mph, channel_name, signal_start_utc = read_vcu_speed(args.input)
    window_label = None
    start_s = args.start_s
    finish_s = args.finish_s
    fixed_rolling_acceleration_mps2 = None
    rolling_coefficient = None
    rolling_force_n = None
    if args.fit_road_load:
        (
            rolling_coefficient,
            rolling_force_n,
            fixed_rolling_acceleration_mps2,
        ) = model_rolling_resistance_acceleration()
    if args.start_eastern is not None:
        if args.start_s is not None or args.finish_s is not None:
            raise ValueError("Use either elapsed-time or Eastern wall-clock window arguments")
        start_utc = parse_eastern_datetime(args.start_eastern)
        finish_utc = parse_eastern_datetime(args.finish_eastern)
        start_s = (start_utc - signal_start_utc).total_seconds()
        finish_s = (finish_utc - signal_start_utc).total_seconds()
        window_label = (
            f"Eastern: {args.start_eastern} to {args.finish_eastern}"
        )
    effective_start_s = float(time_s[0]) if start_s is None else start_s
    effective_finish_s = float(time_s[-1]) if finish_s is None else finish_s
    output_path = plot_vcu_speed(
        time_s,
        speed_mph,
        input_path=args.input,
        output_path=args.output,
        start_s=effective_start_s,
        finish_s=effective_finish_s,
        window_label=window_label,
        rolling_average_s=args.rolling_average_s,
        include_acceleration=args.include_acceleration,
        fit_road_load=args.fit_road_load,
        fixed_rolling_acceleration_mps2=fixed_rolling_acceleration_mps2,
    )
    print(f"channel: {channel_name}")
    print(f"samples: {len(speed_mph):,}")
    print(f"time range: {time_s[0]:.3f}-{time_s[-1]:.3f} s")
    print(f"speed range: {np.min(speed_mph):.2f}-{np.max(speed_mph):.2f} mph")
    if args.start_s is not None or args.finish_s is not None or window_label:
        print(
            "selected window: "
            f"{effective_start_s:.3f}-{effective_finish_s:.3f} s "
            "after first VCU sample"
        )
    if args.rolling_average_s is not None:
        _, window_samples = rolling_average(
            time_s[(time_s >= effective_start_s) & (time_s <= effective_finish_s)],
            speed_mph[(time_s >= effective_start_s) & (time_s <= effective_finish_s)],
            args.rolling_average_s,
        )
        print(
            f"rolling average: {args.rolling_average_s:g} s "
            f"({window_samples} samples)"
        )
    if args.include_acceleration:
        acceleration_window_s = args.rolling_average_s or 0.75
        selected = (time_s >= effective_start_s) & (time_s <= effective_finish_s)
        _, derivative_samples = savgol_derivative(
            time_s[selected],
            (
                rolling_average(
                    time_s[selected], speed_mph[selected], acceleration_window_s
                )[0]
                if args.rolling_average_s is not None
                else speed_mph[selected]
            ),
            acceleration_window_s,
        )
        print(
            f"acceleration derivative: Savitzky-Golay, "
            f"{acceleration_window_s:g} s ({derivative_samples} samples)"
        )
        if args.fit_road_load:
            selected = (time_s >= effective_start_s) & (time_s <= effective_finish_s)
            speed_for_fit = speed_mph[selected]
            if args.rolling_average_s is not None:
                speed_for_fit = rolling_average(
                    time_s[selected], speed_for_fit, args.rolling_average_s
                )[0]
            acceleration_for_fit, _ = savgol_derivative(
                time_s[selected],
                speed_for_fit * 0.44704,
                acceleration_window_s,
            )
            selected_time_for_fit = time_s[selected] - effective_start_s
            fit_edge_s = min(
                acceleration_window_s / 2.0,
                (selected_time_for_fit[-1] - selected_time_for_fit[0]) / 4.0,
            )
            fit_mask = (selected_time_for_fit >= selected_time_for_fit[0] + fit_edge_s) & (
                selected_time_for_fit <= selected_time_for_fit[-1] - fit_edge_s
            )
            _, intercept, linear, quadratic, r_squared = fit_road_load_acceleration(
                speed_for_fit * 0.44704,
                acceleration_for_fit,
                fixed_intercept_mps2=fixed_rolling_acceleration_mps2,
                fit_mask=fit_mask,
            )
            print(
                f"road-load fit (excluding +/-{fit_edge_s:g} s filter edges): "
                f"a = {intercept:.6f} {linear:+.6f} v "
                f"{quadratic:+.6f} v² (R²={r_squared:.3f}); v in m/s"
            )
            drag_area_m2, drag_coefficient = apparent_drag_from_quadratic_coefficient(
                quadratic
            )
            print(
                "apparent aerodynamic drag after modeled downforce rolling loss: "
                f"CdA={drag_area_m2:.4f} m², Cd={drag_coefficient:.4f} "
                "(assuming zero wind/grade variation and no other v² losses)"
            )
            if fixed_rolling_acceleration_mps2 is not None:
                print(
                    "fitted constant loss beyond rolling resistance: "
                    f"{intercept - fixed_rolling_acceleration_mps2:+.6f} m/s²"
                )
            print(
                f"model rolling resistance: Crr={rolling_coefficient:.3f}, "
                f"force={rolling_force_n:.2f} N, "
                f"acceleration={fixed_rolling_acceleration_mps2:.4f} m/s²"
            )
    print(f"plot: {output_path.resolve()}")


if __name__ == "__main__":
    main()
