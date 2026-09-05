"""Compare baseline and straight-line active aero in endurance/efficiency.

The active package adds the configured mass penalty and automatically selects
its low-drag position on cells whose absolute curvature is below the straight
threshold. Both scenarios use the same torque profile so the reported delta
isolates the modeled hardware change rather than a control retune. Defaults
match the capped endurance study: full torque request with 40 kW pack/motor
power and 40 mph vehicle-speed ceilings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.events.common import coarsen_track  # noqa: E402
from lapsim import (  # noqa: E402
    EnduranceRunConfig,
    EventResult,
    SpatialTrack,
    UniformPeriodicTorqueParameterization,
    simulate_endurance,
)
from utils.units import (  # noqa: E402
    KILOGRAMS_PER_POUND,
    meters_per_second_to_miles_per_hour,
    miles_per_hour_to_meters_per_second,
)
from vehicle_model import ActiveAero, Vehicle  # noqa: E402


DEFAULT_TRACK = ROOT / "analysis/data/track/gnss_imu_endurance_track.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/active_aero"
DEFAULT_INITIAL_STATE_OF_CHARGE = 0.9815


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--torque-fraction", type=float, default=1.0)
    parser.add_argument("--power-cap-kw", type=float, default=40.0)
    parser.add_argument("--speed-cap-mph", type=float, default=40.0)
    parser.add_argument("--cell-length-m", type=float, default=1.0)
    parser.add_argument("--laps", type=int, default=22)
    parser.add_argument("--drag-reduction", type=float, default=0.30)
    parser.add_argument("--downforce-reduction", type=float, default=0.30)
    parser.add_argument("--mass-penalty-lb", type=float, default=3.0)
    parser.add_argument(
        "--straight-curvature-threshold-per-m",
        type=float,
        default=0.005,
        help="Low-drag deploys when |curvature| is at or below this value.",
    )
    parser.add_argument("--maximum-brake-pressure-psi", type=float, default=300.0)
    parser.add_argument("--path-speed-tolerance-mps", type=float, default=0.2)
    return parser.parse_args()


def apply_endurance_caps(
    vehicle: Vehicle,
    *,
    power_cap_kw: float,
    speed_cap_mph: float,
) -> None:
    """Apply the repository's pack/motor power and vehicle-speed caps."""

    power_cap_w = 1_000.0 * power_cap_kw
    vehicle.battery.max_discharge_power_w = power_cap_w
    vehicle.drivetrain.motor.peak_power_w = power_cap_w
    vehicle.drivetrain.motor.continuous_power_w = power_cap_w
    vehicle.drivetrain.configured_speed_limit_mps = (
        miles_per_hour_to_meters_per_second(speed_cap_mph)
    )


def calibrated_vehicle(
    *,
    power_cap_kw: float = 40.0,
    speed_cap_mph: float = 40.0,
) -> Vehicle:
    vehicle = Vehicle()
    vehicle.battery.initial_state_of_charge = DEFAULT_INITIAL_STATE_OF_CHARGE
    apply_endurance_caps(
        vehicle,
        power_cap_kw=power_cap_kw,
        speed_cap_mph=speed_cap_mph,
    )
    vehicle.validate()
    return vehicle


def calibrated_active_aero_vehicle(
    *,
    drag_reduction_fraction: float = 0.30,
    downforce_reduction_fraction: float = 0.30,
    mass_penalty_lb: float = 3.0,
    straight_curvature_threshold_per_m: float = 0.005,
    power_cap_kw: float = 40.0,
    speed_cap_mph: float = 40.0,
) -> Vehicle:
    aero = ActiveAero(
        drag_reduction_fraction=drag_reduction_fraction,
        downforce_reduction_fraction=downforce_reduction_fraction,
        straight_curvature_threshold_per_m=(
            straight_curvature_threshold_per_m
        ),
        deployment_mode="automatic",
    )
    vehicle = Vehicle.with_active_aero(
        active_aero_mass_penalty_lb=mass_penalty_lb,
        aero=aero,
    )
    vehicle.battery.initial_state_of_charge = DEFAULT_INITIAL_STATE_OF_CHARGE
    apply_endurance_caps(
        vehicle,
        power_cap_kw=power_cap_kw,
        speed_cap_mph=speed_cap_mph,
    )
    vehicle.validate()
    return vehicle


def run_scenario(
    vehicle: Vehicle,
    track: SpatialTrack,
    *,
    torque_fraction: float,
    laps: int,
    maximum_brake_pressure_psi: float,
    path_speed_tolerance_mps: float,
) -> EventResult:
    profile = UniformPeriodicTorqueParameterization(8).build(
        (torque_fraction,) * 8,
        track,
    )
    return simulate_endurance(
        vehicle,
        track,
        profile,
        config=EnduranceRunConfig(
            laps=laps,
            maximum_brake_pressure_psi=maximum_brake_pressure_psi,
            path_speed_tolerance_mps=path_speed_tolerance_mps,
        ),
    )


def _active_aero_fractions(result: EventResult) -> tuple[float, float]:
    channel = "aero.active_aero.low_drag_mode_active"
    if channel not in result.telemetry or result.telemetry.sample_count == 0:
        return 0.0, 0.0
    active = result.telemetry[channel]
    times_s = result.telemetry["vehicle.time_s"]
    distances_m = result.telemetry["vehicle.distance_m"]
    previous_time_s = 0.0
    previous_distance_m = 0.0
    active_time_s = 0.0
    active_distance_m = 0.0
    for flag, time_s, distance_m in zip(
        active,
        times_s,
        distances_m,
        strict=True,
    ):
        timestep_s = time_s - previous_time_s
        distance_step_m = distance_m - previous_distance_m
        if flag > 0.5:
            active_time_s += timestep_s
            active_distance_m += distance_step_m
        previous_time_s = time_s
        previous_distance_m = distance_m
    time_fraction = active_time_s / times_s[-1] if times_s[-1] > 0.0 else 0.0
    distance_fraction = (
        active_distance_m / distances_m[-1] if distances_m[-1] > 0.0 else 0.0
    )
    return time_fraction, distance_fraction


def result_row(
    name: str,
    vehicle: Vehicle,
    result: EventResult,
) -> dict[str, str | float | int | bool | None]:
    time_fraction, distance_fraction = _active_aero_fractions(result)
    breakdown = result.point_breakdown
    average_lap_time_s = (
        result.elapsed_time_s / result.completed_laps
        if result.completed_laps > 0
        else None
    )
    maximum_speed_mph = max(
        meters_per_second_to_miles_per_hour(speed_mps)
        for speed_mps in result.telemetry["vehicle.speed_mps"]
    )
    maximum_pack_power_kw = max(result.telemetry["battery.power_w"]) / 1_000.0
    maximum_motor_power_kw = (
        max(result.telemetry["motor.mechanical_power_w"]) / 1_000.0
    )
    return {
        "scenario": name,
        "completed": result.completed,
        "failure_reason": result.failure_reason,
        "completed_laps": result.completed_laps,
        "vehicle_mass_lb": vehicle.mass_kg / KILOGRAMS_PER_POUND,
        "endurance_time_s": result.elapsed_time_s,
        "average_lap_time_s": average_lap_time_s,
        "pack_energy_kwh": result.energy_kwh,
        "maximum_vehicle_speed_mph": maximum_speed_mph,
        "maximum_pack_power_kw": maximum_pack_power_kw,
        "maximum_motor_mechanical_power_kw": maximum_motor_power_kw,
        "endurance_points": float(breakdown["endurance_points"]),
        "efficiency_points": float(breakdown["efficiency_points"]),
        "combined_points": result.estimated_points,
        "low_drag_time_fraction": time_fraction,
        "low_drag_distance_fraction": distance_fraction,
    }


def delta_row(
    baseline: dict[str, str | float | int | bool | None],
    active: dict[str, str | float | int | bool | None],
) -> dict[str, float]:
    numeric_keys = (
        "vehicle_mass_lb",
        "endurance_time_s",
        "average_lap_time_s",
        "pack_energy_kwh",
        "maximum_vehicle_speed_mph",
        "maximum_pack_power_kw",
        "maximum_motor_mechanical_power_kw",
        "endurance_points",
        "efficiency_points",
        "combined_points",
    )
    return {
        key: float(active[key]) - float(baseline[key])
        for key in numeric_keys
        if active[key] is not None and baseline[key] is not None
    }


def write_outputs(
    output_dir: Path,
    *,
    assumptions: dict[str, float | int | str],
    rows: list[dict[str, str | float | int | bool | None]],
    deltas: dict[str, float],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "active_aero_comparison.json"
    csv_path = output_dir / "active_aero_comparison.csv"
    json_path.write_text(
        json.dumps(
            {
                "assumptions": assumptions,
                "scenarios": rows,
                "active_minus_baseline": deltas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path.resolve(), csv_path.resolve()


def print_comparison(
    baseline: dict[str, str | float | int | bool | None],
    active: dict[str, str | float | int | bool | None],
    deltas: dict[str, float],
) -> None:
    print("\nEndurance / efficiency active-aero estimate")
    print(f"{'Metric':<28}{'Baseline':>14}{'Active aero':>14}{'Delta':>14}")
    for label, key, unit in (
        ("Vehicle mass", "vehicle_mass_lb", " lb"),
        ("Endurance time", "endurance_time_s", " s"),
        ("Average lap time", "average_lap_time_s", " s"),
        ("Pack energy", "pack_energy_kwh", " kWh"),
        ("Maximum speed", "maximum_vehicle_speed_mph", " mph"),
        ("Maximum pack power", "maximum_pack_power_kw", " kW"),
        (
            "Maximum motor power",
            "maximum_motor_mechanical_power_kw",
            " kW",
        ),
        ("Endurance points", "endurance_points", " pt"),
        ("Efficiency points", "efficiency_points", " pt"),
        ("Combined points", "combined_points", " pt"),
    ):
        baseline_raw = baseline[key]
        active_raw = active[key]
        if baseline_raw is None or active_raw is None or key not in deltas:
            baseline_text = (
                "n/a" if baseline_raw is None else f"{float(baseline_raw):.3f}{unit}"
            )
            active_text = (
                "n/a" if active_raw is None else f"{float(active_raw):.3f}{unit}"
            )
            print(f"{label:<28}{baseline_text:>14}{active_text:>14}{'n/a':>14}")
        else:
            baseline_value = float(baseline_raw)
            active_value = float(active_raw)
            delta_value = deltas[key]
            print(
                f"{label:<28}"
                f"{baseline_value:>13.3f}{unit}"
                f"{active_value:>13.3f}{unit}"
                f"{delta_value:>+13.3f}{unit}"
            )
    for row in (baseline, active):
        if not bool(row["completed"]):
            print(f"{row['scenario']} failure: {row['failure_reason']}")
    print(
        "Low-drag usage: "
        f"{100.0 * float(active['low_drag_distance_fraction']):.1f}% of distance, "
        f"{100.0 * float(active['low_drag_time_fraction']):.1f}% of driving time"
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.torque_fraction <= 1.0:
        raise ValueError("--torque-fraction must be between zero and one")
    if args.cell_length_m <= 0.0:
        raise ValueError("--cell-length-m must be positive")
    if args.laps <= 0:
        raise ValueError("--laps must be positive")
    if args.power_cap_kw <= 0.0:
        raise ValueError("--power-cap-kw must be positive")
    if args.speed_cap_mph <= 0.0:
        raise ValueError("--speed-cap-mph must be positive")
    if args.path_speed_tolerance_mps <= 0.0:
        raise ValueError("--path-speed-tolerance-mps must be positive")
    for name, value in (
        ("--drag-reduction", args.drag_reduction),
        ("--downforce-reduction", args.downforce_reduction),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    if args.mass_penalty_lb < 0.0:
        raise ValueError("--mass-penalty-lb cannot be negative")
    if args.straight_curvature_threshold_per_m < 0.0:
        raise ValueError(
            "--straight-curvature-threshold-per-m cannot be negative"
        )

    track = coarsen_track(
        SpatialTrack.from_csv(args.track.resolve()),
        args.cell_length_m,
    )
    baseline_vehicle = calibrated_vehicle(
        power_cap_kw=args.power_cap_kw,
        speed_cap_mph=args.speed_cap_mph,
    )
    active_vehicle = calibrated_active_aero_vehicle(
        drag_reduction_fraction=args.drag_reduction,
        downforce_reduction_fraction=args.downforce_reduction,
        mass_penalty_lb=args.mass_penalty_lb,
        straight_curvature_threshold_per_m=(
            args.straight_curvature_threshold_per_m
        ),
        power_cap_kw=args.power_cap_kw,
        speed_cap_mph=args.speed_cap_mph,
    )
    baseline_result = run_scenario(
        baseline_vehicle,
        track,
        torque_fraction=args.torque_fraction,
        laps=args.laps,
        maximum_brake_pressure_psi=args.maximum_brake_pressure_psi,
        path_speed_tolerance_mps=args.path_speed_tolerance_mps,
    )
    active_result = run_scenario(
        active_vehicle,
        track,
        torque_fraction=args.torque_fraction,
        laps=args.laps,
        maximum_brake_pressure_psi=args.maximum_brake_pressure_psi,
        path_speed_tolerance_mps=args.path_speed_tolerance_mps,
    )
    rows = [
        result_row("baseline", baseline_vehicle, baseline_result),
        result_row("active_aero", active_vehicle, active_result),
    ]
    deltas = delta_row(rows[0], rows[1])
    print_comparison(rows[0], rows[1], deltas)
    assumptions: dict[str, float | int | str] = {
        "track": str(args.track.resolve()),
        "laps": args.laps,
        "torque_fraction": args.torque_fraction,
        "pack_and_motor_power_cap_kw": args.power_cap_kw,
        "vehicle_speed_cap_mph": args.speed_cap_mph,
        "cell_length_m": args.cell_length_m,
        "path_speed_tolerance_mps": args.path_speed_tolerance_mps,
        "drag_reduction_fraction": args.drag_reduction,
        "downforce_reduction_fraction": args.downforce_reduction,
        "active_aero_mass_penalty_lb": args.mass_penalty_lb,
        "straight_curvature_threshold_per_m": (
            args.straight_curvature_threshold_per_m
        ),
        "deployment_mode": "automatic",
    }
    json_path, csv_path = write_outputs(
        args.output_dir.resolve(),
        assumptions=assumptions,
        rows=rows,
        deltas=deltas,
    )
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
