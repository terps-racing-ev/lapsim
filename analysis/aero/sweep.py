"""Sweep aerodynamic lift and drag coefficients and plot modeled event points.

Each grid point solves the canonical endurance track at several global speed
caps and keeps the pace with the highest projected score. Lap time and energy
are projected over 22 identical laps, then scored with the current Michigan
2026 endurance and efficiency preset. ``--real-car`` applies the calibrated
replay values for drivetrain efficiency, tire friction, and cornering drag.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from lapsim import (
    FSAE_2026_MI6_SCORING,
    LapTimeSolver,
    SpatialTrack,
    SpeedLimitMap,
    SpeedLimitSolver,
    Track,
)
from vehicle_model import Vehicle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACK = REPOSITORY_ROOT / "analysis/data/track/gnss_imu_endurance_track.csv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/aero_sweep"
EVENT_LAPS = FSAE_2026_MI6_SCORING.event_laps
DEFAULT_FRONTAL_AREA_M2 = Vehicle().aero.frontal_area_m2
LEGACY_FRONTAL_AREA_M2 = 0.65799997432
REFERENCE_AREA_SCALE = LEGACY_FRONTAL_AREA_M2 / DEFAULT_FRONTAL_AREA_M2
MPH_TO_MPS = 0.44704


@dataclass(frozen=True, slots=True)
class ProjectedRun:
    completed_laps: int
    driving_time_s: float
    pack_energy_kwh: float
    failure_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frontal-area-m2", type=float, default=DEFAULT_FRONTAL_AREA_M2)
    parser.add_argument("--lift-min", type=float, default=-4.0)
    parser.add_argument("--lift-max", type=float, default=-2.0)
    parser.add_argument("--drag-min", type=float, default=1.0)
    parser.add_argument("--drag-max", type=float, default=1.8)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--cell-length-m", type=float, default=2.0)
    parser.add_argument("--starting-speed-mps", type=float, default=10.0)
    parser.add_argument("--pace-min-mps", type=float, default=14.0)
    parser.add_argument("--pace-max-mps", type=float, default=36.0)
    parser.add_argument("--pace-count", type=int, default=24)
    parser.add_argument("--fixed-speed-cap-mph", type=float)
    parser.add_argument("--motor-power-kw", type=float)
    parser.add_argument("--event-energy-ceiling-kwh", type=float)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--real-car",
        action="store_true",
        help="Use calibrated replay drivetrain, tire, and cornering-loss values.",
    )
    parser.add_argument("--motor-to-wheel-efficiency", type=float)
    parser.add_argument("--constant-tire-mu", type=float)
    parser.add_argument("--cornering-drag-coefficient", type=float, default=0.0)
    parser.add_argument(
        "--metric",
        choices=("combined", "endurance", "efficiency"),
        default="combined",
    )
    return parser.parse_args()


def coarsen_track(track: SpatialTrack, target_cell_length_m: float) -> Track:
    cell_lengths: list[float] = []
    curvatures: list[float] = []
    group_length_m = 0.0
    group_turn_rad = 0.0

    for length_m, curvature_per_m in zip(
        track.cell_length_m, track.curvature_per_m, strict=True
    ):
        group_length_m += length_m
        group_turn_rad += curvature_per_m * length_m
        if group_length_m >= target_cell_length_m:
            cell_lengths.append(group_length_m)
            curvatures.append(group_turn_rad / group_length_m)
            group_length_m = 0.0
            group_turn_rad = 0.0

    if group_length_m:
        cell_lengths.append(group_length_m)
        curvatures.append(group_turn_rad / group_length_m)

    return SpatialTrack.from_cells(
        cell_length_m=cell_lengths,
        curvature_per_m=curvatures,
    ).to_track()


def capped_map(speed_map: SpeedLimitMap, speed_cap_mps: float) -> SpeedLimitMap:
    return SpeedLimitMap(
        distance_m=speed_map.distance_m,
        x_m=speed_map.x_m,
        y_m=speed_map.y_m,
        speed_limit_mps=tuple(
            min(speed_mps, speed_cap_mps)
            for speed_mps in speed_map.speed_limit_mps
        ),
        cell_length_m=speed_map.cell_length_m,
        curvature_per_m=speed_map.curvature_per_m,
    )


def solve_point(
    track: Track,
    frontal_area_m2: float,
    lift_coefficient: float,
    drag_coefficient: float,
    starting_speed_mps: float,
    pace_values_mps: np.ndarray,
    metric: str,
    cell_length_m: float,
    motor_power_w: float | None,
    motor_to_wheel_efficiency: float | None,
    constant_tire_mu: float | None,
    cornering_drag_coefficient: float,
    scoring_model=FSAE_2026_MI6_SCORING,
) -> dict[str, float]:
    vehicle = Vehicle()
    vehicle.aero.frontal_area_m2 = frontal_area_m2
    vehicle.aero.lift_coefficient = lift_coefficient
    vehicle.aero.drag_coefficient = drag_coefficient
    if motor_power_w is not None:
        vehicle.drivetrain.motor.peak_power_w = motor_power_w
        vehicle.drivetrain.motor.continuous_power_w = min(
            vehicle.drivetrain.motor.continuous_power_w,
            motor_power_w,
        )
    if motor_to_wheel_efficiency is not None:
        vehicle.drivetrain.chain_drive.efficiency = motor_to_wheel_efficiency
    if constant_tire_mu is not None:
        vehicle.tire.constant_friction_coefficient = constant_tire_mu
    vehicle.cornering_drag_coefficient = cornering_drag_coefficient
    vehicle.validate()

    physical_limits = SpeedLimitSolver(vehicle, max_step_m=cell_length_m).solve(
        track
    )
    candidates: list[dict[str, float]] = []
    for pace_mps in pace_values_mps:
        lap = LapTimeSolver(vehicle).solve(
            capped_map(physical_limits, pace_mps),
            min(starting_speed_mps, pace_mps),
        )
        run = ProjectedRun(
            completed_laps=EVENT_LAPS,
            driving_time_s=EVENT_LAPS * lap.lap_time_s,
            pack_energy_kwh=EVENT_LAPS * lap.telemetry.total_energy_kwh,
        )
        score = scoring_model.score(run)
        candidates.append(
            {
                "lift_coefficient": lift_coefficient,
                "drag_coefficient": drag_coefficient,
                "frontal_area_m2": frontal_area_m2,
                "drag_area_m2": vehicle.aero.drag_area_m2,
                "downforce_area_m2": vehicle.aero.downforce_area_m2,
                "motor_peak_power_w": vehicle.drivetrain.motor.peak_power_w,
                "speed_cap_mps": pace_mps,
                "lap_time_s": lap.lap_time_s,
                "lap_energy_kwh": lap.telemetry.total_energy_kwh,
                "endurance_points": score.endurance_points,
                "efficiency_points": score.efficiency_points,
                "combined_points": score.combined_points,
            }
        )
    return max(
        candidates,
        key=lambda row: (
            row[f"{metric}_points"],
            -row["lap_energy_kwh"],
            -row["lap_time_s"],
        ),
    )


def solve_point_task(arguments: tuple[object, ...]) -> dict[str, float]:
    """Process-pool adapter for one independent aero grid point."""

    return solve_point(*arguments)


def write_results(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(
    path: Path,
    rows: list[dict[str, float]],
    lift_values: np.ndarray,
    drag_values: np.ndarray,
    metric: str,
    *,
    title: str | None = None,
) -> dict[str, float]:
    metric_key = f"{metric}_points"
    values = np.asarray([row[metric_key] for row in rows]).reshape(
        len(lift_values), len(drag_values)
    )
    downforce_values = -lift_values[::-1]
    display_values = values[::-1, :]
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if np.isclose(value_min, value_max):
        color_min = value_min - 0.5
        color_max = value_max + 0.5
    else:
        color_min = value_min
        color_max = value_max
    best_row = max(rows, key=lambda row: row[metric_key])
    baseline = Vehicle().aero

    figure, axes = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)
    image = axes.pcolormesh(
        drag_values,
        downforce_values,
        display_values,
        cmap="viridis",
        norm=Normalize(vmin=color_min, vmax=color_max),
        shading="nearest",
    )
    if np.ptp(values) > 1e-9:
        contours = axes.contour(
            drag_values,
            downforce_values,
            display_values,
            colors="white",
            alpha=0.42,
            linewidths=0.8,
        )
        axes.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    axes.scatter(
        best_row["drag_coefficient"],
        -best_row["lift_coefficient"],
        marker="*",
        s=190,
        color="white",
        edgecolor="black",
        linewidth=0.9,
        label=f"Best: {best_row[metric_key]:.2f} pt",
        clip_on=False,
        zorder=3,
    )
    axes.scatter(
        baseline.drag_coefficient,
        -baseline.lift_coefficient,
        marker="o",
        s=60,
        facecolor="none",
        edgecolor="white",
        linewidth=1.6,
        label="Current baseline",
        clip_on=False,
        zorder=3,
    )
    axes.set(
        title=title or f"Modeled {metric.title()} Points — Aero Sweep",
        xlabel="Drag coefficient, $C_d$",
        ylabel="Downforce coefficient magnitude, $C_L$",
    )
    axes.set_xlim(drag_values[0], drag_values[-1])
    axes.set_ylim(downforce_values[0], downforce_values[-1])
    axes.legend(loc="upper right", frameon=True)
    colorbar = figure.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_label("Points")
    colorbar.set_ticks(
        [value_min]
        if np.isclose(value_min, value_max)
        else np.linspace(value_min, value_max, 6)
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return best_row


def main() -> None:
    args = parse_args()
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2")
    if args.frontal_area_m2 <= 0:
        raise ValueError("--frontal-area-m2 must be positive")
    if args.cell_length_m <= 0:
        raise ValueError("--cell-length-m must be positive")
    if args.fixed_speed_cap_mph is None and args.pace_count < 2:
        raise ValueError("--pace-count must be at least 2")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.lift_min >= args.lift_max or args.drag_min >= args.drag_max:
        raise ValueError("Sweep minimums must be smaller than maximums")
    if args.fixed_speed_cap_mph is None and (
        args.pace_min_mps <= 0 or args.pace_min_mps >= args.pace_max_mps
    ):
        raise ValueError("Pace bounds must be positive and increasing")
    if args.fixed_speed_cap_mph is not None and args.fixed_speed_cap_mph <= 0:
        raise ValueError("--fixed-speed-cap-mph must be positive")
    if args.motor_power_kw is not None and args.motor_power_kw <= 0:
        raise ValueError("--motor-power-kw must be positive")
    if (
        args.event_energy_ceiling_kwh is not None
        and args.event_energy_ceiling_kwh <= 0
    ):
        raise ValueError("--event-energy-ceiling-kwh must be positive")
    if args.motor_to_wheel_efficiency is not None and not (
        0.0 < args.motor_to_wheel_efficiency <= 1.0
    ):
        raise ValueError("--motor-to-wheel-efficiency must be in (0, 1]")
    if args.constant_tire_mu is not None and args.constant_tire_mu <= 0:
        raise ValueError("--constant-tire-mu must be positive")
    if args.cornering_drag_coefficient < 0:
        raise ValueError("--cornering-drag-coefficient cannot be negative")

    motor_to_wheel_efficiency = args.motor_to_wheel_efficiency
    constant_tire_mu = args.constant_tire_mu
    cornering_drag_coefficient = args.cornering_drag_coefficient
    if args.real_car:
        motor_to_wheel_efficiency = (
            0.80
            if motor_to_wheel_efficiency is None
            else motor_to_wheel_efficiency
        )
        if cornering_drag_coefficient == 0.0:
            cornering_drag_coefficient = 0.036

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_track = SpatialTrack.from_csv(args.track.resolve())
    track = coarsen_track(spatial_track, args.cell_length_m)
    lift_values = np.linspace(args.lift_min, args.lift_max, args.grid_size)
    drag_values = np.linspace(args.drag_min, args.drag_max, args.grid_size)
    fixed_speed_cap_mps = (
        None
        if args.fixed_speed_cap_mph is None
        else args.fixed_speed_cap_mph * MPH_TO_MPS
    )
    pace_values_mps = (
        np.linspace(args.pace_min_mps, args.pace_max_mps, args.pace_count)
        if fixed_speed_cap_mps is None
        else np.asarray([fixed_speed_cap_mps])
    )
    motor_power_w = (
        None if args.motor_power_kw is None else args.motor_power_kw * 1_000.0
    )
    scoring_model = FSAE_2026_MI6_SCORING
    if args.event_energy_ceiling_kwh is not None:
        scoring_model = replace(
            scoring_model,
            maximum_adjusted_energy_per_lap_kg=(
                args.event_energy_ceiling_kwh
                * scoring_model.electric_energy_to_adjusted_kg_per_kwh
                / EVENT_LAPS
            ),
        )
    tasks = [
        (
            track,
            args.frontal_area_m2,
            lift,
            drag,
            args.starting_speed_mps,
            pace_values_mps,
            args.metric,
            args.cell_length_m,
            motor_power_w,
            motor_to_wheel_efficiency,
            constant_tire_mu,
            cornering_drag_coefficient,
            scoring_model,
        )
        for lift in lift_values
        for drag in drag_values
    ]
    if args.workers == 1:
        rows = [solve_point_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(solve_point_task, tasks, chunksize=1))

    csv_path = output_dir / "aero_sweep.csv"
    plot_path = output_dir / f"aero_{args.metric}_points_heatmap.png"
    write_results(csv_path, rows)
    best = plot_heatmap(
        plot_path,
        rows,
        lift_values,
        drag_values,
        args.metric,
        title=(
            f"Endurance + Efficiency Points — Aero Sweep "
            f"($A={args.frontal_area_m2:.9f}\\,m^2$)"
            + (
                f" — {args.motor_power_kw:g} kW / "
                f"{args.fixed_speed_cap_mph:g} mph cap"
                if args.motor_power_kw is not None
                and args.fixed_speed_cap_mph is not None
                else ""
            )
            + (
                f" — {args.event_energy_ceiling_kwh:g} kWh ceiling"
                if args.event_energy_ceiling_kwh is not None
                else ""
            )
            if args.metric == "combined"
            else None
        ),
    )

    scenario_vehicle = Vehicle()
    scenario_vehicle.aero.frontal_area_m2 = args.frontal_area_m2
    if motor_power_w is not None:
        scenario_vehicle.drivetrain.motor.peak_power_w = motor_power_w
        scenario_vehicle.drivetrain.motor.continuous_power_w = min(
            scenario_vehicle.drivetrain.motor.continuous_power_w,
            motor_power_w,
        )
    if motor_to_wheel_efficiency is not None:
        scenario_vehicle.drivetrain.chain_drive.efficiency = motor_to_wheel_efficiency
    if constant_tire_mu is not None:
        scenario_vehicle.tire.constant_friction_coefficient = constant_tire_mu
    scenario_vehicle.cornering_drag_coefficient = cornering_drag_coefficient
    scenario_vehicle.validate()
    metadata_path = output_dir / "aero_sweep_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "track": str(args.track.resolve()),
                "track_cell_length_m": args.cell_length_m,
                "frontal_area_m2": args.frontal_area_m2,
                "legacy_reference_area_m2": LEGACY_FRONTAL_AREA_M2,
                "coefficient_reference_area_scale": REFERENCE_AREA_SCALE,
                "lift_coefficient_range": [args.lift_min, args.lift_max],
                "downforce_coefficient_magnitude_range": [
                    -args.lift_max,
                    -args.lift_min,
                ],
                "drag_coefficient_range": [args.drag_min, args.drag_max],
                "grid_size": args.grid_size,
                "optimization": {
                    "method": (
                        "fixed speed cap at every aero grid point"
                        if fixed_speed_cap_mps is not None
                        else "exhaustive global speed-cap search at every aero grid point"
                    ),
                    "objective": f"maximize {args.metric} points",
                    "pace_range_mps": [
                        float(pace_values_mps[0]),
                        float(pace_values_mps[-1]),
                    ],
                    "pace_count": len(pace_values_mps),
                    "fixed_speed_cap_mph": args.fixed_speed_cap_mph,
                    "fixed_speed_cap_mps": fixed_speed_cap_mps,
                    "workers": args.workers,
                },
                "projection": {
                    "laps": EVENT_LAPS,
                    "method": "solve one physical lap, project identical time and energy over the event",
                },
                "scoring": asdict(scoring_model),
                "scoring_override": {
                    "event_energy_ceiling_kwh": args.event_energy_ceiling_kwh,
                    "per_lap_energy_ceiling_kwh": (
                        None
                        if args.event_energy_ceiling_kwh is None
                        else args.event_energy_ceiling_kwh / EVENT_LAPS
                    ),
                    "official_preset_modified": False,
                },
                "vehicle": {
                    "mass_kg": scenario_vehicle.mass_kg,
                    "chain_drive_ratio": scenario_vehicle.drivetrain.chain_drive.ratio,
                    "chain_drive_efficiency": scenario_vehicle.drivetrain.chain_drive.efficiency,
                    "motor_peak_power_w": scenario_vehicle.drivetrain.motor.peak_power_w,
                    "motor_continuous_power_w": scenario_vehicle.drivetrain.motor.continuous_power_w,
                    "motor_efficiency": scenario_vehicle.drivetrain.motor.efficiency,
                    "inverter_efficiency": scenario_vehicle.drivetrain.inverter.efficiency,
                    "tire_model": (
                        "load-sensitive table"
                        if constant_tire_mu is None
                        else f"constant mu {constant_tire_mu}"
                    ),
                    "cornering_drag_coefficient": scenario_vehicle.cornering_drag_coefficient,
                    "rolling_resistance_coefficient": scenario_vehicle.rolling_resistance_coefficient,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metric_key = f"{args.metric}_points"
    print(f"Best {args.metric} score: {best[metric_key]:.3f} points")
    print(f"Cl={best['lift_coefficient']:.4f}, Cd={best['drag_coefficient']:.4f}")
    print(f"Speed cap={best['speed_cap_mps']:.3f} m/s")
    print(
        "Scenario: "
        f"area={args.frontal_area_m2} m^2, "
        f"motor power={scenario_vehicle.drivetrain.motor.peak_power_w / 1_000.0:g} kW, "
        f"chain efficiency={motor_to_wheel_efficiency}, "
        "tire model="
        f"{'load-sensitive table' if constant_tire_mu is None else f'constant mu {constant_tire_mu}'}, "
        f"cornering drag={cornering_drag_coefficient}"
    )
    print(
        f"Lap time={best['lap_time_s']:.3f} s, "
        f"energy={best['lap_energy_kwh']:.4f} kWh"
    )
    print(f"CSV: {csv_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
