"""Sweep longitudinal CG position and height against modeled dynamic-event score.

The sweep reports acceleration, skidpad, predicted autocross, endurance, and
efficiency points.  Endurance pace is re-optimized at every CG point over a
small global speed-cap grid.  Longitudinal position is expressed as CG distance
forward of the rear axle; internally this maps directly to static front weight
fraction through ``x_cg / wheelbase``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import json
from math import pi
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.aero.sweep import ProjectedRun, capped_map, coarsen_track  # noqa: E402
from lapsim import (  # noqa: E402
    ConstantControlsProfile,
    Controls,
    Curve,
    FSAE_2026_MI6_SCORING,
    LapTimeSolver,
    SkidpadConfig,
    SpatialTrack,
    SpeedLimitSolver,
    Track,
    simulate_acceleration,
    simulate_skidpad,
)
from vehicle_model import Vehicle  # noqa: E402


DEFAULT_TRACK = ROOT / "analysis/data/track/gnss_imu_endurance_track.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/cg_sweep"
INCHES_PER_METER = 1.0 / 0.0254
EVENT_LAPS = FSAE_2026_MI6_SCORING.event_laps

# 2026 Michigan EV least-squares correlation documented in analysis/autox_corr.
AUTOX_INTERCEPT_POINTS = 24.636470
AUTOX_ACCELERATION_WEIGHT = 0.133094
AUTOX_SKIDPAD_WEIGHT = 0.743173


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--longitudinal-min-in", type=float, default=24.0)
    parser.add_argument("--longitudinal-max-in", type=float, default=36.0)
    parser.add_argument("--height-min-in", type=float, default=6.0)
    parser.add_argument("--height-max-in", type=float, default=16.0)
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--track-cell-length-m", type=float, default=3.0)
    parser.add_argument("--pace-min-mps", type=float, default=14.0)
    parser.add_argument("--pace-max-mps", type=float, default=36.0)
    parser.add_argument("--pace-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def acceleration_track(cell_length_m: float = 1.0) -> SpatialTrack:
    cell_count = round(75.0 / cell_length_m)
    return SpatialTrack.from_cells(
        cell_length_m=(75.0 / cell_count,) * cell_count,
        curvature_per_m=(0.0,) * cell_count,
        closed=False,
    )


def skidpad_track(cell_length_m: float = 0.5) -> SpatialTrack:
    return SpatialTrack.from_track(
        Track.from_segments([Curve(radius_m=9.125, span_rad=2.0 * pi)]),
        maximum_cell_length_m=cell_length_m,
        closed=True,
    )


def predicted_autocross_points(
    acceleration_points: float,
    skidpad_points: float,
) -> float:
    prediction = (
        AUTOX_INTERCEPT_POINTS
        + 125.0 * AUTOX_ACCELERATION_WEIGHT * acceleration_points / 100.0
        + 125.0 * AUTOX_SKIDPAD_WEIGHT * skidpad_points / 75.0
    )
    return float(np.clip(prediction, 0.0, 125.0))


def configured_vehicle(
    longitudinal_position_in: float,
    cg_height_in: float,
) -> Vehicle:
    vehicle = Vehicle()
    longitudinal_position_m = longitudinal_position_in / INCHES_PER_METER
    vehicle.chassis.static_front_weight_fraction = (
        longitudinal_position_m / vehicle.chassis.wheelbase_m
    )
    vehicle.chassis.cg_height_m = cg_height_in / INCHES_PER_METER
    vehicle.validate()
    return vehicle


def solve_point(
    track,
    acceleration_course: SpatialTrack,
    skidpad_course: SpatialTrack,
    longitudinal_position_in: float,
    cg_height_in: float,
    pace_values_mps: np.ndarray,
    track_cell_length_m: float,
) -> dict[str, float]:
    acceleration = simulate_acceleration(
        configured_vehicle(longitudinal_position_in, cg_height_in),
        acceleration_course,
        ConstantControlsProfile(Controls(motor_torque_request_nm=230.0)),
    )
    skidpad = None
    skidpad_error: RuntimeError | None = None
    for starting_speed_mps in (9.0, 8.0, 7.0, 6.0):
        try:
            skidpad = simulate_skidpad(
                configured_vehicle(longitudinal_position_in, cg_height_in),
                skidpad_course,
                ConstantControlsProfile(Controls(motor_torque_request_nm=5.0)),
                config=SkidpadConfig(starting_speed_mps=starting_speed_mps),
            )
            break
        except RuntimeError as error:
            skidpad_error = error
    if skidpad is None:
        raise RuntimeError(
            "skidpad solve failed for CG "
            f"x={longitudinal_position_in:.3f} in, z={cg_height_in:.3f} in"
        ) from skidpad_error

    vehicle = configured_vehicle(longitudinal_position_in, cg_height_in)
    physical_limits = SpeedLimitSolver(
        vehicle,
        max_step_m=track_cell_length_m,
    ).solve(track)
    endurance_candidates: list[dict[str, float]] = []
    for pace_mps in pace_values_mps:
        lap = LapTimeSolver(vehicle).solve(
            capped_map(physical_limits, float(pace_mps)),
            min(10.0, float(pace_mps)),
        )
        score = FSAE_2026_MI6_SCORING.score(
            ProjectedRun(
                completed_laps=EVENT_LAPS,
                driving_time_s=EVENT_LAPS * lap.lap_time_s,
                pack_energy_kwh=EVENT_LAPS * lap.telemetry.total_energy_kwh,
            )
        )
        endurance_candidates.append(
            {
                "speed_cap_mps": float(pace_mps),
                "lap_time_s": lap.lap_time_s,
                "lap_energy_kwh": lap.telemetry.total_energy_kwh,
                "endurance_points": score.endurance_points,
                "efficiency_points": score.efficiency_points,
                "endurance_efficiency_points": score.combined_points,
            }
        )
    endurance = max(
        endurance_candidates,
        key=lambda row: (
            row["endurance_efficiency_points"],
            -row["lap_energy_kwh"],
            -row["lap_time_s"],
        ),
    )

    acceleration_points = acceleration.estimated_points
    skidpad_points = skidpad.estimated_points
    autocross_points = predicted_autocross_points(
        acceleration_points,
        skidpad_points,
    )
    total_dynamic_points = (
        acceleration_points
        + skidpad_points
        + autocross_points
        + endurance["endurance_efficiency_points"]
    )
    return {
        "longitudinal_position_from_rear_axle_in": longitudinal_position_in,
        "static_front_weight_percent": (
            vehicle.chassis.static_front_weight_fraction * 100.0
        ),
        "cg_height_in": cg_height_in,
        "acceleration_time_s": float(acceleration.scoring_time_s or np.nan),
        "acceleration_points": acceleration_points,
        "skidpad_time_s": float(skidpad.scoring_time_s or np.nan),
        "skidpad_points": skidpad_points,
        "predicted_autocross_points": autocross_points,
        **endurance,
        "total_dynamic_points": total_dynamic_points,
    }


def solve_point_task(arguments: tuple[object, ...]) -> dict[str, float]:
    return solve_point(*arguments)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(
    path: Path,
    rows: list[dict[str, float]],
    longitudinal_values_in: np.ndarray,
    height_values_in: np.ndarray,
    baseline: dict[str, float],
    wheelbase_in: float,
) -> dict[str, float]:
    values = np.asarray([row["total_dynamic_points"] for row in rows]).reshape(
        len(height_values_in), len(longitudinal_values_in)
    )
    best = max(rows, key=lambda row: row["total_dynamic_points"])
    value_min = float(np.min(values))
    value_max = float(np.max(values))

    figure = plt.figure(figsize=(12.2, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(4.3, 1.4))
    axis = figure.add_subplot(grid[0, 0])
    breakdown_axis = figure.add_subplot(grid[0, 1])
    image = axis.pcolormesh(
        longitudinal_values_in,
        height_values_in,
        values,
        cmap="viridis",
        norm=Normalize(vmin=value_min, vmax=value_max),
        shading="nearest",
    )
    contour_levels = np.linspace(value_min, value_max, 7)
    contours = axis.contour(
        longitudinal_values_in,
        height_values_in,
        values,
        levels=contour_levels,
        colors="white",
        alpha=0.48,
        linewidths=0.8,
    )
    axis.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    axis.scatter(
        best["longitudinal_position_from_rear_axle_in"],
        best["cg_height_in"],
        marker="*",
        s=240,
        color="#ffdd57",
        edgecolor="black",
        linewidth=0.9,
        label=f"Best grid point: {best['total_dynamic_points']:.1f} pt",
        zorder=4,
    )
    axis.scatter(
        baseline["longitudinal_position_from_rear_axle_in"],
        baseline["cg_height_in"],
        marker="o",
        s=82,
        facecolor="none",
        edgecolor="white",
        linewidth=2.0,
        label=f"Current baseline: {baseline['total_dynamic_points']:.1f} pt",
        zorder=4,
    )
    longitudinal_step_in = float(np.diff(longitudinal_values_in).mean())
    height_step_in = float(np.diff(height_values_in).mean())
    axis.set(
        title="Modeled Formula SAE Dynamic Score — CG Sweep",
        xlabel="Longitudinal CG position forward of rear axle [in]",
        ylabel="CG height [in]",
        xlim=(
            longitudinal_values_in[0] - 0.5 * longitudinal_step_in,
            longitudinal_values_in[-1] + 0.5 * longitudinal_step_in,
        ),
        ylim=(
            height_values_in[0] - 0.5 * height_step_in,
            height_values_in[-1] + 0.5 * height_step_in,
        ),
    )
    axis.legend(loc="upper right", framealpha=0.92)
    axis.grid(color="white", alpha=0.10, linewidth=0.6)
    top_axis = axis.secondary_xaxis(
        "top",
        functions=(
            lambda position_in: 100.0 * position_in / wheelbase_in,
            lambda front_percent: wheelbase_in * front_percent / 100.0,
        ),
    )
    top_axis.set_xlabel("Static front weight [%]")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Total modeled dynamic points")

    labels = ["Acceleration", "Skidpad", "Pred. autocross", "Endurance", "Efficiency"]
    keys = [
        "acceleration_points",
        "skidpad_points",
        "predicted_autocross_points",
        "endurance_points",
        "efficiency_points",
    ]
    baseline_values = [baseline[key] for key in keys]
    best_values = [best[key] for key in keys]
    y = np.arange(len(labels))
    bar_height = 0.36
    breakdown_axis.barh(
        y + bar_height / 2,
        baseline_values,
        height=bar_height,
        color="#9aa0a6",
        label="Baseline",
    )
    breakdown_axis.barh(
        y - bar_height / 2,
        best_values,
        height=bar_height,
        color="#2a9d8f",
        label="Best grid point",
    )
    breakdown_axis.set_yticks(y, labels)
    breakdown_axis.invert_yaxis()
    breakdown_axis.set_xlabel("Points")
    breakdown_axis.set_title("Event breakdown")
    breakdown_axis.grid(axis="x", alpha=0.25)
    breakdown_axis.legend(loc="lower right", fontsize=8)
    breakdown_axis.spines[["top", "right"]].set_visible(False)

    delta = best["total_dynamic_points"] - baseline["total_dynamic_points"]
    figure.suptitle(
        f"Best: {best['longitudinal_position_from_rear_axle_in']:.1f} in from rear, "
        f"{best['cg_height_in']:.1f} in high  |  "
        f"{best['static_front_weight_percent']:.1f}% front  |  "
        f"Δ {delta:+.1f} points vs baseline",
        fontsize=11,
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return best


def plot_position_sweep(
    path: Path,
    rows: list[dict[str, float]],
    longitudinal_values_in: np.ndarray,
    height_values_in: np.ndarray,
    baseline: dict[str, float],
    wheelbase_in: float,
) -> None:
    """Plot longitudinal CG sensitivity as one curve per fixed CG height."""

    figure, axis = plt.subplots(figsize=(11.2, 7.0), constrained_layout=True)
    color_norm = Normalize(
        vmin=float(height_values_in[0]),
        vmax=float(height_values_in[-1]),
    )
    colormap = plt.get_cmap("viridis")
    for height_in in height_values_in:
        height_rows = sorted(
            (
                row
                for row in rows
                if np.isclose(row["cg_height_in"], height_in)
            ),
            key=lambda row: row["longitudinal_position_from_rear_axle_in"],
        )
        is_reference_height = any(
            np.isclose(height_in, reference_height_in)
            for reference_height_in in (
                height_values_in[0],
                baseline["cg_height_in"],
                height_values_in[-1],
            )
        )
        axis.plot(
            [row["longitudinal_position_from_rear_axle_in"] for row in height_rows],
            [row["total_dynamic_points"] for row in height_rows],
            color=colormap(color_norm(float(height_in))),
            linewidth=2.5 if is_reference_height else 1.35,
            alpha=1.0 if is_reference_height else 0.72,
            marker="o",
            markersize=4.2 if is_reference_height else 2.7,
            markeredgewidth=0.0,
            zorder=2 if is_reference_height else 1,
        )

    best = max(rows, key=lambda row: row["total_dynamic_points"])
    axis.scatter(
        best["longitudinal_position_from_rear_axle_in"],
        best["total_dynamic_points"],
        marker="*",
        s=250,
        color="#ffdd57",
        edgecolor="black",
        linewidth=1.0,
        label=(
            f"Best: {best['longitudinal_position_from_rear_axle_in']:.1f} in, "
            f"{best['cg_height_in']:.0f} in high — "
            f"{best['total_dynamic_points']:.1f} pt"
        ),
        zorder=5,
    )
    axis.scatter(
        baseline["longitudinal_position_from_rear_axle_in"],
        baseline["total_dynamic_points"],
        marker="o",
        s=95,
        facecolor="white",
        edgecolor="black",
        linewidth=1.8,
        label=(
            f"Current: {baseline['longitudinal_position_from_rear_axle_in']:.2f} in, "
            f"{baseline['cg_height_in']:.0f} in high — "
            f"{baseline['total_dynamic_points']:.1f} pt"
        ),
        zorder=5,
    )
    longitudinal_step_in = float(np.diff(longitudinal_values_in).mean())
    axis.set(
        title="Modeled Dynamic Score vs. Longitudinal CG Position",
        xlabel="Longitudinal CG position forward of rear axle [in]",
        ylabel="Total modeled dynamic points",
        xlim=(
            longitudinal_values_in[0] - 0.25 * longitudinal_step_in,
            longitudinal_values_in[-1] + 0.25 * longitudinal_step_in,
        ),
    )
    axis.grid(alpha=0.24, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower left", framealpha=0.94)
    top_axis = axis.secondary_xaxis(
        "top",
        functions=(
            lambda position_in: 100.0 * position_in / wheelbase_in,
            lambda front_percent: wheelbase_in * front_percent / 100.0,
        ),
    )
    top_axis.set_xlabel("Static front weight [%]")
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=colormap),
        ax=axis,
        pad=0.02,
    )
    colorbar.set_label("CG height [in]")
    colorbar.set_ticks(height_values_in)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2")
    if args.longitudinal_min_in >= args.longitudinal_max_in:
        raise ValueError("longitudinal sweep minimum must be below maximum")
    if args.height_min_in <= 0 or args.height_min_in >= args.height_max_in:
        raise ValueError("height bounds must be positive and increasing")
    if args.track_cell_length_m <= 0:
        raise ValueError("--track-cell-length-m must be positive")
    if args.pace_count < 2 or args.pace_min_mps >= args.pace_max_mps:
        raise ValueError("pace sweep bounds/count are invalid")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    track = coarsen_track(
        SpatialTrack.from_csv(args.track.resolve()),
        args.track_cell_length_m,
    )
    acceleration_course = acceleration_track()
    skidpad_course = skidpad_track()
    longitudinal_values_in = np.linspace(
        args.longitudinal_min_in,
        args.longitudinal_max_in,
        args.grid_size,
    )
    height_values_in = np.linspace(
        args.height_min_in,
        args.height_max_in,
        args.grid_size,
    )
    pace_values_mps = np.linspace(
        args.pace_min_mps,
        args.pace_max_mps,
        args.pace_count,
    )
    tasks = [
        (
            track,
            acceleration_course,
            skidpad_course,
            float(longitudinal_position_in),
            float(height_in),
            pace_values_mps,
            args.track_cell_length_m,
        )
        for height_in in height_values_in
        for longitudinal_position_in in longitudinal_values_in
    ]
    if args.workers == 1:
        rows = [solve_point_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(solve_point_task, tasks, chunksize=1))

    baseline_vehicle = Vehicle()
    baseline_position_in = (
        baseline_vehicle.chassis.rear_axle_distance_from_cg_m * INCHES_PER_METER
    )
    baseline_height_in = baseline_vehicle.chassis.cg_height_m * INCHES_PER_METER
    baseline = solve_point(
        track,
        acceleration_course,
        skidpad_course,
        baseline_position_in,
        baseline_height_in,
        pace_values_mps,
        args.track_cell_length_m,
    )

    csv_path = output_dir / "cg_score_sweep.csv"
    plot_path = output_dir / "cg_total_dynamic_score.png"
    position_plot_path = output_dir / "cg_position_sweep_by_height.png"
    metadata_path = output_dir / "cg_score_sweep_metadata.json"
    write_csv(csv_path, rows)
    best = plot_sweep(
        plot_path,
        rows,
        longitudinal_values_in,
        height_values_in,
        baseline,
        baseline_vehicle.chassis.wheelbase_m * INCHES_PER_METER,
    )
    plot_position_sweep(
        position_plot_path,
        rows,
        longitudinal_values_in,
        height_values_in,
        baseline,
        baseline_vehicle.chassis.wheelbase_m * INCHES_PER_METER,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "track": str(args.track.resolve()),
                "grid_size": args.grid_size,
                "longitudinal_position_from_rear_axle_in": [
                    args.longitudinal_min_in,
                    args.longitudinal_max_in,
                ],
                "cg_height_in": [args.height_min_in, args.height_max_in],
                "track_cell_length_m": args.track_cell_length_m,
                "pace_values_mps": pace_values_mps.tolist(),
                "score_definition": (
                    "acceleration + skidpad + predicted autocross + endurance + efficiency"
                ),
                "autocross_correlation": {
                    "intercept_points": AUTOX_INTERCEPT_POINTS,
                    "acceleration_weight": AUTOX_ACCELERATION_WEIGHT,
                    "skidpad_weight": AUTOX_SKIDPAD_WEIGHT,
                    "source": "analysis/autox_corr/README.md",
                },
                "endurance_scoring": asdict(FSAE_2026_MI6_SCORING),
                "baseline": baseline,
                "best_grid_point": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Best total dynamic score: {best['total_dynamic_points']:.3f} points")
    print(
        "Best CG: "
        f"{best['longitudinal_position_from_rear_axle_in']:.3f} in forward of rear axle, "
        f"{best['cg_height_in']:.3f} in high "
        f"({best['static_front_weight_percent']:.2f}% front)"
    )
    print(f"Baseline total dynamic score: {baseline['total_dynamic_points']:.3f}")
    print(f"CSV: {csv_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Plot: {plot_path}")
    print(f"Position plot: {position_plot_path}")


if __name__ == "__main__":
    main()
