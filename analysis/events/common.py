"""Output and track helpers shared by event analysis scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from lapsim import EventResult, SpatialTrack


def coarsen_track(
    source: SpatialTrack,
    maximum_cell_length_m: float,
) -> SpatialTrack:
    """Combine source cells while preserving integrated heading."""

    if maximum_cell_length_m <= 0.0:
        raise ValueError("maximum_cell_length_m must be positive")
    lengths: list[float] = []
    curvatures: list[float] = []
    group_length_m = 0.0
    group_turn_rad = 0.0
    for length_m, curvature_per_m in zip(
        source.cell_length_m,
        source.curvature_per_m,
        strict=True,
    ):
        group_length_m += length_m
        group_turn_rad += curvature_per_m * length_m
        if group_length_m >= maximum_cell_length_m - 1e-12:
            lengths.append(group_length_m)
            curvatures.append(group_turn_rad / group_length_m)
            group_length_m = 0.0
            group_turn_rad = 0.0
    if group_length_m > 0.0:
        lengths.append(group_length_m)
        curvatures.append(group_turn_rad / group_length_m)
    return SpatialTrack.from_cells(
        cell_length_m=lengths,
        curvature_per_m=curvatures,
        closed=source.closed,
    )


def write_event_outputs(result: EventResult, output_dir: Path) -> tuple[Path, Path]:
    """Write a compact points summary and complete aligned telemetry CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{result.event}_summary.json"
    telemetry_path = output_dir / f"{result.event}_telemetry.csv"
    summary = {
        "event": result.event,
        "completed": result.completed,
        "elapsed_time_s": result.elapsed_time_s,
        "scoring_time_s": result.scoring_time_s,
        "distance_m": result.distance_m,
        "energy_kwh": result.energy_kwh,
        "estimated_points": result.estimated_points,
        "maximum_points": result.maximum_points,
        "completed_laps": result.completed_laps,
        "lap_times_s": result.lap_times_s,
        "failure_reason": result.failure_reason,
        "point_breakdown": result.point_breakdown,
        "telemetry_sample_count": result.telemetry.sample_count,
        "telemetry_channels": tuple(result.telemetry),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    channels = result.telemetry.as_dict()
    with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        names = tuple(channels)
        writer.writerow(names)
        for sample_index in range(result.telemetry.sample_count):
            writer.writerow(channels[name][sample_index] for name in names)
    return summary_path.resolve(), telemetry_path.resolve()


def print_result(result: EventResult) -> None:
    status = "completed" if result.completed else f"failed: {result.failure_reason}"
    scoring_time = (
        f"{result.scoring_time_s:.3f} s"
        if result.scoring_time_s is not None
        else "not eligible"
    )
    print(f"{result.event}: {status}")
    print(f"scoring time: {scoring_time}")
    print(f"estimated points: {result.points:.3f} / {result.maximum_points:.1f}")
    print(f"energy: {result.energy_kwh:.6f} kWh")
    print(f"telemetry: {result.telemetry.sample_count} samples, {len(result.telemetry)} channels")


__all__ = ["coarsen_track", "print_result", "write_event_outputs"]
