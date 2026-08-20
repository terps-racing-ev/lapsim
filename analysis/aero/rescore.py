"""Rescore an aero sweep after uniformly scaling its simulated lap times."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from lapsim import FSAE_2026_MI6_SCORING
from sweep import ProjectedRun, plot_heatmap, write_results


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "outputs/aero_sweep/aero_sweep.csv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs/aero_sweep/aero_sweep_rescaled.csv"
DEFAULT_PLOT = (
    REPOSITORY_ROOT
    / "outputs/aero_sweep/aero_rescaled_combined_points_heatmap.png"
)
EVENT_LAPS = FSAE_2026_MI6_SCORING.event_laps
MINIMUM_LAP_TIME_S = (
    FSAE_2026_MI6_SCORING.endurance_minimum_time_s / EVENT_LAPS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument(
        "--target-fastest-lap-s",
        type=float,
        default=MINIMUM_LAP_TIME_S,
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {name: float(value) for name, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def rescore_rows(
    rows: list[dict[str, float]], target_fastest_lap_s: float
) -> tuple[list[dict[str, float]], float]:
    if not rows:
        raise ValueError("Input sweep contains no rows")
    if target_fastest_lap_s <= 0:
        raise ValueError("Target fastest lap time must be positive")

    scale = target_fastest_lap_s / min(row["lap_time_s"] for row in rows)
    rescored: list[dict[str, float]] = []
    for row in rows:
        raw_lap_time_s = row["lap_time_s"]
        scaled_lap_time_s = scale * raw_lap_time_s
        score = FSAE_2026_MI6_SCORING.score(
            ProjectedRun(
                completed_laps=EVENT_LAPS,
                driving_time_s=EVENT_LAPS * scaled_lap_time_s,
                pack_energy_kwh=EVENT_LAPS * row["lap_energy_kwh"],
            )
        )
        rescored.append(
            {
                **row,
                "raw_lap_time_s": raw_lap_time_s,
                "lap_time_s": scaled_lap_time_s,
                "time_scale_factor": scale,
                "endurance_points": score.endurance_points,
                "efficiency_points": score.efficiency_points,
                "combined_points": score.combined_points,
            }
        )
    return rescored, scale


def main() -> None:
    args = parse_args()
    rows, scale = rescore_rows(
        load_rows(args.input.resolve()), args.target_fastest_lap_s
    )
    output_path = args.output.resolve()
    plot_path = args.plot.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    write_results(output_path, rows)

    lift_values = np.asarray(sorted({row["lift_coefficient"] for row in rows}))
    drag_values = np.asarray(sorted({row["drag_coefficient"] for row in rows}))
    best = plot_heatmap(
        plot_path,
        rows,
        lift_values,
        drag_values,
        "combined",
        title="Time-Rescaled Endurance + Efficiency Points",
    )

    print(f"Time scale factor: {scale:.6f}")
    print(f"Fastest scaled lap: {min(row['lap_time_s'] for row in rows):.3f} s")
    print(f"Best combined score: {best['combined_points']:.3f} points")
    print(f"Cl={best['lift_coefficient']:.4f}, Cd={best['drag_coefficient']:.4f}")
    print(f"CSV: {output_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
