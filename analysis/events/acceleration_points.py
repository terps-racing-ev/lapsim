"""Estimate acceleration points from a controls profile and open track."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.events.common import print_result, write_event_outputs  # noqa: E402
from lapsim import (  # noqa: E402
    AccelerationConfig,
    ConstantControlsProfile,
    Controls,
    ControlsProfile,
    EventResult,
    FSAE_2026_MI_ACCELERATION_SCORING,
    SpatialTrack,
    TimedEventScoring,
    simulate_acceleration,
)
from vehicle_model import Vehicle  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs/events/acceleration"


def standard_acceleration_track(cell_length_m: float = 0.5) -> SpatialTrack:
    cell_count = round(75.0 / cell_length_m)
    if cell_count <= 0:
        raise ValueError("cell_length_m is too large")
    actual_cell_length_m = 75.0 / cell_count
    return SpatialTrack.from_cells(
        cell_length_m=(actual_cell_length_m,) * cell_count,
        curvature_per_m=(0.0,) * cell_count,
        closed=False,
    )


def run(
    profile: ControlsProfile,
    track: SpatialTrack,
    *,
    vehicle: Vehicle | None = None,
    config: AccelerationConfig | None = None,
    scoring: TimedEventScoring = FSAE_2026_MI_ACCELERATION_SCORING,
) -> EventResult:
    """Programmatic entry point used by analysis sweeps."""

    return simulate_acceleration(
        vehicle if vehicle is not None else Vehicle(),
        track,
        profile,
        config=config,
        scoring=scoring,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor-torque-nm", type=float, default=230.0)
    parser.add_argument("--cell-length-m", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        ConstantControlsProfile(
            Controls(motor_torque_request_nm=args.motor_torque_nm)
        ),
        standard_acceleration_track(args.cell_length_m),
    )
    print_result(result)
    summary_path, telemetry_path = write_event_outputs(result, args.output_dir)
    print(summary_path)
    print(telemetry_path)


if __name__ == "__main__":
    main()
