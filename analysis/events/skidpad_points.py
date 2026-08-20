"""Estimate skidpad points from a controls profile and closed circle track."""

from __future__ import annotations

import argparse
from math import pi
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.events.common import print_result, write_event_outputs  # noqa: E402
from lapsim import (  # noqa: E402
    ConstantControlsProfile,
    Controls,
    ControlsProfile,
    EventResult,
    FSAE_2026_MI_SKIDPAD_SCORING,
    SkidpadConfig,
    SpatialTrack,
    TimedEventScoring,
    simulate_skidpad,
)
from lapsim import Curve, Track  # noqa: E402
from vehicle_model import Vehicle  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs/events/skidpad"
STANDARD_SKIDPAD_RACING_LINE_RADIUS_M = 9.125


def standard_skidpad_circle(cell_length_m: float = 0.25) -> SpatialTrack:
    """Return the centerline between the 15.25 m and 21.25 m circles."""

    return SpatialTrack.from_track(
        Track.from_segments(
            [Curve(radius_m=STANDARD_SKIDPAD_RACING_LINE_RADIUS_M, span_rad=2.0 * pi)]
        ),
        maximum_cell_length_m=cell_length_m,
        closed=True,
    )


def run(
    profile: ControlsProfile,
    track: SpatialTrack,
    *,
    vehicle: Vehicle | None = None,
    config: SkidpadConfig | None = None,
    scoring: TimedEventScoring = FSAE_2026_MI_SKIDPAD_SCORING,
) -> EventResult:
    """Programmatic entry point used by analysis sweeps."""

    return simulate_skidpad(
        vehicle if vehicle is not None else Vehicle(),
        track,
        profile,
        config=config,
        scoring=scoring,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor-torque-nm", type=float, default=5.0)
    parser.add_argument("--starting-speed-mps", type=float, default=9.0)
    parser.add_argument("--cell-length-m", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        ConstantControlsProfile(
            Controls(motor_torque_request_nm=args.motor_torque_nm)
        ),
        standard_skidpad_circle(args.cell_length_m),
        config=SkidpadConfig(starting_speed_mps=args.starting_speed_mps),
    )
    print_result(result)
    summary_path, telemetry_path = write_event_outputs(result, args.output_dir)
    print(summary_path)
    print(telemetry_path)


if __name__ == "__main__":
    main()
