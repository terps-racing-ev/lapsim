"""Estimate endurance/efficiency points from a profile and closed track."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.events.common import (  # noqa: E402
    coarsen_track,
    print_result,
    write_event_outputs,
)
from lapsim import (  # noqa: E402
    ControlsProfile,
    EnduranceRunConfig,
    EventResult,
    FSAEEnduranceEfficiencyScoring,
    FSAE_2026_MI6_SCORING,
    SpatialTrack,
    UniformPeriodicTorqueParameterization,
    simulate_endurance,
)
from lapsim import TorqueProfile  # noqa: E402
from vehicle_model import Vehicle  # noqa: E402


DEFAULT_TRACK = ROOT / "analysis/data/track/gnss_imu_endurance_track.csv"
DEFAULT_OUTPUT = ROOT / "outputs/events/endurance"


def calibrated_vehicle() -> Vehicle:
    vehicle = Vehicle()
    vehicle.aero.drag_coefficient = 2.5
    vehicle.drivetrain.chain_drive.efficiency = 0.86
    vehicle.tire.constant_friction_coefficient = 1.8
    vehicle.cornering_drag_coefficient = 0.036
    vehicle.battery.initial_state_of_charge = 0.9815
    vehicle.validate()
    return vehicle


def run(
    profile: ControlsProfile | TorqueProfile,
    track: SpatialTrack,
    *,
    vehicle: Vehicle | None = None,
    config: EnduranceRunConfig | None = None,
    scoring: FSAEEnduranceEfficiencyScoring = FSAE_2026_MI6_SCORING,
) -> EventResult:
    """Programmatic entry point used by analysis sweeps."""

    return simulate_endurance(
        vehicle if vehicle is not None else calibrated_vehicle(),
        track,
        profile,
        config=config,
        scoring=scoring,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--torque-fraction", type=float, default=0.26)
    parser.add_argument("--cell-length-m", type=float, default=2.0)
    parser.add_argument("--laps", type=int, default=22)
    parser.add_argument("--maximum-brake-pressure-psi", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    track = coarsen_track(
        SpatialTrack.from_csv(args.track),
        args.cell_length_m,
    )
    profile = UniformPeriodicTorqueParameterization(8).build(
        (args.torque_fraction,) * 8,
        track,
    )
    result = run(
        profile,
        track,
        config=EnduranceRunConfig(
            laps=args.laps,
            maximum_brake_pressure_psi=args.maximum_brake_pressure_psi,
        ),
    )
    print_result(result)
    summary_path, telemetry_path = write_event_outputs(result, args.output_dir)
    print(summary_path)
    print(telemetry_path)


if __name__ == "__main__":
    main()
