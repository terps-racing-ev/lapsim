"""Estimate acceleration points from a controls profile and open track."""

from __future__ import annotations

import argparse
from math import radians
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
from vehicle_model import (  # noqa: E402
    Pacejka52UpcR20LateralModel,
    Pacejka52UpcR20LongitudinalModel,
    Tire,
    Vehicle,
)


DEFAULT_OUTPUT = ROOT / "outputs/events/acceleration"
PSI_TO_PA = 6_894.757293168
DEFAULT_TIRE_PRESSURE_PSI = 10.0
DEFAULT_TIRE_CAMBER_DEG = -1.0


def upc_r20_vehicle(
    *,
    tire_pressure_psi: float = DEFAULT_TIRE_PRESSURE_PSI,
    camber_deg: float = DEFAULT_TIRE_CAMBER_DEG,
) -> Vehicle:
    """Build the acceleration-test vehicle with the published UPC R20 fit."""

    return Vehicle(
        tire=Tire(
            pacejka_lateral=Pacejka52UpcR20LateralModel(),
            pacejka_longitudinal=Pacejka52UpcR20LongitudinalModel(),
            camber_angle_rad=radians(camber_deg),
            inflation_pressure_pa=tire_pressure_psi * PSI_TO_PA,
        )
    )


def standard_acceleration_track(cell_length_m: float = 0.5) -> SpatialTrack:
    """Return the 75 m timed course; the event API adds the 0.3 m rollout."""

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
    parser.add_argument(
        "--tire-pressure-psi",
        type=float,
        default=DEFAULT_TIRE_PRESSURE_PSI,
    )
    parser.add_argument(
        "--camber-deg",
        type=float,
        default=DEFAULT_TIRE_CAMBER_DEG,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        ConstantControlsProfile(
            Controls(motor_torque_request_nm=args.motor_torque_nm)
        ),
        standard_acceleration_track(args.cell_length_m),
        vehicle=upc_r20_vehicle(
            tire_pressure_psi=args.tire_pressure_psi,
            camber_deg=args.camber_deg,
        ),
    )
    print(
        "tire: UPC Hoosier 16x7.5-10 R20; "
        f"pressure={args.tire_pressure_psi:.3f} psi; "
        f"camber={args.camber_deg:.3f} deg per wheel"
    )
    print_result(result)
    summary_path, telemetry_path = write_event_outputs(result, args.output_dir)
    print(summary_path)
    print(telemetry_path)


if __name__ == "__main__":
    main()
