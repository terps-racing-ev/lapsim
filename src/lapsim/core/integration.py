"""Adapters between spatial tracks and the time-domain vehicle model."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite

from scipy.optimize import brentq

from vehicle_model.vehicle import Vehicle

from .controls import Controls


def advance_vehicle_by_distance(
    vehicle: Vehicle,
    controls: Controls,
    distance_step_m: float,
) -> float:
    """Advance one spatial cell using a time step solved outside the vehicle.

    Spatial event solvers still need to land on track-cell boundaries. This
    adapter previews time-domain updates on copies of the current state, solves
    for the elapsed time that covers ``distance_step_m``, and commits exactly
    one time-based update to ``vehicle``. The returned value is that time step.
    """

    if not isfinite(distance_step_m) or distance_step_m <= 0.0:
        raise ValueError("distance_step_m must be finite and positive")

    initial_distance_m = vehicle.distance_m

    def traveled_distance_m(timestep_s: float) -> tuple[float, float]:
        candidate = deepcopy(vehicle)
        candidate.update_state(controls, timestep_s)
        return (
            candidate.distance_m - initial_distance_m,
            candidate.speed_mps,
        )

    upper_timestep_s = max(
        distance_step_m / max(vehicle.speed_mps, 1.0),
        1.0e-6,
    )
    for _ in range(60):
        traveled_m, final_speed_mps = traveled_distance_m(upper_timestep_s)
        if traveled_m >= distance_step_m:
            break
        if final_speed_mps <= 0.0:
            raise ValueError(
                "cannot traverse distance_step_m: vehicle stops before the "
                "end of the cell"
            )
        upper_timestep_s *= 2.0
    else:
        raise ValueError("cannot traverse distance_step_m with the supplied controls")

    timestep_s = brentq(
        lambda candidate_timestep_s: (
            traveled_distance_m(candidate_timestep_s)[0] - distance_step_m
        ),
        distance_step_m
        / (2.0 * max(vehicle.drivetrain.vehicle_speed_limit_mps, 1.0)),
        upper_timestep_s,
        xtol=max(5.0e-324, upper_timestep_s * 1.0e-12),
    )
    vehicle.update_state(controls, timestep_s)
    # Keep the spatial boundary exact despite accumulated floating-point noise.
    vehicle.distance_m = initial_distance_m + distance_step_m
    return timestep_s
