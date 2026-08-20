"""Distance-indexed replay of explicit driver controls."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from vehicle_model.vehicle import Vehicle

from ..core.controls import Controls
from ..core.telemetry import Telemetry, TelemetryRecorder

# Compatibility alias. Replays now return the same mapping-style telemetry as
# every other simulation path.
ReplayTelemetry = Telemetry


def replay_controls(
    vehicle: Vehicle,
    controls: Sequence[Controls],
    distance_step_m: float | Sequence[float],
) -> ReplayTelemetry:
    """Reset a vehicle and replay controls at successive track distances."""

    if not controls:
        raise ValueError("controls cannot be empty")
    if isinstance(distance_step_m, (int, float)):
        if not isfinite(distance_step_m) or distance_step_m <= 0:
            raise ValueError("distance_step_m must be finite and positive")
        distance_steps_m = (float(distance_step_m),) * len(controls)
    else:
        distance_steps_m = tuple(float(value) for value in distance_step_m)
        if len(distance_steps_m) != len(controls):
            raise ValueError("distance_step_m must contain one value per control")
        if any(not isfinite(value) or value <= 0 for value in distance_steps_m):
            raise ValueError("Every distance step must be finite and positive")

    vehicle.validate()
    vehicle.reset_state()
    recorder = TelemetryRecorder()
    for control, step_m in zip(controls, distance_steps_m, strict=True):
        previous_time_s = vehicle.time_s
        vehicle.update_state(control, step_m)
        recorder.record(
            vehicle.telemetry_snapshot(),
            timestep_s=vehicle.time_s - previous_time_s,
        )

    return recorder.freeze()
