"""Time-indexed replay of explicit driver controls."""

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
    timestep_s: float | Sequence[float],
) -> ReplayTelemetry:
    """Reset a vehicle and replay controls at successive time samples."""

    if not controls:
        raise ValueError("controls cannot be empty")
    if isinstance(timestep_s, (int, float)):
        if not isfinite(timestep_s) or timestep_s <= 0:
            raise ValueError("timestep_s must be finite and positive")
        timesteps_s = (float(timestep_s),) * len(controls)
    else:
        timesteps_s = tuple(float(value) for value in timestep_s)
        if len(timesteps_s) != len(controls):
            raise ValueError("timestep_s must contain one value per control")
        if any(not isfinite(value) or value <= 0 for value in timesteps_s):
            raise ValueError("Every time step must be finite and positive")

    vehicle.validate()
    vehicle.reset_state()
    recorder = TelemetryRecorder()
    for control, step_s in zip(controls, timesteps_s, strict=True):
        vehicle.update_state(control, step_s)
        recorder.record(
            vehicle.telemetry_snapshot(),
            timestep_s=step_s,
        )

    return recorder.freeze()
