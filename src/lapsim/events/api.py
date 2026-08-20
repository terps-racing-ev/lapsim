"""High-level Formula SAE event simulations with points and telemetry.

Analysis code should enter the simulator through ``simulate_endurance``,
``simulate_acceleration``, or ``simulate_skidpad``.  Every function accepts a
vehicle, a spatial track, and a distance-indexed controls profile and returns
the same :class:`EventResult` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan, isfinite
from statistics import fmean
from typing import Any, Literal

from vehicle_model.vehicle import Vehicle

from ..core.controls import Controls
from ..core.profiles import ControlsProfile
from ..core.telemetry import JOULES_PER_KILOWATT_HOUR, Telemetry, TelemetryRecorder
from ..courses.spatial_track import SpatialTrack
from ..optimization.torque_profile import TorqueProfile
from ..solvers.path_constraints import PathConstraintSolver
from .endurance import EnduranceRunConfig, EnduranceSimulator
from .scoring import (
    FSAEEnduranceEfficiencyScoring,
    FSAE_2026_MI_ACCELERATION_SCORING,
    FSAE_2026_MI6_SCORING,
    FSAE_2026_MI_SKIDPAD_SCORING,
    TimedEventScoring,
)


EventName = Literal["endurance", "acceleration", "skidpad"]


@dataclass(frozen=True, slots=True)
class EventResult:
    """Common output contract for every points-producing event simulation."""

    event: EventName
    completed: bool
    elapsed_time_s: float
    scoring_time_s: float | None
    distance_m: float
    energy_kwh: float
    estimated_points: float
    maximum_points: float
    completed_laps: int
    lap_times_s: tuple[float, ...]
    failure_reason: str | None
    telemetry: Telemetry
    point_breakdown: dict[str, float | bool | None]

    @property
    def points(self) -> float:
        """Short alias used by analysis tables and parameter sweeps."""

        return self.estimated_points


@dataclass(frozen=True, slots=True)
class AccelerationConfig:
    """Numerical setup for one open acceleration course."""

    starting_speed_mps: float = 0.0
    maximum_driving_time_s: float = 30.0

    def __post_init__(self) -> None:
        if not isfinite(self.starting_speed_mps) or self.starting_speed_mps < 0.0:
            raise ValueError("starting_speed_mps must be finite and nonnegative")
        if (
            not isfinite(self.maximum_driving_time_s)
            or self.maximum_driving_time_s <= 0.0
        ):
            raise ValueError("maximum_driving_time_s must be finite and positive")


@dataclass(frozen=True, slots=True)
class SkidpadConfig:
    """Warmup and scored laps on one closed skidpad circle."""

    warmup_laps: int = 1
    scored_laps: int = 1
    starting_speed_mps: float = 0.0
    maximum_driving_time_s: float = 60.0

    def __post_init__(self) -> None:
        if self.warmup_laps < 0:
            raise ValueError("warmup_laps cannot be negative")
        if self.scored_laps <= 0:
            raise ValueError("scored_laps must be positive")
        if not isfinite(self.starting_speed_mps) or self.starting_speed_mps < 0.0:
            raise ValueError("starting_speed_mps must be finite and nonnegative")
        if (
            not isfinite(self.maximum_driving_time_s)
            or self.maximum_driving_time_s <= 0.0
        ):
            raise ValueError("maximum_driving_time_s must be finite and positive")

    @property
    def total_laps(self) -> int:
        return self.warmup_laps + self.scored_laps


@dataclass(frozen=True, slots=True)
class _PathRunResult:
    completed: bool
    elapsed_time_s: float
    distance_m: float
    energy_kwh: float
    completed_laps: int
    lap_times_s: tuple[float, ...]
    failure_reason: str | None
    telemetry: Telemetry


@dataclass(frozen=True, slots=True)
class _TrackSteeringProfile:
    """Make track geometry authoritative while preserving longitudinal inputs."""

    profile: ControlsProfile
    track: SpatialTrack
    wheelbase_m: float

    def controls_at(self, distance_m: float) -> Controls:
        supplied = self.profile.controls_at(distance_m)
        if not isinstance(supplied, Controls):
            raise TypeError("controls_at must return Controls")
        wrapped_m = self.track.wrap_distance_m(distance_m)
        cell_index = _cell_index_at(self.track, wrapped_m)
        return replace(
            supplied,
            steering_angle_rad=atan(
                self.track.curvature_per_m[cell_index] * self.wheelbase_m
            ),
        )


def _cell_index_at(track: SpatialTrack, distance_m: float) -> int:
    lower = 0
    upper = track.cell_count
    while lower < upper:
        middle = (lower + upper) // 2
        if track.distance_m[middle + 1] <= distance_m:
            lower = middle + 1
        else:
            upper = middle
    return min(lower, track.cell_count - 1)


def _require_controls_profile(profile: Any) -> ControlsProfile:
    if not isinstance(profile, ControlsProfile):
        raise TypeError("profile must implement controls_at(distance_m) -> Controls")
    return profile


def _run_prescribed_path(
    vehicle: Vehicle,
    track: SpatialTrack,
    profile: ControlsProfile,
    *,
    laps: int,
    starting_speed_mps: float,
    maximum_driving_time_s: float,
) -> _PathRunResult:
    """Integrate explicit driver controls and always retain full telemetry."""

    if laps <= 0:
        raise ValueError("laps must be positive")
    if laps > 1 and not track.closed:
        raise ValueError("multiple laps require a closed track")
    vehicle.validate()
    vehicle.reset_state()
    vehicle.speed_mps = starting_speed_mps
    recorder = TelemetryRecorder()
    energy_j = 0.0
    completed_laps = 0
    lap_times_s: list[float] = []
    failure_reason: str | None = None
    lap_start_time_s = vehicle.time_s
    steering_profile = _TrackSteeringProfile(
        profile=profile,
        track=track,
        wheelbase_m=vehicle.chassis.wheelbase_m,
    )

    for lap_index in range(laps):
        for cell_index, cell_length_m in enumerate(track.cell_length_m):
            lap_distance_m = track.cell_center_distance_m[cell_index]
            controls = steering_profile.controls_at(lap_distance_m)
            time_before_step_s = vehicle.time_s
            try:
                vehicle.update_state(controls, cell_length_m)
            except ValueError as error:
                failure_reason = f"vehicle could not traverse the event path: {error}"
                break
            timestep_s = vehicle.time_s - time_before_step_s
            if timestep_s <= 0.0:
                failure_reason = "vehicle did not advance in time"
                break
            energy_j += vehicle.battery.current_power_w * timestep_s
            snapshot = vehicle.telemetry_snapshot()
            snapshot.update(
                {
                    "event.lap_index": float(lap_index),
                    "event.cell_index": float(cell_index),
                    "event.lap_distance_m": track.distance_m[cell_index + 1],
                    "event.cumulative_net_energy_j": energy_j,
                }
            )
            recorder.record(snapshot, timestep_s=timestep_s)

            if snapshot.get("limits.lateral_saturated", 0.0) > 0.5:
                failure_reason = (
                    f"vehicle exceeded lateral path capacity on lap {lap_index + 1}, "
                    f"cell {cell_index}"
                )
                break
            if vehicle.time_s > maximum_driving_time_s:
                failure_reason = "maximum configured driving time exceeded"
                break
            if vehicle.battery.state_of_charge <= 0.0:
                failure_reason = "battery state of charge depleted"
                break

        if failure_reason is not None:
            break
        completed_laps += 1
        lap_times_s.append(vehicle.time_s - lap_start_time_s)
        lap_start_time_s = vehicle.time_s

    return _PathRunResult(
        completed=completed_laps == laps and failure_reason is None,
        elapsed_time_s=vehicle.time_s,
        distance_m=vehicle.distance_m,
        energy_kwh=energy_j / JOULES_PER_KILOWATT_HOUR,
        completed_laps=completed_laps,
        lap_times_s=tuple(lap_times_s),
        failure_reason=failure_reason,
        telemetry=recorder.freeze(),
    )


def simulate_acceleration(
    vehicle: Vehicle,
    track: SpatialTrack,
    profile: ControlsProfile,
    *,
    config: AccelerationConfig | None = None,
    scoring: TimedEventScoring = FSAE_2026_MI_ACCELERATION_SCORING,
) -> EventResult:
    """Run an open acceleration track and estimate event points."""

    if track.closed:
        raise ValueError("acceleration requires an open track")
    if scoring.event_name != "acceleration":
        raise ValueError("acceleration requires an acceleration scoring model")
    setup = config if config is not None else AccelerationConfig()
    run = _run_prescribed_path(
        vehicle,
        track,
        _require_controls_profile(profile),
        laps=1,
        starting_speed_mps=setup.starting_speed_mps,
        maximum_driving_time_s=setup.maximum_driving_time_s,
    )
    score = scoring.score(run.elapsed_time_s, completed=run.completed)
    return EventResult(
        event="acceleration",
        completed=run.completed,
        elapsed_time_s=run.elapsed_time_s,
        scoring_time_s=run.elapsed_time_s if run.completed else None,
        distance_m=run.distance_m,
        energy_kwh=run.energy_kwh,
        estimated_points=score.points,
        maximum_points=score.maximum_points,
        completed_laps=run.completed_laps,
        lap_times_s=run.lap_times_s,
        failure_reason=run.failure_reason,
        telemetry=run.telemetry,
        point_breakdown={
            "acceleration_points": score.points,
            "minimum_time_s": score.minimum_time_s,
            "maximum_time_s": score.maximum_time_s,
            "time_eligible": score.time_eligible,
        },
    )


def simulate_skidpad(
    vehicle: Vehicle,
    track: SpatialTrack,
    profile: ControlsProfile,
    *,
    config: SkidpadConfig | None = None,
    scoring: TimedEventScoring = FSAE_2026_MI_SKIDPAD_SCORING,
) -> EventResult:
    """Run warmup/scored laps of a closed circle and estimate skidpad points."""

    if not track.closed:
        raise ValueError("skidpad requires a closed track")
    if scoring.event_name != "skidpad":
        raise ValueError("skidpad requires a skidpad scoring model")
    setup = config if config is not None else SkidpadConfig()
    run = _run_prescribed_path(
        vehicle,
        track,
        _require_controls_profile(profile),
        laps=setup.total_laps,
        starting_speed_mps=setup.starting_speed_mps,
        maximum_driving_time_s=setup.maximum_driving_time_s,
    )
    scoring_time_s = (
        fmean(run.lap_times_s[-setup.scored_laps :]) if run.completed else None
    )
    score = scoring.score(scoring_time_s, completed=run.completed)
    return EventResult(
        event="skidpad",
        completed=run.completed,
        elapsed_time_s=run.elapsed_time_s,
        scoring_time_s=scoring_time_s,
        distance_m=run.distance_m,
        energy_kwh=run.energy_kwh,
        estimated_points=score.points,
        maximum_points=score.maximum_points,
        completed_laps=run.completed_laps,
        lap_times_s=run.lap_times_s,
        failure_reason=run.failure_reason,
        telemetry=run.telemetry,
        point_breakdown={
            "skidpad_points": score.points,
            "minimum_time_s": score.minimum_time_s,
            "maximum_time_s": score.maximum_time_s,
            "warmup_laps": float(setup.warmup_laps),
            "scored_laps": float(setup.scored_laps),
            "time_eligible": score.time_eligible,
        },
    )


def simulate_endurance(
    vehicle: Vehicle,
    track: SpatialTrack,
    profile: ControlsProfile | TorqueProfile,
    *,
    config: EnduranceRunConfig | None = None,
    scoring: FSAEEnduranceEfficiencyScoring = FSAE_2026_MI6_SCORING,
    constraint_solver: PathConstraintSolver | None = None,
) -> EventResult:
    """Run a stateful endurance event and return combined estimated points."""

    if not track.closed:
        raise ValueError("endurance requires a closed track")
    setup = (
        config
        if config is not None
        else EnduranceRunConfig()
    )
    solver = (
        constraint_solver
        if constraint_solver is not None
        else PathConstraintSolver(
            maximum_brake_pressure_psi=setup.maximum_brake_pressure_psi
        )
    )
    constraints = solver.solve(track, vehicle)
    if isinstance(profile, ControlsProfile):
        event_profile: ControlsProfile | TorqueProfile = _TrackSteeringProfile(
            profile=profile,
            track=track,
            wheelbase_m=vehicle.chassis.wheelbase_m,
        )
    elif isinstance(profile, TorqueProfile):
        event_profile = profile
    else:
        raise TypeError(
            "profile must implement controls_at(distance_m) or request_fraction(distance_m)"
        )
    run = EnduranceSimulator().run(
        vehicle,
        constraints,
        event_profile,
        setup,
        record_telemetry=True,
    )
    if run.telemetry is None:
        raise RuntimeError("endurance simulation did not return telemetry")
    telemetry_channels = run.telemetry.as_dict()
    common_channel_aliases = {
        "event.lap_index": "endurance.lap_index",
        "event.cell_index": "endurance.cell_index",
        "event.lap_distance_m": "endurance.lap_distance_m",
        "event.cumulative_net_energy_j": "energy.cumulative_net_j",
    }
    for common_name, endurance_name in common_channel_aliases.items():
        if endurance_name in telemetry_channels:
            telemetry_channels.setdefault(
                common_name,
                telemetry_channels[endurance_name],
            )
    telemetry = Telemetry(telemetry_channels)
    score = scoring.score(run)
    distance_m = (
        telemetry["vehicle.distance_m"][-1]
        if telemetry.sample_count > 0
        else 0.0
    )
    return EventResult(
        event="endurance",
        completed=run.completed,
        elapsed_time_s=run.driving_time_s,
        scoring_time_s=run.driving_time_s if run.completed else None,
        distance_m=distance_m,
        energy_kwh=run.pack_energy_kwh,
        estimated_points=score.combined_points,
        maximum_points=(
            scoring.maximum_endurance_time_points
            + 25.0
            + scoring.maximum_efficiency_points
        ),
        completed_laps=run.completed_laps,
        lap_times_s=run.lap_times_s,
        failure_reason=run.failure_reason,
        telemetry=telemetry,
        point_breakdown={
            "endurance_points": score.endurance_points,
            "endurance_time_points": score.endurance_time_points,
            "endurance_lap_points": score.endurance_lap_points,
            "efficiency_points": score.efficiency_points,
            "adjusted_energy_per_lap_kg": score.adjusted_energy_per_lap_kg,
            "efficiency_factor": score.efficiency_factor,
            "endurance_time_eligible": score.endurance_time_eligible,
            "efficiency_time_eligible": score.efficiency_time_eligible,
            "efficiency_energy_eligible": score.efficiency_energy_eligible,
        },
    )


__all__ = [
    "AccelerationConfig",
    "EventName",
    "EventResult",
    "SkidpadConfig",
    "simulate_acceleration",
    "simulate_endurance",
    "simulate_skidpad",
]
