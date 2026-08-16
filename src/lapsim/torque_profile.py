"""Reusable periodic torque-request parameterizations."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from vehicle_model.vehicle import Vehicle

from .controls import Controls
from .spatial_track import SpatialTrack


@runtime_checkable
class EnduranceControlProfile(Protocol):
    """Distance-indexed driver controls supplied to endurance simulation."""

    def controls_at(self, lap_distance_m: float) -> Controls: ...


@runtime_checkable
class TorqueProfile(Protocol):
    """Distance-indexed normalized driver torque request."""

    def request_fraction(self, lap_distance_m: float) -> float: ...


@dataclass(frozen=True, slots=True)
class PeriodicPiecewiseLinearTorqueProfile:
    """Periodic linear interpolation of normalized torque-request knots."""

    track_length_m: float
    knot_distance_m: tuple[float, ...]
    request_fraction_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.track_length_m <= 0:
            raise ValueError("track_length_m must be positive")
        if len(self.knot_distance_m) != len(self.request_fraction_values):
            raise ValueError("Knot distance and request arrays must match")
        if len(self.knot_distance_m) < 2:
            raise ValueError("At least two periodic torque knots are required")
        if abs(self.knot_distance_m[0]) > 1e-9:
            raise ValueError("The first torque knot must be at distance zero")
        if any(
            upper <= lower
            for lower, upper in zip(self.knot_distance_m, self.knot_distance_m[1:])
        ):
            raise ValueError("Torque-knot distances must strictly increase")
        if self.knot_distance_m[-1] >= self.track_length_m:
            raise ValueError("The final knot must be before the periodic endpoint")
        if any(
            not isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.request_fraction_values
        ):
            raise ValueError("Torque-request fractions must be finite and in [0, 1]")

    def request_fraction(self, lap_distance_m: float) -> float:
        if not isfinite(lap_distance_m):
            raise ValueError("lap_distance_m must be finite")
        wrapped_distance_m = lap_distance_m % self.track_length_m
        upper_index = bisect_right(self.knot_distance_m, wrapped_distance_m)
        if upper_index < len(self.knot_distance_m):
            lower_index = upper_index - 1
            lower_distance_m = self.knot_distance_m[lower_index]
            upper_distance_m = self.knot_distance_m[upper_index]
            lower_value = self.request_fraction_values[lower_index]
            upper_value = self.request_fraction_values[upper_index]
        else:
            lower_index = len(self.knot_distance_m) - 1
            lower_distance_m = self.knot_distance_m[lower_index]
            upper_distance_m = self.track_length_m
            lower_value = self.request_fraction_values[lower_index]
            upper_value = self.request_fraction_values[0]
        fraction = (wrapped_distance_m - lower_distance_m) / (
            upper_distance_m - lower_distance_m
        )
        return lower_value + fraction * (upper_value - lower_value)


@runtime_checkable
class TorqueProfileParameterization(Protocol):
    """Map optimizer variables to a track-specific torque profile."""

    @property
    def variable_count(self) -> int: ...

    def bounds(self, vehicle: Vehicle) -> tuple[tuple[float, float], ...]: ...

    def build(
        self,
        variables: Sequence[float],
        track: SpatialTrack,
    ) -> TorqueProfile: ...


@dataclass(frozen=True, slots=True)
class UniformPeriodicTorqueParameterization:
    """Evenly spaced periodic normalized-torque control points.

    Normalized requests keep the optimization variables valid when a vehicle
    sweep changes the motor map, gear ratio, battery power limit, or mass.
    """

    control_point_count: int = 12

    def __post_init__(self) -> None:
        if self.control_point_count < 2:
            raise ValueError("control_point_count must be at least two")

    @property
    def variable_count(self) -> int:
        return self.control_point_count

    def bounds(self, vehicle: Vehicle) -> tuple[tuple[float, float], ...]:
        vehicle.validate()
        return ((0.0, 1.0),) * self.control_point_count

    def build(
        self,
        variables: Sequence[float],
        track: SpatialTrack,
    ) -> PeriodicPiecewiseLinearTorqueProfile:
        values = tuple(float(value) for value in variables)
        if len(values) != self.control_point_count:
            raise ValueError(
                f"Expected {self.control_point_count} torque variables, got {len(values)}"
            )
        spacing_m = track.length_m / self.control_point_count
        return PeriodicPiecewiseLinearTorqueProfile(
            track_length_m=track.length_m,
            knot_distance_m=tuple(
                index * spacing_m for index in range(self.control_point_count)
            ),
            request_fraction_values=values,
        )


__all__ = [
    "EnduranceControlProfile",
    "PeriodicPiecewiseLinearTorqueProfile",
    "TorqueProfile",
    "TorqueProfileParameterization",
    "UniformPeriodicTorqueParameterization",
]
