"""Reusable distance-indexed driver-control profiles."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from .controls import Controls


@runtime_checkable
class ControlsProfile(Protocol):
    """Return driver requests at a distance along an event track."""

    def controls_at(self, distance_m: float) -> Controls: ...


@dataclass(frozen=True, slots=True)
class ConstantControlsProfile:
    """Hold one set of controls for the complete event."""

    controls: Controls

    def controls_at(self, distance_m: float) -> Controls:
        if not isfinite(distance_m):
            raise ValueError("distance_m must be finite")
        return self.controls


@dataclass(frozen=True, slots=True)
class PiecewiseLinearControlsProfile:
    """Linearly interpolate controls along an open or periodic distance axis."""

    distance_m: tuple[float, ...]
    controls: tuple[Controls, ...]
    period_m: float | None = None

    def __post_init__(self) -> None:
        if len(self.distance_m) != len(self.controls) or len(self.distance_m) < 2:
            raise ValueError("distance and controls need matching arrays of length >= 2")
        if abs(self.distance_m[0]) > 1e-9:
            raise ValueError("the first controls distance must be zero")
        if any(
            not isfinite(value) for value in self.distance_m
        ) or any(
            upper <= lower
            for lower, upper in zip(self.distance_m, self.distance_m[1:])
        ):
            raise ValueError("controls distances must be finite and increasing")
        if self.period_m is not None:
            if not isfinite(self.period_m) or self.period_m <= self.distance_m[-1]:
                raise ValueError("period_m must be finite and after the final knot")

    @staticmethod
    def _interpolate(lower: Controls, upper: Controls, fraction: float) -> Controls:
        def blend(lower_value: float, upper_value: float) -> float:
            return lower_value + fraction * (upper_value - lower_value)

        return Controls(
            motor_torque_request_nm=blend(
                lower.motor_torque_request_nm,
                upper.motor_torque_request_nm,
            ),
            front_brake_pressure_psi=blend(
                lower.front_brake_pressure_psi,
                upper.front_brake_pressure_psi,
            ),
            rear_brake_pressure_psi=blend(
                lower.rear_brake_pressure_psi,
                upper.rear_brake_pressure_psi,
            ),
            rear_regenerative_brake_force_request_n=blend(
                lower.rear_regenerative_brake_force_request_n,
                upper.rear_regenerative_brake_force_request_n,
            ),
            steering_angle_rad=blend(
                lower.steering_angle_rad,
                upper.steering_angle_rad,
            ),
        )

    def controls_at(self, distance_m: float) -> Controls:
        if not isfinite(distance_m):
            raise ValueError("distance_m must be finite")
        if self.period_m is None:
            query_m = min(max(distance_m, self.distance_m[0]), self.distance_m[-1])
        else:
            query_m = distance_m % self.period_m

        upper_index = bisect_right(self.distance_m, query_m)
        if upper_index == 0:
            return self.controls[0]
        if upper_index < len(self.distance_m):
            lower_index = upper_index - 1
            upper_distance_m = self.distance_m[upper_index]
            upper_controls = self.controls[upper_index]
        elif self.period_m is None:
            return self.controls[-1]
        else:
            lower_index = len(self.distance_m) - 1
            upper_distance_m = self.period_m
            upper_controls = self.controls[0]

        lower_distance_m = self.distance_m[lower_index]
        fraction = (query_m - lower_distance_m) / (
            upper_distance_m - lower_distance_m
        )
        return self._interpolate(
            self.controls[lower_index],
            upper_controls,
            fraction,
        )


__all__ = [
    "ConstantControlsProfile",
    "ControlsProfile",
    "PiecewiseLinearControlsProfile",
]
