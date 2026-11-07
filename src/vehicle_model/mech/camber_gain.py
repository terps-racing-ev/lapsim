"""Measured camber-versus-travel curves and clamped interpolation.

Camber is tire-local (negative inward); positive suspension travel is bump.
The source workbook and zero-travel reference are documented with the data.
"""

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite, radians


@dataclass(frozen=True, slots=True)
class CamberGainCurve:
    """Camber change versus suspension travel with clamped interpolation."""

    suspension_travel_m: tuple[float, ...]
    camber_change_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.suspension_travel_m) < 2:
            raise ValueError("camber gain curve requires at least two points")
        if len(self.suspension_travel_m) != len(self.camber_change_rad):
            raise ValueError("camber gain curve travel and camber lengths must match")
        if not all(
            isfinite(value)
            for value in (*self.suspension_travel_m, *self.camber_change_rad)
        ):
            raise ValueError("camber gain curve values must be finite")
        if any(
            next_travel_m <= travel_m
            for travel_m, next_travel_m in zip(
                self.suspension_travel_m,
                self.suspension_travel_m[1:],
            )
        ):
            raise ValueError("camber gain curve travel must be strictly increasing")

    def camber_change_at(self, suspension_travel_m: float) -> float:
        """Return interpolated camber change, clamped to measured endpoints."""

        if not isfinite(suspension_travel_m):
            raise ValueError("suspension_travel_m must be finite")
        if suspension_travel_m <= self.suspension_travel_m[0]:
            return self.camber_change_rad[0]
        if suspension_travel_m >= self.suspension_travel_m[-1]:
            return self.camber_change_rad[-1]

        upper_index = bisect_right(self.suspension_travel_m, suspension_travel_m)
        lower_index = upper_index - 1
        lower_travel_m = self.suspension_travel_m[lower_index]
        upper_travel_m = self.suspension_travel_m[upper_index]
        fraction = (suspension_travel_m - lower_travel_m) / (
            upper_travel_m - lower_travel_m
        )
        return self.camber_change_rad[lower_index] + fraction * (
            self.camber_change_rad[upper_index] - self.camber_change_rad[lower_index]
        )


# Measured suspension sweep from camber gain.xlsx, Sheet1 E4:H12. Each curve is
# referenced to its static value at zero travel so static alignment remains an
# independent setup input. Positive suspension travel is bump.
_MEASURED_TRAVEL_M = (
    -0.0254,
    -0.01905,
    -0.0127,
    -0.00635,
    0.0,
    0.00635,
    0.0127,
    0.01905,
    0.0254,
)
_MEASURED_FRONT_CAMBER_CHANGE_DEG = (1.8, 1.5, 0.9, 0.5, 0.0, -0.3, -0.7, -1.0, -1.3)
_MEASURED_REAR_LEFT_CAMBER_CHANGE_DEG = (
    2.3,
    1.4,
    1.1,
    0.4,
    0.0,
    -0.7,
    -1.1,
    -1.6,
    -2.7,
)
_MEASURED_REAR_RIGHT_CAMBER_CHANGE_DEG = (
    2.1,
    1.3,
    0.6,
    0.2,
    0.0,
    -0.7,
    -1.1,
    -1.6,
    -2.7,
)


def _measured_curve(camber_change_deg: tuple[float, ...]) -> CamberGainCurve:
    return CamberGainCurve(
        suspension_travel_m=_MEASURED_TRAVEL_M,
        camber_change_rad=tuple(radians(value) for value in camber_change_deg),
    )


def front_camber_gain_curve() -> CamberGainCurve:
    # The workbook contains one measured front curve, so it applies to both sides.
    return _measured_curve(_MEASURED_FRONT_CAMBER_CHANGE_DEG)


def rear_left_camber_gain_curve() -> CamberGainCurve:
    return _measured_curve(_MEASURED_REAR_LEFT_CAMBER_CHANGE_DEG)


def rear_right_camber_gain_curve() -> CamberGainCurve:
    return _measured_curve(_MEASURED_REAR_RIGHT_CAMBER_CHANGE_DEG)
