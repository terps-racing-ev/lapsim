"""Per-wheel steering and camber geometry."""

from dataclasses import dataclass, field
from math import atan2, cos, isfinite, pi, radians, sin, tan
from typing import TYPE_CHECKING, Sequence

from .camber_gain import (
    CamberGainCurve,
    front_camber_gain_curve,
    rear_left_camber_gain_curve,
    rear_right_camber_gain_curve,
)
from .wheels import WHEEL_NAMES, WHEEL_SIDE_SIGNS

if TYPE_CHECKING:
    from ..interfaces import ChassisModel

# Compatibility name used by existing analysis scripts.
WHEEL_POSITIONS = WHEEL_NAMES
DEFAULT_FRONT_CASTER_ANGLE_DEG = 7.3
DEFAULT_FRONT_CASTER_ANGLE_RAD = radians(DEFAULT_FRONT_CASTER_ANGLE_DEG)


@dataclass(frozen=True, slots=True)
class WheelAlignmentCorner:
    """Solved orientation and suspension travel at one contact patch.

    Camber uses a tire-local convention on both sides of the car: negative
    camber means the top of the tire leans toward the vehicle centerline.
    Suspension travel is positive in bump and negative in droop.
    """

    steering_angle_rad: float = 0.0
    camber_angle_rad: float = 0.0
    suspension_travel_m: float = 0.0
    camber_gain_angle_rad: float = 0.0


@dataclass(frozen=True, slots=True)
class WheelAlignmentState:
    """Solved front-left through rear-right wheel alignment."""

    front_left: WheelAlignmentCorner = field(default_factory=WheelAlignmentCorner)
    front_right: WheelAlignmentCorner = field(default_factory=WheelAlignmentCorner)
    rear_left: WheelAlignmentCorner = field(default_factory=WheelAlignmentCorner)
    rear_right: WheelAlignmentCorner = field(default_factory=WheelAlignmentCorner)

    @property
    def all(
        self,
    ) -> tuple[
        WheelAlignmentCorner,
        WheelAlignmentCorner,
        WheelAlignmentCorner,
        WheelAlignmentCorner,
    ]:
        return self.front_left, self.front_right, self.rear_left, self.rear_right

    @property
    def steering_angles_rad(self) -> tuple[float, float, float, float]:
        return tuple(corner.steering_angle_rad for corner in self.all)

    @property
    def camber_angles_rad(self) -> tuple[float, float, float, float]:
        return tuple(corner.camber_angle_rad for corner in self.all)


@dataclass(slots=True)
class WheelAlignment:
    """Static alignment plus quasi-static steering/suspension kinematics.

    ``anti_ackermann_fraction`` blends parallel steer (0) with ideal
    anti-Ackermann (1), where the outside front wheel has the larger steering
    angle. Positive toe values mean toe-in on both sides. Camber gain curves
    provide tire-local camber change as a function of suspension travel.
    """

    anti_ackermann_fraction: float = 1.0
    front_toe_in_angle_rad: float = 0.0
    rear_toe_in_angle_rad: float = 0.0
    front_static_camber_angle_rad: float = 0.0
    rear_static_camber_angle_rad: float = 0.0
    front_caster_angle_rad: float = DEFAULT_FRONT_CASTER_ANGLE_RAD
    front_left_camber_gain_curve: CamberGainCurve = field(
        default_factory=front_camber_gain_curve
    )
    front_right_camber_gain_curve: CamberGainCurve = field(
        default_factory=front_camber_gain_curve
    )
    rear_left_camber_gain_curve: CamberGainCurve = field(
        default_factory=rear_left_camber_gain_curve
    )
    rear_right_camber_gain_curve: CamberGainCurve = field(
        default_factory=rear_right_camber_gain_curve
    )
    current_state: WheelAlignmentState = field(
        init=False,
        default_factory=WheelAlignmentState,
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isfinite(self.anti_ackermann_fraction) or not (
            0.0 <= self.anti_ackermann_fraction <= 1.0
        ):
            raise ValueError("anti_ackermann_fraction must be between 0 and 1")
        for name, angle_rad in (
            ("front_toe_in_angle_rad", self.front_toe_in_angle_rad),
            ("rear_toe_in_angle_rad", self.rear_toe_in_angle_rad),
            ("front_static_camber_angle_rad", self.front_static_camber_angle_rad),
            ("rear_static_camber_angle_rad", self.rear_static_camber_angle_rad),
            ("front_caster_angle_rad", self.front_caster_angle_rad),
        ):
            if not isfinite(angle_rad) or not -pi / 2 < angle_rad < pi / 2:
                raise ValueError(f"{name} must be finite and within (-pi/2, pi/2)")

    def reset_state(self) -> None:
        self.current_state = WheelAlignmentState()

    @staticmethod
    def _ideal_ackermann_angles_rad(
        chassis: "ChassisModel",
        steering_angle_rad: float,
    ) -> tuple[float, float]:
        if steering_angle_rad == 0.0:
            return 0.0, 0.0
        turn_radius_m = chassis.wheelbase_m / abs(tan(steering_angle_rad))
        inner_angle_rad = atan2(
            chassis.wheelbase_m,
            turn_radius_m - 0.5 * chassis.front_track_width_m,
        )
        outer_angle_rad = atan2(
            chassis.wheelbase_m,
            turn_radius_m + 0.5 * chassis.front_track_width_m,
        )
        if steering_angle_rad > 0.0:
            return inner_angle_rad, outer_angle_rad
        return -outer_angle_rad, -inner_angle_rad

    @staticmethod
    def _caster_camber_body_rad(
        steering_angle_rad: float,
        caster_angle_rad: float,
    ) -> float:
        """Return road-wheel camber caused by rotation about a caster axis."""

        caster_sin = sin(caster_angle_rad)
        return atan2(
            caster_sin * sin(steering_angle_rad),
            cos(caster_angle_rad) ** 2 + caster_sin**2 * cos(steering_angle_rad),
        )

    def calculate(
        self,
        chassis: "ChassisModel",
        steering_angle_rad: float,
        body_roll_angle_rad: float,
        suspension_travel_m: Sequence[float],
    ) -> WheelAlignmentState:
        """Resolve control input and suspension motion into four wheel angles."""

        if not isfinite(steering_angle_rad):
            raise ValueError("steering_angle_rad must be finite")
        if not isfinite(body_roll_angle_rad):
            raise ValueError("body_roll_angle_rad must be finite")
        if len(suspension_travel_m) != 4 or not all(
            isfinite(travel_m) for travel_m in suspension_travel_m
        ):
            raise ValueError("four finite suspension travel values are required")

        ideal_left_rad, ideal_right_rad = self._ideal_ackermann_angles_rad(
            chassis,
            steering_angle_rad,
        )
        # Swap the ideal inner/outer angles to produce anti-Ackermann.
        front_left_rad = steering_angle_rad + self.anti_ackermann_fraction * (
            ideal_right_rad - steering_angle_rad
        )
        front_right_rad = steering_angle_rad + self.anti_ackermann_fraction * (
            ideal_left_rad - steering_angle_rad
        )
        steering_angles_rad = (
            front_left_rad - self.front_toe_in_angle_rad,
            front_right_rad + self.front_toe_in_angle_rad,
            -self.rear_toe_in_angle_rad,
            self.rear_toe_in_angle_rad,
        )
        if any(abs(angle_rad) >= pi / 2 for angle_rad in steering_angles_rad):
            raise ValueError(
                "resolved wheel steering angle must be within (-pi/2, pi/2)"
            )

        static_camber_angles_rad = (
            self.front_static_camber_angle_rad,
            self.front_static_camber_angle_rad,
            self.rear_static_camber_angle_rad,
            self.rear_static_camber_angle_rad,
        )
        camber_gain_curves = (
            self.front_left_camber_gain_curve,
            self.front_right_camber_gain_curve,
            self.rear_left_camber_gain_curve,
            self.rear_right_camber_gain_curve,
        )
        camber_gain_angles_rad = tuple(
            curve.camber_change_at(travel_m)
            for curve, travel_m in zip(
                camber_gain_curves,
                suspension_travel_m,
                strict=True,
            )
        )
        caster_camber_body_rad = (
            self._caster_camber_body_rad(
                steering_angles_rad[0],
                self.front_caster_angle_rad,
            ),
            self._caster_camber_body_rad(
                steering_angles_rad[1],
                self.front_caster_angle_rad,
            ),
            0.0,
            0.0,
        )
        camber_angles_rad = tuple(
            static_camber_rad
            + side_sign * (caster_camber_rad - body_roll_angle_rad)
            + camber_gain_angle_rad
            for (
                static_camber_rad,
                side_sign,
                caster_camber_rad,
                camber_gain_angle_rad,
            ) in zip(
                static_camber_angles_rad,
                WHEEL_SIDE_SIGNS,
                caster_camber_body_rad,
                camber_gain_angles_rad,
                strict=True,
            )
        )
        return WheelAlignmentState(
            *(
                WheelAlignmentCorner(
                    steering_angle_rad=steering_rad,
                    camber_angle_rad=camber_rad,
                    suspension_travel_m=travel_m,
                    camber_gain_angle_rad=camber_gain_angle_rad,
                )
                for steering_rad, camber_rad, travel_m, camber_gain_angle_rad in zip(
                    steering_angles_rad,
                    camber_angles_rad,
                    suspension_travel_m,
                    camber_gain_angles_rad,
                    strict=True,
                )
            )
        )

    def update_state(self, state: WheelAlignmentState, timestep_s: float) -> None:
        if timestep_s <= 0.0:
            raise ValueError("timestep_s must be positive")
        self.current_state = state

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry.update(
            {
                "alignment.anti_ackermann_fraction": (self.anti_ackermann_fraction),
                "alignment.front_toe_in_angle_rad": self.front_toe_in_angle_rad,
                "alignment.rear_toe_in_angle_rad": self.rear_toe_in_angle_rad,
                "alignment.front_static_camber_angle_rad": (
                    self.front_static_camber_angle_rad
                ),
                "alignment.rear_static_camber_angle_rad": (
                    self.rear_static_camber_angle_rad
                ),
                "alignment.front_caster_angle_rad": self.front_caster_angle_rad,
            }
        )
        for position, corner in zip(
            WHEEL_POSITIONS,
            self.current_state.all,
            strict=True,
        ):
            prefix = f"alignment.{position}"
            telemetry.update(
                {
                    f"{prefix}.steering_angle_rad": corner.steering_angle_rad,
                    f"{prefix}.camber_angle_rad": corner.camber_angle_rad,
                    f"{prefix}.suspension_travel_m": corner.suspension_travel_m,
                    f"{prefix}.camber_gain_angle_rad": (corner.camber_gain_angle_rad),
                }
            )


__all__ = [
    "CamberGainCurve",
    "WHEEL_POSITIONS",
    "WHEEL_SIDE_SIGNS",
    "WheelAlignment",
    "WheelAlignmentCorner",
    "WheelAlignmentState",
]
