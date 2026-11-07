"""Contact-patch velocities, wheel forces, and immutable four-wheel results.

All four-wheel values use front-left, front-right, rear-left, rear-right order.
Body coordinates are x forward and y left; positive yaw turns left.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import atan2, cos, hypot, sin


@dataclass(frozen=True, slots=True)
class WheelKinematics:
    """Body-frame contact-patch velocity and wheel steering orientation."""

    steering_angle_rad: float = 0.0
    longitudinal_speed_mps: float = 0.0
    lateral_speed_mps: float = 0.0
    camber_angle_rad: float = 0.0

    @property
    def speed_mps(self) -> float:
        return hypot(self.longitudinal_speed_mps, self.lateral_speed_mps)

    @property
    def slip_angle_rad(self) -> float:
        if self.speed_mps <= 1e-9:
            return 0.0
        return (
            atan2(self.lateral_speed_mps, self.longitudinal_speed_mps)
            - self.steering_angle_rad
        )


@dataclass(frozen=True, slots=True)
class TireKinematics:
    """Kinematics at the four contact patches, front-left through rear-right."""

    front_left: WheelKinematics = field(default_factory=WheelKinematics)
    front_right: WheelKinematics = field(default_factory=WheelKinematics)
    rear_left: WheelKinematics = field(default_factory=WheelKinematics)
    rear_right: WheelKinematics = field(default_factory=WheelKinematics)

    @classmethod
    def from_vehicle_motion(
        cls,
        speed_mps: float,
        body_slip_angle_rad: float,
        curvature_per_m: float,
        steering_angles_rad: Sequence[float],
        wheel_positions_m: Sequence[tuple[float, float]],
        camber_angles_rad: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    ) -> "TireKinematics":
        """Resolve body motion into a velocity and slip input at each wheel."""

        if (
            len(steering_angles_rad) != 4
            or len(wheel_positions_m) != 4
            or len(camber_angles_rad) != 4
        ):
            raise ValueError(
                "four steering angles, wheel positions, and camber angles are required"
            )
        body_longitudinal_speed_mps = speed_mps * cos(body_slip_angle_rad)
        body_lateral_speed_mps = speed_mps * sin(body_slip_angle_rad)
        yaw_rate_radps = speed_mps * curvature_per_m
        return cls(
            *(
                WheelKinematics(
                    steering_angle_rad=steering_angle_rad,
                    longitudinal_speed_mps=(
                        body_longitudinal_speed_mps - yaw_rate_radps * y_m
                    ),
                    lateral_speed_mps=(body_lateral_speed_mps + yaw_rate_radps * x_m),
                    camber_angle_rad=camber_angle_rad,
                )
                for steering_angle_rad, (x_m, y_m), camber_angle_rad in zip(
                    steering_angles_rad,
                    wheel_positions_m,
                    camber_angles_rad,
                    strict=True,
                )
            )
        )

    @property
    def all(
        self,
    ) -> tuple[WheelKinematics, WheelKinematics, WheelKinematics, WheelKinematics]:
        return self.front_left, self.front_right, self.rear_left, self.rear_right

    @property
    def vehicle_speed_mps(self) -> float:
        return 0.25 * sum(wheel.speed_mps for wheel in self.all)


@dataclass(frozen=True, slots=True)
class TireState:
    """One contact patch's solved force, capacity, and slip state."""

    normal_load_n: float = 0.0
    lateral_force_n: float = 0.0
    lateral_capacity_n: float = 0.0
    longitudinal_capacity_n: float = 0.0
    drive_force_n: float = 0.0
    braking_force_n: float = 0.0
    steering_angle_rad: float = 0.0
    camber_angle_rad: float = 0.0
    local_longitudinal_speed_mps: float = 0.0
    local_lateral_speed_mps: float = 0.0
    slip_angle_rad: float = 0.0
    slip_ratio: float = 0.0
    wheel_surface_speed_mps: float = 0.0
    wheel_angular_speed_radps: float = 0.0
    body_longitudinal_force_n: float = 0.0
    body_lateral_force_n: float = 0.0

    @property
    def longitudinal_force_n(self) -> float:
        """Signed force on the vehicle: propulsion positive, braking negative."""

        return self.drive_force_n - self.braking_force_n

    @property
    def force_utilization(self) -> float:
        if self.longitudinal_capacity_n <= 0.0:
            return 0.0
        return min(
            (self.drive_force_n + self.braking_force_n) / self.longitudinal_capacity_n,
            1.0,
        )


@dataclass(frozen=True, slots=True)
class TireStates:
    """Solved operating points for the four contact patches."""

    vehicle_speed_mps: float = 0.0
    front_left: TireState = field(default_factory=TireState)
    front_right: TireState = field(default_factory=TireState)
    rear_left: TireState = field(default_factory=TireState)
    rear_right: TireState = field(default_factory=TireState)

    @property
    def all(self) -> tuple[TireState, TireState, TireState, TireState]:
        return self.front_left, self.front_right, self.rear_left, self.rear_right

    @property
    def front(self) -> tuple[TireState, TireState]:
        return self.front_left, self.front_right

    @property
    def rear(self) -> tuple[TireState, TireState]:
        return self.rear_left, self.rear_right

    @staticmethod
    def _sum(states: tuple[TireState, ...], name: str) -> float:
        return sum(getattr(state, name) for state in states)

    @property
    def drive_force_n(self) -> float:
        return self._sum(self.all, "drive_force_n")

    @property
    def braking_force_n(self) -> float:
        return self._sum(self.all, "braking_force_n")

    @property
    def longitudinal_force_n(self) -> float:
        return self._sum(self.all, "longitudinal_force_n")

    @property
    def lateral_force_n(self) -> float:
        return self._sum(self.all, "lateral_force_n")

    @property
    def body_longitudinal_force_n(self) -> float:
        return self._sum(self.all, "body_longitudinal_force_n")

    @property
    def body_lateral_force_n(self) -> float:
        return self._sum(self.all, "body_lateral_force_n")

    def tangent_force_n(self, body_slip_angle_rad: float) -> float:
        """Return total tire force along the vehicle velocity vector."""

        return self.body_longitudinal_force_n * cos(
            body_slip_angle_rad
        ) + self.body_lateral_force_n * sin(body_slip_angle_rad)

    def normal_force_n(self, body_slip_angle_rad: float) -> float:
        """Return total tire force normal to the vehicle velocity vector."""

        return -self.body_longitudinal_force_n * sin(
            body_slip_angle_rad
        ) + self.body_lateral_force_n * cos(body_slip_angle_rad)

    def yaw_moment_nm(
        self,
        wheel_positions_m: Sequence[tuple[float, float]],
    ) -> float:
        """Return contact-patch yaw moment about the vehicle CG."""

        if len(wheel_positions_m) != 4:
            raise ValueError("four wheel positions are required")
        return sum(
            x_m * state.body_lateral_force_n - y_m * state.body_longitudinal_force_n
            for state, (x_m, y_m) in zip(
                self.all,
                wheel_positions_m,
                strict=True,
            )
        )

    @property
    def longitudinal_capacity_n(self) -> float:
        return self._sum(self.all, "longitudinal_capacity_n")

    @property
    def front_braking_force_n(self) -> float:
        return self._sum(self.front, "braking_force_n")

    @property
    def rear_braking_force_n(self) -> float:
        return self._sum(self.rear, "braking_force_n")

    @property
    def rear_longitudinal_capacity_n(self) -> float:
        return self._sum(self.rear, "longitudinal_capacity_n")

    @property
    def driven_wheel_surface_speed_mps(self) -> float:
        return 0.5 * sum(state.wheel_surface_speed_mps for state in self.rear)

    @property
    def driven_slip_ratio(self) -> float:
        return 0.5 * sum(state.slip_ratio for state in self.rear)
