"""Driver control inputs for distance-domain vehicle simulation."""

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True, slots=True)
class Controls:
    """Driver requests held across one spatial simulation cell."""

    motor_torque_request_nm: float = 0.0
    front_brake_pressure_psi: float = 0.0
    rear_brake_pressure_psi: float = 0.0
    rear_regenerative_brake_force_request_n: float = 0.0
    steering_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.motor_torque_request_nm < 0:
            raise ValueError(
                "motor_torque_request_nm cannot be negative until regen is implemented"
            )
        if self.front_brake_pressure_psi < 0:
            raise ValueError("front_brake_pressure_psi cannot be negative")
        if self.rear_brake_pressure_psi < 0:
            raise ValueError("rear_brake_pressure_psi cannot be negative")
        if self.rear_regenerative_brake_force_request_n < 0:
            raise ValueError(
                "rear_regenerative_brake_force_request_n cannot be negative"
            )
        if not -pi / 2 < self.steering_angle_rad < pi / 2:
            raise ValueError(
                "steering_angle_rad must be strictly between -pi/2 and pi/2"
            )
