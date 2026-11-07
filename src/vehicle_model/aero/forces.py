"""Immutable aerodynamic force result shared by aero and load transfer."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AeroForces:
    """Aerodynamic forces at one vehicle speed."""

    drag_n: float
    downforce_n: float
    front_downforce_n: float
    rear_downforce_n: float
    downforce_multiplier: float = 1.0
    body_roll_angle_rad: float = 0.0
    drag_coefficient_multiplier: float = 1.0
    downforce_coefficient_multiplier: float = 1.0
    low_drag_mode_active: bool = False
