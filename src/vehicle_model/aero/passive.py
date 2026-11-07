"""Passive speed-squared aero law and its retained force state."""

from dataclasses import dataclass, field
from math import degrees, isfinite

from .forces import AeroForces
from .parameters import (
    DEFAULT_FRONTAL_AREA_M2,
    DEFAULT_DRAG_COEFFICIENT,
    DEFAULT_LIFT_COEFFICIENT,
    DEFAULT_FRONT_DOWNFORCE_FRACTION,
    DEFAULT_ROLL_LIMIT_RAD,
    DEFAULT_DOWNFORCE_RETENTION_AT_ROLL_LIMIT,
)


@dataclass(slots=True)
class Aero:
    """Speed-squared drag and lift model with explicit reference area."""

    frontal_area_m2: float = DEFAULT_FRONTAL_AREA_M2
    drag_coefficient: float = DEFAULT_DRAG_COEFFICIENT
    lift_coefficient: float = DEFAULT_LIFT_COEFFICIENT
    front_downforce_fraction: float = DEFAULT_FRONT_DOWNFORCE_FRACTION
    roll_limit_rad: float = DEFAULT_ROLL_LIMIT_RAD
    downforce_retention_at_roll_limit: float = DEFAULT_DOWNFORCE_RETENTION_AT_ROLL_LIMIT
    current_forces_n: AeroForces = field(
        init=False,
        default_factory=lambda: AeroForces(0.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable aerodynamic parameters."""

        if not isfinite(self.frontal_area_m2) or self.frontal_area_m2 <= 0:
            raise ValueError("frontal_area_m2 must be finite and positive")
        if not isfinite(self.drag_coefficient) or self.drag_coefficient < 0:
            raise ValueError("drag_coefficient must be finite and nonnegative")
        if not isfinite(self.lift_coefficient):
            raise ValueError("lift_coefficient must be finite")
        if not 0 <= self.front_downforce_fraction <= 1:
            raise ValueError("front_downforce_fraction must be between 0 and 1")
        if not isfinite(self.roll_limit_rad) or self.roll_limit_rad <= 0.0:
            raise ValueError("roll_limit_rad must be finite and positive")
        if not 0.0 <= self.downforce_retention_at_roll_limit <= 1.0:
            raise ValueError(
                "downforce_retention_at_roll_limit must be between 0 and 1"
            )

    def reset_state(self) -> None:
        """Clear the most recently calculated aerodynamic forces."""

        self.current_forces_n = AeroForces(0.0, 0.0, 0.0, 0.0)

    def select_configuration(self, curvature_per_m: float) -> bool:
        """Retain the fixed configuration used by the passive aero model."""

        if not isfinite(curvature_per_m):
            raise ValueError("curvature_per_m must be finite")
        return False

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        forces = self.current_forces_n
        telemetry.update(
            {
                "aero.drag_n": forces.drag_n,
                "aero.downforce_n": forces.downforce_n,
                "aero.front_downforce_n": forces.front_downforce_n,
                "aero.rear_downforce_n": forces.rear_downforce_n,
                "aero.frontal_area_m2": self.frontal_area_m2,
                "aero.drag_coefficient": self.drag_coefficient,
                "aero.lift_coefficient": self.lift_coefficient,
                "aero.body_roll_angle_rad": forces.body_roll_angle_rad,
                "aero.body_roll_angle_deg": degrees(forces.body_roll_angle_rad),
                "aero.downforce_multiplier": forces.downforce_multiplier,
                "aero.roll_limit_rad": self.roll_limit_rad,
                "aero.roll_limit_deg": degrees(self.roll_limit_rad),
                "aero.downforce_retention_at_roll_limit": (
                    self.downforce_retention_at_roll_limit
                ),
                "aero.effective_drag_coefficient": (
                    self.drag_coefficient * forces.drag_coefficient_multiplier
                ),
                "aero.effective_lift_coefficient": (
                    self.lift_coefficient * forces.downforce_coefficient_multiplier
                ),
                "aero.low_drag_mode_active": float(forces.low_drag_mode_active),
            }
        )

    def update_state(
        self,
        vehicle_speed_mps: float,
        air_density_kgpm3: float,
        timestep_s: float,
        body_roll_angle_rad: float = 0.0,
    ) -> AeroForces:
        """Calculate and retain the aerodynamic forces for one timestep."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.current_forces_n = self.forces_n(
            vehicle_speed_mps,
            air_density_kgpm3,
            body_roll_angle_rad,
        )
        return self.current_forces_n

    def forces_n(
        self,
        vehicle_speed_mps: float,
        air_density_kgpm3: float,
        body_roll_angle_rad: float = 0.0,
    ) -> AeroForces:
        """Calculate drag and axle downforce at the requested speed."""

        if vehicle_speed_mps < 0:
            raise ValueError("vehicle_speed_mps cannot be negative")
        if air_density_kgpm3 <= 0:
            raise ValueError("air_density_kgpm3 must be positive")
        if not isfinite(body_roll_angle_rad):
            raise ValueError("body_roll_angle_rad must be finite")

        dynamic_pressure_pa = 0.5 * air_density_kgpm3 * vehicle_speed_mps**2
        drag_n = dynamic_pressure_pa * self.drag_coefficient * self.frontal_area_m2
        downforce_multiplier = self.downforce_multiplier(body_roll_angle_rad)
        downforce_n = downforce_multiplier * (
            -dynamic_pressure_pa * self.lift_coefficient * self.frontal_area_m2
        )
        front_downforce_n = self.front_downforce_fraction * downforce_n
        rear_downforce_n = downforce_n - front_downforce_n
        return AeroForces(
            drag_n=drag_n,
            downforce_n=downforce_n,
            front_downforce_n=front_downforce_n,
            rear_downforce_n=rear_downforce_n,
            downforce_multiplier=downforce_multiplier,
            body_roll_angle_rad=body_roll_angle_rad,
        )

    def downforce_multiplier(self, body_roll_angle_rad: float) -> float:
        """Linearly reduce downforce to the retained fraction at the roll limit."""

        if not isfinite(body_roll_angle_rad):
            raise ValueError("body_roll_angle_rad must be finite")
        roll_fraction = min(abs(body_roll_angle_rad) / self.roll_limit_rad, 1.0)
        return 1.0 - roll_fraction * (1.0 - self.downforce_retention_at_roll_limit)
