"""Aero-subteam parameters and force calculations."""

from dataclasses import dataclass, field
from math import isfinite

from utils.units import square_inches_to_square_meters

DEFAULT_FRONTAL_AREA_IN2 = 1019.902
DEFAULT_FRONTAL_AREA_M2 = square_inches_to_square_meters(DEFAULT_FRONTAL_AREA_IN2)
# Vehicle aero data supplies a positive downforce coefficient and L/D. Convert
# it to this model's SAE signed-lift convention and derive Cd = Cl_downforce/L/D.
DEFAULT_DOWNFORCE_COEFFICIENT = 3.62
DEFAULT_LIFT_TO_DRAG_RATIO = 2.946
DEFAULT_DRAG_COEFFICIENT = (
    DEFAULT_DOWNFORCE_COEFFICIENT / DEFAULT_LIFT_TO_DRAG_RATIO
)
DEFAULT_LIFT_COEFFICIENT = -DEFAULT_DOWNFORCE_COEFFICIENT
DEFAULT_FRONT_DOWNFORCE_FRACTION = 0.5269293255


@dataclass(frozen=True, slots=True)
class AeroForces:
    """Aerodynamic forces at one vehicle speed."""

    drag_n: float
    downforce_n: float
    front_downforce_n: float
    rear_downforce_n: float


@dataclass(slots=True)
class Aero:
    """Speed-squared drag and lift model with explicit reference area."""

    frontal_area_m2: float = DEFAULT_FRONTAL_AREA_M2
    drag_coefficient: float = DEFAULT_DRAG_COEFFICIENT
    lift_coefficient: float = DEFAULT_LIFT_COEFFICIENT
    front_downforce_fraction: float = DEFAULT_FRONT_DOWNFORCE_FRACTION
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

    @property
    def drag_area_m2(self) -> float:
        """Return CdA for compatibility with coefficient-area consumers."""

        return self.drag_coefficient * self.frontal_area_m2

    @drag_area_m2.setter
    def drag_area_m2(self, value: float) -> None:
        if value < 0:
            raise ValueError("drag_area_m2 cannot be negative")
        self.drag_coefficient = value / self.frontal_area_m2

    @property
    def downforce_area_m2(self) -> float:
        """Return positive ClA magnitude when lift coefficient is negative."""

        return -self.lift_coefficient * self.frontal_area_m2

    @downforce_area_m2.setter
    def downforce_area_m2(self, value: float) -> None:
        if value < 0:
            raise ValueError("downforce_area_m2 cannot be negative")
        self.lift_coefficient = -value / self.frontal_area_m2

    def reset_state(self) -> None:
        """Clear the most recently calculated aerodynamic forces."""

        self.current_forces_n = AeroForces(0.0, 0.0, 0.0, 0.0)

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
                "aero.drag_area_m2": self.drag_area_m2,
                "aero.downforce_area_m2": self.downforce_area_m2,
            }
        )

    def update_state(
        self,
        vehicle_speed_mps: float,
        air_density_kgpm3: float,
        timestep_s: float,
    ) -> AeroForces:
        """Calculate and retain the aerodynamic forces for one timestep."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.current_forces_n = self.forces_n(
            vehicle_speed_mps,
            air_density_kgpm3,
        )
        return self.current_forces_n

    def forces_n(
        self,
        vehicle_speed_mps: float,
        air_density_kgpm3: float,
    ) -> AeroForces:
        """Calculate drag and axle downforce at the requested speed."""

        if vehicle_speed_mps < 0:
            raise ValueError("vehicle_speed_mps cannot be negative")
        if air_density_kgpm3 <= 0:
            raise ValueError("air_density_kgpm3 must be positive")

        dynamic_pressure_pa = 0.5 * air_density_kgpm3 * vehicle_speed_mps**2
        drag_n = dynamic_pressure_pa * self.drag_coefficient * self.frontal_area_m2
        downforce_n = (
            -dynamic_pressure_pa * self.lift_coefficient * self.frontal_area_m2
        )
        front_downforce_n = self.front_downforce_fraction * downforce_n
        rear_downforce_n = downforce_n - front_downforce_n
        return AeroForces(
            drag_n=drag_n,
            downforce_n=downforce_n,
            front_downforce_n=front_downforce_n,
            rear_downforce_n=rear_downforce_n,
        )
