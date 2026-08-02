"""Aerodynamic parameters and force calculations."""

from dataclasses import dataclass, field


DEFAULT_DRAG_AREA_M2 = 1.803
DEFAULT_DOWNFORCE_AREA_M2 = 3.43
DEFAULT_FRONT_DOWNFORCE_FRACTION = 0.49


@dataclass(frozen=True, slots=True)
class AeroForces:
    """Aerodynamic forces at one vehicle speed."""

    drag_n: float
    downforce_n: float
    front_downforce_n: float
    rear_downforce_n: float


@dataclass(slots=True)
class Aero:
    """Speed-squared drag and downforce model using coefficient areas."""

    drag_area_m2: float = DEFAULT_DRAG_AREA_M2
    downforce_area_m2: float = DEFAULT_DOWNFORCE_AREA_M2
    front_downforce_fraction: float = DEFAULT_FRONT_DOWNFORCE_FRACTION
    current_forces_n: AeroForces = field(
        init=False,
        default_factory=lambda: AeroForces(0.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable aerodynamic parameters."""

        if self.drag_area_m2 < 0:
            raise ValueError("drag_area_m2 cannot be negative")
        if self.downforce_area_m2 < 0:
            raise ValueError("downforce_area_m2 cannot be negative")
        if not 0 <= self.front_downforce_fraction <= 1:
            raise ValueError("front_downforce_fraction must be between 0 and 1")

    def reset_state(self) -> None:
        """Clear the most recently calculated aerodynamic forces."""

        self.current_forces_n = AeroForces(0.0, 0.0, 0.0, 0.0)

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

        dynamic_pressure_pa = (
            0.5 * air_density_kgpm3 * vehicle_speed_mps**2
        )
        drag_n = dynamic_pressure_pa * self.drag_area_m2
        downforce_n = dynamic_pressure_pa * self.downforce_area_m2
        front_downforce_n = self.front_downforce_fraction * downforce_n
        rear_downforce_n = downforce_n - front_downforce_n
        return AeroForces(
            drag_n=drag_n,
            downforce_n=downforce_n,
            front_downforce_n=front_downforce_n,
            rear_downforce_n=rear_downforce_n,
        )
