"""Chassis geometry, weight distribution, and tire normal loads."""

from dataclasses import dataclass, field

from .aero import AeroForces
from .utils.units import inches_to_meters


DEFAULT_WHEELBASE_IN = 61.0
DEFAULT_CG_HEIGHT_IN = 11.0
DEFAULT_WHEELBASE_M = inches_to_meters(DEFAULT_WHEELBASE_IN)
DEFAULT_CG_HEIGHT_M = inches_to_meters(DEFAULT_CG_HEIGHT_IN)
DEFAULT_STATIC_FRONT_WEIGHT_FRACTION = 0.475


@dataclass(frozen=True, slots=True)
class TireNormalLoads:
    """Normal force carried by each tire."""

    front_left_n: float
    front_right_n: float
    rear_left_n: float
    rear_right_n: float

    @property
    def front_n(self) -> tuple[float, float]:
        return self.front_left_n, self.front_right_n

    @property
    def rear_n(self) -> tuple[float, float]:
        return self.rear_left_n, self.rear_right_n

    @property
    def all_n(self) -> tuple[float, float, float, float]:
        return self.front_n + self.rear_n

    @property
    def front_axle_n(self) -> float:
        return sum(self.front_n)

    @property
    def rear_axle_n(self) -> float:
        return sum(self.rear_n)


@dataclass(slots=True)
class Chassis:
    """Basic chassis model with longitudinal load transfer."""

    wheelbase_m: float = DEFAULT_WHEELBASE_M
    cg_height_m: float = DEFAULT_CG_HEIGHT_M
    static_front_weight_fraction: float = (
        DEFAULT_STATIC_FRONT_WEIGHT_FRACTION
    )
    current_tire_normal_loads_n: TireNormalLoads = field(
        init=False,
        default_factory=lambda: TireNormalLoads(0.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable chassis parameters."""

        if self.wheelbase_m <= 0:
            raise ValueError("wheelbase_m must be positive")
        if self.cg_height_m <= 0:
            raise ValueError("cg_height_m must be positive")
        if not 0 <= self.static_front_weight_fraction <= 1:
            raise ValueError(
                "static_front_weight_fraction must be between 0 and 1"
            )

    def reset_state(self) -> None:
        """Clear the most recently calculated tire normal loads."""

        self.current_tire_normal_loads_n = TireNormalLoads(0.0, 0.0, 0.0, 0.0)

    def update_state(
        self,
        tire_normal_loads_n: TireNormalLoads,
        timestep_s: float,
    ) -> None:
        """Retain the tire normal loads for one timestep."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.current_tire_normal_loads_n = tire_normal_loads_n

    def tire_normal_loads_n(
        self,
        mass_kg: float,
        gravity_mps2: float,
        aero_forces: AeroForces,
        longitudinal_acceleration_mps2: float = 0.0,
    ) -> TireNormalLoads:
        """Calculate tire loads; positive acceleration transfers load rearward."""

        static_front_axle_n = (
            self.static_front_weight_fraction * mass_kg * gravity_mps2
        )
        static_rear_axle_n = (
            (1.0 - self.static_front_weight_fraction)
            * mass_kg
            * gravity_mps2
        )
        front_without_transfer_n = (
            static_front_axle_n + aero_forces.front_downforce_n
        )
        rear_without_transfer_n = (
            static_rear_axle_n + aero_forces.rear_downforce_n
        )
        rearward_load_transfer_n = (
            mass_kg
            * longitudinal_acceleration_mps2
            * self.cg_height_m
            / self.wheelbase_m
        )
        rearward_load_transfer_n = min(
            max(rearward_load_transfer_n, -rear_without_transfer_n),
            front_without_transfer_n,
        )
        front_axle_n = front_without_transfer_n - rearward_load_transfer_n
        rear_axle_n = rear_without_transfer_n + rearward_load_transfer_n

        # Lateral load transfer is not modeled yet, so each axle is split evenly.
        return TireNormalLoads(
            front_left_n=front_axle_n / 2.0,
            front_right_n=front_axle_n / 2.0,
            rear_left_n=rear_axle_n / 2.0,
            rear_right_n=rear_axle_n / 2.0,
        )
