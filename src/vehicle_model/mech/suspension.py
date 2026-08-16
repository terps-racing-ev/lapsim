"""Mechanical-subteam rigid-suspension load-distribution model."""

from dataclasses import dataclass, field

from ..aero import AeroForces
from ..interfaces import ChassisModel
from .loads import TireNormalLoads


@dataclass(slots=True)
class Suspension:
    """Longitudinal load transfer with equal left/right axle loads.

    More detailed implementations can add track widths, roll stiffness,
    springs, dampers, heave, pitch, and lateral load transfer while returning
    the same ``TireNormalLoads`` data contract.
    """

    current_tire_normal_loads_n: TireNormalLoads = field(
        init=False,
        default_factory=lambda: TireNormalLoads(0.0, 0.0, 0.0, 0.0),
    )

    def validate(self) -> None:
        """The baseline suspension has no configurable parameters."""

    def reset_state(self) -> None:
        self.current_tire_normal_loads_n = TireNormalLoads(0.0, 0.0, 0.0, 0.0)

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        loads = self.current_tire_normal_loads_n
        telemetry.update(
            {
                "suspension.front_left_normal_load_n": loads.front_left_n,
                "suspension.front_right_normal_load_n": loads.front_right_n,
                "suspension.rear_left_normal_load_n": loads.rear_left_n,
                "suspension.rear_right_normal_load_n": loads.rear_right_n,
                "suspension.front_axle_normal_load_n": loads.front_axle_n,
                "suspension.rear_axle_normal_load_n": loads.rear_axle_n,
            }
        )

    def update_state(
        self,
        tire_normal_loads_n: TireNormalLoads,
        timestep_s: float,
    ) -> None:
        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.current_tire_normal_loads_n = tire_normal_loads_n

    def tire_normal_loads_n(
        self,
        mass_kg: float,
        gravity_mps2: float,
        aero_forces: AeroForces,
        chassis: ChassisModel,
        longitudinal_acceleration_mps2: float = 0.0,
        lateral_acceleration_mps2: float = 0.0,
    ) -> TireNormalLoads:
        """Calculate loads; lateral acceleration is unused in this model."""

        static_front_axle_n = (
            chassis.static_front_weight_fraction * mass_kg * gravity_mps2
        )
        static_rear_axle_n = (
            (1.0 - chassis.static_front_weight_fraction) * mass_kg * gravity_mps2
        )
        front_without_transfer_n = static_front_axle_n + aero_forces.front_downforce_n
        rear_without_transfer_n = static_rear_axle_n + aero_forces.rear_downforce_n
        rearward_transfer_n = (
            mass_kg
            * longitudinal_acceleration_mps2
            * chassis.cg_height_m
            / chassis.wheelbase_m
        )
        rearward_transfer_n = min(
            max(rearward_transfer_n, -rear_without_transfer_n),
            front_without_transfer_n,
        )
        front_axle_n = front_without_transfer_n - rearward_transfer_n
        rear_axle_n = rear_without_transfer_n + rearward_transfer_n
        return TireNormalLoads(
            front_left_n=front_axle_n / 2.0,
            front_right_n=front_axle_n / 2.0,
            rear_left_n=rear_axle_n / 2.0,
            rear_right_n=rear_axle_n / 2.0,
        )
