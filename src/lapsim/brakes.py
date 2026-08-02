"""Braking-system model and maximum-deceleration calculation."""

from dataclasses import dataclass, field
from math import sqrt
from typing import TYPE_CHECKING

from scipy.optimize import brentq

if TYPE_CHECKING:
    from .tire import Tire
    from .vehicle import Vehicle


DEFAULT_SOLVER_TOLERANCE_MPS2 = 1e-6


@dataclass(slots=True)
class Brakes:
    """Ideal four-wheel brakes limited by the available tire grip.

    This initial model assumes ideal front/rear brake-force allocation. It
    includes longitudinal load transfer, tire load sensitivity, and the grip
    already consumed by cornering.
    """

    solver_tolerance_mps2: float = DEFAULT_SOLVER_TOLERANCE_MPS2
    current_force_request_n: float = field(init=False, default=0.0)
    current_friction_force_n: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable braking-model parameters."""

        if self.solver_tolerance_mps2 <= 0:
            raise ValueError("solver_tolerance_mps2 must be positive")

    def reset_state(self) -> None:
        """Clear the current brake command and friction force."""

        self.current_force_request_n = 0.0
        self.current_friction_force_n = 0.0

    def update_state(
        self,
        force_request_n: float,
        friction_force_n: float,
        timestep_s: float,
    ) -> None:
        """Retain the brake operating point for one timestep."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        if force_request_n < 0:
            raise ValueError("force_request_n cannot be negative")
        if friction_force_n < 0:
            raise ValueError("friction_force_n cannot be negative")
        if friction_force_n > force_request_n:
            raise ValueError("friction_force_n cannot exceed force_request_n")
        self.current_force_request_n = force_request_n
        self.current_friction_force_n = friction_force_n


    def maximum_deceleration_mps2(
        self,
        vehicle: "Vehicle",
        speed_mps: float,
        curvature_per_m: float,
        gravity_mps2: float,
        air_density_kgpm3: float,
    ) -> float:
        """Return the greatest feasible positive deceleration magnitude."""

        self.validate()
        aero_forces = vehicle.aero.forces_n(speed_mps, air_density_kgpm3)
        total_normal_force_n = (
            vehicle.mass_kg * gravity_mps2 + aero_forces.downforce_n
        )
        drag_force_n = aero_forces.drag_n
        rolling_force_n = (
            vehicle.rolling_resistance_coefficient * total_normal_force_n
        )

        total_lateral_force_n = (
            vehicle.mass_kg * speed_mps**2 * abs(curvature_per_m)
        )
        front_lateral_force_n = (
            vehicle.chassis.static_front_weight_fraction
            * total_lateral_force_n
        )
        rear_lateral_force_n = total_lateral_force_n - front_lateral_force_n

        def braking_residual(deceleration_mps2: float) -> float:
            tire_normal_loads = vehicle.chassis.tire_normal_loads_n(
                vehicle.mass_kg,
                gravity_mps2,
                aero_forces,
                longitudinal_acceleration_mps2=-deceleration_mps2,
            )
            front_braking_force_n = self._axle_braking_force_n(
                vehicle.tire,
                tire_normal_loads.front_n,
                front_lateral_force_n,
            )
            rear_braking_force_n = self._axle_braking_force_n(
                vehicle.tire,
                tire_normal_loads.rear_n,
                rear_lateral_force_n,
            )
            force_limited_deceleration_mps2 = (
                front_braking_force_n
                + rear_braking_force_n
                + drag_force_n
                + rolling_force_n
            ) / vehicle.effective_longitudinal_mass_kg
            return deceleration_mps2 - force_limited_deceleration_mps2

        maximum_tire_force_n = (
            max(vehicle.tire.longitudinal_coefficients)
            * total_normal_force_n
        )
        upper_deceleration_mps2 = (
            maximum_tire_force_n + drag_force_n + rolling_force_n
        ) / vehicle.effective_longitudinal_mass_kg

        return brentq(
            braking_residual,
            0.0,
            upper_deceleration_mps2,
            xtol=self.solver_tolerance_mps2,
        )

    @classmethod
    def _axle_braking_force_n(
        cls,
        tire: "Tire",
        tire_normal_loads_n: tuple[float, float],
        axle_lateral_force_n: float,
    ) -> float:
        """Return combined braking grip for the two tires on one axle."""

        lateral_force_limit_n = cls._tire_force_capacity_n(
            tire,
            tire_normal_loads_n,
            lateral=True,
        )
        if lateral_force_limit_n <= 0:
            return 0.0

        lateral_utilization = min(
            1.0,
            axle_lateral_force_n / lateral_force_limit_n,
        )
        remaining_longitudinal_fraction = sqrt(
            max(0.0, 1.0 - lateral_utilization**2)
        )
        longitudinal_force_limit_n = cls._tire_force_capacity_n(
            tire,
            tire_normal_loads_n,
            lateral=False,
        )
        return longitudinal_force_limit_n * remaining_longitudinal_fraction

    @staticmethod
    def _tire_force_capacity_n(
        tire: "Tire",
        tire_normal_loads_n: tuple[float, ...],
        *,
        lateral: bool,
    ) -> float:
        if lateral:
            return sum(
                tire.lateral_force_capacity_n(normal_load_n)
                for normal_load_n in tire_normal_loads_n
            )
        return sum(
            tire.longitudinal_force_capacity_n(normal_load_n)
            for normal_load_n in tire_normal_loads_n
        )
