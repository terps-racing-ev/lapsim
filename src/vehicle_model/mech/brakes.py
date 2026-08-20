"""Mechanical-subteam braking model and deceleration calculation."""

from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Literal

from scipy.optimize import brentq

from utils.units import pound_force_inches_to_newton_meters

if TYPE_CHECKING:
    from ..vehicle import Vehicle


DEFAULT_SOLVER_TOLERANCE_MPS2 = 1e-6
DEFAULT_MAXIMUM_BRAKE_PRESSURE_PSI = 300.0
DEFAULT_FRONT_BRAKE_TORQUE_PER_PSI_LBFIN = 10.12849472
DEFAULT_REAR_BRAKE_TORQUE_PER_PSI_LBFIN = 5.390972994


@dataclass(slots=True)
class Brakes:
    """Pressure-actuated four-wheel brakes limited by available tire grip.

    The configured axle gains convert measured line pressure to wheel torque
    and then vehicle force. The resulting requests remain subject to
    longitudinal load transfer, tire load sensitivity, and grip already
    consumed by cornering.
    """

    solver_tolerance_mps2: float = DEFAULT_SOLVER_TOLERANCE_MPS2
    front_torque_per_pressure_lbfin_per_psi: float = (
        DEFAULT_FRONT_BRAKE_TORQUE_PER_PSI_LBFIN
    )
    rear_torque_per_pressure_lbfin_per_psi: float = (
        DEFAULT_REAR_BRAKE_TORQUE_PER_PSI_LBFIN
    )
    maximum_pressure_psi: float = DEFAULT_MAXIMUM_BRAKE_PRESSURE_PSI
    pressure_deadband_psi: float = 0.0
    pressure_force_model: Literal[
        "linear-hardware-gains", "firmware-force-map"
    ] = "linear-hardware-gains"
    maximum_force_request_n: float | None = None
    current_front_pressure_psi: float = field(init=False, default=0.0)
    current_rear_pressure_psi: float = field(init=False, default=0.0)
    current_force_request_n: float = field(init=False, default=0.0)
    current_friction_force_n: float = field(init=False, default=0.0)
    current_front_force_request_n: float = field(init=False, default=0.0)
    current_rear_force_request_n: float = field(init=False, default=0.0)
    current_front_friction_force_n: float = field(init=False, default=0.0)
    current_rear_friction_force_n: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable braking-model parameters."""

        if self.solver_tolerance_mps2 <= 0:
            raise ValueError("solver_tolerance_mps2 must be positive")
        if self.front_torque_per_pressure_lbfin_per_psi < 0:
            raise ValueError("front brake torque gain cannot be negative")
        if self.rear_torque_per_pressure_lbfin_per_psi < 0:
            raise ValueError("rear brake torque gain cannot be negative")
        if (
            not isfinite(self.maximum_pressure_psi)
            or self.maximum_pressure_psi <= 0
        ):
            raise ValueError("maximum_pressure_psi must be finite and positive")
        if self.pressure_deadband_psi < 0:
            raise ValueError("pressure_deadband_psi cannot be negative")
        if self.pressure_force_model not in {
            "linear-hardware-gains",
            "firmware-force-map",
        }:
            raise ValueError(
                f"Unknown pressure_force_model: {self.pressure_force_model}"
            )
        if (
            self.maximum_force_request_n is not None
            and self.maximum_force_request_n <= 0
        ):
            raise ValueError("maximum_force_request_n must be positive")

    @property
    def front_torque_per_pressure_nm_per_psi(self) -> float:
        return pound_force_inches_to_newton_meters(
            self.front_torque_per_pressure_lbfin_per_psi
        )

    @property
    def rear_torque_per_pressure_nm_per_psi(self) -> float:
        return pound_force_inches_to_newton_meters(
            self.rear_torque_per_pressure_lbfin_per_psi
        )

    @property
    def total_torque_per_pressure_nm_per_psi(self) -> float:
        """Return the combined front-plus-rear axle brake-torque gain."""

        return (
            self.front_torque_per_pressure_nm_per_psi
            + self.rear_torque_per_pressure_nm_per_psi
        )

    def equivalent_vehicle_force_per_pressure_n_per_psi(
        self, rolling_radius_m: float
    ) -> float:
        """Convert the combined axle torque gain to vehicle brake force."""

        if rolling_radius_m <= 0:
            raise ValueError("rolling_radius_m must be positive")
        return self.total_torque_per_pressure_nm_per_psi / rolling_radius_m

    def axle_force_requests_from_pressures_n(
        self,
        front_pressure_psi: float,
        rear_pressure_psi: float,
        rolling_radius_m: float,
    ) -> tuple[float, float]:
        """Convert measured hydraulic pressures into axle force requests."""

        if front_pressure_psi < 0 or rear_pressure_psi < 0:
            raise ValueError("brake pressures cannot be negative")
        if rolling_radius_m <= 0:
            raise ValueError("rolling_radius_m must be positive")

        front_pressure_psi = min(front_pressure_psi, self.maximum_pressure_psi)
        rear_pressure_psi = min(rear_pressure_psi, self.maximum_pressure_psi)
        front_effective_psi = max(
            front_pressure_psi - self.pressure_deadband_psi, 0.0
        )
        rear_effective_psi = max(
            rear_pressure_psi - self.pressure_deadband_psi, 0.0
        )
        if self.pressure_force_model == "linear-hardware-gains":
            front_force_n = (
                front_effective_psi
                * self.front_torque_per_pressure_nm_per_psi
                / rolling_radius_m
            )
            rear_force_n = (
                rear_effective_psi
                * self.rear_torque_per_pressure_nm_per_psi
                / rolling_radius_m
            )
        else:
            front_force_n = (
                183.0
                + 8.7696 * front_pressure_psi
                - 0.0141145 * front_pressure_psi**2
                + 0.0000068383 * front_pressure_psi**3
                if front_effective_psi > 0.0
                else 0.0
            )
            rear_force_n = (
                5.6077 * rear_pressure_psi if rear_effective_psi > 0.0 else 0.0
            )

        total_force_n = front_force_n + rear_force_n
        if (
            self.maximum_force_request_n is not None
            and total_force_n > self.maximum_force_request_n
        ):
            scale = self.maximum_force_request_n / total_force_n
            front_force_n *= scale
            rear_force_n *= scale
        return front_force_n, rear_force_n

    def axle_pressures_for_force_requests_psi(
        self,
        front_force_n: float,
        rear_force_n: float,
        rolling_radius_m: float,
    ) -> tuple[float, float]:
        """Invert the configured pressure map for an offline driver command."""

        if front_force_n < 0 or rear_force_n < 0:
            raise ValueError("brake force requests cannot be negative")
        if rolling_radius_m <= 0:
            raise ValueError("rolling_radius_m must be positive")

        if self.pressure_force_model == "linear-hardware-gains":
            def pressure(force_n: float, gain_nm_per_psi: float) -> float:
                if force_n <= 0.0:
                    return 0.0
                if gain_nm_per_psi <= 0.0:
                    raise ValueError(
                        "cannot request brake force with a zero pressure gain"
                    )
                return min(
                    force_n * rolling_radius_m / gain_nm_per_psi
                    + self.pressure_deadband_psi,
                    self.maximum_pressure_psi,
                )

            return (
                pressure(front_force_n, self.front_torque_per_pressure_nm_per_psi),
                pressure(rear_force_n, self.rear_torque_per_pressure_nm_per_psi),
            )

        def invert(force_n: float, *, front: bool) -> float:
            if force_n <= 0.0:
                return 0.0
            lower_psi = self.pressure_deadband_psi
            upper_psi = min(max(lower_psi + 1.0, 1.0), self.maximum_pressure_psi)
            while True:
                forces = self.axle_force_requests_from_pressures_n(
                    upper_psi if front else 0.0,
                    0.0 if front else upper_psi,
                    rolling_radius_m,
                )
                if forces[0 if front else 1] >= force_n:
                    break
                if upper_psi >= self.maximum_pressure_psi:
                    return self.maximum_pressure_psi
                upper_psi = min(2.0 * upper_psi, self.maximum_pressure_psi)
            for _ in range(32):
                candidate_psi = 0.5 * (lower_psi + upper_psi)
                forces = self.axle_force_requests_from_pressures_n(
                    candidate_psi if front else 0.0,
                    0.0 if front else candidate_psi,
                    rolling_radius_m,
                )
                if forces[0 if front else 1] >= force_n:
                    upper_psi = candidate_psi
                else:
                    lower_psi = candidate_psi
            return upper_psi

        return invert(front_force_n, front=True), invert(rear_force_n, front=False)

    @property
    def front_brake_force_fraction(self) -> float:
        """Hydraulic/hardware brake-force split implied by the axle gains."""

        total = self.total_torque_per_pressure_nm_per_psi
        return self.front_torque_per_pressure_nm_per_psi / total if total > 0 else 0.5

    def reset_state(self) -> None:
        """Clear the current brake command and friction force."""

        self.current_front_pressure_psi = 0.0
        self.current_rear_pressure_psi = 0.0
        self.current_force_request_n = 0.0
        self.current_friction_force_n = 0.0
        self.current_front_force_request_n = 0.0
        self.current_rear_force_request_n = 0.0
        self.current_front_friction_force_n = 0.0
        self.current_rear_friction_force_n = 0.0

    def update_state(
        self,
        force_request_n: float,
        friction_force_n: float,
        timestep_s: float,
        *,
        front_pressure_psi: float = 0.0,
        rear_pressure_psi: float = 0.0,
        front_friction_force_n: float | None = None,
        rear_friction_force_n: float | None = None,
        front_force_request_n: float | None = None,
        rear_force_request_n: float | None = None,
    ) -> None:
        """Retain the brake operating point for one timestep."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        if force_request_n < 0:
            raise ValueError("force_request_n cannot be negative")
        if friction_force_n < 0:
            raise ValueError("friction_force_n cannot be negative")
        if friction_force_n > force_request_n + 1e-9:
            raise ValueError("friction_force_n cannot exceed force_request_n")
        if front_pressure_psi < 0 or rear_pressure_psi < 0:
            raise ValueError("brake pressures cannot be negative")
        self.current_front_pressure_psi = min(
            front_pressure_psi, self.maximum_pressure_psi
        )
        self.current_rear_pressure_psi = min(
            rear_pressure_psi, self.maximum_pressure_psi
        )
        self.current_force_request_n = force_request_n
        self.current_friction_force_n = friction_force_n
        front_fraction = self.front_brake_force_fraction
        self.current_front_force_request_n = (
            force_request_n * front_fraction
            if front_force_request_n is None
            else front_force_request_n
        )
        self.current_rear_force_request_n = (
            force_request_n * (1.0 - front_fraction)
            if rear_force_request_n is None
            else rear_force_request_n
        )
        if (
            abs(
                self.current_front_force_request_n
                + self.current_rear_force_request_n
                - force_request_n
            )
            > 1e-6
        ):
            raise ValueError(
                "front and rear force requests must sum to force_request_n"
            )
        self.current_front_friction_force_n = (
            friction_force_n * front_fraction
            if front_friction_force_n is None
            else front_friction_force_n
        )
        self.current_rear_friction_force_n = (
            friction_force_n * (1.0 - front_fraction)
            if rear_friction_force_n is None
            else rear_friction_force_n
        )

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry.update(
            {
                "brakes.front_pressure_psi": self.current_front_pressure_psi,
                "brakes.rear_pressure_psi": self.current_rear_pressure_psi,
                "brakes.maximum_pressure_psi": self.maximum_pressure_psi,
                "brakes.force_request_n": self.current_force_request_n,
                "brakes.friction_force_n": self.current_friction_force_n,
                "brakes.front_force_request_n": self.current_front_force_request_n,
                "brakes.rear_force_request_n": self.current_rear_force_request_n,
                "brakes.front_friction_force_n": self.current_front_friction_force_n,
                "brakes.rear_friction_force_n": self.current_rear_friction_force_n,
                "brakes.front_force_fraction": self.front_brake_force_fraction,
                "brakes.force_limited": float(
                    self.current_friction_force_n < self.current_force_request_n - 1e-9
                ),
                "brakes.front_torque_per_pressure_nm_per_psi": (
                    self.front_torque_per_pressure_nm_per_psi
                ),
                "brakes.rear_torque_per_pressure_nm_per_psi": (
                    self.rear_torque_per_pressure_nm_per_psi
                ),
                "brakes.total_torque_per_pressure_nm_per_psi": (
                    self.total_torque_per_pressure_nm_per_psi
                ),
            }
        )

    def maximum_deceleration_mps2(
        self,
        vehicle: "Vehicle",
        speed_mps: float,
        curvature_per_m: float,
        gravity_mps2: float,
        air_density_kgpm3: float,
    ) -> float:
        """Return the greatest feasible positive deceleration magnitude."""

        aero_forces = vehicle.aero_forces_n(
            speed_mps,
            speed_mps**2 * abs(curvature_per_m),
            air_density_kgpm3,
        )
        total_normal_force_n = vehicle.mass_kg * gravity_mps2 + aero_forces.downforce_n
        drag_force_n = aero_forces.drag_n
        rolling_force_n = vehicle.rolling_resistance_coefficient * total_normal_force_n

        total_lateral_force_n = vehicle.mass_kg * speed_mps**2 * abs(curvature_per_m)
        def braking_residual(deceleration_mps2: float) -> float:
            tire_normal_loads = vehicle.suspension.tire_normal_loads_n(
                vehicle.mass_kg,
                gravity_mps2,
                aero_forces,
                vehicle.chassis,
                longitudinal_acceleration_mps2=-deceleration_mps2,
                lateral_acceleration_mps2=(speed_mps**2 * abs(curvature_per_m)),
            )
            tire_lateral_forces_n = vehicle.tire.lateral_forces_n(
                tire_normal_loads, total_lateral_force_n
            )
            braking_force_n = sum(
                vehicle.tire.combined_longitudinal_force_capacity_n(
                    normal_load_n, lateral_force_n
                )
                for normal_load_n, lateral_force_n in zip(
                    tire_normal_loads.all_n,
                    tire_lateral_forces_n.all_n,
                    strict=True,
                )
            )
            force_limited_deceleration_mps2 = (
                braking_force_n + drag_force_n + rolling_force_n
            ) / vehicle.effective_longitudinal_mass_kg
            return deceleration_mps2 - force_limited_deceleration_mps2

        maximum_tire_force_n = (
            vehicle.tire.maximum_longitudinal_coefficient * total_normal_force_n
        )
        upper_deceleration_mps2 = (
            maximum_tire_force_n + drag_force_n + rolling_force_n
        ) / vehicle.effective_longitudinal_mass_kg
        # The ideal constant-mu solution can land exactly on this analytical
        # bound. Give Brent a small strict bracket so floating-point roundoff
        # cannot leave both residual endpoints on the same side of zero.
        upper_deceleration_mps2 = (
            1.01 * upper_deceleration_mps2 + self.solver_tolerance_mps2
        )

        return brentq(
            braking_residual,
            0.0,
            upper_deceleration_mps2,
            xtol=self.solver_tolerance_mps2,
        )
