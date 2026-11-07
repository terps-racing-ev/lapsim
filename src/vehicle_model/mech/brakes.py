"""Brake hardware configuration, hydraulic requests, and retained brake state."""

from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Literal

from utils.units import pound_force_inches_to_newton_meters

from ..limits import maximum_braking_deceleration_mps2
from .brake_hydraulics import BrakePressureMap

if TYPE_CHECKING:
    from ..vehicle import Vehicle


DEFAULT_SOLVER_TOLERANCE_MPS2 = 1e-6
DEFAULT_MAXIMUM_BRAKE_PRESSURE_PSI = 800.0
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
    pressure_force_model: Literal["linear-hardware-gains", "firmware-force-map"] = (
        "linear-hardware-gains"
    )
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
        if not isfinite(self.maximum_pressure_psi) or self.maximum_pressure_psi <= 0:
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

    def _pressure_map(self) -> BrakePressureMap:
        """Read current gains each time so setup edits take effect immediately."""

        return BrakePressureMap(
            front_torque_per_pressure_nm_per_psi=self.front_torque_per_pressure_nm_per_psi,
            rear_torque_per_pressure_nm_per_psi=self.rear_torque_per_pressure_nm_per_psi,
            maximum_pressure_psi=self.maximum_pressure_psi,
            pressure_deadband_psi=self.pressure_deadband_psi,
            pressure_force_model=self.pressure_force_model,
            maximum_force_request_n=self.maximum_force_request_n,
        )

    def axle_force_requests_from_pressures_n(
        self,
        front_pressure_psi: float,
        rear_pressure_psi: float,
        rolling_radius_m: float,
    ) -> tuple[float, float]:
        """Convert measured pressures to axle requests before tire grip limits."""

        return self._pressure_map().axle_force_requests_from_pressures_n(
            front_pressure_psi, rear_pressure_psi, rolling_radius_m
        )

    def axle_pressures_for_force_requests_psi(
        self,
        front_force_n: float,
        rear_force_n: float,
        rolling_radius_m: float,
    ) -> tuple[float, float]:
        """Invert the configured pressure map for an offline driver command."""

        return self._pressure_map().axle_pressures_for_force_requests_psi(
            front_force_n, rear_force_n, rolling_radius_m
        )

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
        """Query the whole-vehicle braking envelope, including tire grip.

        This compatibility entry point delegates to vehicle_model.limits;
        pressure maps and brake state stay in the mechanical subsystem.
        """

        return maximum_braking_deceleration_mps2(
            vehicle,
            speed_mps,
            curvature_per_m,
            gravity_mps2,
            air_density_kgpm3,
            self.solver_tolerance_mps2,
        )
