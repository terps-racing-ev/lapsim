"""Load-sensitive, combined-slip force model for all four tires."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import copysign, exp, isfinite, sqrt

from utils.units import inches_to_meters, pounds_force_to_newtons

from .loads import TireCornerValues, TireForces, TireNormalLoads
from .pacejka import Pacejka61LateralModel

DEFAULT_ROLLING_RADIUS_IN = 8.0
DEFAULT_ROLLING_RADIUS_M = inches_to_meters(DEFAULT_ROLLING_RADIUS_IN)
DEFAULT_PEAK_LONGITUDINAL_SLIP_RATIO = 0.10
DEFAULT_NORMAL_LOADS_N = tuple(
    pounds_force_to_newtons(load_lbf)
    for load_lbf in (50.0, 100.0, 150.0, 200.0, 250.0)
)
DEFAULT_LATERAL_COEFFICIENTS = (1.625, 1.575, 1.525, 1.475, 1.425)
DEFAULT_LONGITUDINAL_COEFFICIENTS = DEFAULT_LATERAL_COEFFICIENTS
TIRE_POSITIONS = ("front_left", "front_right", "rear_left", "rear_right")


@dataclass(frozen=True, slots=True)
class TireState:
    """One contact patch's solved force, capacity, and slip state."""

    normal_load_n: float = 0.0
    lateral_force_n: float = 0.0
    lateral_capacity_n: float = 0.0
    longitudinal_capacity_n: float = 0.0
    drive_force_n: float = 0.0
    braking_force_n: float = 0.0
    slip_ratio: float = 0.0
    wheel_surface_speed_mps: float = 0.0

    @property
    def longitudinal_force_n(self) -> float:
        """Signed force on the vehicle: propulsion positive, braking negative."""

        return self.drive_force_n - self.braking_force_n

    @property
    def force_utilization(self) -> float:
        if self.longitudinal_capacity_n <= 0.0:
            return 0.0
        return min(
            (self.drive_force_n + self.braking_force_n)
            / self.longitudinal_capacity_n,
            1.0,
        )


@dataclass(frozen=True, slots=True)
class TireStates:
    """Solved operating points for the four contact patches."""

    vehicle_speed_mps: float = 0.0
    front_left: TireState = field(default_factory=TireState)
    front_right: TireState = field(default_factory=TireState)
    rear_left: TireState = field(default_factory=TireState)
    rear_right: TireState = field(default_factory=TireState)

    @property
    def all(self) -> tuple[TireState, TireState, TireState, TireState]:
        return self.front_left, self.front_right, self.rear_left, self.rear_right

    @property
    def front(self) -> tuple[TireState, TireState]:
        return self.front_left, self.front_right

    @property
    def rear(self) -> tuple[TireState, TireState]:
        return self.rear_left, self.rear_right

    @staticmethod
    def _sum(states: tuple[TireState, ...], name: str) -> float:
        return sum(getattr(state, name) for state in states)

    @property
    def drive_force_n(self) -> float:
        return self._sum(self.all, "drive_force_n")

    @property
    def braking_force_n(self) -> float:
        return self._sum(self.all, "braking_force_n")

    @property
    def longitudinal_force_n(self) -> float:
        return self._sum(self.all, "longitudinal_force_n")

    @property
    def lateral_force_n(self) -> float:
        return self._sum(self.all, "lateral_force_n")

    @property
    def longitudinal_capacity_n(self) -> float:
        return self._sum(self.all, "longitudinal_capacity_n")

    @property
    def front_braking_force_n(self) -> float:
        return self._sum(self.front, "braking_force_n")

    @property
    def rear_braking_force_n(self) -> float:
        return self._sum(self.rear, "braking_force_n")

    @property
    def rear_longitudinal_capacity_n(self) -> float:
        return self._sum(self.rear, "longitudinal_capacity_n")

    @property
    def driven_wheel_surface_speed_mps(self) -> float:
        return 0.5 * sum(state.wheel_surface_speed_mps for state in self.rear)

    @property
    def driven_slip_ratio(self) -> float:
        return 0.5 * sum(state.slip_ratio for state in self.rear)


@dataclass(slots=True)
class Tire:
    """Resolve load-sensitive combined forces and slip at every tire."""

    rolling_radius_m: float = DEFAULT_ROLLING_RADIUS_M
    normal_loads_n: list[float] = field(
        default_factory=lambda: list(DEFAULT_NORMAL_LOADS_N)
    )
    lateral_coefficients: list[float] = field(
        default_factory=lambda: list(DEFAULT_LATERAL_COEFFICIENTS)
    )
    longitudinal_coefficients: list[float] = field(
        default_factory=lambda: list(DEFAULT_LONGITUDINAL_COEFFICIENTS)
    )
    # The lookup-table lateral scale is the endurance-validated default.
    # Pacejka remains opt-in until its absolute force scale is validated.
    pacejka_lateral: Pacejka61LateralModel | None = None
    camber_angle_rad: float = 0.0
    inflation_pressure_pa: float = 98_000.0
    constant_friction_coefficient: float | None = None
    peak_longitudinal_slip_ratio: float = DEFAULT_PEAK_LONGITUDINAL_SLIP_RATIO
    longitudinal_slip_relaxation_length_m: float = 0.0
    current_states: TireStates = field(init=False, default_factory=TireStates)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate mutable lookup tables and slip parameters."""

        if self.rolling_radius_m <= 0:
            raise ValueError("rolling_radius_m must be positive")
        table_length = len(self.normal_loads_n)
        if table_length < 2:
            raise ValueError("A tire table requires at least two load points")
        if len(self.lateral_coefficients) != table_length:
            raise ValueError("lateral_coefficients must match normal_loads_n")
        if len(self.longitudinal_coefficients) != table_length:
            raise ValueError("longitudinal_coefficients must match normal_loads_n")
        if any(load <= 0 for load in self.normal_loads_n):
            raise ValueError("normal loads must be positive")
        if any(
            upper <= lower
            for lower, upper in zip(self.normal_loads_n, self.normal_loads_n[1:])
        ):
            raise ValueError("normal loads must be strictly increasing")
        if any(coefficient <= 0 for coefficient in self.lateral_coefficients):
            raise ValueError("lateral coefficients must be positive")
        if any(coefficient <= 0 for coefficient in self.longitudinal_coefficients):
            raise ValueError("longitudinal coefficients must be positive")
        if not isfinite(self.camber_angle_rad):
            raise ValueError("camber_angle_rad must be finite")
        if not isfinite(self.inflation_pressure_pa) or self.inflation_pressure_pa <= 0:
            raise ValueError("inflation_pressure_pa must be finite and positive")
        if self.pacejka_lateral is not None:
            self.pacejka_lateral.validate()
        if (
            self.constant_friction_coefficient is not None
            and self.constant_friction_coefficient <= 0
        ):
            raise ValueError("constant_friction_coefficient must be positive")
        if not 0.0 <= self.peak_longitudinal_slip_ratio < 1.0:
            raise ValueError("peak_longitudinal_slip_ratio must be in [0, 1)")
        if self.longitudinal_slip_relaxation_length_m < 0.0:
            raise ValueError("longitudinal slip relaxation length cannot be negative")

    def reset_state(self) -> None:
        self.current_states = TireStates()

    @property
    def current_longitudinal_force_n(self) -> float:
        return self.current_states.longitudinal_force_n

    @property
    def current_lateral_force_n(self) -> float:
        return self.current_states.lateral_force_n

    @property
    def maximum_longitudinal_coefficient(self) -> float:
        return (
            self.constant_friction_coefficient
            if self.constant_friction_coefficient is not None
            else max(self.longitudinal_coefficients)
        )

    def update_state(self, states: TireStates, timestep_s: float) -> None:
        """Commit one already-solved set of contact-patch states."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.current_states = states

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        states = self.current_states
        telemetry.update(
            {
                "tire.longitudinal_force_n": states.longitudinal_force_n,
                "tire.lateral_force_n": states.lateral_force_n,
                "tire.drive_force_n": states.drive_force_n,
                "tire.braking_force_n": states.braking_force_n,
                "tire.driven_slip_ratio": states.driven_slip_ratio,
                "tire.driven_slip_percent": 100.0 * states.driven_slip_ratio,
                "tire.vehicle_speed_mps": states.vehicle_speed_mps,
                "tire.driven_wheel_surface_speed_mps": (
                    states.driven_wheel_surface_speed_mps
                ),
                "tire.driven_slip_speed_mps": (
                    states.driven_wheel_surface_speed_mps - states.vehicle_speed_mps
                ),
                "tire.rolling_radius_m": self.rolling_radius_m,
                "tire.camber_angle_rad": self.camber_angle_rad,
                "tire.inflation_pressure_pa": self.inflation_pressure_pa,
                "tire.pacejka_lateral_active": float(
                    self.pacejka_lateral is not None
                    and self.constant_friction_coefficient is None
                ),
                "tire.constant_friction_coefficient": (
                    self.constant_friction_coefficient or 0.0
                ),
            }
        )
        for position, state in zip(TIRE_POSITIONS, states.all, strict=True):
            prefix = f"tire.{position}"
            telemetry.update(
                {
                    f"{prefix}.normal_load_n": state.normal_load_n,
                    f"{prefix}.longitudinal_force_n": state.longitudinal_force_n,
                    f"{prefix}.lateral_force_n": state.lateral_force_n,
                    f"{prefix}.drive_force_n": state.drive_force_n,
                    f"{prefix}.braking_force_n": state.braking_force_n,
                    f"{prefix}.longitudinal_capacity_n": (
                        state.longitudinal_capacity_n
                    ),
                    f"{prefix}.lateral_capacity_n": state.lateral_capacity_n,
                    f"{prefix}.force_utilization": state.force_utilization,
                    f"{prefix}.slip_ratio": state.slip_ratio,
                    f"{prefix}.slip_percent": 100.0 * state.slip_ratio,
                    f"{prefix}.wheel_surface_speed_mps": (
                        state.wheel_surface_speed_mps
                    ),
                }
            )

    def lateral_coefficient(self, normal_load_n: float) -> float:
        """Return lateral friction coefficient at one tire's normal load."""

        if self.constant_friction_coefficient is not None:
            return self.constant_friction_coefficient
        if self.pacejka_lateral is not None:
            if normal_load_n <= 0.0:
                return 0.0
            return self.lateral_force_capacity_n(normal_load_n) / normal_load_n
        return self._interpolate(normal_load_n, self.lateral_coefficients)

    def longitudinal_coefficient(self, normal_load_n: float) -> float:
        """Return longitudinal friction coefficient at one tire's normal load."""

        if self.constant_friction_coefficient is not None:
            return self.constant_friction_coefficient
        return self._interpolate(normal_load_n, self.longitudinal_coefficients)

    def lateral_force_capacity_n(self, normal_load_n: float) -> float:
        """Return one tire's lateral force capacity."""

        if self.constant_friction_coefficient is None and self.pacejka_lateral:
            return self.pacejka_lateral.peak_force_n(
                normal_load_n,
                camber_angle_rad=self.camber_angle_rad,
                inflation_pressure_pa=self.inflation_pressure_pa,
            )
        return self.lateral_coefficient(normal_load_n) * max(normal_load_n, 0.0)

    def pure_lateral_force_n(
        self,
        normal_load_n: float,
        slip_angle_rad: float,
        *,
        camber_angle_rad: float | None = None,
        inflation_pressure_pa: float | None = None,
    ) -> float:
        """Evaluate the configured MF-Tyre 6.1 pure-lateral force curve."""

        if self.pacejka_lateral is None:
            raise RuntimeError("pure lateral force requires a Pacejka lateral model")
        return self.pacejka_lateral.force_n(
            normal_load_n,
            slip_angle_rad,
            camber_angle_rad=(
                self.camber_angle_rad
                if camber_angle_rad is None
                else camber_angle_rad
            ),
            inflation_pressure_pa=(
                self.inflation_pressure_pa
                if inflation_pressure_pa is None
                else inflation_pressure_pa
            ),
        )

    def longitudinal_force_capacity_n(self, normal_load_n: float) -> float:
        """Return one tire's longitudinal force capacity."""

        return self.longitudinal_coefficient(normal_load_n) * max(normal_load_n, 0.0)

    def combined_longitudinal_force_capacity_n(
        self,
        normal_load_n: float,
        lateral_force_n: float,
    ) -> float:
        """Return longitudinal grip remaining after lateral-force demand."""

        lateral_capacity_n = self.lateral_force_capacity_n(normal_load_n)
        if lateral_capacity_n <= 0:
            return 0.0
        lateral_utilization = min(abs(lateral_force_n) / lateral_capacity_n, 1.0)
        return self.longitudinal_force_capacity_n(normal_load_n) * sqrt(
            max(0.0, 1.0 - lateral_utilization**2)
        )

    def lateral_forces_n(
        self,
        normal_loads_n: TireNormalLoads,
        total_lateral_force_n: float,
    ) -> TireForces:
        """Distribute lateral demand at equal per-tire capacity utilization."""

        capacities = tuple(
            self.lateral_force_capacity_n(load_n) for load_n in normal_loads_n.all_n
        )
        total_capacity_n = sum(capacities)
        if total_capacity_n <= 0.0:
            return TireCornerValues.zeros()
        force_n = copysign(
            min(abs(total_lateral_force_n), total_capacity_n),
            total_lateral_force_n,
        )
        return TireCornerValues.from_iterable(
            force_n * capacity_n / total_capacity_n for capacity_n in capacities
        )

    def calculate_forces(
        self,
        normal_loads_n: TireNormalLoads,
        total_lateral_force_n: float,
        drive_force_request_n: float,
        front_brake_force_request_n: float,
        rear_brake_force_request_n: float,
        vehicle_speed_mps: float,
        timestep_s: float,
    ) -> TireStates:
        """Solve all four contact patches without mutating tire state."""

        if min(
            drive_force_request_n,
            front_brake_force_request_n,
            rear_brake_force_request_n,
            vehicle_speed_mps,
        ) < 0.0:
            raise ValueError("tire force requests and vehicle speed cannot be negative")
        if timestep_s <= 0.0:
            raise ValueError("timestep_s must be positive")

        lateral_forces_n = self.lateral_forces_n(
            normal_loads_n, total_lateral_force_n
        )
        capacities_n = tuple(
            self.combined_longitudinal_force_capacity_n(load_n, lateral_n)
            for load_n, lateral_n in zip(
                normal_loads_n.all_n, lateral_forces_n.all_n, strict=True
            )
        )
        brake_requests_n = (
            front_brake_force_request_n / 2.0,
            front_brake_force_request_n / 2.0,
            rear_brake_force_request_n / 2.0,
            rear_brake_force_request_n / 2.0,
        )
        braking_forces_n = tuple(
            min(request_n, capacity_n)
            for request_n, capacity_n in zip(
                brake_requests_n, capacities_n, strict=True
            )
        )
        rear_drive_capacities_n = tuple(
            max(capacity_n - brake_force_n, 0.0)
            for capacity_n, brake_force_n in zip(
                capacities_n[2:], braking_forces_n[2:], strict=True
            )
        )
        total_rear_drive_capacity_n = sum(rear_drive_capacities_n)
        # Preserve the endurance baseline's ideal limited-slip assumption by
        # biasing rear drive force toward the contact patch with available grip.
        rear_drive_forces_n = (
            tuple(
                min(
                    drive_force_request_n
                    * capacity_n
                    / total_rear_drive_capacity_n,
                    capacity_n,
                )
                for capacity_n in rear_drive_capacities_n
            )
            if total_rear_drive_capacity_n > 0.0
            else (0.0, 0.0)
        )
        drive_forces_n = (0.0, 0.0) + rear_drive_forces_n

        states = []
        for load_n, lateral_n, capacity_n, drive_n, brake_n, previous in zip(
            normal_loads_n.all_n,
            lateral_forces_n.all_n,
            capacities_n,
            drive_forces_n,
            braking_forces_n,
            self.current_states.all,
            strict=True,
        ):
            drive_n, brake_n, slip_ratio = self._longitudinal_response(
                drive_n,
                brake_n,
                capacity_n,
                previous.slip_ratio,
                vehicle_speed_mps,
                timestep_s,
            )
            states.append(
                TireState(
                    normal_load_n=load_n,
                    lateral_force_n=lateral_n,
                    lateral_capacity_n=self.lateral_force_capacity_n(load_n),
                    longitudinal_capacity_n=capacity_n,
                    drive_force_n=drive_n,
                    braking_force_n=brake_n,
                    slip_ratio=slip_ratio,
                    wheel_surface_speed_mps=vehicle_speed_mps * (1.0 + slip_ratio),
                )
            )
        return TireStates(vehicle_speed_mps, *states)

    def _longitudinal_response(
        self,
        drive_force_n: float,
        braking_force_n: float,
        capacity_n: float,
        previous_slip_ratio: float,
        vehicle_speed_mps: float,
        timestep_s: float,
    ) -> tuple[float, float, float]:
        """Apply the tire-owned force/slip curve and optional relaxation."""

        force_magnitude_n = drive_force_n + braking_force_n
        if capacity_n <= 0.0 or self.peak_longitudinal_slip_ratio == 0.0:
            return drive_force_n, braking_force_n, 0.0
        direction = 1.0 if drive_force_n >= braking_force_n else -1.0
        target_slip = (
            direction
            * self.peak_longitudinal_slip_ratio
            * min(force_magnitude_n / capacity_n, 1.0)
        )
        if self.longitudinal_slip_relaxation_length_m <= 0.0:
            return drive_force_n, braking_force_n, target_slip
        relaxation_fraction = 1.0 - exp(
            -max(vehicle_speed_mps, 0.1)
            * timestep_s
            / self.longitudinal_slip_relaxation_length_m
        )
        slip_ratio = previous_slip_ratio + relaxation_fraction * (
            target_slip - previous_slip_ratio
        )
        available_force_n = capacity_n * min(
            abs(slip_ratio) / self.peak_longitudinal_slip_ratio, 1.0
        )
        scale = (
            min(available_force_n / force_magnitude_n, 1.0)
            if force_magnitude_n
            else 0.0
        )
        return drive_force_n * scale, braking_force_n * scale, slip_ratio

    def _interpolate(
        self,
        normal_load_n: float,
        coefficients: Sequence[float],
    ) -> float:
        clamped_load_n = min(
            max(normal_load_n, self.normal_loads_n[0]), self.normal_loads_n[-1]
        )
        for index, upper_load_n in enumerate(self.normal_loads_n[1:], start=1):
            if clamped_load_n <= upper_load_n:
                lower_load_n = self.normal_loads_n[index - 1]
                fraction = (clamped_load_n - lower_load_n) / (
                    upper_load_n - lower_load_n
                )
                return coefficients[index - 1] + fraction * (
                    coefficients[index] - coefficients[index - 1]
                )
        return coefficients[-1]
