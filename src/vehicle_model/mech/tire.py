"""Load-sensitive, combined-slip force model for all four tires."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import copysign, isfinite, radians, sqrt

from utils.units import inches_to_meters, pounds_force_to_newtons

from .wheels import WHEEL_NAMES, WHEEL_SIDE_SIGNS
from .wheel_state import TireKinematics, TireState, TireStates, WheelKinematics
from .tire_forces import calculate_contact_forces, longitudinal_response
from .loads import TireCornerValues, TireForces, TireNormalLoads
from .pacejka import (
    Pacejka52UpcR20LateralModel,
    Pacejka52UpcR20LongitudinalModel,
    Pacejka61LateralModel,
)

DEFAULT_ROLLING_RADIUS_IN = 8.0
DEFAULT_ROLLING_RADIUS_M = inches_to_meters(DEFAULT_ROLLING_RADIUS_IN)
DEFAULT_PEAK_LONGITUDINAL_SLIP_RATIO = 0.10
DEFAULT_PEAK_LATERAL_SLIP_ANGLE_RAD = radians(10.0)
DEFAULT_NORMAL_LOADS_N = tuple(
    pounds_force_to_newtons(load_lbf) for load_lbf in (50.0, 100.0, 150.0, 200.0, 250.0)
)
DEFAULT_LATERAL_COEFFICIENTS = (1.625, 1.575, 1.525, 1.475, 1.425)
DEFAULT_LONGITUDINAL_COEFFICIENTS = DEFAULT_LATERAL_COEFFICIENTS
TIRE_MODEL_ORIENTATIONS = WHEEL_SIDE_SIGNS


@dataclass(slots=True)
class Tire:
    """Resolve load-sensitive combined forces and slip at every tire.

    All camber inputs use vehicle-local alignment: negative is inward lean.
    The fitted Pacejka models use the opposite inclination sign, so convert
    with ``gamma = -camber`` at every coefficient-model call. Left/right
    mirroring separately transforms slip and lateral force in calculate_forces.
    """

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
    pacejka_lateral: Pacejka61LateralModel | Pacejka52UpcR20LateralModel | None = field(
        default_factory=Pacejka52UpcR20LateralModel
    )
    pacejka_longitudinal: Pacejka52UpcR20LongitudinalModel | None = field(
        default_factory=Pacejka52UpcR20LongitudinalModel
    )
    inflation_pressure_pa: float = 98_000.0
    constant_friction_coefficient: float | None = None
    peak_longitudinal_slip_ratio: float = DEFAULT_PEAK_LONGITUDINAL_SLIP_RATIO
    peak_lateral_slip_angle_rad: float = DEFAULT_PEAK_LATERAL_SLIP_ANGLE_RAD
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
        if not isfinite(self.inflation_pressure_pa) or self.inflation_pressure_pa <= 0:
            raise ValueError("inflation_pressure_pa must be finite and positive")
        if self.pacejka_lateral is not None:
            self.pacejka_lateral.validate()
        if self.pacejka_longitudinal is not None:
            self.pacejka_longitudinal.validate()
        if (
            self.constant_friction_coefficient is not None
            and self.constant_friction_coefficient <= 0
        ):
            raise ValueError("constant_friction_coefficient must be positive")
        if not 0.0 <= self.peak_longitudinal_slip_ratio < 1.0:
            raise ValueError("peak_longitudinal_slip_ratio must be in [0, 1)")
        if (
            not isfinite(self.peak_lateral_slip_angle_rad)
            or self.peak_lateral_slip_angle_rad <= 0.0
        ):
            raise ValueError("peak_lateral_slip_angle_rad must be finite and positive")
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
        if (
            self.constant_friction_coefficient is None
            and self.pacejka_longitudinal is not None
        ):
            reference_load_n = (
                self.pacejka_longitudinal.nominal_load_n
                * self.pacejka_longitudinal.nominal_load_scale
            )
            return (
                self.pacejka_longitudinal.peak_force_n(
                    reference_load_n,
                    camber_angle_rad=0.0,
                    inflation_pressure_pa=self.inflation_pressure_pa,
                )
                / reference_load_n
            )
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
                "tire.body_longitudinal_force_n": states.body_longitudinal_force_n,
                "tire.body_lateral_force_n": states.body_lateral_force_n,
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
                "tire.inflation_pressure_pa": self.inflation_pressure_pa,
                "tire.pacejka_lateral_active": float(
                    self.pacejka_lateral is not None
                    and self.constant_friction_coefficient is None
                ),
                "tire.lateral_correlation_factor": (
                    self.pacejka_lateral.correlation_factor
                    if self.pacejka_lateral is not None
                    else 1.0
                ),
                "tire.pacejka_longitudinal_active": float(
                    self.pacejka_longitudinal is not None
                    and self.constant_friction_coefficient is None
                ),
                "tire.longitudinal_correlation_factor": (
                    self.pacejka_longitudinal.correlation_factor
                    if self.pacejka_longitudinal is not None
                    else 1.0
                ),
                "tire.constant_friction_coefficient": (
                    self.constant_friction_coefficient or 0.0
                ),
            }
        )
        for position, state in zip(WHEEL_NAMES, states.all, strict=True):
            prefix = f"tire.{position}"
            telemetry.update(
                {
                    f"{prefix}.normal_load_n": state.normal_load_n,
                    f"{prefix}.longitudinal_force_n": state.longitudinal_force_n,
                    f"{prefix}.lateral_force_n": state.lateral_force_n,
                    f"{prefix}.body_longitudinal_force_n": (
                        state.body_longitudinal_force_n
                    ),
                    f"{prefix}.body_lateral_force_n": state.body_lateral_force_n,
                    f"{prefix}.drive_force_n": state.drive_force_n,
                    f"{prefix}.braking_force_n": state.braking_force_n,
                    f"{prefix}.steering_angle_rad": state.steering_angle_rad,
                    f"{prefix}.camber_angle_rad": state.camber_angle_rad,
                    f"{prefix}.local_longitudinal_speed_mps": (
                        state.local_longitudinal_speed_mps
                    ),
                    f"{prefix}.local_lateral_speed_mps": (
                        state.local_lateral_speed_mps
                    ),
                    f"{prefix}.slip_angle_rad": state.slip_angle_rad,
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
                    f"{prefix}.wheel_angular_speed_radps": (
                        state.wheel_angular_speed_radps
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
        if self.pacejka_longitudinal is not None:
            if normal_load_n <= 0.0:
                return 0.0
            return self.longitudinal_force_capacity_n(normal_load_n) / normal_load_n
        return self._interpolate(normal_load_n, self.longitudinal_coefficients)

    def lateral_force_capacity_n(
        self,
        normal_load_n: float,
        camber_angle_rad: float = 0.0,
    ) -> float:
        """Return one tire's lateral force capacity."""

        if not isfinite(camber_angle_rad):
            raise ValueError("camber_angle_rad must be finite")
        if self.constant_friction_coefficient is None and self.pacejka_lateral:
            return self.pacejka_lateral.peak_force_n(
                normal_load_n,
                camber_angle_rad=-camber_angle_rad,
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
        """Evaluate pure lateral force using vehicle-local camber (negative inward)."""

        if self.pacejka_lateral is None:
            raise RuntimeError("pure lateral force requires a Pacejka lateral model")
        return self.pacejka_lateral.force_n(
            normal_load_n,
            slip_angle_rad,
            camber_angle_rad=(0.0 if camber_angle_rad is None else -camber_angle_rad),
            inflation_pressure_pa=(
                self.inflation_pressure_pa
                if inflation_pressure_pa is None
                else inflation_pressure_pa
            ),
        )

    def longitudinal_force_capacity_n(
        self,
        normal_load_n: float,
        camber_angle_rad: float = 0.0,
    ) -> float:
        """Return one tire's longitudinal force capacity."""

        if not isfinite(camber_angle_rad):
            raise ValueError("camber_angle_rad must be finite")
        if self.constant_friction_coefficient is None and self.pacejka_longitudinal:
            return self.pacejka_longitudinal.peak_force_n(
                normal_load_n,
                camber_angle_rad=-camber_angle_rad,
                inflation_pressure_pa=self.inflation_pressure_pa,
            )
        return self.longitudinal_coefficient(normal_load_n) * max(normal_load_n, 0.0)

    def pure_longitudinal_force_n(
        self,
        normal_load_n: float,
        slip_ratio: float,
        *,
        camber_angle_rad: float | None = None,
        inflation_pressure_pa: float | None = None,
    ) -> float:
        """Evaluate pure longitudinal force using vehicle-local camber."""

        if self.pacejka_longitudinal is None:
            raise RuntimeError(
                "pure longitudinal force requires a Pacejka longitudinal model"
            )
        return self.pacejka_longitudinal.force_n(
            normal_load_n,
            slip_ratio,
            camber_angle_rad=(0.0 if camber_angle_rad is None else -camber_angle_rad),
            inflation_pressure_pa=(
                self.inflation_pressure_pa
                if inflation_pressure_pa is None
                else inflation_pressure_pa
            ),
        )

    def combined_longitudinal_force_capacity_n(
        self,
        normal_load_n: float,
        lateral_force_n: float,
        camber_angle_rad: float = 0.0,
    ) -> float:
        """Return longitudinal grip remaining after lateral-force demand."""

        lateral_capacity_n = self.lateral_force_capacity_n(
            normal_load_n,
            camber_angle_rad,
        )
        if lateral_capacity_n <= 0:
            return 0.0
        lateral_utilization = min(abs(lateral_force_n) / lateral_capacity_n, 1.0)
        return self.longitudinal_force_capacity_n(
            normal_load_n,
            camber_angle_rad,
        ) * sqrt(max(0.0, 1.0 - lateral_utilization**2))

    def lateral_forces_n(
        self,
        normal_loads_n: TireNormalLoads,
        total_lateral_force_n: float,
        *,
        front_lateral_force_fraction: float,
        camber_angles_rad: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    ) -> TireForces:
        """Distribute lateral demand while preserving the requested axle split.

        Within each axle, equal capacity utilization is retained as the simple
        left/right model.  Supplying the front fraction prevents grip on one
        axle from taking over lateral-force demand belonging to the other.
        """

        if len(camber_angles_rad) != 4 or not all(
            isfinite(angle_rad) for angle_rad in camber_angles_rad
        ):
            raise ValueError("four finite camber angles are required")
        capacities = tuple(
            self.lateral_force_capacity_n(load_n, camber_angle_rad)
            for load_n, camber_angle_rad in zip(
                normal_loads_n.all_n,
                camber_angles_rad,
                strict=True,
            )
        )

        total_capacity_n = self.total_lateral_force_capacity_n(
            normal_loads_n,
            front_lateral_force_fraction,
            camber_angles_rad,
        )
        if total_capacity_n <= 0.0:
            return TireCornerValues.zeros()
        force_n = copysign(
            min(abs(total_lateral_force_n), total_capacity_n),
            total_lateral_force_n,
        )
        front_force_n = force_n * front_lateral_force_fraction
        rear_force_n = force_n - front_force_n
        front_capacity_n = sum(capacities[:2])
        rear_capacity_n = sum(capacities[2:])
        return TireCornerValues(
            *(
                front_force_n * capacity_n / front_capacity_n
                for capacity_n in capacities[:2]
            ),
            *(
                rear_force_n * capacity_n / rear_capacity_n
                for capacity_n in capacities[2:]
            ),
        )

    def lateral_force_from_slip_angle_n(
        self,
        normal_load_n: float,
        slip_angle_rad: float,
        camber_angle_rad: float = 0.0,
    ) -> float:
        """Return signed lateral force generated by one tire's slip angle."""

        if not isfinite(slip_angle_rad):
            raise ValueError("slip_angle_rad must be finite")
        if normal_load_n <= 0.0:
            return 0.0
        capacity_n = self.lateral_force_capacity_n(normal_load_n, camber_angle_rad)
        if (
            self.pacejka_lateral is not None
            and self.constant_friction_coefficient is None
        ):
            force_n = self.pure_lateral_force_n(
                normal_load_n,
                slip_angle_rad,
                camber_angle_rad=camber_angle_rad,
            )
            return max(-capacity_n, min(force_n, capacity_n))
        utilization = min(
            abs(slip_angle_rad) / self.peak_lateral_slip_angle_rad,
            1.0,
        )
        return -copysign(capacity_n * utilization, slip_angle_rad)

    def total_lateral_force_capacity_n(
        self,
        normal_loads_n: TireNormalLoads,
        front_lateral_force_fraction: float,
        camber_angles_rad: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    ) -> float:
        """Return total lateral capacity for a fixed yaw-balanced axle split."""

        if not 0.0 < front_lateral_force_fraction < 1.0:
            raise ValueError("front_lateral_force_fraction must be between 0 and 1")
        if len(camber_angles_rad) != 4 or not all(
            isfinite(angle_rad) for angle_rad in camber_angles_rad
        ):
            raise ValueError("four finite camber angles are required")
        capacities = tuple(
            self.lateral_force_capacity_n(load_n, camber_angle_rad)
            for load_n, camber_angle_rad in zip(
                normal_loads_n.all_n,
                camber_angles_rad,
                strict=True,
            )
        )
        return min(
            sum(capacities[:2]) / front_lateral_force_fraction,
            sum(capacities[2:]) / (1.0 - front_lateral_force_fraction),
        )

    def calculate_forces(
        self,
        normal_loads_n: TireNormalLoads,
        kinematics: TireKinematics,
        drive_force_request_n: float,
        front_brake_force_request_n: float,
        rear_brake_force_request_n: float,
        timestep_s: float,
    ) -> TireStates:
        """Solve contact forces using this tire's grip laws and previous state."""

        return calculate_contact_forces(
            self,
            normal_loads_n,
            kinematics,
            drive_force_request_n,
            front_brake_force_request_n,
            rear_brake_force_request_n,
            timestep_s,
        )

    def _longitudinal_response(
        self,
        drive_force_n: float,
        braking_force_n: float,
        capacity_n: float,
        previous_slip_ratio: float,
        vehicle_speed_mps: float,
        timestep_s: float,
    ) -> tuple[float, float, float]:
        """Apply the tire's force/slip curve and optional relaxation."""

        return longitudinal_response(
            self,
            drive_force_n,
            braking_force_n,
            capacity_n,
            previous_slip_ratio,
            vehicle_speed_mps,
            timestep_s,
        )

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
