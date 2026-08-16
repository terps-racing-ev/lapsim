"""Mechanical-subteam lateral and longitudinal tire-force model."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite, sqrt

from utils.units import inches_to_meters, pounds_force_to_newtons

from .pacejka import Pacejka61LateralModel

DEFAULT_ROLLING_RADIUS_IN = 8.0
DEFAULT_ROLLING_RADIUS_M = inches_to_meters(DEFAULT_ROLLING_RADIUS_IN)

DEFAULT_NORMAL_LOADS_N = tuple(
    pounds_force_to_newtons(load_lbf) for load_lbf in (50.0, 100.0, 150.0, 200.0, 250.0)
)
DEFAULT_LATERAL_COEFFICIENTS = (1.625, 1.575, 1.525, 1.475, 1.425)
DEFAULT_LONGITUDINAL_COEFFICIENTS = DEFAULT_LATERAL_COEFFICIENTS


@dataclass(slots=True)
class Tire:
    """Pacejka lateral tire model plus a longitudinal friction approximation."""

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
    pacejka_lateral: Pacejka61LateralModel | None = field(
        default_factory=Pacejka61LateralModel
    )
    camber_angle_rad: float = 0.0
    inflation_pressure_pa: float = 98_000.0
    constant_friction_coefficient: float | None = None
    current_longitudinal_force_n: float = field(init=False, default=0.0)
    current_lateral_force_n: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable tire lookup tables."""

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
            upper_load <= lower_load
            for lower_load, upper_load in zip(
                self.normal_loads_n,
                self.normal_loads_n[1:],
            )
        ):
            raise ValueError("normal loads must be strictly increasing")
        if any(coefficient <= 0 for coefficient in self.lateral_coefficients):
            raise ValueError("lateral coefficients must be positive")
        if any(coefficient <= 0 for coefficient in self.longitudinal_coefficients):
            raise ValueError("longitudinal coefficients must be positive")
        if not isfinite(self.camber_angle_rad):
            raise ValueError("camber_angle_rad must be finite")
        if (
            not isfinite(self.inflation_pressure_pa)
            or self.inflation_pressure_pa <= 0
        ):
            raise ValueError("inflation_pressure_pa must be finite and positive")
        if self.pacejka_lateral is not None:
            self.pacejka_lateral.validate()
        if (
            self.constant_friction_coefficient is not None
            and self.constant_friction_coefficient <= 0
        ):
            raise ValueError("constant_friction_coefficient must be positive")

    def reset_state(self) -> None:
        """Clear the current aggregate tire forces."""

        self.current_longitudinal_force_n = 0.0
        self.current_lateral_force_n = 0.0

    @property
    def maximum_longitudinal_coefficient(self) -> float:
        """Largest configured coefficient, used to bound brake solving."""

        return (
            self.constant_friction_coefficient
            if self.constant_friction_coefficient is not None
            else max(self.longitudinal_coefficients)
        )

    def update_state(
        self,
        longitudinal_force_n: float,
        lateral_force_n: float,
        timestep_s: float,
    ) -> None:
        """Retain aggregate tire forces for one timestep."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.current_longitudinal_force_n = longitudinal_force_n
        self.current_lateral_force_n = lateral_force_n

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry.update(
            {
                "tire.longitudinal_force_n": self.current_longitudinal_force_n,
                "tire.lateral_force_n": self.current_lateral_force_n,
                "tire.rolling_radius_m": self.rolling_radius_m,
                "tire.camber_angle_rad": self.camber_angle_rad,
                "tire.inflation_pressure_pa": self.inflation_pressure_pa,
                "tire.pacejka_lateral_active": float(
                    self.pacejka_lateral is not None
                    and self.constant_friction_coefficient is None
                ),
                "tire.constant_friction_coefficient": (
                    0.0
                    if self.constant_friction_coefficient is None
                    else self.constant_friction_coefficient
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

        if (
            self.constant_friction_coefficient is None
            and self.pacejka_lateral is not None
        ):
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

        return self.longitudinal_coefficient(normal_load_n) * max(
            normal_load_n,
            0.0,
        )

    def combined_longitudinal_force_capacity_n(
        self,
        normal_load_n: float,
        lateral_force_n: float,
    ) -> float:
        """Return longitudinal grip remaining after lateral-force demand."""

        lateral_capacity_n = self.lateral_force_capacity_n(normal_load_n)
        if lateral_capacity_n <= 0:
            return 0.0
        lateral_utilization = min(
            1.0,
            abs(lateral_force_n) / lateral_capacity_n,
        )
        return self.longitudinal_force_capacity_n(normal_load_n) * sqrt(
            max(0.0, 1.0 - lateral_utilization**2)
        )

    def _interpolate(
        self,
        normal_load_n: float,
        coefficients: Sequence[float],
    ) -> float:
        clamped_load_n = min(
            max(normal_load_n, self.normal_loads_n[0]),
            self.normal_loads_n[-1],
        )

        for index, upper_load_n in enumerate(self.normal_loads_n[1:], start=1):
            if clamped_load_n <= upper_load_n:
                lower_load_n = self.normal_loads_n[index - 1]
                interpolation_fraction = (clamped_load_n - lower_load_n) / (
                    upper_load_n - lower_load_n
                )
                lower_coefficient = coefficients[index - 1]
                return lower_coefficient + interpolation_fraction * (
                    coefficients[index] - lower_coefficient
                )

        return coefficients[-1]
