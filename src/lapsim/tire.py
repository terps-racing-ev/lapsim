"""Load-sensitive tire-force model."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import sqrt

from .utils.units import pounds_force_to_newtons


DEFAULT_NORMAL_LOADS_N = tuple(
    pounds_force_to_newtons(load_lbf)
    for load_lbf in (50.0, 100.0, 150.0, 200.0, 250.0)
)
DEFAULT_LATERAL_COEFFICIENTS = (1.625, 1.575, 1.525, 1.475, 1.425)
DEFAULT_LONGITUDINAL_COEFFICIENTS = DEFAULT_LATERAL_COEFFICIENTS
@dataclass(slots=True)
class Tire:
    """Linear load-sensitive friction model from Ryder's MATLAB simulator."""

    normal_loads_n: list[float] = field(
        default_factory=lambda: list(DEFAULT_NORMAL_LOADS_N)
    )
    lateral_coefficients: list[float] = field(
        default_factory=lambda: list(DEFAULT_LATERAL_COEFFICIENTS)
    )
    longitudinal_coefficients: list[float] = field(
        default_factory=lambda: list(DEFAULT_LONGITUDINAL_COEFFICIENTS)
    )
    current_longitudinal_force_n: float = field(init=False, default=0.0)
    current_lateral_force_n: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable tire lookup tables."""

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
        if any(
            coefficient <= 0
            for coefficient in self.longitudinal_coefficients
        ):
            raise ValueError("longitudinal coefficients must be positive")

    def reset_state(self) -> None:
        """Clear the current aggregate tire forces."""

        self.current_longitudinal_force_n = 0.0
        self.current_lateral_force_n = 0.0

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


    def lateral_coefficient(self, normal_load_n: float) -> float:
        """Return lateral friction coefficient at one tire's normal load."""

        return self._interpolate(normal_load_n, self.lateral_coefficients)

    def longitudinal_coefficient(self, normal_load_n: float) -> float:
        """Return longitudinal friction coefficient at one tire's normal load."""

        return self._interpolate(normal_load_n, self.longitudinal_coefficients)

    def lateral_force_capacity_n(self, normal_load_n: float) -> float:
        """Return one tire's lateral force capacity."""

        return self.lateral_coefficient(normal_load_n) * max(normal_load_n, 0.0)

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
                interpolation_fraction = (
                    (clamped_load_n - lower_load_n)
                    / (upper_load_n - lower_load_n)
                )
                lower_coefficient = coefficients[index - 1]
                return lower_coefficient + interpolation_fraction * (
                    coefficients[index] - lower_coefficient
                )

        return coefficients[-1]
