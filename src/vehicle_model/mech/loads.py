"""Mechanical-subteam force and load result objects."""

from dataclasses import dataclass


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
