"""Small immutable per-corner value containers."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TireCornerValues:
    """Four scalar values ordered front-left through rear-right."""

    front_left_n: float
    front_right_n: float
    rear_left_n: float
    rear_right_n: float

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "TireCornerValues":
        front_left_n, front_right_n, rear_left_n, rear_right_n = values
        return cls(front_left_n, front_right_n, rear_left_n, rear_right_n)

    @classmethod
    def zeros(cls) -> "TireCornerValues":
        return cls(0.0, 0.0, 0.0, 0.0)

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

    @property
    def total_n(self) -> float:
        return sum(self.all_n)


# The aliases retain domain meaning at call sites without duplicating the same
# four-value implementation for loads, forces, capacities, and requests.
TireNormalLoads = TireCornerValues
TireForces = TireCornerValues
