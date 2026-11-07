"""Digitized motor-efficiency bands and explicit handling of missing data."""

from bisect import bisect_left
import csv
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path


DEFAULT_EFFICIENCY_MAP_PATH = (
    Path(__file__).resolve().parents[3]
    / "data/motor_data/emrax_228_cc_efficiency_bands.csv"
)


@dataclass(frozen=True, slots=True)
class MotorEfficiencyMap:
    """Bilinear interpolation of the digitized bands' lower edges.

    This scalar surface is a simulation assumption, not measured efficiency.
    Missing/boundary-uncertain cells and out-of-range points return None, so
    the caller can explicitly apply and report its configured fallback.
    Rows are indexed by torque, then speed. Only positive motoring is mapped.
    """

    speed_rpm: tuple[float, ...]
    torque_nm: tuple[float, ...]
    lower_fraction: tuple[tuple[float | None, ...], ...]
    source_path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "speed_rpm", tuple(float(v) for v in self.speed_rpm))
        object.__setattr__(self, "torque_nm", tuple(float(v) for v in self.torque_nm))
        object.__setattr__(self, "lower_fraction", tuple(
            tuple(None if v is None else float(v) for v in row)
            for row in self.lower_fraction
        ))
        for axis in (self.speed_rpm, self.torque_nm):
            if len(axis) < 2 or any(not isfinite(v) or v < 0.0 for v in axis):
                raise ValueError("Efficiency-map axes require at least two finite nonnegative values")
            if any(b <= a for a, b in zip(axis, axis[1:])):
                raise ValueError("Efficiency-map axes must strictly increase")
        if len(self.lower_fraction) != len(self.torque_nm) or any(
            len(row) != len(self.speed_rpm) for row in self.lower_fraction
        ):
            raise ValueError("Efficiency-map values must match both axes")
        if any(
            value is not None and (not isfinite(value) or not 0.0 < value <= 1.0)
            for row in self.lower_fraction for value in row
        ):
            raise ValueError("Mapped efficiencies must be finite fractions in (0, 1]")

    def __deepcopy__(self, memo):
        # Immutable map data can be shared by every solver preview vehicle.
        return self

    @classmethod
    def from_csv(cls, path: str | Path) -> "MotorEfficiencyMap":
        source = Path(path)
        points = {}
        with source.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                key = (float(row["torque_nm"]), float(row["speed_rpm"]))
                if key in points:
                    raise ValueError(f"Duplicate efficiency-map operating point: {key}")
                status = row["status"]
                if status == "digitized_band":
                    value = float(row["efficiency_min_fraction"])
                    upper = row["efficiency_max_fraction"]
                    if upper and not value <= float(upper) <= 1.0:
                        raise ValueError("Invalid efficiency-band upper bound")
                elif status in ("boundary_uncertain", "outside_published_region"):
                    value = None
                else:
                    raise ValueError(f"Unknown efficiency-map status: {status}")
                points[key] = value
        speeds = tuple(sorted({key[1] for key in points}))
        torques = tuple(sorted({key[0] for key in points}))
        if len(points) != len(speeds) * len(torques):
            raise ValueError("Efficiency-map CSV must contain a complete rectangular grid")
        values = tuple(tuple(points[(torque, speed)] for speed in speeds) for torque in torques)
        return cls(speeds, torques, values, str(source.resolve()))

    @staticmethod
    def _weights(axis: tuple[float, ...], value: float):
        if value < axis[0] or value > axis[-1]:
            return ()
        upper = bisect_left(axis, value)
        if axis[upper] == value:
            return ((upper, 1.0),)
        lower = upper - 1
        fraction = (value - axis[lower]) / (axis[upper] - axis[lower])
        return ((lower, 1.0 - fraction), (upper, fraction))

    def lookup_fraction(self, speed_rpm: float, torque_nm: float) -> float | None:
        if not isfinite(speed_rpm) or not isfinite(torque_nm):
            raise ValueError("Efficiency-map coordinates must be finite")
        if speed_rpm <= 0.0 or torque_nm <= 0.0:
            return None
        speed_weights = self._weights(self.speed_rpm, speed_rpm)
        torque_weights = self._weights(self.torque_nm, torque_nm)
        if not speed_weights or not torque_weights:
            return None
        efficiency = 0.0
        for row, torque_weight in torque_weights:
            for column, speed_weight in speed_weights:
                value = self.lower_fraction[row][column]
                if value is None:
                    return None
                efficiency += value * torque_weight * speed_weight
        return efficiency


@lru_cache(maxsize=1)
def default_motor_efficiency_map() -> MotorEfficiencyMap:
    return MotorEfficiencyMap.from_csv(DEFAULT_EFFICIENCY_MAP_PATH)
