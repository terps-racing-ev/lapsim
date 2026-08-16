"""Schema-free telemetry snapshots and accumulated channel histories."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import isfinite, nan
from numbers import Real

JOULES_PER_KILOWATT_HOUR = 3_600_000.0


@dataclass(frozen=True, slots=True)
class Telemetry(Mapping[str, tuple[float, ...]]):
    """Namespaced scalar channels accumulated over simulation samples.

    Components own their channel names and write one scalar snapshot at a
    time. The final telemetry behaves like a read-only mapping from channel
    name to an equally sized tuple. New component models can add channels
    without modifying this class.
    """

    channels: dict[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        normalized = {
            str(name): tuple(float(value) for value in values)
            for name, values in self.channels.items()
        }
        lengths = {len(values) for values in normalized.values()}
        if len(lengths) > 1:
            raise ValueError("All telemetry channels must have the same length")
        object.__setattr__(self, "channels", normalized)

    def __getitem__(self, name: str) -> tuple[float, ...]:
        return self.channels[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.channels)

    def __len__(self) -> int:
        return len(self.channels)

    @property
    def sample_count(self) -> int:
        if not self.channels:
            return 0
        return len(next(iter(self.channels.values())))

    def as_dict(self) -> dict[str, tuple[float, ...]]:
        """Return a shallow copy suitable for NumPy, pandas, or CSV analysis."""

        return dict(self.channels)

    # Compatibility accessors for existing solvers, validation, and plots.
    @property
    def time_s(self) -> tuple[float, ...]:
        return self.channels["vehicle.time_s"]

    @property
    def distance_m(self) -> tuple[float, ...]:
        return self.channels["vehicle.distance_m"]

    @property
    def x_m(self) -> tuple[float, ...]:
        return self.channels["vehicle.x_m"]

    @property
    def y_m(self) -> tuple[float, ...]:
        return self.channels["vehicle.y_m"]

    @property
    def speed_mps(self) -> tuple[float, ...]:
        return self.channels["vehicle.speed_mps"]

    @property
    def longitudinal_acceleration_mps2(self) -> tuple[float, ...]:
        return self.channels["vehicle.longitudinal_acceleration_mps2"]

    @property
    def lateral_acceleration_mps2(self) -> tuple[float, ...]:
        return self.channels["vehicle.lateral_acceleration_mps2"]

    @property
    def propulsion_force_n(self) -> tuple[float, ...]:
        return self.channels["drivetrain.wheel_force_n"]

    @property
    def motor_speed_rpm(self) -> tuple[float, ...]:
        return self.channels["motor.speed_rpm"]

    @property
    def motor_torque_nm(self) -> tuple[float, ...]:
        return self.channels["motor.torque_nm"]

    @property
    def battery_power_w(self) -> tuple[float, ...]:
        return self.channels["battery.power_w"]

    @property
    def cumulative_energy_j(self) -> tuple[float, ...]:
        return self.channels["energy.cumulative_j"]

    @property
    def cumulative_positive_energy_j(self) -> tuple[float, ...]:
        return self.channels.get(
            "energy.cumulative_positive_j", self.cumulative_energy_j
        )

    @property
    def cumulative_net_energy_j(self) -> tuple[float, ...]:
        return self.channels.get("energy.cumulative_net_j", self.cumulative_energy_j)

    @property
    def cumulative_energy_kwh(self) -> tuple[float, ...]:
        return tuple(
            energy_j / JOULES_PER_KILOWATT_HOUR for energy_j in self.cumulative_energy_j
        )

    @property
    def total_energy_j(self) -> float:
        if not self.cumulative_energy_j:
            return 0.0
        return self.cumulative_energy_j[-1]

    @property
    def total_energy_kwh(self) -> float:
        return self.total_energy_j / JOULES_PER_KILOWATT_HOUR


class TelemetryRecorder:
    """Accumulate arbitrary scalar snapshots into aligned channel histories."""

    def __init__(self) -> None:
        self._channels: dict[str, list[float]] = {}
        self._sample_count = 0
        self._cumulative_positive_energy_j = 0.0
        self._cumulative_net_energy_j = 0.0

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def record(
        self,
        snapshot: Mapping[str, Real],
        *,
        timestep_s: float | None = None,
    ) -> None:
        """Append one snapshot, backfilling optional component signals with NaN."""

        if timestep_s is not None and timestep_s < 0:
            raise ValueError("timestep_s cannot be negative")
        normalized: dict[str, float] = {}
        for name, value in snapshot.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Telemetry channel names must be nonempty strings")
            if not isinstance(value, Real):
                raise TypeError(f"Telemetry channel {name!r} must be scalar")
            scalar = float(value)
            if not isfinite(scalar):
                raise ValueError(f"Telemetry channel {name!r} must be finite")
            normalized[name] = scalar

        if timestep_s is not None:
            battery_power_w = normalized.get("battery.power_w", 0.0)
            self._cumulative_positive_energy_j += max(battery_power_w, 0.0) * timestep_s
            self._cumulative_net_energy_j += battery_power_w * timestep_s
            normalized.setdefault(
                "energy.cumulative_j",
                self._cumulative_positive_energy_j,
            )
            normalized.setdefault(
                "energy.cumulative_positive_j",
                self._cumulative_positive_energy_j,
            )
            normalized.setdefault(
                "energy.cumulative_net_j",
                self._cumulative_net_energy_j,
            )

        for name in normalized:
            if name not in self._channels:
                self._channels[name] = [nan] * self._sample_count
        for name, values in self._channels.items():
            values.append(normalized.get(name, nan))
        self._sample_count += 1

    def freeze(self) -> Telemetry:
        """Return an immutable-by-convention mapping of all recorded channels."""

        return Telemetry(
            {name: tuple(values) for name, values in self._channels.items()}
        )


__all__ = ["Telemetry", "TelemetryRecorder"]
