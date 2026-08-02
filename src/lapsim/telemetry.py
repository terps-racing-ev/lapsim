"""Calculated telemetry channels for a converged lap."""

from dataclasses import dataclass


JOULES_PER_KILOWATT_HOUR = 3_600_000.0


@dataclass(frozen=True, slots=True)
class Telemetry:
    """Cell-centered vehicle and powertrain data around one lap."""

    distance_m: tuple[float, ...]
    speed_mps: tuple[float, ...]
    longitudinal_acceleration_mps2: tuple[float, ...]
    propulsion_force_n: tuple[float, ...]
    motor_torque_nm: tuple[float, ...]
    battery_power_w: tuple[float, ...]
    cumulative_energy_j: tuple[float, ...]

    def __post_init__(self) -> None:
        channel_lengths = {
            len(self.distance_m),
            len(self.speed_mps),
            len(self.longitudinal_acceleration_mps2),
            len(self.propulsion_force_n),
            len(self.motor_torque_nm),
            len(self.battery_power_w),
            len(self.cumulative_energy_j),
        }
        if len(channel_lengths) != 1:
            raise ValueError("All telemetry channels must have the same length")

    @property
    def cumulative_energy_kwh(self) -> tuple[float, ...]:
        """Cumulative positive battery energy in kilowatt-hours."""

        return tuple(
            energy_j / JOULES_PER_KILOWATT_HOUR
            for energy_j in self.cumulative_energy_j
        )

    @property
    def total_energy_j(self) -> float:
        """Total positive battery energy consumed during the lap."""

        if not self.cumulative_energy_j:
            return 0.0
        return self.cumulative_energy_j[-1]

    @property
    def total_energy_kwh(self) -> float:
        """Total positive battery energy consumed in kilowatt-hours."""

        return self.total_energy_j / JOULES_PER_KILOWATT_HOUR
