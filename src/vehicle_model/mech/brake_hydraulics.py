"""Hydraulic pressure-to-force maps and their offline inverse.

These maps describe brake hardware or its firmware calibration. Tire grip and
whole-vehicle deceleration are handled outside the hydraulic model.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class BrakePressureMap:
    """Snapshot of the mutable brake setup used for one map evaluation."""

    front_torque_per_pressure_nm_per_psi: float
    rear_torque_per_pressure_nm_per_psi: float
    maximum_pressure_psi: float
    pressure_deadband_psi: float
    pressure_force_model: Literal["linear-hardware-gains", "firmware-force-map"]
    maximum_force_request_n: float | None

    def axle_force_requests_from_pressures_n(
        self,
        front_pressure_psi: float,
        rear_pressure_psi: float,
        rolling_radius_m: float,
    ) -> tuple[float, float]:
        """Convert measured hydraulic pressures into axle force requests."""

        if front_pressure_psi < 0 or rear_pressure_psi < 0:
            raise ValueError("brake pressures cannot be negative")
        if rolling_radius_m <= 0:
            raise ValueError("rolling_radius_m must be positive")

        front_pressure_psi = min(front_pressure_psi, self.maximum_pressure_psi)
        rear_pressure_psi = min(rear_pressure_psi, self.maximum_pressure_psi)
        front_effective_psi = max(front_pressure_psi - self.pressure_deadband_psi, 0.0)
        rear_effective_psi = max(rear_pressure_psi - self.pressure_deadband_psi, 0.0)
        if self.pressure_force_model == "linear-hardware-gains":
            front_force_n = (
                front_effective_psi
                * self.front_torque_per_pressure_nm_per_psi
                / rolling_radius_m
            )
            rear_force_n = (
                rear_effective_psi
                * self.rear_torque_per_pressure_nm_per_psi
                / rolling_radius_m
            )
        else:
            front_force_n = (
                183.0
                + 8.7696 * front_pressure_psi
                - 0.0141145 * front_pressure_psi**2
                + 0.0000068383 * front_pressure_psi**3
                if front_effective_psi > 0.0
                else 0.0
            )
            rear_force_n = (
                5.6077 * rear_pressure_psi if rear_effective_psi > 0.0 else 0.0
            )

        total_force_n = front_force_n + rear_force_n
        if (
            self.maximum_force_request_n is not None
            and total_force_n > self.maximum_force_request_n
        ):
            scale = self.maximum_force_request_n / total_force_n
            front_force_n *= scale
            rear_force_n *= scale
        return front_force_n, rear_force_n

    def axle_pressures_for_force_requests_psi(
        self,
        front_force_n: float,
        rear_force_n: float,
        rolling_radius_m: float,
    ) -> tuple[float, float]:
        """Invert the configured pressure map for an offline driver command."""

        if front_force_n < 0 or rear_force_n < 0:
            raise ValueError("brake force requests cannot be negative")
        if rolling_radius_m <= 0:
            raise ValueError("rolling_radius_m must be positive")

        if self.pressure_force_model == "linear-hardware-gains":

            def pressure(force_n: float, gain_nm_per_psi: float) -> float:
                if force_n <= 0.0:
                    return 0.0
                if gain_nm_per_psi <= 0.0:
                    raise ValueError(
                        "cannot request brake force with a zero pressure gain"
                    )
                return min(
                    force_n * rolling_radius_m / gain_nm_per_psi
                    + self.pressure_deadband_psi,
                    self.maximum_pressure_psi,
                )

            return (
                pressure(front_force_n, self.front_torque_per_pressure_nm_per_psi),
                pressure(rear_force_n, self.rear_torque_per_pressure_nm_per_psi),
            )

        def invert(force_n: float, *, front: bool) -> float:
            if force_n <= 0.0:
                return 0.0
            lower_psi = self.pressure_deadband_psi
            upper_psi = min(max(lower_psi + 1.0, 1.0), self.maximum_pressure_psi)
            while True:
                forces = self.axle_force_requests_from_pressures_n(
                    upper_psi if front else 0.0,
                    0.0 if front else upper_psi,
                    rolling_radius_m,
                )
                if forces[0 if front else 1] >= force_n:
                    break
                if upper_psi >= self.maximum_pressure_psi:
                    return self.maximum_pressure_psi
                upper_psi = min(2.0 * upper_psi, self.maximum_pressure_psi)
            for _ in range(32):
                candidate_psi = 0.5 * (lower_psi + upper_psi)
                forces = self.axle_force_requests_from_pressures_n(
                    candidate_psi if front else 0.0,
                    0.0 if front else candidate_psi,
                    rolling_radius_m,
                )
                if forces[0 if front else 1] >= force_n:
                    upper_psi = candidate_psi
                else:
                    lower_psi = candidate_psi
            return upper_psi

        return invert(front_force_n, front=True), invert(rear_force_n, front=False)
