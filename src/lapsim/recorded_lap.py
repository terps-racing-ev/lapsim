"""Recorded track and control data for distance-indexed validation replays."""

from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
from math import atan
from pathlib import Path
from collections.abc import Sequence

from .controls import Controls


@dataclass(frozen=True, slots=True)
class RecordedLap:
    """Synchronized measured track, controls, and validation channels."""

    time_s: tuple[float, ...]
    x_m: tuple[float, ...]
    y_m: tuple[float, ...]
    speed_mps: tuple[float, ...]
    distance_trip_m: tuple[float, ...]
    motor_speed_rpm: tuple[float, ...]
    motor_torque_command_nm: tuple[float, ...]
    motor_torque_feedback_nm: tuple[float, ...]
    inverter_torque_feedback_nm: tuple[float, ...]
    inverter_motor_speed_rpm: tuple[float, ...]
    accelerator_percent: tuple[float, ...]
    brake_pressure_psi: tuple[float, ...]
    pack_power_w: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(values)
            for values in (
                self.time_s,
                self.x_m,
                self.y_m,
                self.speed_mps,
                self.distance_trip_m,
                self.motor_speed_rpm,
                self.motor_torque_command_nm,
                self.motor_torque_feedback_nm,
                self.inverter_torque_feedback_nm,
                self.inverter_motor_speed_rpm,
                self.accelerator_percent,
                self.brake_pressure_psi,
                self.pack_power_w,
            )
        }
        if lengths != {len(self.time_s)} or len(self.time_s) < 2:
            raise ValueError("Recorded-lap channels must have the same length")
        if any(
            upper_time_s <= lower_time_s
            for lower_time_s, upper_time_s in zip(
                self.time_s,
                self.time_s[1:],
            )
        ):
            raise ValueError("Recorded-lap timestamps must strictly increase")

    @classmethod
    def from_csv(cls, path: str | Path) -> "RecordedLap":
        """Load the synchronized CSV emitted by the MF4 lap extractor."""

        source_path = Path(path)
        with source_path.open(newline="") as handle:
            rows = list(DictReader(handle))
        if not rows:
            raise ValueError(f"Recorded lap is empty: {source_path}")

        def column(name: str, *, scale: float = 1.0) -> tuple[float, ...]:
            if name not in rows[0]:
                raise ValueError(f"Recorded lap is missing required column {name!r}")
            return tuple(float(row[name]) * scale for row in rows)

        return cls(
            time_s=column("time_s"),
            x_m=column("gps_x_m"),
            y_m=column("gps_y_m"),
            speed_mps=column("gps_speed_mps"),
            distance_trip_m=column("gps_distance_trip_m"),
            motor_speed_rpm=column("motor_rpm"),
            motor_torque_command_nm=column("torque_command_nm"),
            motor_torque_feedback_nm=column("torque_feedback_nm"),
            inverter_torque_feedback_nm=column("inv_torque_feedback_nm"),
            inverter_motor_speed_rpm=column("inv_motor_rpm"),
            accelerator_percent=column("apps_percent"),
            brake_pressure_psi=column("brake_pressure_psi"),
            pack_power_w=column("battery_power_kw", scale=1_000.0),
        )

    @property
    def duration_s(self) -> float:
        return self.time_s[-1] - self.time_s[0]

    @property
    def distance_m(self) -> float:
        return self.distance_trip_m[-1] - self.distance_trip_m[0]

    def controls(
        self,
        curvature_per_m: Sequence[float],
        *,
        wheelbase_m: float,
        use_torque_feedback: bool = False,
    ) -> tuple[Controls, ...]:
        """Convert recorded channels to the simulator's control interface.

        Steering was not logged in the 6.20 endurance recording. The supplied
        curvature therefore comes from the GNSS racing line and is converted
        to an equivalent road-wheel angle using the bicycle model.
        """

        if len(curvature_per_m) != len(self.time_s):
            raise ValueError("curvature_per_m must match the recorded lap")
        if wheelbase_m <= 0:
            raise ValueError("wheelbase_m must be positive")
        torque_nm = (
            self.motor_torque_feedback_nm
            if use_torque_feedback
            else self.motor_torque_command_nm
        )
        return tuple(
            Controls(
                motor_torque_request_nm=max(torque, 0.0),
                front_brake_pressure_psi=max(pressure, 0.0),
                rear_brake_pressure_psi=max(pressure, 0.0),
                steering_angle_rad=atan(curvature * wheelbase_m),
            )
            for torque, pressure, curvature in zip(
                torque_nm,
                self.brake_pressure_psi,
                curvature_per_m,
                strict=True,
            )
        )
