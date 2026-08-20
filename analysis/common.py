"""Shared, explicit adapters for recorded-lap analysis scripts.

This module owns data alignment and conversion from recorded signals to the
simulator's control interface.  In particular, it never silently treats logged
braking or negative torque as zero: callers must select those approximations.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from math import atan
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from lapsim import Controls, SpatialTrack

if TYPE_CHECKING:
    from vehicle_model.vehicle import Vehicle

DEFAULT_GNSS_LAG_S = 0.3072
TIME_COLUMNS = ("time_s", "mf4_time_s")
LONG_ACCEL_COLUMNS = (
    "corrected_longitudinal_accel_mps2",
    "corrected_accel_longitudinal_mps2",
    "corrected_accel_x_mps2",
    "accel_x_gravity_removed_mps2",
    "longitudinal_acceleration_mps2",
)
LAT_ACCEL_COLUMNS = (
    "corrected_lateral_accel_mps2",
    "corrected_accel_lateral_mps2",
    "corrected_accel_y_mps2",
    "accel_y_gravity_removed_mps2",
)
VERT_ACCEL_COLUMNS = (
    "corrected_vertical_accel_mps2",
    "corrected_accel_vertical_mps2",
    "corrected_accel_z_mps2",
    "accel_z_gravity_removed_mps2",
)


class UnmodeledRecordedControlError(ValueError):
    """A recorded control cannot be represented without an explicit policy."""


@dataclass(frozen=True, slots=True)
class RecordedControlAdapter:
    """Convert logged torque and brake pressure into :class:`Controls`.

    Pressure remains a pressure at the control boundary. Optional calibration
    overrides configure the vehicle's brake model; the adapter never converts
    a per-sample pressure into force itself.
    """

    brake_force_per_psi_n: float | None = None
    maximum_brake_force_request_n: float | None = None
    brake_deadband_psi: float = 0.0
    negative_torque_policy: Literal["error", "clip"] = "error"
    negative_torque_tolerance_nm: float = 0.5

    def __post_init__(self) -> None:
        if self.brake_force_per_psi_n is not None and self.brake_force_per_psi_n < 0:
            raise ValueError("brake_force_per_psi_n cannot be negative")
        if (
            self.maximum_brake_force_request_n is not None
            and self.maximum_brake_force_request_n <= 0
        ):
            raise ValueError("maximum_brake_force_request_n must be positive")
        if self.brake_deadband_psi < 0:
            raise ValueError("brake_deadband_psi cannot be negative")
        if self.negative_torque_tolerance_nm < 0:
            raise ValueError("negative_torque_tolerance_nm cannot be negative")

    def validate_trace(
        self,
        motor_torque_nm: np.ndarray,
        brake_pressure_psi: np.ndarray,
    ) -> None:
        minimum_torque = float(np.nanmin(motor_torque_nm))
        if (
            minimum_torque < -self.negative_torque_tolerance_nm
            and self.negative_torque_policy == "error"
        ):
            raise UnmodeledRecordedControlError(
                f"Recorded torque reaches {minimum_torque:.2f} Nm, but negative "
                "torque/regen is not modeled. Pass --negative-torque-policy clip "
                "only to request that approximation explicitly."
            )

    def configure_vehicle(self, vehicle: "Vehicle") -> None:
        """Apply optional replay calibration to the vehicle brake component."""

        brakes = vehicle.brakes
        brakes.pressure_deadband_psi = self.brake_deadband_psi
        brakes.maximum_force_request_n = self.maximum_brake_force_request_n
        if self.brake_force_per_psi_n is not None:
            current_gain_n_per_psi = (
                brakes.equivalent_vehicle_force_per_pressure_n_per_psi(
                    vehicle.tire.rolling_radius_m
                )
            )
            if current_gain_n_per_psi <= 0.0:
                if self.brake_force_per_psi_n > 0.0:
                    raise ValueError("cannot calibrate zero brake pressure gains")
            else:
                scale = self.brake_force_per_psi_n / current_gain_n_per_psi
                brakes.front_torque_per_pressure_lbfin_per_psi *= scale
                brakes.rear_torque_per_pressure_lbfin_per_psi *= scale

    def controls(
        self,
        *,
        motor_torque_nm: float,
        brake_pressure_psi: float,
        curvature_per_m: float,
        wheelbase_m: float,
    ) -> Controls:
        if wheelbase_m <= 0:
            raise ValueError("wheelbase_m must be positive")
        if motor_torque_nm < -self.negative_torque_tolerance_nm:
            if self.negative_torque_policy == "error":
                raise UnmodeledRecordedControlError(
                    "Negative motor torque is not modeled"
                )
            motor_torque_nm = 0.0
        return Controls(
            motor_torque_request_nm=max(motor_torque_nm, 0.0),
            front_brake_pressure_psi=max(brake_pressure_psi, 0.0),
            rear_brake_pressure_psi=max(brake_pressure_psi, 0.0),
            steering_angle_rad=atan(curvature_per_m * wheelbase_m),
        )


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    """Read numeric CSV columns, ignoring nonnumeric metadata columns."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    data: dict[str, np.ndarray] = {}
    for name in rows[0]:
        try:
            data[name] = np.asarray([float(row[name]) for row in rows], dtype=float)
        except (TypeError, ValueError):
            continue
    return data


def first_present(data: dict[str, np.ndarray], names: Iterable[str], label: str) -> str:
    for name in names:
        if name in data:
            return name
    raise ValueError(f"Missing {label}; tried: {', '.join(names)}")


def interpolate_channel(
    source_time_s: np.ndarray,
    source: np.ndarray,
    target_time_s: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(source_time_s) & np.isfinite(source)
    if finite.sum() < 2:
        raise ValueError("A channel needs at least two finite timestamped samples")
    order = np.argsort(source_time_s[finite])
    return np.interp(target_time_s, source_time_s[finite][order], source[finite][order])


def corrected_imu_at_lap_times(
    lap: dict[str, np.ndarray], corrected_imu: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    lap_time_name = first_present(lap, TIME_COLUMNS, "first-lap timestamp")
    imu_time_name = first_present(
        corrected_imu, TIME_COLUMNS, "corrected-IMU timestamp"
    )
    selected = {"time": imu_time_name}
    result: dict[str, np.ndarray] = {}
    for output_name, aliases in (
        ("imu_longitudinal_accel_mps2", LONG_ACCEL_COLUMNS),
        ("imu_lateral_accel_mps2", LAT_ACCEL_COLUMNS),
        ("imu_vertical_accel_mps2", VERT_ACCEL_COLUMNS),
    ):
        source_name = first_present(corrected_imu, aliases, output_name)
        selected[output_name] = source_name
        result[output_name] = interpolate_channel(
            corrected_imu[imu_time_name], corrected_imu[source_name], lap[lap_time_name]
        )
    return result, selected


def project_to_track_distance(
    track: SpatialTrack,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> np.ndarray:
    """Project points to the centerline and unwrap station through one lap."""

    points = np.column_stack((x_m, y_m))
    starts = np.column_stack((track.x_m[:-1], track.y_m[:-1]))
    vectors = np.column_stack((np.diff(track.x_m), np.diff(track.y_m)))
    length_sq = np.einsum("ij,ij->i", vectors, vectors)
    starts_station = np.asarray(track.distance_m[:-1])
    segment_length = np.asarray(track.cell_length_m)
    stations = np.empty(len(points))
    for index, point in enumerate(points):
        fractions = np.clip(
            np.einsum("ij,ij->i", point - starts, vectors) / length_sq,
            0.0,
            1.0,
        )
        closest = starts + fractions[:, None] * vectors
        cell = int(np.argmin(np.einsum("ij,ij->i", point - closest, point - closest)))
        stations[index] = starts_station[cell] + fractions[cell] * segment_length[cell]
    angular_station = stations * (2.0 * np.pi / track.length_m)
    return np.unwrap(angular_station) * (track.length_m / (2.0 * np.pi))


def periodic_curvature_at_station(
    track: SpatialTrack,
    station_m: np.ndarray,
) -> np.ndarray:
    """Periodically interpolate track-cell curvature at arbitrary stations."""

    centers = np.asarray(track.cell_center_distance_m)
    curvature = np.asarray(track.curvature_per_m)
    wrapped = np.mod(station_m, track.length_m)
    extended_station = np.r_[
        centers[-1] - track.length_m,
        centers,
        centers[0] + track.length_m,
    ]
    extended_curvature = np.r_[curvature[-1], curvature, curvature[0]]
    return np.interp(wrapped, extended_station, extended_curvature)
