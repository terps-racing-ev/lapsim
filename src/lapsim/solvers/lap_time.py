"""Minimum-time speed-profile solver within a local speed-limit map."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from vehicle_model.vehicle import Vehicle

from .speed_limit import (
    STANDARD_AIR_DENSITY_KGPM3,
    STANDARD_GRAVITY_MPS2,
    SpeedLimitMap,
)
from ..core.telemetry import Telemetry

DEFAULT_CONVERGENCE_TOLERANCE_MPS = 1e-6
DEFAULT_MAX_PASSES = 1_000
LOAD_TRANSFER_SOLVER_ITERATIONS = 10


@dataclass(frozen=True, slots=True)
class LapResult:
    """Minimum-time profile and telemetry for one prescribed start speed."""

    speed_limit_map: SpeedLimitMap
    speed_mps: tuple[float, ...]
    lap_time_s: float
    starting_speed_mps: float
    passes: int
    telemetry: Telemetry

    @property
    def distance_m(self) -> tuple[float, ...]:
        return self.speed_limit_map.distance_m

    @property
    def x_m(self) -> tuple[float, ...]:
        return self.speed_limit_map.x_m

    @property
    def y_m(self) -> tuple[float, ...]:
        return self.speed_limit_map.y_m

    def plot_speed_map(
        self,
        speeds_mps: Sequence[float] | None = None,
        *,
        title: str = "Minimum-time speed map",
    ):
        """Plot the solved speeds on the underlying track map."""

        return self.speed_limit_map.plot_speed_map(
            self.speed_mps if speeds_mps is None else speeds_mps,
            title=title,
        )

    def plot_telemetry_summary(self):
        """Plot track speed, torque, acceleration, power, and energy."""

        from .plotting import plot_lap_telemetry_summary

        return plot_lap_telemetry_summary(self)


class LapTimeSolver:
    """Solve a minimum-time profile under speed, acceleration, and braking limits."""

    def __init__(
        self,
        vehicle: Vehicle,
        convergence_tolerance_mps: float = DEFAULT_CONVERGENCE_TOLERANCE_MPS,
        max_passes: int = DEFAULT_MAX_PASSES,
        gravity_mps2: float = STANDARD_GRAVITY_MPS2,
        air_density_kgpm3: float = STANDARD_AIR_DENSITY_KGPM3,
    ) -> None:
        if convergence_tolerance_mps <= 0:
            raise ValueError("convergence_tolerance_mps must be positive")
        if max_passes <= 0:
            raise ValueError("max_passes must be positive")
        if gravity_mps2 <= 0:
            raise ValueError("gravity_mps2 must be positive")
        if air_density_kgpm3 <= 0:
            raise ValueError("air_density_kgpm3 must be positive")
        self.vehicle = vehicle
        self.convergence_tolerance_mps = convergence_tolerance_mps
        self.max_passes = max_passes
        self.gravity_mps2 = gravity_mps2
        self.air_density_kgpm3 = air_density_kgpm3

    def solve(
        self,
        speed_limit_map: SpeedLimitMap,
        starting_speed_mps: float,
    ) -> LapResult:
        """Find the fastest feasible profile from the prescribed start speed."""

        self.vehicle.validate()
        self._validate_inputs(speed_limit_map, starting_speed_mps)
        speeds = list(speed_limit_map.speed_limit_mps)
        speeds[0] = starting_speed_mps
        cell_count = len(speed_limit_map.cell_length_m)

        for pass_number in range(1, self.max_passes + 1):
            largest_change_mps = 0.0

            for cell_index in range(cell_count):
                acceleration_mps2 = self._forward_acceleration_mps2(
                    speeds[cell_index],
                    speed_limit_map.curvature_per_m[cell_index],
                )
                reachable_speed_mps = self._speed_after_distance(
                    speeds[cell_index],
                    acceleration_mps2,
                    speed_limit_map.cell_length_m[cell_index],
                )
                new_speed_mps = min(
                    speeds[cell_index + 1],
                    speed_limit_map.speed_limit_mps[cell_index + 1],
                    reachable_speed_mps,
                )
                largest_change_mps = max(
                    largest_change_mps,
                    speeds[cell_index + 1] - new_speed_mps,
                )
                speeds[cell_index + 1] = new_speed_mps

            for cell_index in range(cell_count - 1, 0, -1):
                braking_deceleration_mps2 = (
                    self.vehicle.brakes.maximum_deceleration_mps2(
                        self.vehicle,
                        speeds[cell_index + 1],
                        speed_limit_map.curvature_per_m[cell_index],
                        self.gravity_mps2,
                        self.air_density_kgpm3,
                    )
                )
                reachable_speed_mps = self._speed_before_braking(
                    speeds[cell_index + 1],
                    braking_deceleration_mps2,
                    speed_limit_map.cell_length_m[cell_index],
                )
                new_speed_mps = min(
                    speeds[cell_index],
                    speed_limit_map.speed_limit_mps[cell_index],
                    reachable_speed_mps,
                )
                largest_change_mps = max(
                    largest_change_mps,
                    speeds[cell_index] - new_speed_mps,
                )
                speeds[cell_index] = new_speed_mps

            maximum_feasible_start_mps = self._speed_before_braking(
                speeds[1],
                self.vehicle.brakes.maximum_deceleration_mps2(
                    self.vehicle,
                    speeds[1],
                    speed_limit_map.curvature_per_m[0],
                    self.gravity_mps2,
                    self.air_density_kgpm3,
                ),
                speed_limit_map.cell_length_m[0],
            )
            if starting_speed_mps > (
                maximum_feasible_start_mps + self.convergence_tolerance_mps
            ):
                raise ValueError(
                    "starting_speed_mps is too high to satisfy the upcoming "
                    "speed limits"
                )

            if largest_change_mps < self.convergence_tolerance_mps:
                return self._make_result(
                    speed_limit_map,
                    speeds,
                    starting_speed_mps,
                    pass_number,
                )

        raise RuntimeError(
            f"Speed profile did not converge after {self.max_passes} passes"
        )

    @staticmethod
    def _validate_inputs(
        speed_limit_map: SpeedLimitMap,
        starting_speed_mps: float,
    ) -> None:
        point_count = len(speed_limit_map.distance_m)
        cell_count = point_count - 1
        if point_count < 2:
            raise ValueError("A speed-limit map requires at least two points")
        if len(speed_limit_map.speed_limit_mps) != point_count:
            raise ValueError("Speed limits must match the map point count")
        if len(speed_limit_map.cell_length_m) != cell_count:
            raise ValueError("Cell lengths must match the map cell count")
        if len(speed_limit_map.curvature_per_m) != cell_count:
            raise ValueError("Curvatures must match the map cell count")
        if starting_speed_mps < 0:
            raise ValueError("starting_speed_mps cannot be negative")
        if starting_speed_mps > speed_limit_map.speed_limit_mps[0]:
            raise ValueError(
                "starting_speed_mps exceeds the local starting-point limit"
            )

    def _forward_acceleration_mps2(
        self,
        speed_mps: float,
        curvature_per_m: float,
    ) -> float:
        drag_force_n, rolling_force_n, _ = self._resistance_and_downforce(
            speed_mps,
            curvature_per_m,
        )
        aero_forces = self.vehicle.aero_forces_n(
            speed_mps,
            speed_mps**2 * abs(curvature_per_m),
            self.air_density_kgpm3,
        )
        tire_normal_loads = self.vehicle.suspension.tire_normal_loads_n(
            self.vehicle.mass_kg,
            self.gravity_mps2,
            aero_forces,
            self.vehicle.chassis,
            lateral_acceleration_mps2=(speed_mps**2 * abs(curvature_per_m)),
        )
        motor_force_n = self.vehicle.drivetrain.available_wheel_force_n(
            speed_mps,
            self.vehicle.battery,
        )
        combined_tire_force_n = self._combined_longitudinal_tire_force_n(
            speed_mps,
            curvature_per_m,
            tire_normal_loads.all_n,
        )

        acceleration_mps2 = 0.0
        for _ in range(LOAD_TRANSFER_SOLVER_ITERATIONS):
            tire_normal_loads = self.vehicle.suspension.tire_normal_loads_n(
                self.vehicle.mass_kg,
                self.gravity_mps2,
                aero_forces,
                self.vehicle.chassis,
                longitudinal_acceleration_mps2=acceleration_mps2,
                lateral_acceleration_mps2=(speed_mps**2 * abs(curvature_per_m)),
            )
            rear_traction_force_n = self._tire_force_capacity_n(
                tire_normal_loads.rear_n,
                lateral=False,
            )
            drive_force_n = min(
                motor_force_n,
                rear_traction_force_n,
                combined_tire_force_n,
            )
            acceleration_mps2 = (
                drive_force_n - drag_force_n - rolling_force_n
            ) / self.vehicle.effective_longitudinal_mass_kg
        return acceleration_mps2

    def _combined_longitudinal_tire_force_n(
        self,
        speed_mps: float,
        curvature_per_m: float,
        tire_normal_loads_n: Sequence[float],
    ) -> float:
        lateral_force_n = self.vehicle.mass_kg * speed_mps**2 * abs(curvature_per_m)
        lateral_force_limit_n = self._tire_force_capacity_n(
            tire_normal_loads_n,
            lateral=True,
        )
        if lateral_force_limit_n <= 0:
            return 0.0
        lateral_utilization = min(1.0, lateral_force_n / lateral_force_limit_n)
        remaining_longitudinal_fraction = sqrt(max(0.0, 1.0 - lateral_utilization**2))
        longitudinal_force_limit_n = self._tire_force_capacity_n(
            tire_normal_loads_n,
            lateral=False,
        )
        return longitudinal_force_limit_n * remaining_longitudinal_fraction

    def _tire_force_capacity_n(
        self,
        tire_normal_loads_n: Sequence[float],
        *,
        lateral: bool,
    ) -> float:
        if lateral:
            return sum(
                self.vehicle.tire.lateral_force_capacity_n(normal_load_n)
                for normal_load_n in tire_normal_loads_n
            )
        return sum(
            self.vehicle.tire.longitudinal_force_capacity_n(normal_load_n)
            for normal_load_n in tire_normal_loads_n
        )

    def _resistance_and_downforce(
        self,
        speed_mps: float,
        curvature_per_m: float = 0.0,
    ) -> tuple[float, float, float]:
        aero_forces = self.vehicle.aero_forces_n(
            speed_mps,
            speed_mps**2 * abs(curvature_per_m),
            self.air_density_kgpm3,
        )
        rolling_force_n = self.vehicle.rolling_resistance_coefficient * (
            self.vehicle.mass_kg * self.gravity_mps2 + aero_forces.downforce_n
        )
        lateral_acceleration_mps2 = speed_mps**2 * abs(curvature_per_m)
        normal_loads = self.vehicle.suspension.tire_normal_loads_n(
            self.vehicle.mass_kg,
            self.gravity_mps2,
            aero_forces,
            self.vehicle.chassis,
            lateral_acceleration_mps2=lateral_acceleration_mps2,
        )
        total_normal_force_n = sum(normal_loads.all_n)
        lateral_force_n = self.vehicle.mass_kg * lateral_acceleration_mps2
        cornering_drag_force_n = 0.0
        if total_normal_force_n > 0.0:
            cornering_drag_force_n = (
                self.vehicle.cornering_drag_coefficient
                * lateral_force_n**2
                / total_normal_force_n
            )
        return (
            aero_forces.drag_n + cornering_drag_force_n,
            rolling_force_n,
            aero_forces.downforce_n,
        )

    @staticmethod
    def _speed_after_distance(
        initial_speed_mps: float,
        acceleration_mps2: float,
        distance_m: float,
    ) -> float:
        return sqrt(
            max(
                0.0,
                initial_speed_mps**2 + 2.0 * acceleration_mps2 * distance_m,
            )
        )

    @staticmethod
    def _speed_before_braking(
        final_speed_mps: float,
        braking_deceleration_mps2: float,
        distance_m: float,
    ) -> float:
        return sqrt(
            max(
                0.0,
                final_speed_mps**2 + 2.0 * braking_deceleration_mps2 * distance_m,
            )
        )

    def _make_result(
        self,
        speed_limit_map: SpeedLimitMap,
        speeds: list[float],
        starting_speed_mps: float,
        passes: int,
    ) -> LapResult:
        lap_time_s = sum(
            cell_length_m / (0.5 * (speeds[index] + speeds[index + 1]))
            for index, cell_length_m in enumerate(speed_limit_map.cell_length_m)
        )
        telemetry = self._make_telemetry(speed_limit_map, speeds)
        return LapResult(
            speed_limit_map=speed_limit_map,
            speed_mps=tuple(speeds),
            lap_time_s=lap_time_s,
            starting_speed_mps=starting_speed_mps,
            passes=passes,
            telemetry=telemetry,
        )

    def _make_telemetry(
        self,
        speed_limit_map: SpeedLimitMap,
        speeds: list[float],
    ) -> Telemetry:
        telemetry_distance_m: list[float] = []
        average_speed_mps: list[float] = []
        acceleration_mps2: list[float] = []
        propulsion_force_n: list[float] = []
        motor_torque_nm: list[float] = []
        battery_power_w: list[float] = []
        cumulative_energy_j: list[float] = []
        energy_used_j = 0.0
        drivetrain = self.vehicle.drivetrain

        for index, cell_length_m in enumerate(speed_limit_map.cell_length_m):
            cell_average_speed_mps = 0.5 * (speeds[index] + speeds[index + 1])
            cell_acceleration_mps2 = (speeds[index + 1] ** 2 - speeds[index] ** 2) / (
                2.0 * cell_length_m
            )
            drag_force_n, rolling_force_n, _ = self._resistance_and_downforce(
                cell_average_speed_mps,
                speed_limit_map.curvature_per_m[index],
            )
            requested_force_n = max(
                0.0,
                self.vehicle.effective_longitudinal_mass_kg * cell_acceleration_mps2
                + drag_force_n
                + rolling_force_n,
            )
            requested_torque_nm = drivetrain.motor_torque_for_wheel_force_nm(
                requested_force_n
            )
            cell_motor_torque_nm = min(
                requested_torque_nm,
                drivetrain.available_motor_torque_nm(
                    cell_average_speed_mps,
                    self.vehicle.battery,
                ),
            )
            cell_propulsion_force_n = drivetrain.wheel_force_from_motor_torque_n(
                cell_motor_torque_nm
            )
            cell_battery_power_w = drivetrain.positive_battery_power_w(
                cell_propulsion_force_n,
                cell_average_speed_mps,
                self.vehicle.battery,
            )
            energy_used_j += (
                cell_battery_power_w * cell_length_m / cell_average_speed_mps
            )

            telemetry_distance_m.append(
                0.5
                * (
                    speed_limit_map.distance_m[index]
                    + speed_limit_map.distance_m[index + 1]
                )
            )
            average_speed_mps.append(cell_average_speed_mps)
            acceleration_mps2.append(cell_acceleration_mps2)
            propulsion_force_n.append(cell_propulsion_force_n)
            motor_torque_nm.append(cell_motor_torque_nm)
            battery_power_w.append(cell_battery_power_w)
            cumulative_energy_j.append(energy_used_j)

        return Telemetry(
            {
                "vehicle.distance_m": tuple(telemetry_distance_m),
                "vehicle.speed_mps": tuple(average_speed_mps),
                "vehicle.longitudinal_acceleration_mps2": tuple(acceleration_mps2),
                "drivetrain.wheel_force_n": tuple(propulsion_force_n),
                "motor.torque_nm": tuple(motor_torque_nm),
                "battery.power_w": tuple(battery_power_w),
                "energy.cumulative_j": tuple(cumulative_energy_j),
            }
        )
