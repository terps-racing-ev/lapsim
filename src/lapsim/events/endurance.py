"""Stateful prescribed-path endurance simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, isfinite

from vehicle_model.mech.brakes import DEFAULT_MAXIMUM_BRAKE_PRESSURE_PSI
from vehicle_model.vehicle import Vehicle

from ..core.controls import Controls
from ..core.telemetry import JOULES_PER_KILOWATT_HOUR, Telemetry, TelemetryRecorder
from ..optimization.torque_profile import EnduranceControlProfile, TorqueProfile
from ..solvers.path_constraints import PathSpeedConstraints


@dataclass(frozen=True, slots=True)
class EnduranceRunConfig:
    """Event-independent run length and numerical behavior."""

    laps: int = 22
    starting_speed_mps: float | None = None
    maximum_driving_time_s: float = 3_600.0
    minimum_moving_speed_mps: float = 0.05
    path_speed_tolerance_mps: float = 0.01
    maximum_brake_pressure_psi: float | None = DEFAULT_MAXIMUM_BRAKE_PRESSURE_PSI

    def __post_init__(self) -> None:
        if self.laps <= 0:
            raise ValueError("laps must be positive")
        if self.starting_speed_mps is not None and self.starting_speed_mps < 0:
            raise ValueError("starting_speed_mps cannot be negative")
        if self.maximum_driving_time_s <= 0:
            raise ValueError("maximum_driving_time_s must be positive")
        if self.minimum_moving_speed_mps <= 0:
            raise ValueError("minimum_moving_speed_mps must be positive")
        if self.path_speed_tolerance_mps <= 0:
            raise ValueError("path_speed_tolerance_mps must be positive")
        if self.maximum_brake_pressure_psi is not None and (
            not isfinite(self.maximum_brake_pressure_psi)
            or self.maximum_brake_pressure_psi <= 0.0
        ):
            raise ValueError("maximum_brake_pressure_psi must be finite and positive")


@dataclass(frozen=True, slots=True)
class EnduranceRunResult:
    """Performance summary with optional full component-owned telemetry."""

    completed_laps: int
    driving_time_s: float
    lap_times_s: tuple[float, ...]
    pack_energy_kwh: float
    final_state_of_charge: float
    failure_reason: str | None
    telemetry: Telemetry | None

    @property
    def completed(self) -> bool:
        return self.failure_reason is None


class EnduranceSimulator:
    """Apply supplied driver controls while preserving component state.

    A torque-only profile receives path-following steering plus a deterministic
    longitudinal controller.  The controller reduces requested drive torque,
    coasts, or requests friction braking as needed to reach the next cyclic
    path-speed ceiling.  A full control profile supplies motor, brake-pressure,
    regen, and steering commands directly and is only validated against the
    path constraints.
    """

    @staticmethod
    def _path_resistance_and_lateral_force_n(
        vehicle: Vehicle,
        *,
        speed_mps: float,
        curvature_per_m: float,
    ) -> tuple[float, float]:
        """Return entry resistance and lateral force used by ``Vehicle``."""

        lateral_acceleration_mps2 = speed_mps**2 * curvature_per_m
        aero_forces = vehicle.aero_forces_n(
            speed_mps,
            lateral_acceleration_mps2,
        )
        rolling_force_n = vehicle.rolling_resistance_coefficient * (
            vehicle.mass_kg * vehicle.gravity_mps2 + aero_forces.downforce_n
        )
        tire_normal_loads = vehicle.suspension.tire_normal_loads_n(
            vehicle.mass_kg,
            vehicle.gravity_mps2,
            aero_forces,
            vehicle.chassis,
            longitudinal_acceleration_mps2=vehicle.longitudinal_acceleration_mps2,
            lateral_acceleration_mps2=lateral_acceleration_mps2,
        )
        lateral_capacity_n = sum(
            vehicle.tire.lateral_force_capacity_n(normal_load_n)
            for normal_load_n in tire_normal_loads.all_n
        )
        requested_lateral_force_n = (
            vehicle.mass_kg * speed_mps**2 * abs(curvature_per_m)
        )
        lateral_force_n = min(requested_lateral_force_n, lateral_capacity_n)
        total_normal_force_n = (
            vehicle.mass_kg * vehicle.gravity_mps2 + aero_forces.downforce_n
        )
        cornering_drag_force_n = (
            vehicle.cornering_drag_coefficient
            * lateral_force_n**2
            / total_normal_force_n
        )
        return (
            aero_forces.drag_n + rolling_force_n + cornering_drag_force_n,
            lateral_force_n,
        )

    @staticmethod
    def _brake_pressures_for_target_force(
        vehicle: Vehicle,
        *,
        brake_force_request_n: float,
        target_acceleration_mps2: float,
        lateral_force_n: float,
    ) -> tuple[float, float]:
        """Allocate a brake request between axles without stranding tire grip."""

        lateral_acceleration_mps2 = lateral_force_n / vehicle.mass_kg
        aero_forces = vehicle.aero_forces_n(
            vehicle.speed_mps,
            lateral_acceleration_mps2,
        )
        tire_normal_loads = vehicle.suspension.tire_normal_loads_n(
            vehicle.mass_kg,
            vehicle.gravity_mps2,
            aero_forces,
            vehicle.chassis,
            longitudinal_acceleration_mps2=target_acceleration_mps2,
            lateral_acceleration_mps2=lateral_acceleration_mps2,
        )
        tire_lateral_forces_n = vehicle.tire.lateral_forces_n(
            tire_normal_loads, lateral_force_n
        )
        tire_capacities_n = tuple(
            vehicle.tire.combined_longitudinal_force_capacity_n(load_n, lateral_n)
            for load_n, lateral_n in zip(
                tire_normal_loads.all_n,
                tire_lateral_forces_n.all_n,
                strict=True,
            )
        )
        front_capacity_n = sum(tire_capacities_n[:2])
        rear_capacity_n = sum(tire_capacities_n[2:])
        total_capacity_n = front_capacity_n + rear_capacity_n
        allocated_force_n = min(brake_force_request_n, total_capacity_n)
        front_fraction = vehicle.brakes.front_brake_force_fraction
        front_force_n = min(
            allocated_force_n * front_fraction,
            front_capacity_n,
        )
        rear_force_n = min(
            allocated_force_n * (1.0 - front_fraction),
            rear_capacity_n,
        )
        remaining_force_n = allocated_force_n - front_force_n - rear_force_n
        if remaining_force_n > 0.0:
            front_headroom_n = front_capacity_n - front_force_n
            rear_headroom_n = rear_capacity_n - rear_force_n
            total_headroom_n = front_headroom_n + rear_headroom_n
            if total_headroom_n > 0.0:
                front_addition_n = min(
                    remaining_force_n * front_headroom_n / total_headroom_n,
                    front_headroom_n,
                )
                front_force_n += front_addition_n
                rear_force_n += min(
                    remaining_force_n - front_addition_n,
                    rear_headroom_n,
                )
        return vehicle.brakes.axle_pressures_for_force_requests_psi(
            max(front_force_n, 0.0),
            max(rear_force_n, 0.0),
            vehicle.tire.rolling_radius_m,
        )

    def _torque_profile_controls(
        self,
        *,
        vehicle: Vehicle,
        request_fraction: float,
        curvature_per_m: float,
        cell_length_m: float,
        target_speed_mps: float,
        maximum_brake_pressure_psi: float | None,
    ) -> tuple[Controls, float, bool, float, bool]:
        """Convert a torque profile into ceiling-following controls for one cell.

        Returns the controls, unconstrained profile torque, whether drive torque
        was reduced, the requested hydraulic brake force, and whether either
        hydraulic command was pressure limited.
        """

        initial_speed_mps = vehicle.speed_mps
        profile_torque_nm = request_fraction * (
            vehicle.drivetrain.motor.torque_limit_nm(
                vehicle.drivetrain.motor_speed_rpm(initial_speed_mps)
            )
        )
        target_acceleration_mps2 = (
            target_speed_mps**2 - initial_speed_mps**2
        ) / (2.0 * cell_length_m)
        resistance_force_n, lateral_force_n = (
            self._path_resistance_and_lateral_force_n(
                vehicle,
                speed_mps=initial_speed_mps,
                curvature_per_m=curvature_per_m,
            )
        )
        force_for_target_n = (
            vehicle.effective_longitudinal_mass_kg * target_acceleration_mps2
            + resistance_force_n
        )

        brake_force_request_n = max(-force_for_target_n, 0.0)
        brake_pressure_limited = False
        if brake_force_request_n > 0.0:
            motor_torque_request_nm = 0.0
            front_pressure_psi, rear_pressure_psi = (
                self._brake_pressures_for_target_force(
                    vehicle,
                    brake_force_request_n=brake_force_request_n,
                    target_acceleration_mps2=target_acceleration_mps2,
                    lateral_force_n=lateral_force_n,
                )
            )
            brake_pressure_limited = (
                front_pressure_psi >= vehicle.brakes.maximum_pressure_psi
                or rear_pressure_psi >= vehicle.brakes.maximum_pressure_psi
            )
            if maximum_brake_pressure_psi is not None:
                brake_pressure_limited |= (
                    front_pressure_psi > maximum_brake_pressure_psi
                    or rear_pressure_psi > maximum_brake_pressure_psi
                )
                front_pressure_psi = min(
                    front_pressure_psi,
                    maximum_brake_pressure_psi,
                )
                rear_pressure_psi = min(
                    rear_pressure_psi,
                    maximum_brake_pressure_psi,
                )
        else:
            target_motor_torque_nm = (
                vehicle.drivetrain.motor_torque_for_wheel_force_nm(
                    max(force_for_target_n, 0.0)
                )
            )
            motor_torque_request_nm = min(
                profile_torque_nm,
                target_motor_torque_nm,
            )
            front_pressure_psi = 0.0
            rear_pressure_psi = 0.0

        torque_limited = motor_torque_request_nm < profile_torque_nm - 1e-9
        return (
            Controls(
                motor_torque_request_nm=motor_torque_request_nm,
                front_brake_pressure_psi=front_pressure_psi,
                rear_brake_pressure_psi=rear_pressure_psi,
                steering_angle_rad=atan(
                    curvature_per_m * vehicle.chassis.wheelbase_m
                ),
            ),
            profile_torque_nm,
            torque_limited,
            brake_force_request_n,
            brake_pressure_limited,
        )

    def run(
        self,
        vehicle: Vehicle,
        constraints: PathSpeedConstraints,
        profile: TorqueProfile | EnduranceControlProfile,
        config: EnduranceRunConfig,
        *,
        record_telemetry: bool = False,
    ) -> EnduranceRunResult:
        track = constraints.track
        if not track.closed:
            raise ValueError("Endurance simulation requires a closed track")
        vehicle.validate()
        vehicle.reset_state()
        start_ceiling_mps = constraints.braking_speed_ceiling_mps[0]
        vehicle.speed_mps = (
            start_ceiling_mps
            if config.starting_speed_mps is None
            else config.starting_speed_mps
        )

        recorder = TelemetryRecorder() if record_telemetry else None
        energy_j = 0.0
        lap_times_s: list[float] = []
        completed_laps = 0
        failure_reason: str | None = None
        lap_start_time_s = 0.0
        cell_lengths_m = track.cell_length_m

        for lap_index in range(config.laps):
            for cell_index, cell_length_m in enumerate(cell_lengths_m):
                next_cell_index = (cell_index + 1) % track.cell_count
                curvature_per_m = track.curvature_per_m[cell_index]
                target_speed_mps = constraints.braking_speed_ceiling_mps[
                    next_cell_index
                ]
                initial_speed_mps = vehicle.speed_mps
                if initial_speed_mps > (
                    constraints.local_corner_speed_mps[cell_index]
                    + config.path_speed_tolerance_mps
                ):
                    failure_reason = (
                        f"Car would spin out: supplied controls entered path cell "
                        f"{cell_index} on lap {lap_index + 1} above its local "
                        "corner-speed limit"
                    )
                    break
                if initial_speed_mps > (
                    constraints.braking_speed_ceiling_mps[cell_index]
                    + config.path_speed_tolerance_mps
                ):
                    failure_reason = (
                        "supplied controls entered a path cell above its braking "
                        "ceiling"
                    )
                    break

                lap_distance_m = track.cell_center_distance_m[cell_index]
                if isinstance(profile, EnduranceControlProfile):
                    controls = profile.controls_at(lap_distance_m)
                    if not isinstance(controls, Controls):
                        raise TypeError("controls_at must return Controls")
                    request_fraction = None
                    profile_torque_nm = None
                    path_torque_limited = False
                    path_brake_force_request_n = 0.0
                    path_brake_pressure_limited = False
                else:
                    request_fraction = profile.request_fraction(lap_distance_m)
                    (
                        controls,
                        profile_torque_nm,
                        path_torque_limited,
                        path_brake_force_request_n,
                        path_brake_pressure_limited,
                    ) = self._torque_profile_controls(
                        vehicle=vehicle,
                        request_fraction=request_fraction,
                        curvature_per_m=curvature_per_m,
                        cell_length_m=cell_length_m,
                        target_speed_mps=target_speed_mps,
                        maximum_brake_pressure_psi=(
                            config.maximum_brake_pressure_psi
                        ),
                    )
                time_before_step_s = vehicle.time_s
                try:
                    vehicle.update_state(controls, cell_length_m)
                except ValueError as error:
                    failure_reason = f"vehicle stalled on the endurance path: {error}"
                    break
                timestep_s = vehicle.time_s - time_before_step_s
                if timestep_s <= 0.0:
                    failure_reason = "vehicle stalled on the endurance path"
                    break
                if (
                    vehicle.speed_mps
                    > target_speed_mps + config.path_speed_tolerance_mps
                ):
                    failure_reason = (
                        "supplied controls exceeded the path ceiling "
                        f"at lap {lap_index + 1}, cell {cell_index}: "
                        f"entered {initial_speed_mps:.6f} m/s, "
                        f"reached {vehicle.speed_mps:.6f} m/s, "
                        f"ceiling {target_speed_mps:.6f} m/s; "
                        f"front pressure {controls.front_brake_pressure_psi:.2f} psi, "
                        f"rear pressure {controls.rear_brake_pressure_psi:.2f} psi"
                    )
                    break
                if vehicle.time_s > config.maximum_driving_time_s:
                    failure_reason = "maximum configured driving time exceeded"
                    break
                if vehicle.speed_mps < config.minimum_moving_speed_mps:
                    failure_reason = "vehicle stalled on the endurance path"
                    break
                energy_j += vehicle.battery.current_power_w * timestep_s
                if recorder is not None:
                    snapshot = vehicle.telemetry_snapshot()
                    snapshot.update(
                        {
                            "endurance.lap_index": float(lap_index),
                            "endurance.cell_index": float(cell_index),
                            "endurance.lap_distance_m": track.distance_m[
                                cell_index + 1
                            ],
                            "endurance.path_speed_ceiling_mps": target_speed_mps,
                            "energy.cumulative_net_j": energy_j,
                        }
                    )
                    if request_fraction is not None:
                        snapshot["endurance.torque_request_fraction"] = (
                            request_fraction
                        )
                        snapshot["endurance.profile_motor_torque_nm"] = (
                            profile_torque_nm
                        )
                        snapshot["endurance.path_torque_limited"] = float(
                            path_torque_limited
                        )
                        snapshot["endurance.path_brake_active"] = float(
                            path_brake_force_request_n > 0.0
                        )
                        snapshot["endurance.path_brake_force_request_n"] = (
                            path_brake_force_request_n
                        )
                        snapshot["endurance.path_brake_pressure_limited"] = float(
                            path_brake_pressure_limited
                        )
                    recorder.record(snapshot, timestep_s=timestep_s)
                if vehicle.battery.state_of_charge <= 0.0:
                    failure_reason = "battery state of charge depleted"
                    break

            if failure_reason is not None:
                break
            completed_laps += 1
            lap_times_s.append(vehicle.time_s - lap_start_time_s)
            lap_start_time_s = vehicle.time_s

        telemetry = recorder.freeze() if recorder is not None else None
        return EnduranceRunResult(
            completed_laps=completed_laps,
            driving_time_s=vehicle.time_s,
            lap_times_s=tuple(lap_times_s),
            pack_energy_kwh=energy_j / JOULES_PER_KILOWATT_HOUR,
            final_state_of_charge=vehicle.battery.state_of_charge,
            failure_reason=failure_reason,
            telemetry=telemetry,
        )

__all__ = ["EnduranceRunConfig", "EnduranceRunResult", "EnduranceSimulator"]
