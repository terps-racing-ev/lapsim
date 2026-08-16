"""Stateful prescribed-path endurance simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan

from vehicle_model.vehicle import Vehicle

from .controls import Controls
from .path_constraints import PathSpeedConstraints
from .telemetry import JOULES_PER_KILOWATT_HOUR, Telemetry, TelemetryRecorder
from .torque_profile import EnduranceControlProfile, TorqueProfile


@dataclass(frozen=True, slots=True)
class EnduranceRunConfig:
    """Event-independent run length and numerical behavior."""

    laps: int = 22
    starting_speed_mps: float | None = None
    maximum_driving_time_s: float = 3_600.0
    minimum_moving_speed_mps: float = 0.05
    path_speed_tolerance_mps: float = 0.01

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

    Path curvature and the cyclic braking ceiling are validation constraints,
    never control commands. A torque-only profile receives path-following
    steering and zero brake pressure. A full control profile supplies motor,
    brake-pressure, regen, and steering commands directly.
    """

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
                else:
                    request_fraction = profile.request_fraction(lap_distance_m)
                    controls = Controls(
                        motor_torque_request_nm=request_fraction
                        * vehicle.drivetrain.motor.torque_limit_nm(
                            vehicle.drivetrain.motor_speed_rpm(initial_speed_mps)
                        ),
                        steering_angle_rad=atan(
                            curvature_per_m * vehicle.chassis.wheelbase_m
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
