"""Translate controls into axle requests and solve the vehicle force balance.

Each solver instance describes one time step at its starting speed. Candidate
force evaluations read the previous tire state without advancing component
history. Aero configuration is selected for each candidate curvature.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, tan
from typing import TYPE_CHECKING

from scipy.optimize import brentq, least_squares

from lapsim.core.controls import Controls

from .aero import AeroForces
from .mech import TireKinematics, TireNormalLoads, TireStates, WheelAlignmentState

if TYPE_CHECKING:
    from .vehicle import Vehicle

MAX_ABS_ACCELERATION_MPS2 = 100.0
MAX_ABS_BODY_SLIP_ANGLE_RAD = 0.8
MAX_ABS_CURVATURE_PER_M = 1.0
OPERATING_POINT_TOLERANCE = 1e-7


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """Forces and tire geometry at a solved or candidate vehicle motion."""

    acceleration_mps2: float
    lateral_acceleration_mps2: float
    body_slip_angle_rad: float
    curvature_per_m: float
    yaw_rate_radps: float
    yaw_moment_nm: float
    aero_forces: AeroForces
    tire_states: TireStates
    alignment_state: WheelAlignmentState
    lateral_capacity_n: float
    rear_drive_capacity_n: float
    speed_limited_drive_force_n: float
    cornering_drag_force_n: float
    resistance_force_n: float
    normal_loads_n: TireNormalLoads


@dataclass(frozen=True, slots=True)
class ForceRequests:
    """Axle requests before tire grip and the vehicle speed cap are applied.

    Rear braking includes hydraulic and regenerative requests. The motor
    request has already been limited by available motor and battery power.
    """

    available_motor_torque_nm: float
    applied_motor_torque_nm: float
    drive_force_n: float
    front_brake_force_n: float
    rear_brake_force_n: float

    @property
    def total_brake_force_n(self) -> float:
        return self.front_brake_force_n + self.rear_brake_force_n

    @classmethod
    def from_controls(cls, vehicle: Vehicle, controls: Controls) -> ForceRequests:
        """Convert motor torque and hydraulic pressures to wheel forces."""

        available_motor_torque_nm = vehicle.drivetrain.available_motor_torque_nm(
            vehicle.speed_mps, vehicle.battery
        )
        applied_motor_torque_nm = min(
            controls.motor_torque_request_nm, available_motor_torque_nm
        )
        drive_force_n = vehicle.drivetrain.wheel_force_from_motor_torque_n(
            applied_motor_torque_nm
        )
        front_friction_force_n, rear_friction_force_n = (
            vehicle.brakes.axle_force_requests_from_pressures_n(
                controls.front_brake_pressure_psi,
                controls.rear_brake_pressure_psi,
                vehicle.tire.rolling_radius_m,
            )
        )
        return cls(
            available_motor_torque_nm=available_motor_torque_nm,
            applied_motor_torque_nm=applied_motor_torque_nm,
            drive_force_n=drive_force_n,
            front_brake_force_n=front_friction_force_n,
            rear_brake_force_n=(
                rear_friction_force_n + controls.rear_regenerative_brake_force_request_n
            ),
        )


class ForceBalanceSolver:
    """Find acceleration, body slip, and curvature that balance tire forces.

    Straight motion needs only a longitudinal root solve. Cornering also
    balances lateral acceleration and steady yaw moment. Solver settings live
    here so they are separate from motion integration and component updates.
    """

    def __init__(
        self,
        vehicle: Vehicle,
        controls: Controls,
        requests: ForceRequests,
        timestep_s: float,
    ) -> None:
        self.vehicle = vehicle
        self.controls = controls
        self.requests = requests
        self.timestep_s = timestep_s
        self.initial_speed_mps = vehicle.speed_mps
        self.wheel_positions_m = vehicle.chassis.wheel_positions_m

    def evaluate(
        self,
        assumed_longitudinal_acceleration_mps2: float,
        body_slip_angle_rad: float,
        curvature_per_m: float,
    ) -> OperatingPoint:
        """Evaluate loads, tire forces, road resistance, and force balance."""

        vehicle = self.vehicle
        assumed_lateral_acceleration_mps2 = self.initial_speed_mps**2 * curvature_per_m
        aero_forces = vehicle.aero_forces_n(
            self.initial_speed_mps,
            assumed_lateral_acceleration_mps2,
            curvature_per_m=curvature_per_m,
        )
        # Resolve candidate body motion into loads and wheel geometry.
        tire_normal_loads = vehicle.suspension.tire_normal_loads_n(
            vehicle.mass_kg,
            vehicle.gravity_mps2,
            aero_forces,
            vehicle.chassis,
            longitudinal_acceleration_mps2=(assumed_longitudinal_acceleration_mps2),
            lateral_acceleration_mps2=assumed_lateral_acceleration_mps2,
        )
        alignment_state = vehicle.alignment.calculate(
            vehicle.chassis,
            self.controls.steering_angle_rad,
            aero_forces.body_roll_angle_rad,
            vehicle.suspension.wheel_travel_m(
                vehicle.chassis,
                aero_forces.body_roll_angle_rad,
            ),
        )
        tire_kinematics = TireKinematics.from_vehicle_motion(
            self.initial_speed_mps,
            body_slip_angle_rad,
            curvature_per_m,
            alignment_state.steering_angles_rad,
            self.wheel_positions_m,
            alignment_state.camber_angles_rad,
        )
        # Apply tire grip and longitudinal slip before limiting drive by speed.
        tire_states = vehicle.tire.calculate_forces(
            tire_normal_loads,
            tire_kinematics,
            self.requests.drive_force_n,
            self.requests.front_brake_force_n,
            self.requests.rear_brake_force_n,
            self.timestep_s,
        )
        lateral_force_n = abs(tire_states.normal_force_n(body_slip_angle_rad))
        lateral_capacity_n = vehicle.tire.total_lateral_force_capacity_n(
            tire_normal_loads,
            vehicle.chassis.static_front_weight_fraction,
            alignment_state.camber_angles_rad,
        )
        cornering_drag_force_n = (
            vehicle.cornering_drag_coefficient
            * lateral_force_n**2
            / (vehicle.mass_kg * vehicle.gravity_mps2 + aero_forces.downforce_n)
        )
        rolling_force_n = vehicle.rolling_resistance_coefficient * (
            vehicle.mass_kg * vehicle.gravity_mps2 + aero_forces.downforce_n
        )
        resistance_force_n = (
            aero_forces.drag_n + rolling_force_n + cornering_drag_force_n
        )
        maximum_acceleration_to_speed_limit_mps2 = max(
            0.0,
            (vehicle.drivetrain.vehicle_speed_limit_mps - self.initial_speed_mps)
            / self.timestep_s,
        )
        speed_limited_drive_force_n = (
            vehicle.effective_longitudinal_mass_kg
            * maximum_acceleration_to_speed_limit_mps2
            + tire_states.braking_force_n
            + resistance_force_n
        )
        if speed_limited_drive_force_n < self.requests.drive_force_n:
            tire_states = vehicle.tire.calculate_forces(
                tire_normal_loads,
                tire_kinematics,
                speed_limited_drive_force_n,
                self.requests.front_brake_force_n,
                self.requests.rear_brake_force_n,
                self.timestep_s,
            )
        # Project contact forces along the ground path and balance acceleration.
        tangent_tire_force_n = tire_states.tangent_force_n(body_slip_angle_rad)
        normal_tire_force_n = tire_states.normal_force_n(body_slip_angle_rad)
        force_balance_acceleration_mps2 = (
            tangent_tire_force_n - resistance_force_n
        ) / vehicle.effective_longitudinal_mass_kg
        if self.initial_speed_mps <= 0.0:
            force_balance_acceleration_mps2 = max(
                force_balance_acceleration_mps2,
                0.0,
            )
        yaw_moment_nm = tire_states.yaw_moment_nm(self.wheel_positions_m)
        return OperatingPoint(
            acceleration_mps2=force_balance_acceleration_mps2,
            lateral_acceleration_mps2=normal_tire_force_n / vehicle.mass_kg,
            body_slip_angle_rad=body_slip_angle_rad,
            curvature_per_m=curvature_per_m,
            yaw_rate_radps=self.initial_speed_mps * curvature_per_m,
            yaw_moment_nm=yaw_moment_nm,
            aero_forces=aero_forces,
            tire_states=tire_states,
            alignment_state=alignment_state,
            lateral_capacity_n=lateral_capacity_n,
            rear_drive_capacity_n=max(
                tire_states.rear_longitudinal_capacity_n
                - tire_states.rear_braking_force_n,
                0.0,
            ),
            speed_limited_drive_force_n=speed_limited_drive_force_n,
            cornering_drag_force_n=cornering_drag_force_n,
            resistance_force_n=resistance_force_n,
            normal_loads_n=tire_normal_loads,
        )

    def residuals(
        self, state: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Residuals in longitudinal force, lateral force, and yaw balance."""

        assumed_acceleration_mps2, body_slip_angle_rad, curvature_per_m = state
        operating_point = self.evaluate(
            assumed_acceleration_mps2,
            body_slip_angle_rad,
            curvature_per_m,
        )
        if self.initial_speed_mps <= 1e-6:
            return (
                assumed_acceleration_mps2 - operating_point.acceleration_mps2,
                body_slip_angle_rad,
                curvature_per_m,
            )
        return (
            assumed_acceleration_mps2 - operating_point.acceleration_mps2,
            operating_point.lateral_acceleration_mps2
            - self.initial_speed_mps**2 * curvature_per_m,
            operating_point.yaw_moment_nm
            / (self.vehicle.mass_kg * self.vehicle.chassis.wheelbase_m),
        )

    def solve(self) -> OperatingPoint:
        """Return the final force evaluation at the converged operating point."""

        if (
            abs(self.controls.steering_angle_rad) <= 1e-12
            or self.initial_speed_mps <= 1e-6
        ):
            return self._solve_straight()
        return self._solve_cornering()

    def _solve_straight(self) -> OperatingPoint:
        """Hold body slip and curvature at zero and solve acceleration."""

        longitudinal_acceleration_mps2 = brentq(
            lambda acceleration_mps2: (
                acceleration_mps2
                - self.evaluate(acceleration_mps2, 0.0, 0.0).acceleration_mps2
            ),
            -MAX_ABS_ACCELERATION_MPS2,
            MAX_ABS_ACCELERATION_MPS2,
            xtol=OPERATING_POINT_TOLERANCE,
        )
        return self.evaluate(
            longitudinal_acceleration_mps2,
            0.0,
            0.0,
        )

    def _solve_cornering(self) -> OperatingPoint:
        """Solve all three balances using prior motion or steering as a seed."""

        initial_curvature_per_m = (
            self.vehicle.curvature_per_m
            if abs(self.vehicle.curvature_per_m) > 1e-12
            else tan(self.controls.steering_angle_rad)
            / self.vehicle.chassis.wheelbase_m
        )
        initial_body_slip_angle_rad = (
            self.vehicle.body_slip_angle_rad
            if abs(self.vehicle.body_slip_angle_rad) > 1e-12
            else atan2(
                -self.wheel_positions_m[2][0] * tan(self.controls.steering_angle_rad),
                self.vehicle.chassis.wheelbase_m,
            )
        )
        initial_solution = (
            0.0,
            max(
                -MAX_ABS_BODY_SLIP_ANGLE_RAD,
                min(
                    initial_body_slip_angle_rad,
                    MAX_ABS_BODY_SLIP_ANGLE_RAD,
                ),
            ),
            max(
                -MAX_ABS_CURVATURE_PER_M,
                min(initial_curvature_per_m, MAX_ABS_CURVATURE_PER_M),
            ),
        )
        solution = least_squares(
            self.residuals,
            initial_solution,
            bounds=(
                (
                    -MAX_ABS_ACCELERATION_MPS2,
                    -MAX_ABS_BODY_SLIP_ANGLE_RAD,
                    -MAX_ABS_CURVATURE_PER_M,
                ),
                (
                    MAX_ABS_ACCELERATION_MPS2,
                    MAX_ABS_BODY_SLIP_ANGLE_RAD,
                    MAX_ABS_CURVATURE_PER_M,
                ),
            ),
            xtol=OPERATING_POINT_TOLERANCE,
            ftol=OPERATING_POINT_TOLERANCE,
            gtol=OPERATING_POINT_TOLERANCE,
            max_nfev=100,
        )
        if not solution.success:
            raise RuntimeError(
                "vehicle force-balance solve did not converge: "
                f"status={solution.status}, state={tuple(solution.x)}, "
                f"residuals={tuple(solution.fun)}"
            )
        return self.evaluate(*solution.x)
