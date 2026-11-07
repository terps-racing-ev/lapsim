"""Steady-state maximum-speed solver for a constant-radius skidpad."""

from copy import deepcopy
from dataclasses import dataclass
from math import atan, isfinite

from scipy.optimize import least_squares

from lapsim.core.controls import Controls
from vehicle_model.dynamics import (
    ForceBalanceSolver,
    ForceRequests,
    MAX_ABS_BODY_SLIP_ANGLE_RAD,
)
from vehicle_model.vehicle import Vehicle

CONTROL_SOLVE_TIMESTEP_S = 0.05
CONTROL_RESIDUAL_TOLERANCE = 1e-4
YAW_RESIDUAL_TOLERANCE_MPS2 = 1e-4
SPEED_SEARCH_STEP_MPS = 0.5
SPEED_SEARCH_TOLERANCE_MPS = 1e-3
MAX_ABS_STEERING_ANGLE_RAD = 0.8


@dataclass(frozen=True, slots=True)
class SkidpadOperatingPoint:
    """Maximum feasible steady state on one constant-curvature circle."""

    speed_mps: float
    controls: Controls
    vehicle: Vehicle
    longitudinal_residual_mps2: float
    lateral_residual_mps2: float

    @property
    def lateral_utilization(self) -> float:
        if self.vehicle.lateral_force_capacity_n <= 0.0:
            return 0.0
        return min(
            abs(self.vehicle.current_lateral_force_n)
            / self.vehicle.lateral_force_capacity_n,
            1.0,
        )

    @property
    def feasible(self) -> bool:
        yaw_residual_mps2 = abs(self.vehicle.yaw_moment_nm) / (
            self.vehicle.mass_kg * self.vehicle.chassis.wheelbase_m
        )
        return (
            abs(self.longitudinal_residual_mps2) <= CONTROL_RESIDUAL_TOLERANCE
            and abs(self.lateral_residual_mps2) <= CONTROL_RESIDUAL_TOLERANCE
            and yaw_residual_mps2 <= YAW_RESIDUAL_TOLERANCE_MPS2
        )


def _evaluate_controls(
    vehicle: Vehicle,
    speed_mps: float,
    target_curvature_per_m: float,
    motor_torque_nm: float,
    steering_angle_rad: float,
    body_slip_angle_rad: float,
) -> SkidpadOperatingPoint:
    candidate = deepcopy(vehicle)
    candidate.initial_speed_mps = speed_mps
    candidate.reset_state()
    candidate.speed_mps = speed_mps
    candidate.body_slip_angle_rad = body_slip_angle_rad
    candidate.curvature_per_m = target_curvature_per_m
    controls = Controls(
        motor_torque_request_nm=float(motor_torque_nm),
        steering_angle_rad=float(steering_angle_rad),
    )
    candidate.update_state(controls, CONTROL_SOLVE_TIMESTEP_S)
    return SkidpadOperatingPoint(
        speed_mps=speed_mps,
        controls=controls,
        vehicle=candidate,
        longitudinal_residual_mps2=candidate.longitudinal_acceleration_mps2,
        lateral_residual_mps2=(
            candidate.lateral_acceleration_mps2 - speed_mps**2 * target_curvature_per_m
        ),
    )


def _solve_controls_at_speed(
    vehicle: Vehicle,
    speed_mps: float,
    target_curvature_per_m: float,
    initial_controls: Controls,
    initial_body_slip_angle_rad: float = 0.0,
) -> SkidpadOperatingPoint:
    # Solve the three steady balances at a prescribed speed and curvature.
    # Searching controls through update_state can jump to a different turning
    # branch and reject an otherwise feasible speed near peak lateral grip.
    candidate = deepcopy(vehicle)
    candidate.initial_speed_mps = speed_mps
    candidate.reset_state()
    candidate.speed_mps = speed_mps
    torque_limit_nm = candidate.drivetrain.available_motor_torque_nm(
        speed_mps, candidate.battery
    )
    if torque_limit_nm <= 0.0:
        torque_limit_nm = 1e-9
    turn_sign = 1.0 if target_curvature_per_m > 0.0 else -1.0
    lower_steering_rad, upper_steering_rad = (
        (0.0, MAX_ABS_STEERING_ANGLE_RAD)
        if turn_sign > 0.0
        else (-MAX_ABS_STEERING_ANGLE_RAD, 0.0)
    )
    initial_torque_nm = min(
        max(initial_controls.motor_torque_request_nm, 0.0),
        torque_limit_nm,
    )
    initial_steering_rad = min(
        max(initial_controls.steering_angle_rad, lower_steering_rad),
        upper_steering_rad,
    )

    def residuals(state: tuple[float, float, float]) -> tuple[float, float, float]:
        controls = Controls(
            motor_torque_request_nm=float(state[0]),
            steering_angle_rad=float(state[1]),
        )
        requests = ForceRequests.from_controls(candidate, controls)
        operating_point = ForceBalanceSolver(
            candidate, controls, requests, CONTROL_SOLVE_TIMESTEP_S
        ).evaluate(
            0.0, float(state[2]), target_curvature_per_m
        )
        return (
            operating_point.acceleration_mps2,
            operating_point.lateral_acceleration_mps2
            - speed_mps**2 * target_curvature_per_m,
            operating_point.yaw_moment_nm
            / (candidate.mass_kg * candidate.chassis.wheelbase_m),
        )

    solutions = []
    for beta in (initial_body_slip_angle_rad, -turn_sign * 0.05, turn_sign * 0.05):
        solution = least_squares(
            residuals,
            (initial_torque_nm, initial_steering_rad, beta),
            bounds=(
                (0.0, lower_steering_rad, -MAX_ABS_BODY_SLIP_ANGLE_RAD),
                (torque_limit_nm, upper_steering_rad, MAX_ABS_BODY_SLIP_ANGLE_RAD),
            ),
            x_scale=(max(torque_limit_nm * 0.25, 1.0), 0.2, 0.1),
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
            max_nfev=120,
        )
        solutions.append(solution)
        if max(abs(value) for value in solution.fun) < CONTROL_RESIDUAL_TOLERANCE:
            break
    solution = min(solutions, key=lambda item: sum(value**2 for value in item.fun))
    return _evaluate_controls(
        vehicle,
        speed_mps,
        target_curvature_per_m,
        solution.x[0],
        solution.x[1],
        solution.x[2],
    )


def solve_skidpad_operating_point(
    vehicle: Vehicle,
    target_curvature_per_m: float,
    initial_controls: Controls | None = None,
    initial_speed_mps: float = 0.0,
) -> SkidpadOperatingPoint:
    """Find maximum speed while holding circle curvature and zero acceleration."""

    if not isfinite(target_curvature_per_m) or abs(target_curvature_per_m) <= 1e-12:
        raise ValueError("skidpad curvature must be finite and nonzero")
    if not isfinite(initial_speed_mps) or initial_speed_mps < 0.0:
        raise ValueError("initial_speed_mps must be finite and nonnegative")
    vehicle.validate()
    seed = initial_controls or Controls()
    steering_seed_rad = atan(target_curvature_per_m * vehicle.chassis.wheelbase_m)
    controls = Controls(
        motor_torque_request_nm=min(
            max(seed.motor_torque_request_nm, 20.0),
            vehicle.drivetrain.motor.maximum_torque_nm,
        ),
        steering_angle_rad=steering_seed_rad,
    )

    maximum_speed_mps = vehicle.drivetrain.vehicle_speed_limit_mps
    minimum_search_speed_mps = min(1.0, 0.1 * maximum_speed_mps)
    lower_speed_mps = min(
        max(initial_speed_mps, minimum_search_speed_mps),
        maximum_speed_mps,
    )
    lower_point = _solve_controls_at_speed(
        vehicle,
        lower_speed_mps,
        target_curvature_per_m,
        controls,
    )
    while not lower_point.feasible and lower_speed_mps > minimum_search_speed_mps:
        lower_speed_mps = max(
            lower_speed_mps - SPEED_SEARCH_STEP_MPS,
            minimum_search_speed_mps,
        )
        lower_point = _solve_controls_at_speed(
            vehicle,
            lower_speed_mps,
            target_curvature_per_m,
            controls,
        )
    if not lower_point.feasible:
        raise RuntimeError("no feasible steady skidpad operating point")

    upper_speed_mps = lower_speed_mps
    while upper_speed_mps < maximum_speed_mps:
        candidate_speed_mps = min(
            upper_speed_mps + SPEED_SEARCH_STEP_MPS,
            maximum_speed_mps,
        )
        candidate = _solve_controls_at_speed(
            vehicle,
            candidate_speed_mps,
            target_curvature_per_m,
            lower_point.controls,
            lower_point.vehicle.body_slip_angle_rad,
        )
        if not candidate.feasible:
            upper_speed_mps = candidate_speed_mps
            break
        lower_speed_mps = candidate_speed_mps
        upper_speed_mps = candidate_speed_mps
        lower_point = candidate
        if candidate_speed_mps >= maximum_speed_mps:
            return candidate
    else:
        return lower_point

    while upper_speed_mps - lower_speed_mps > SPEED_SEARCH_TOLERANCE_MPS:
        candidate_speed_mps = 0.5 * (lower_speed_mps + upper_speed_mps)
        candidate = _solve_controls_at_speed(
            vehicle,
            candidate_speed_mps,
            target_curvature_per_m,
            lower_point.controls,
            lower_point.vehicle.body_slip_angle_rad,
        )
        if candidate.feasible:
            lower_speed_mps = candidate_speed_mps
            lower_point = candidate
        else:
            upper_speed_mps = candidate_speed_mps
    return lower_point


__all__ = ["SkidpadOperatingPoint", "solve_skidpad_operating_point"]
