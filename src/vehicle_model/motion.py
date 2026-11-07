"""Integrate vehicle motion after the forces for a time step are known.

This module has no component state or numerical force solver. Distances are
in meters, angles in radians, and positive curvature turns left.
"""

from dataclasses import dataclass
from math import cos, sin


@dataclass(frozen=True, slots=True)
class MotionStep:
    """End speed and pose changes over one complete time interval."""

    final_speed_mps: float
    distance_m: float
    x_displacement_m: float
    y_displacement_m: float
    final_heading_rad: float
    curvature_per_m: float
    yaw_rate_radps: float

    def average_speed_mps(self, timestep_s: float) -> float:
        """Include any stationary part of the interval in the average."""

        return self.distance_m / timestep_s


def integrate_motion(
    *,
    initial_speed_mps: float,
    initial_heading_rad: float,
    longitudinal_acceleration_mps2: float,
    lateral_acceleration_mps2: float,
    body_slip_angle_rad: float,
    speed_limit_mps: float,
    timestep_s: float,
) -> MotionStep:
    """Integrate constant acceleration and a constant-curvature ground path.

    The caller supplies a positive time step and nonnegative starting speed.
    Braking can stop the vehicle partway through the step; it stays at rest
    for the remaining time. Body heading and direction of travel differ by
    the solved body slip angle.
    """

    unconstrained_final_speed_mps = (
        initial_speed_mps + longitudinal_acceleration_mps2 * timestep_s
    )
    if unconstrained_final_speed_mps >= 0.0:
        final_speed_mps = min(unconstrained_final_speed_mps, speed_limit_mps)
        distance_m = 0.5 * (initial_speed_mps + final_speed_mps) * timestep_s
    else:
        stopping_time_s = -initial_speed_mps / longitudinal_acceleration_mps2
        final_speed_mps = 0.0
        distance_m = 0.5 * initial_speed_mps * stopping_time_s

    curvature_per_m = (
        lateral_acceleration_mps2 / initial_speed_mps**2
        if initial_speed_mps > 1e-6
        else 0.0
    )
    final_heading_rad = initial_heading_rad + curvature_per_m * distance_m
    initial_course_rad = initial_heading_rad + body_slip_angle_rad
    final_course_rad = final_heading_rad + body_slip_angle_rad
    if abs(curvature_per_m) > 1e-12:
        x_displacement_m = (
            sin(final_course_rad) - sin(initial_course_rad)
        ) / curvature_per_m
        y_displacement_m = (
            -(cos(final_course_rad) - cos(initial_course_rad)) / curvature_per_m
        )
    else:
        x_displacement_m = distance_m * cos(initial_course_rad)
        y_displacement_m = distance_m * sin(initial_course_rad)

    return MotionStep(
        final_speed_mps=final_speed_mps,
        distance_m=distance_m,
        x_displacement_m=x_displacement_m,
        y_displacement_m=y_displacement_m,
        final_heading_rad=final_heading_rad,
        curvature_per_m=curvature_per_m,
        yaw_rate_radps=initial_speed_mps * curvature_per_m,
    )
