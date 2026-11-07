"""Allocate axle requests to contact patches and apply longitudinal relaxation.

The Tire supplies grip laws and previous wheel state. These calculations return
new wheel results without committing history. Brakes get priority over rear
propulsion, which is biased toward the rear wheel with remaining grip.
"""

from math import cos, exp, isfinite, sin
from typing import TYPE_CHECKING

from .loads import TireCornerValues, TireNormalLoads
from .wheel_state import TireKinematics, TireState, TireStates
from .wheels import WHEEL_SIDE_SIGNS

if TYPE_CHECKING:
    from .tire import Tire

TIRE_MODEL_ORIENTATIONS = WHEEL_SIDE_SIGNS


def calculate_contact_forces(
    tire: "Tire",
    normal_loads_n: TireNormalLoads,
    kinematics: TireKinematics,
    drive_force_request_n: float,
    front_brake_force_request_n: float,
    rear_brake_force_request_n: float,
    timestep_s: float,
) -> TireStates:
    """Solve all four contact patches without mutating tire state."""

    if (
        min(
            drive_force_request_n,
            front_brake_force_request_n,
            rear_brake_force_request_n,
        )
        < 0.0
    ):
        raise ValueError("tire force requests cannot be negative")
    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be positive")
    if any(
        not all(
            isfinite(value)
            for value in (
                wheel.steering_angle_rad,
                wheel.longitudinal_speed_mps,
                wheel.lateral_speed_mps,
                wheel.camber_angle_rad,
            )
        )
        for wheel in kinematics.all
    ):
        raise ValueError("wheel kinematics must be finite")

    lateral_forces_n = TireCornerValues.from_iterable(
        model_orientation
        * tire.lateral_force_from_slip_angle_n(
            load_n,
            model_orientation * wheel.slip_angle_rad,
            wheel.camber_angle_rad,
        )
        for load_n, wheel, model_orientation in zip(
            normal_loads_n.all_n,
            kinematics.all,
            TIRE_MODEL_ORIENTATIONS,
            strict=True,
        )
    )
    capacities_n = tuple(
        tire.combined_longitudinal_force_capacity_n(
            load_n,
            lateral_n,
            wheel.camber_angle_rad,
        )
        for load_n, lateral_n, wheel in zip(
            normal_loads_n.all_n,
            lateral_forces_n.all_n,
            kinematics.all,
            strict=True,
        )
    )
    drive_forces_n, braking_forces_n = _allocate_longitudinal_forces(
        capacities_n,
        drive_force_request_n,
        front_brake_force_request_n,
        rear_brake_force_request_n,
    )

    states = []
    for (
        load_n,
        lateral_n,
        capacity_n,
        drive_n,
        brake_n,
        wheel,
        previous,
    ) in zip(
        normal_loads_n.all_n,
        lateral_forces_n.all_n,
        capacities_n,
        drive_forces_n,
        braking_forces_n,
        kinematics.all,
        tire.current_states.all,
        strict=True,
    ):
        drive_n, brake_n, slip_ratio = tire._longitudinal_response(
            drive_n,
            brake_n,
            capacity_n,
            previous.slip_ratio,
            abs(wheel.longitudinal_speed_mps),
            timestep_s,
        )
        wheel_longitudinal_force_n = drive_n - brake_n
        steering_cos = cos(wheel.steering_angle_rad)
        steering_sin = sin(wheel.steering_angle_rad)
        body_longitudinal_force_n = (
            wheel_longitudinal_force_n * steering_cos - lateral_n * steering_sin
        )
        body_lateral_force_n = (
            wheel_longitudinal_force_n * steering_sin + lateral_n * steering_cos
        )
        wheel_surface_speed_mps = wheel.longitudinal_speed_mps * (1.0 + slip_ratio)
        states.append(
            TireState(
                normal_load_n=load_n,
                lateral_force_n=lateral_n,
                lateral_capacity_n=tire.lateral_force_capacity_n(
                    load_n,
                    wheel.camber_angle_rad,
                ),
                longitudinal_capacity_n=capacity_n,
                drive_force_n=drive_n,
                braking_force_n=brake_n,
                steering_angle_rad=wheel.steering_angle_rad,
                camber_angle_rad=wheel.camber_angle_rad,
                local_longitudinal_speed_mps=wheel.longitudinal_speed_mps,
                local_lateral_speed_mps=wheel.lateral_speed_mps,
                slip_angle_rad=wheel.slip_angle_rad,
                slip_ratio=slip_ratio,
                wheel_surface_speed_mps=wheel_surface_speed_mps,
                wheel_angular_speed_radps=(
                    wheel_surface_speed_mps / tire.rolling_radius_m
                ),
                body_longitudinal_force_n=body_longitudinal_force_n,
                body_lateral_force_n=body_lateral_force_n,
            )
        )
    return TireStates(kinematics.vehicle_speed_mps, *states)


def longitudinal_response(
    tire: "Tire",
    drive_force_n: float,
    braking_force_n: float,
    capacity_n: float,
    previous_slip_ratio: float,
    vehicle_speed_mps: float,
    timestep_s: float,
) -> tuple[float, float, float]:
    """Apply the tire-owned force/slip curve and optional relaxation."""

    force_magnitude_n = drive_force_n + braking_force_n
    if capacity_n <= 0.0 or tire.peak_longitudinal_slip_ratio == 0.0:
        return drive_force_n, braking_force_n, 0.0
    direction = 1.0 if drive_force_n >= braking_force_n else -1.0
    target_slip = (
        direction
        * tire.peak_longitudinal_slip_ratio
        * min(force_magnitude_n / capacity_n, 1.0)
    )
    if tire.longitudinal_slip_relaxation_length_m <= 0.0:
        return drive_force_n, braking_force_n, target_slip
    relaxation_fraction = 1.0 - exp(
        -max(vehicle_speed_mps, 0.1)
        * timestep_s
        / tire.longitudinal_slip_relaxation_length_m
    )
    slip_ratio = previous_slip_ratio + relaxation_fraction * (
        target_slip - previous_slip_ratio
    )
    available_force_n = capacity_n * min(
        abs(slip_ratio) / tire.peak_longitudinal_slip_ratio, 1.0
    )
    scale = (
        min(available_force_n / force_magnitude_n, 1.0) if force_magnitude_n else 0.0
    )
    return drive_force_n * scale, braking_force_n * scale, slip_ratio


def _allocate_longitudinal_forces(
    capacities_n: tuple[float, ...],
    drive_force_request_n: float,
    front_brake_force_request_n: float,
    rear_brake_force_request_n: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Apply each axle's brakes, then distribute drive over remaining rear grip.

    Each axle's brake request is split equally between its two wheels. The
    ideal limited-slip baseline allocates rear drive in proportion to grip.
    """

    brake_requests_n = (
        front_brake_force_request_n / 2.0,
        front_brake_force_request_n / 2.0,
        rear_brake_force_request_n / 2.0,
        rear_brake_force_request_n / 2.0,
    )
    braking_forces_n = tuple(
        min(request_n, capacity_n)
        for request_n, capacity_n in zip(brake_requests_n, capacities_n, strict=True)
    )
    rear_drive_capacities_n = tuple(
        max(capacity_n - brake_force_n, 0.0)
        for capacity_n, brake_force_n in zip(
            capacities_n[2:], braking_forces_n[2:], strict=True
        )
    )
    total_rear_drive_capacity_n = sum(rear_drive_capacities_n)
    # Preserve the endurance baseline's ideal limited-slip assumption by
    # biasing rear drive force toward the contact patch with available grip.
    rear_drive_forces_n = (
        tuple(
            min(
                drive_force_request_n * capacity_n / total_rear_drive_capacity_n,
                capacity_n,
            )
            for capacity_n in rear_drive_capacities_n
        )
        if total_rear_drive_capacity_n > 0.0
        else (0.0, 0.0)
    )
    drive_forces_n = (0.0, 0.0) + rear_drive_forces_n

    return drive_forces_n, braking_forces_n
