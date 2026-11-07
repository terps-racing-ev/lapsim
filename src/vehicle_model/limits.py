"""Whole-vehicle performance limits used by spatial lap solvers.

A braking envelope couples aero, load transfer, alignment, and tire grip; it is
separate from the hydraulic brake hardware and its retained operating state.
"""

from math import atan
from typing import TYPE_CHECKING

from scipy.optimize import brentq

if TYPE_CHECKING:
    from .vehicle import Vehicle


def maximum_braking_deceleration_mps2(
    vehicle: "Vehicle",
    speed_mps: float,
    curvature_per_m: float,
    gravity_mps2: float,
    air_density_kgpm3: float,
    solver_tolerance_mps2: float,
) -> float:
    """Return the greatest feasible positive deceleration magnitude."""

    aero_forces = vehicle.aero_forces_n(
        speed_mps,
        speed_mps**2 * curvature_per_m,
        air_density_kgpm3,
        curvature_per_m=curvature_per_m,
    )
    total_normal_force_n = vehicle.mass_kg * gravity_mps2 + aero_forces.downforce_n
    drag_force_n = aero_forces.drag_n
    rolling_force_n = vehicle.rolling_resistance_coefficient * total_normal_force_n

    lateral_acceleration_mps2 = speed_mps**2 * curvature_per_m
    total_lateral_force_n = vehicle.mass_kg * abs(lateral_acceleration_mps2)
    alignment_state = vehicle.wheel_alignment_state(
        atan(curvature_per_m * vehicle.chassis.wheelbase_m),
        lateral_acceleration_mps2,
    )

    def braking_residual(deceleration_mps2: float) -> float:
        tire_normal_loads = vehicle.suspension.tire_normal_loads_n(
            vehicle.mass_kg,
            gravity_mps2,
            aero_forces,
            vehicle.chassis,
            longitudinal_acceleration_mps2=-deceleration_mps2,
            lateral_acceleration_mps2=lateral_acceleration_mps2,
        )
        tire_lateral_forces_n = vehicle.tire.lateral_forces_n(
            tire_normal_loads,
            total_lateral_force_n,
            front_lateral_force_fraction=(vehicle.chassis.static_front_weight_fraction),
            camber_angles_rad=alignment_state.camber_angles_rad,
        )
        braking_force_n = sum(
            vehicle.tire.combined_longitudinal_force_capacity_n(
                normal_load_n,
                lateral_force_n,
                camber_angle_rad,
            )
            for normal_load_n, lateral_force_n, camber_angle_rad in zip(
                tire_normal_loads.all_n,
                tire_lateral_forces_n.all_n,
                alignment_state.camber_angles_rad,
                strict=True,
            )
        )
        force_limited_deceleration_mps2 = (
            braking_force_n + drag_force_n + rolling_force_n
        ) / vehicle.effective_longitudinal_mass_kg
        return deceleration_mps2 - force_limited_deceleration_mps2

    maximum_tire_force_n = (
        vehicle.tire.maximum_longitudinal_coefficient * total_normal_force_n
    )
    upper_deceleration_mps2 = (
        maximum_tire_force_n + drag_force_n + rolling_force_n
    ) / vehicle.effective_longitudinal_mass_kg
    # The ideal constant-mu solution can land exactly on this analytical
    # bound. Give Brent a small strict bracket so floating-point roundoff
    # cannot leave both residual endpoints on the same side of zero.
    upper_deceleration_mps2 = 1.01 * upper_deceleration_mps2 + solver_tolerance_mps2

    return brentq(
        braking_residual,
        0.0,
        upper_deceleration_mps2,
        xtol=solver_tolerance_mps2,
    )
