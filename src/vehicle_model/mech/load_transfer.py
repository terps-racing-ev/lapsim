"""Quasi-static load-transfer calculations, separate from suspension history.

Positive lateral acceleration turns left and transfers load to the right.
All functions return candidate results; only Suspension.update_state commits
those results to telemetry-visible component state.
"""

from dataclasses import dataclass, field
from math import isfinite, tan

from ..aero import AeroForces
from ..interfaces import ChassisModel
from .loads import TireNormalLoads


@dataclass(frozen=True, slots=True)
class LoadTransferResult:
    """Four normal loads and the roll solution that produced them."""

    normal_loads_n: TireNormalLoads = field(
        default_factory=lambda: TireNormalLoads(0.0, 0.0, 0.0, 0.0)
    )
    body_roll_angle_rad: float = 0.0
    total_roll_moment_nm: float = 0.0
    elastic_roll_moment_nm: float = 0.0
    front_roll_stiffness_fraction: float = 0.0
    front_lateral_load_transfer_n: float = 0.0
    rear_lateral_load_transfer_n: float = 0.0


def elastic_roll_arm_m(chassis: ChassisModel) -> float:
    cg_from_front_fraction = 1.0 - chassis.static_front_weight_fraction
    roll_axis_height_at_cg_m = (
        chassis.front_roll_axis_height_m
        + cg_from_front_fraction
        * (chassis.rear_roll_axis_height_m - chassis.front_roll_axis_height_m)
    )
    return max(chassis.cg_height_m - roll_axis_height_at_cg_m, 0.0)


def body_roll_angle_rad(
    mass_kg: float,
    chassis: ChassisModel,
    lateral_acceleration_mps2: float,
    total_roll_stiffness_nm_per_rad: float,
) -> float:
    """Return quasi-static sprung-body roll for a lateral acceleration."""

    return (
        mass_kg
        * lateral_acceleration_mps2
        * elastic_roll_arm_m(chassis)
        / total_roll_stiffness_nm_per_rad
    )


def wheel_travel_m(
    chassis: ChassisModel,
    body_roll_angle_rad: float,
) -> tuple[float, float, float, float]:
    """Return roll-induced bump/droop at each wheel.

    Positive travel is bump. Positive body roll, used for a left turn,
    puts the right-side suspension into bump and the left side into droop.
    """

    if not isfinite(body_roll_angle_rad):
        raise ValueError("body_roll_angle_rad must be finite")
    return tuple(
        -y_m * tan(body_roll_angle_rad) for _, y_m in chassis.wheel_positions_m
    )


def calculate_load_transfer(
    mass_kg: float,
    gravity_mps2: float,
    aero_forces: AeroForces,
    chassis: ChassisModel,
    total_roll_stiffness_nm_per_rad: float,
    front_lateral_load_transfer_fraction: float,
    longitudinal_acceleration_mps2: float = 0.0,
    lateral_acceleration_mps2: float = 0.0,
    *,
    roll_arm_m: float | None = None,
    roll_angle_rad: float | None = None,
) -> LoadTransferResult:
    """Calculate per-tire loads from pitch and quasi-static body roll."""

    front_axle_n, rear_axle_n = _axle_loads_with_pitch_transfer_n(
        mass_kg,
        gravity_mps2,
        aero_forces,
        chassis,
        longitudinal_acceleration_mps2,
    )

    total_lateral_force_n = mass_kg * lateral_acceleration_mps2
    if roll_arm_m is None:
        roll_arm_m = elastic_roll_arm_m(chassis)
    total_roll_moment_nm = total_lateral_force_n * chassis.cg_height_m
    elastic_roll_moment_nm = total_lateral_force_n * roll_arm_m
    front_geometric_roll_moment_nm = (
        chassis.static_front_weight_fraction
        * total_lateral_force_n
        * chassis.front_roll_axis_height_m
    )
    rear_geometric_roll_moment_nm = (
        (1.0 - chassis.static_front_weight_fraction)
        * total_lateral_force_n
        * chassis.rear_roll_axis_height_m
    )
    if roll_angle_rad is None:
        roll_angle_rad = body_roll_angle_rad(
            mass_kg,
            chassis,
            lateral_acceleration_mps2,
            total_roll_stiffness_nm_per_rad,
        )

    front_roll_stiffness_fraction = _front_elastic_stiffness_fraction(
        chassis,
        front_lateral_load_transfer_fraction,
        elastic_roll_moment_nm,
        front_geometric_roll_moment_nm,
        rear_geometric_roll_moment_nm,
    )
    front_roll_stiffness_nm_per_rad = (
        front_roll_stiffness_fraction * total_roll_stiffness_nm_per_rad
    )
    rear_roll_stiffness_nm_per_rad = (
        total_roll_stiffness_nm_per_rad - front_roll_stiffness_nm_per_rad
    )
    front_transfer_n = (
        front_geometric_roll_moment_nm
        + front_roll_stiffness_nm_per_rad * roll_angle_rad
    ) / chassis.front_track_width_m
    rear_transfer_n = (
        rear_geometric_roll_moment_nm + rear_roll_stiffness_nm_per_rad * roll_angle_rad
    ) / chassis.rear_track_width_m
    front_transfer_n = min(
        max(front_transfer_n, -front_axle_n / 2.0),
        front_axle_n / 2.0,
    )
    rear_transfer_n = min(
        max(rear_transfer_n, -rear_axle_n / 2.0),
        rear_axle_n / 2.0,
    )

    normal_loads = TireNormalLoads(
        front_left_n=front_axle_n / 2.0 - front_transfer_n,
        front_right_n=front_axle_n / 2.0 + front_transfer_n,
        rear_left_n=rear_axle_n / 2.0 - rear_transfer_n,
        rear_right_n=rear_axle_n / 2.0 + rear_transfer_n,
    )
    return LoadTransferResult(
        normal_loads_n=normal_loads,
        body_roll_angle_rad=roll_angle_rad,
        total_roll_moment_nm=total_roll_moment_nm,
        elastic_roll_moment_nm=elastic_roll_moment_nm,
        front_roll_stiffness_fraction=front_roll_stiffness_fraction,
        front_lateral_load_transfer_n=front_transfer_n,
        rear_lateral_load_transfer_n=rear_transfer_n,
    )


def _axle_loads_with_pitch_transfer_n(
    mass_kg: float,
    gravity_mps2: float,
    aero_forces: AeroForces,
    chassis: ChassisModel,
    longitudinal_acceleration_mps2: float,
) -> tuple[float, float]:
    """Add static and aerodynamic axle loads, then transfer load rearward.

    Clamp at axle lift so the result cannot contain negative normal loads.
    Positive longitudinal acceleration transfers normal load to the rear.
    """

    static_front_axle_n = chassis.static_front_weight_fraction * mass_kg * gravity_mps2
    static_rear_axle_n = (
        (1.0 - chassis.static_front_weight_fraction) * mass_kg * gravity_mps2
    )
    front_without_transfer_n = static_front_axle_n + aero_forces.front_downforce_n
    rear_without_transfer_n = static_rear_axle_n + aero_forces.rear_downforce_n
    rearward_transfer_n = (
        mass_kg
        * longitudinal_acceleration_mps2
        * chassis.cg_height_m
        / chassis.wheelbase_m
    )
    rearward_transfer_n = min(
        max(rearward_transfer_n, -rear_without_transfer_n),
        front_without_transfer_n,
    )
    front_axle_n = front_without_transfer_n - rearward_transfer_n
    rear_axle_n = rear_without_transfer_n + rearward_transfer_n

    return front_axle_n, rear_axle_n


def _front_elastic_stiffness_fraction(
    chassis: ChassisModel,
    front_lateral_load_transfer_fraction: float,
    elastic_roll_moment_nm: float,
    front_geometric_roll_moment_nm: float,
    rear_geometric_roll_moment_nm: float,
) -> float:
    """Find the elastic stiffness split needed for the target transfer split.

    Unequal track widths and geometric transfer make this differ from the
    target front fraction. Clamp to the physically attainable stiffness range.
    """

    target_front_fraction = front_lateral_load_transfer_fraction
    target_ratio = (
        target_front_fraction / (1.0 - target_front_fraction)
        if target_front_fraction < 1.0
        else float("inf")
    )
    if abs(elastic_roll_moment_nm) <= 1e-12:
        front_roll_stiffness_fraction = target_front_fraction
    elif target_front_fraction >= 1.0:
        front_roll_stiffness_fraction = 1.0
    elif target_front_fraction <= 0.0:
        front_roll_stiffness_fraction = 0.0
    else:
        numerator = (
            target_ratio
            * chassis.front_track_width_m
            * (rear_geometric_roll_moment_nm + elastic_roll_moment_nm)
            - chassis.rear_track_width_m * front_geometric_roll_moment_nm
        )
        denominator = elastic_roll_moment_nm * (
            chassis.rear_track_width_m + target_ratio * chassis.front_track_width_m
        )
        front_roll_stiffness_fraction = min(
            max(numerator / denominator, 0.0),
            1.0,
        )
    return front_roll_stiffness_fraction
