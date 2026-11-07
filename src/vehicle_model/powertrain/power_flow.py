"""Account for actual wheel forces and the power exchanged with the pack."""

from __future__ import annotations

from dataclasses import dataclass

from lapsim.core.controls import Controls

from ..interfaces import BatteryModel, DrivetrainModel
from ..mech.wheel_state import TireStates


@dataclass(frozen=True, slots=True)
class PowerFlow:
    """Applied braking and power conversion for one solved time step.

    Propulsion and recovered power are nonnegative. Net battery power is
    positive on discharge and negative on charge. Motor RPM and pack power
    use driven-wheel surface speed, including longitudinal slip.
    """

    friction_braking_force_n: float
    regenerative_braking_force_n: float
    motor_torque_nm: float
    motor_speed_rpm: float
    propulsion_battery_power_w: float
    regenerative_power_w: float

    @property
    def net_battery_power_w(self) -> float:
        return self.propulsion_battery_power_w - self.regenerative_power_w

    @classmethod
    def calculate(
        cls,
        drivetrain: DrivetrainModel,
        battery: BatteryModel,
        controls: Controls,
        tire_states: TireStates,
        average_speed_mps: float,
    ) -> PowerFlow:
        """Use achieved tire forces, before advancing any component history.

        Preserve the baseline brake allocation: rear braking satisfies the
        regen request first, and remaining braking is hydraulic. Pack charge
        limits constrain recovered electrical power separately.
        """

        regenerative_braking_force_n = min(
            controls.rear_regenerative_brake_force_request_n,
            tire_states.rear_braking_force_n,
        )
        friction_braking_force_n = max(
            tire_states.braking_force_n - regenerative_braking_force_n,
            0.0,
        )
        wheel_surface_speed_mps = average_speed_mps * (
            1.0 + tire_states.driven_slip_ratio
        )
        motor_torque_nm = drivetrain.motor_torque_for_wheel_force_nm(
            tire_states.drive_force_n
        )
        motor_speed_rpm = drivetrain.motor_speed_rpm(wheel_surface_speed_mps)
        propulsion_battery_power_w = drivetrain.positive_battery_power_w(
            tire_states.drive_force_n,
            wheel_surface_speed_mps,
            battery,
        )
        regenerative_power_w = regenerative_battery_power_w(
            drivetrain,
            battery,
            regenerative_braking_force_n,
            wheel_surface_speed_mps,
        )
        return cls(
            friction_braking_force_n=friction_braking_force_n,
            regenerative_braking_force_n=regenerative_braking_force_n,
            motor_torque_nm=motor_torque_nm,
            motor_speed_rpm=motor_speed_rpm,
            propulsion_battery_power_w=propulsion_battery_power_w,
            regenerative_power_w=regenerative_power_w,
        )


def regenerative_efficiency(drivetrain: DrivetrainModel) -> float:
    """Legacy constant regen assumption; the supplied map has no generating data.

    Motor.efficiency is the configured fallback, not mapped motoring efficiency.
    """

    return (
        drivetrain.chain_drive.efficiency
        * drivetrain.motor.efficiency
        * drivetrain.inverter.efficiency
    )


def maximum_regenerative_brake_force_n(
    drivetrain: DrivetrainModel,
    battery: BatteryModel,
    vehicle_speed_mps: float,
) -> float:
    """Return rear-axle regen capacity from motor, pack, and speed limits.

    Tire grip and the driver's requested deceleration are applied separately.
    The baseline retains its constant generating-efficiency assumption; the
    published motor map only supplies positive-torque motoring bands.
    """

    if vehicle_speed_mps <= 0.0 or battery.charge_power_limit_w <= 0.0:
        return 0.0
    motor_speed_rpm = drivetrain.motor_speed_rpm(vehicle_speed_mps)
    if motor_speed_rpm >= drivetrain.motor.max_speed_rpm:
        return 0.0
    motor_torque_limit_nm = drivetrain.motor.torque_limit_nm(motor_speed_rpm)
    wheel_torque_limit_nm = (
        motor_torque_limit_nm
        * drivetrain.chain_drive.ratio
        / drivetrain.chain_drive.efficiency
    )
    torque_limited_force_n = wheel_torque_limit_nm / drivetrain.tire.rolling_radius_m
    wheel_power_limit_w = min(
        drivetrain.motor.peak_power_w / drivetrain.chain_drive.efficiency,
        battery.charge_power_limit_w / regenerative_efficiency(drivetrain),
    )
    return min(
        torque_limited_force_n,
        wheel_power_limit_w / vehicle_speed_mps,
    )


def regenerative_battery_power_w(
    drivetrain: DrivetrainModel,
    battery: BatteryModel,
    regenerative_brake_force_n: float,
    wheel_surface_speed_mps: float,
) -> float:
    """Return nonnegative pack-charge power recovered from rear braking."""

    if regenerative_brake_force_n < 0.0:
        raise ValueError("regenerative_brake_force_n cannot be negative")
    if wheel_surface_speed_mps < 0.0:
        raise ValueError("wheel_surface_speed_mps cannot be negative")
    requested_charge_power_w = (
        regenerative_brake_force_n
        * wheel_surface_speed_mps
        * regenerative_efficiency(drivetrain)
    )
    return battery.limit_charge_power_w(requested_charge_power_w)
