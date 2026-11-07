"""Powertrain coordination and wheel-force conversions."""

from dataclasses import dataclass, field

from utils.units import (
    miles_per_hour_to_meters_per_second,
    radians_per_second_to_revolutions_per_minute,
    revolutions_per_minute_to_radians_per_second,
)

from ..electrical.inverter import Inverter
from ..interfaces import (
    BatteryModel,
    ComponentModel,
    ChainDriveModel,
    InverterModel,
    MotorModel,
    TireModel,
)
from ..mech.tire import Tire
from .chain_drive import ChainDrive
from .motor import Motor

DEFAULT_SPEED_LIMIT_MPH = 100.0
DEFAULT_SPEED_LIMIT_MPS = miles_per_hour_to_meters_per_second(DEFAULT_SPEED_LIMIT_MPH)
DEFAULT_DRIVEN_WHEEL_INERTIA_KGM2 = 0.75


@dataclass(slots=True)
class Drivetrain:
    """Coordinate independently replaceable propulsion components.

    ``Motor``, ``Inverter``, and ``ChainDrive`` are baseline implementations.
    Any object satisfying their protocols can be supplied instead. Component
    parameters and state are accessed through their owning nested model.
    """

    motor: MotorModel = field(default_factory=Motor)
    inverter: InverterModel = field(default_factory=Inverter)
    chain_drive: ChainDriveModel = field(default_factory=ChainDrive)
    tire: TireModel = field(default_factory=Tire)
    driven_wheel_inertia_kgm2: float = DEFAULT_DRIVEN_WHEEL_INERTIA_KGM2
    configured_speed_limit_mps: float | None = DEFAULT_SPEED_LIMIT_MPS
    current_wheel_force_n: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name, component, protocol in (
            ("motor", self.motor, MotorModel),
            ("inverter", self.inverter, InverterModel),
            ("chain_drive", self.chain_drive, ChainDriveModel),
            ("tire", self.tire, TireModel),
        ):
            if not isinstance(component, protocol):
                raise TypeError(f"{name} does not satisfy {protocol.__name__}")
            component.validate()
        if self.driven_wheel_inertia_kgm2 < 0:
            raise ValueError("driven_wheel_inertia_kgm2 cannot be negative")
        if (
            self.configured_speed_limit_mps is not None
            and self.configured_speed_limit_mps <= 0
        ):
            raise ValueError("configured_speed_limit_mps must be positive or None")

    @property
    def components(self) -> tuple[ComponentModel, ...]:
        return self.motor, self.inverter, self.chain_drive

    def reset_state(self) -> None:
        for component in self.components:
            component.reset_state()
        self.current_wheel_force_n = 0.0

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        wheel_torque_nm = self.current_wheel_force_n * self.tire.rolling_radius_m
        telemetry.update(
            {
                "drivetrain.wheel_force_n": self.current_wheel_force_n,
                "drivetrain.wheel_torque_nm": wheel_torque_nm,
                "drivetrain.equivalent_rotating_mass_kg": (
                    self.equivalent_rotating_mass_kg
                ),
                "drivetrain.vehicle_speed_limit_mps": self.vehicle_speed_limit_mps,
            }
        )
        for component in self.components:
            component.update_telemetry(telemetry)

    def update_state(
        self,
        motor_torque_nm: float,
        motor_speed_rpm: float,
        wheel_force_n: float,
        timestep_s: float,
        battery_power_w: float = 0.0,
    ) -> None:
        """Retain operating state in the coordinator and owned submodels."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.motor.update_state(motor_torque_nm, motor_speed_rpm, timestep_s)
        motor_mechanical_power_w = max(
            motor_torque_nm
            * revolutions_per_minute_to_radians_per_second(motor_speed_rpm),
            0.0,
        )
        motor_electrical_power_w = self.motor.electrical_power_for_mechanical_output_w(
            motor_mechanical_power_w,
            motor_speed_rpm,
            motor_torque_nm,
        )
        self.inverter.update_state(
            battery_power_w,
            motor_electrical_power_w,
            timestep_s,
        )
        self.current_wheel_force_n = wheel_force_n

    @property
    def efficiency(self) -> float:
        """Nominal/fallback pack-to-wheel efficiency; actual motoring uses the map."""

        return (
            self.inverter.efficiency
            * self.motor.efficiency
            * self.chain_drive.efficiency
        )

    @property
    def wheel_referenced_rotational_inertia_kgm2(self) -> float:
        """Reflect motor-side inertia through the squared reduction ratio."""

        input_inertia_kgm2 = (
            self.motor.rotor_inertia_kgm2 + self.chain_drive.input_inertia_kgm2
        )
        return (
            input_inertia_kgm2 * self.chain_drive.ratio**2
            + self.chain_drive.output_inertia_kgm2
            + self.driven_wheel_inertia_kgm2
        )

    @property
    def equivalent_rotating_mass_kg(self) -> float:
        """Convert wheel-referenced inertia to longitudinal mass using I/r²."""

        return (
            self.wheel_referenced_rotational_inertia_kgm2
            / self.tire.rolling_radius_m**2
        )

    def max_motor_mechanical_power_w(
        self, battery: BatteryModel, motor_speed_rpm: float | None = None
    ) -> float:
        """Apply pack and inverter limits before the motor's shaft-power cap."""

        motor_electrical_limit_w = self.inverter.motor_electrical_power_limit_w(
            battery.discharge_power_limit_w
        )
        if motor_speed_rpm is None:
            return self.motor.mechanical_power_limit_w(motor_electrical_limit_w)
        return self.motor.mechanical_power_limit_w(
            motor_electrical_limit_w, motor_speed_rpm=motor_speed_rpm
        )

    def max_wheel_power_w(
        self, battery: BatteryModel, motor_speed_rpm: float | None = None
    ) -> float:
        return self.chain_drive.wheel_power_for_motor_power_w(
            self.max_motor_mechanical_power_w(battery, motor_speed_rpm)
        )

    def motor_speed_rad_s(self, vehicle_speed_mps: float) -> float:
        """Convert tire surface speed to motor speed through radius and ratio."""

        if vehicle_speed_mps < 0:
            raise ValueError("vehicle_speed_mps cannot be negative")
        wheel_speed_rad_s = vehicle_speed_mps / self.tire.rolling_radius_m
        return wheel_speed_rad_s * self.chain_drive.ratio

    def motor_speed_rpm(self, vehicle_speed_mps: float) -> float:
        return radians_per_second_to_revolutions_per_minute(
            self.motor_speed_rad_s(vehicle_speed_mps)
        )

    @property
    def rpm_limited_vehicle_speed_mps(self) -> float:
        motor_speed_rad_s = revolutions_per_minute_to_radians_per_second(
            self.motor.max_speed_rpm
        )
        wheel_speed_rad_s = motor_speed_rad_s / self.chain_drive.ratio
        return wheel_speed_rad_s * self.tire.rolling_radius_m

    @property
    def vehicle_speed_limit_mps(self) -> float:
        limit_mps = self.rpm_limited_vehicle_speed_mps
        if self.configured_speed_limit_mps is not None:
            limit_mps = min(limit_mps, self.configured_speed_limit_mps)
        return limit_mps

    def available_motor_torque_nm(
        self,
        vehicle_speed_mps: float,
        battery: BatteryModel,
    ) -> float:
        return self.available_motor_torque_at_speed_rpm(
            self.motor_speed_rpm(vehicle_speed_mps),
            battery,
        )

    def available_motor_torque_at_speed_rpm(
        self,
        motor_speed_rpm: float,
        battery: BatteryModel,
    ) -> float:
        """Combine the torque curve, RPM cutoff, and available electrical power."""

        if motor_speed_rpm < 0:
            raise ValueError("motor_speed_rpm cannot be negative")
        if motor_speed_rpm >= self.motor.max_speed_rpm:
            return 0.0
        torque_limit_nm = self.motor.torque_limit_nm(motor_speed_rpm)
        motor_speed_rad_s = revolutions_per_minute_to_radians_per_second(
            motor_speed_rpm
        )
        if motor_speed_rad_s == 0:
            return torque_limit_nm
        power_limited_torque_nm = (
            self.max_motor_mechanical_power_w(battery, motor_speed_rpm) / motor_speed_rad_s
        )
        return min(torque_limit_nm, power_limited_torque_nm)

    def wheel_force_from_motor_torque_n(self, motor_torque_nm: float) -> float:
        wheel_torque_nm = self.chain_drive.wheel_torque_from_motor_torque_nm(
            motor_torque_nm
        )
        return wheel_torque_nm / self.tire.rolling_radius_m

    def available_wheel_force_n(
        self,
        vehicle_speed_mps: float,
        battery: BatteryModel,
    ) -> float:
        return self.wheel_force_from_motor_torque_n(
            self.available_motor_torque_nm(vehicle_speed_mps, battery)
        )

    def motor_torque_for_wheel_force_nm(self, wheel_force_n: float) -> float:
        if wheel_force_n < 0:
            raise ValueError("wheel_force_n cannot be negative without regen")
        return self.chain_drive.motor_torque_from_wheel_torque_nm(
            wheel_force_n * self.tire.rolling_radius_m
        )

    def positive_battery_power_w(
        self,
        wheel_force_n: float,
        vehicle_speed_mps: float,
        battery: BatteryModel,
    ) -> float:
        """Trace wheel power back through chain, motor, and inverter losses.

        The speed argument is tire surface speed when longitudinal slip is
        modeled. The returned pack-terminal power is limited by the battery.
        """

        if vehicle_speed_mps < 0:
            raise ValueError("vehicle_speed_mps cannot be negative")
        wheel_power_w = max(wheel_force_n * vehicle_speed_mps, 0.0)
        motor_mechanical_power_w = self.chain_drive.motor_power_for_wheel_power_w(
            wheel_power_w
        )
        motor_speed_rpm = self.motor_speed_rpm(vehicle_speed_mps)
        motor_torque_nm = self.motor_torque_for_wheel_force_nm(wheel_force_n)
        motor_electrical_power_w = self.motor.electrical_power_for_mechanical_output_w(
            motor_mechanical_power_w,
            motor_speed_rpm,
            motor_torque_nm,
        )
        requested_battery_power_w = self.inverter.dc_power_for_motor_electrical_power_w(
            motor_electrical_power_w
        )
        return battery.limit_discharge_power_w(requested_battery_power_w)


__all__ = [
    "ChainDrive",
    "Drivetrain",
    "Inverter",
    "Motor",
]
