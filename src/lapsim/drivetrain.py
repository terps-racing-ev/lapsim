"""Electric drivetrain parameters and force/power conversions."""

from dataclasses import dataclass, field

from .battery import Battery
from .utils.units import (
    inches_to_meters,
    miles_per_hour_to_meters_per_second,
    radians_per_second_to_revolutions_per_minute,
    revolutions_per_minute_to_radians_per_second,
)


DEFAULT_MOTOR_TORQUE_CURVE_RPM = (
    0.0,
    2_000.0,
    3_000.0,
    4_000.0,
    5_000.0,
)
DEFAULT_MOTOR_TORQUE_CURVE_NM = (230.0, 230.0, 224.0, 218.0, 206.0)
DEFAULT_CONTINUOUS_MOTOR_TORQUE_CURVE_RPM = (
    0.0,
    5_500.0,
    6_000.0,
    6_500.0,
    7_000.0,
)
DEFAULT_CONTINUOUS_MOTOR_TORQUE_CURVE_NM = (82.0, 82.0, 82.0, 75.0, 70.0)
DEFAULT_PEAK_MOTOR_POWER_W = 80_000.0
DEFAULT_CONTINUOUS_MOTOR_POWER_W = 50_000.0
DEFAULT_MAX_MOTOR_SPEED_RPM = 7_000.0
DEFAULT_SPEED_LIMIT_MPH = 100.0
DEFAULT_SPEED_LIMIT_MPS = miles_per_hour_to_meters_per_second(
    DEFAULT_SPEED_LIMIT_MPH
)
TIRE_DIAMETER_IN = 16.0
DEFAULT_ROLLING_RADIUS_M = inches_to_meters(TIRE_DIAMETER_IN) / 2.0
DEFAULT_FINAL_DRIVE_RATIO = 3.7
DEFAULT_DIFFERENTIAL_EFFICIENCY = 0.95
DEFAULT_INVERTER_EFFICIENCY = 0.97
DEFAULT_MOTOR_EFFICIENCY = 0.95
# Placeholder component inertias from Emrax228_Motor_Parameters.xlsx. Input-side
# inertias rotate at motor speed; output-side inertia rotates at wheel speed.
DEFAULT_MOTOR_ROTOR_INERTIA_KGM2 = 0.01215
DEFAULT_FINAL_DRIVE_INPUT_INERTIA_KGM2 = 0.00005
DEFAULT_FINAL_DRIVE_OUTPUT_INERTIA_KGM2 = 0.003
# Combined rear wheel/tire rotational inertia. This is an initial engineering
# estimate for two 16-inch assemblies and should be replaced by a CAD or
# pendulum measurement when available.
DEFAULT_DRIVEN_WHEEL_INERTIA_KGM2 = 0.75


@dataclass(slots=True)
class Drivetrain:
    """Fixed-ratio rear-wheel-drive electric powertrain."""

    rolling_radius_m: float = DEFAULT_ROLLING_RADIUS_M
    final_drive_ratio: float = DEFAULT_FINAL_DRIVE_RATIO
    inverter_efficiency: float = DEFAULT_INVERTER_EFFICIENCY
    motor_efficiency: float = DEFAULT_MOTOR_EFFICIENCY
    differential_efficiency: float = DEFAULT_DIFFERENTIAL_EFFICIENCY
    motor_rotor_inertia_kgm2: float = DEFAULT_MOTOR_ROTOR_INERTIA_KGM2
    final_drive_input_inertia_kgm2: float = (
        DEFAULT_FINAL_DRIVE_INPUT_INERTIA_KGM2
    )
    final_drive_output_inertia_kgm2: float = (
        DEFAULT_FINAL_DRIVE_OUTPUT_INERTIA_KGM2
    )
    driven_wheel_inertia_kgm2: float = DEFAULT_DRIVEN_WHEEL_INERTIA_KGM2
    motor_torque_curve_rpm: list[float] = field(
        default_factory=lambda: list(DEFAULT_MOTOR_TORQUE_CURVE_RPM)
    )
    motor_torque_curve_nm: list[float] = field(
        default_factory=lambda: list(DEFAULT_MOTOR_TORQUE_CURVE_NM)
    )
    continuous_motor_torque_curve_rpm: list[float] = field(
        default_factory=lambda: list(
            DEFAULT_CONTINUOUS_MOTOR_TORQUE_CURVE_RPM
        )
    )
    continuous_motor_torque_curve_nm: list[float] = field(
        default_factory=lambda: list(
            DEFAULT_CONTINUOUS_MOTOR_TORQUE_CURVE_NM
        )
    )
    peak_motor_power_w: float = DEFAULT_PEAK_MOTOR_POWER_W
    continuous_motor_power_w: float = DEFAULT_CONTINUOUS_MOTOR_POWER_W
    max_motor_speed_rpm: float = DEFAULT_MAX_MOTOR_SPEED_RPM
    configured_speed_limit_mps: float | None = DEFAULT_SPEED_LIMIT_MPS
    current_motor_torque_nm: float = field(init=False, default=0.0)
    current_motor_speed_rpm: float = field(init=False, default=0.0)
    current_wheel_force_n: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable drivetrain parameters."""

        positive_parameters = {
            "rolling_radius_m": self.rolling_radius_m,
            "final_drive_ratio": self.final_drive_ratio,
            "max_motor_speed_rpm": self.max_motor_speed_rpm,
            "peak_motor_power_w": self.peak_motor_power_w,
            "continuous_motor_power_w": self.continuous_motor_power_w,
        }
        for name, value in positive_parameters.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        efficiencies = {
            "inverter_efficiency": self.inverter_efficiency,
            "motor_efficiency": self.motor_efficiency,
            "differential_efficiency": self.differential_efficiency,
        }
        for name, value in efficiencies.items():
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be greater than 0 and at most 1")

        inertias = {
            "motor_rotor_inertia_kgm2": self.motor_rotor_inertia_kgm2,
            "final_drive_input_inertia_kgm2": (
                self.final_drive_input_inertia_kgm2
            ),
            "final_drive_output_inertia_kgm2": (
                self.final_drive_output_inertia_kgm2
            ),
            "driven_wheel_inertia_kgm2": self.driven_wheel_inertia_kgm2,
        }
        for name, value in inertias.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")

        if (
            self.configured_speed_limit_mps is not None
            and self.configured_speed_limit_mps <= 0
        ):
            raise ValueError("configured_speed_limit_mps must be positive or None")

        curve_length = len(self.motor_torque_curve_rpm)
        if curve_length < 2:
            raise ValueError("motor torque curve requires at least two points")
        if len(self.motor_torque_curve_nm) != curve_length:
            raise ValueError(
                "motor_torque_curve_nm must match motor_torque_curve_rpm"
            )
        if any(speed_rpm < 0 for speed_rpm in self.motor_torque_curve_rpm):
            raise ValueError("motor torque curve RPM values cannot be negative")
        if any(
            upper_rpm <= lower_rpm
            for lower_rpm, upper_rpm in zip(
                self.motor_torque_curve_rpm,
                self.motor_torque_curve_rpm[1:],
            )
        ):
            raise ValueError("motor torque curve RPM values must increase")
        if any(torque_nm <= 0 for torque_nm in self.motor_torque_curve_nm):
            raise ValueError("motor torque curve values must be positive")

        continuous_curve_length = len(
            self.continuous_motor_torque_curve_rpm
        )
        if continuous_curve_length < 2:
            raise ValueError(
                "continuous motor torque curve requires at least two points"
            )
        if (
            len(self.continuous_motor_torque_curve_nm)
            != continuous_curve_length
        ):
            raise ValueError(
                "continuous_motor_torque_curve_nm must match "
                "continuous_motor_torque_curve_rpm"
            )
        if any(
            speed_rpm < 0
            for speed_rpm in self.continuous_motor_torque_curve_rpm
        ):
            raise ValueError(
                "continuous motor torque curve RPM values cannot be negative"
            )
        if any(
            upper_rpm <= lower_rpm
            for lower_rpm, upper_rpm in zip(
                self.continuous_motor_torque_curve_rpm,
                self.continuous_motor_torque_curve_rpm[1:],
            )
        ):
            raise ValueError(
                "continuous motor torque curve RPM values must increase"
            )
        if any(
            torque_nm <= 0
            for torque_nm in self.continuous_motor_torque_curve_nm
        ):
            raise ValueError(
                "continuous motor torque curve values must be positive"
            )
        if self.continuous_motor_power_w > self.peak_motor_power_w:
            raise ValueError(
                "continuous_motor_power_w cannot exceed peak_motor_power_w"
            )

    def reset_state(self) -> None:
        """Clear the drivetrain's current operating point."""

        self.current_motor_torque_nm = 0.0
        self.current_motor_speed_rpm = 0.0
        self.current_wheel_force_n = 0.0

    def update_state(
        self,
        motor_torque_nm: float,
        motor_speed_rpm: float,
        wheel_force_n: float,
        timestep_s: float,
    ) -> None:
        """Retain the drivetrain operating point for one timestep."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.current_motor_torque_nm = motor_torque_nm
        self.current_motor_speed_rpm = motor_speed_rpm
        self.current_wheel_force_n = wheel_force_n

    @property
    def max_motor_torque_nm(self) -> float:
        """Highest torque value in the configured motor curve."""

        return max(self.motor_torque_curve_nm)

    def motor_torque_limit_nm(self, motor_speed_rpm: float) -> float:
        """Interpolate the configured peak torque ceiling at requested RPM."""

        return self._interpolate_torque_curve_nm(
            motor_speed_rpm,
            self.motor_torque_curve_rpm,
            self.motor_torque_curve_nm,
        )

    def continuous_motor_torque_limit_nm(
        self,
        motor_speed_rpm: float,
    ) -> float:
        """Interpolate the EMRAX continuous torque ceiling."""

        return self._interpolate_torque_curve_nm(
            motor_speed_rpm,
            self.continuous_motor_torque_curve_rpm,
            self.continuous_motor_torque_curve_nm,
        )

    @staticmethod
    def _interpolate_torque_curve_nm(
        motor_speed_rpm: float,
        curve_speed_rpm: list[float],
        curve_torque_nm: list[float],
    ) -> float:
        """Linearly interpolate and endpoint-clamp one torque curve."""

        if motor_speed_rpm < 0:
            raise ValueError("motor_speed_rpm cannot be negative")
        clamped_speed_rpm = min(
            max(motor_speed_rpm, curve_speed_rpm[0]),
            curve_speed_rpm[-1],
        )
        for index, upper_speed_rpm in enumerate(
            curve_speed_rpm[1:],
            start=1,
        ):
            if clamped_speed_rpm <= upper_speed_rpm:
                lower_speed_rpm = curve_speed_rpm[index - 1]
                interpolation_fraction = (
                    (clamped_speed_rpm - lower_speed_rpm)
                    / (upper_speed_rpm - lower_speed_rpm)
                )
                lower_torque_nm = curve_torque_nm[index - 1]
                return lower_torque_nm + interpolation_fraction * (
                    curve_torque_nm[index] - lower_torque_nm
                )
        return curve_torque_nm[-1]


    @property
    def efficiency(self) -> float:
        """Overall battery-to-wheel efficiency during propulsion."""

        return (
            self.inverter_efficiency
            * self.motor_efficiency
            * self.differential_efficiency
        )

    @property
    def wheel_referenced_rotational_inertia_kgm2(self) -> float:
        """Return all modeled inertias reflected to wheel speed."""

        input_side_inertia_kgm2 = (
            self.motor_rotor_inertia_kgm2
            + self.final_drive_input_inertia_kgm2
        )
        return (
            input_side_inertia_kgm2 * self.final_drive_ratio**2
            + self.final_drive_output_inertia_kgm2
            + self.driven_wheel_inertia_kgm2
        )

    @property
    def equivalent_rotating_mass_kg(self) -> float:
        """Return modeled rotating inertia as equivalent translating mass."""

        return (
            self.wheel_referenced_rotational_inertia_kgm2
            / self.rolling_radius_m**2
        )

    def max_motor_mechanical_power_w(self, battery: Battery) -> float:
        """Maximum motor-shaft power available from the battery."""

        battery_limited_motor_power_w = (
            battery.max_discharge_power_w
            * self.inverter_efficiency
            * self.motor_efficiency
        )
        return min(
            battery_limited_motor_power_w,
            self.peak_motor_power_w,
        )

    def max_wheel_power_w(self, battery: Battery) -> float:
        """Maximum driven-wheel power available from the battery."""

        return (
            self.max_motor_mechanical_power_w(battery)
            * self.differential_efficiency
        )

    def motor_speed_rad_s(self, vehicle_speed_mps: float) -> float:
        """Convert vehicle speed to motor angular speed."""

        if vehicle_speed_mps < 0:
            raise ValueError("vehicle_speed_mps cannot be negative")
        wheel_speed_rad_s = vehicle_speed_mps / self.rolling_radius_m
        return wheel_speed_rad_s * self.final_drive_ratio

    def motor_speed_rpm(self, vehicle_speed_mps: float) -> float:
        """Convert vehicle speed to motor speed in RPM."""

        return radians_per_second_to_revolutions_per_minute(
            self.motor_speed_rad_s(vehicle_speed_mps)
        )

    @property
    def rpm_limited_vehicle_speed_mps(self) -> float:
        """Vehicle speed corresponding to the motor RPM limit."""

        motor_speed_rad_s = revolutions_per_minute_to_radians_per_second(
            self.max_motor_speed_rpm
        )
        wheel_speed_rad_s = motor_speed_rad_s / self.final_drive_ratio
        return wheel_speed_rad_s * self.rolling_radius_m

    @property
    def vehicle_speed_limit_mps(self) -> float:
        """Lowest configured or motor-RPM vehicle speed limit."""

        limit_mps = self.rpm_limited_vehicle_speed_mps
        if self.configured_speed_limit_mps is not None:
            limit_mps = min(limit_mps, self.configured_speed_limit_mps)
        return limit_mps

    def available_motor_torque_nm(
        self,
        vehicle_speed_mps: float,
        battery: Battery,
    ) -> float:
        """Return available torque after electrical power and RPM limits."""

        motor_speed_rpm = self.motor_speed_rpm(vehicle_speed_mps)
        return self.available_motor_torque_at_speed_rpm(
            motor_speed_rpm,
            battery,
        )

    def available_motor_torque_at_speed_rpm(
        self,
        motor_speed_rpm: float,
        battery: Battery,
    ) -> float:
        """Return available torque at an independently supplied motor speed."""

        if motor_speed_rpm < 0:
            raise ValueError("motor_speed_rpm cannot be negative")
        if motor_speed_rpm >= self.max_motor_speed_rpm:
            return 0.0

        torque_curve_limit_nm = self.motor_torque_limit_nm(motor_speed_rpm)

        motor_speed_rad_s = revolutions_per_minute_to_radians_per_second(
            motor_speed_rpm
        )
        if motor_speed_rad_s == 0:
            return torque_curve_limit_nm

        power_limited_torque_nm = (
            self.max_motor_mechanical_power_w(battery) / motor_speed_rad_s
        )
        return min(torque_curve_limit_nm, power_limited_torque_nm)

    def wheel_force_from_motor_torque_n(self, motor_torque_nm: float) -> float:
        """Convert motor-shaft torque to driven-wheel force."""

        wheel_torque_nm = (
            motor_torque_nm
            * self.final_drive_ratio
            * self.differential_efficiency
        )
        return wheel_torque_nm / self.rolling_radius_m

    def available_wheel_force_n(
        self,
        vehicle_speed_mps: float,
        battery: Battery,
    ) -> float:
        """Return motor-limited driven-wheel force at vehicle speed."""

        return self.wheel_force_from_motor_torque_n(
            self.available_motor_torque_nm(vehicle_speed_mps, battery)
        )

    def motor_torque_for_wheel_force_nm(self, wheel_force_n: float) -> float:
        """Convert requested driven-wheel force back to motor torque."""

        if wheel_force_n < 0:
            raise ValueError("wheel_force_n cannot be negative without regen")
        return (
            wheel_force_n
            * self.rolling_radius_m
            / (self.final_drive_ratio * self.differential_efficiency)
        )

    def positive_battery_power_w(
        self,
        wheel_force_n: float,
        vehicle_speed_mps: float,
        battery: Battery,
    ) -> float:
        """Convert positive wheel power to battery electrical power."""

        if vehicle_speed_mps < 0:
            raise ValueError("vehicle_speed_mps cannot be negative")
        positive_wheel_power_w = max(wheel_force_n * vehicle_speed_mps, 0.0)
        requested_battery_power_w = positive_wheel_power_w / self.efficiency
        return battery.limit_discharge_power_w(requested_battery_power_w)
