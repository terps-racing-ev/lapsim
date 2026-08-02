"""Vehicle parameters and physical limits."""

from dataclasses import dataclass, field
from math import cos, sin, tan

from .aero import Aero
from .battery import Battery
from .brakes import Brakes
from .chassis import Chassis
from .controls import Controls
from .drivetrain import Drivetrain
from .environment import STANDARD_AIR_DENSITY_KGPM3, STANDARD_GRAVITY_MPS2
from .state import ResettableComponent
from .tire import Tire

from .utils.units import pounds_to_kilograms

# ================= In SES/Comp/Datasheet docs ======================

DEFAULT_MASS_LB = 650.0
DEFAULT_MASS_KG = pounds_to_kilograms(DEFAULT_MASS_LB)
# ================= Need to ask for =======================

# Tires
DEFAULT_ROLLING_RESISTANCE_COEFFICIENT = 0.012

@dataclass(slots=True)
class Vehicle:
    """Parameters for the initial rear-wheel-drive TR21-based vehicle model."""

    mass_kg: float = DEFAULT_MASS_KG
    tire: Tire = field(default_factory=Tire)
    drivetrain: Drivetrain = field(default_factory=Drivetrain)
    aero: Aero = field(default_factory=Aero)
    battery: Battery = field(default_factory=Battery)
    brakes: Brakes = field(default_factory=Brakes)
    chassis: Chassis = field(default_factory=Chassis)

    # Environment used by chronological update_state simulations
    gravity_mps2: float = STANDARD_GRAVITY_MPS2
    air_density_kgpm3: float = STANDARD_AIR_DENSITY_KGPM3

    # Initial conditions restored by reset_state
    initial_speed_mps: float = 0.0
    initial_x_m: float = 0.0
    initial_y_m: float = 0.0
    initial_heading_rad: float = 0.0

    # Chronological simulation state
    time_s: float = field(init=False, default=0.0)
    distance_m: float = field(init=False, default=0.0)
    speed_mps: float = field(init=False, default=0.0)
    x_m: float = field(init=False, default=0.0)
    y_m: float = field(init=False, default=0.0)
    heading_rad: float = field(init=False, default=0.0)
    longitudinal_acceleration_mps2: float = field(init=False, default=0.0)
    lateral_acceleration_mps2: float = field(init=False, default=0.0)
    curvature_per_m: float = field(init=False, default=0.0)
    current_controls: Controls = field(init=False, default_factory=Controls)

    # Road resistance
    rolling_resistance_coefficient: float = (
        DEFAULT_ROLLING_RESISTANCE_COEFFICIENT
    )

    def __post_init__(self) -> None:
        self.validate()
        self.reset_state()

    def validate(self) -> None:
        """Validate the vehicle and all mutable component models."""

        if not isinstance(self.tire, Tire):
            raise TypeError("tire must be a Tire")
        if not isinstance(self.drivetrain, Drivetrain):
            raise TypeError("drivetrain must be a Drivetrain")
        if not isinstance(self.aero, Aero):
            raise TypeError("aero must be an Aero")
        if not isinstance(self.battery, Battery):
            raise TypeError("battery must be a Battery")
        if not isinstance(self.brakes, Brakes):
            raise TypeError("brakes must be Brakes")
        if not isinstance(self.chassis, Chassis):
            raise TypeError("chassis must be a Chassis")

        self.tire.validate()
        self.drivetrain.validate()
        self.aero.validate()
        self.battery.validate()
        self.brakes.validate()
        self.chassis.validate()

        positive_parameters = {
            "mass_kg": self.mass_kg,
            "gravity_mps2": self.gravity_mps2,
            "air_density_kgpm3": self.air_density_kgpm3,
        }
        for name, value in positive_parameters.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        nonnegative_parameters = {
            "rolling_resistance_coefficient": self.rolling_resistance_coefficient,
            "initial_speed_mps": self.initial_speed_mps,
        }
        for name, value in nonnegative_parameters.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def components(self) -> tuple[ResettableComponent, ...]:
        """Return the independently resettable component models."""

        return (
            self.tire,
            self.drivetrain,
            self.aero,
            self.battery,
            self.brakes,
            self.chassis,
        )

    @property
    def effective_longitudinal_mass_kg(self) -> float:
        """Mass accelerated longitudinally, including rotating inertia."""

        return self.mass_kg + self.drivetrain.equivalent_rotating_mass_kg

    def reset_state(self) -> None:
        """Restore every component to its configured initial state."""

        for component in self.components:
            component.reset_state()
        self.time_s = 0.0
        self.distance_m = 0.0
        self.speed_mps = self.initial_speed_mps
        self.x_m = self.initial_x_m
        self.y_m = self.initial_y_m
        self.heading_rad = self.initial_heading_rad
        self.longitudinal_acceleration_mps2 = 0.0
        self.lateral_acceleration_mps2 = 0.0
        self.curvature_per_m = 0.0
        self.current_controls = Controls()

    def update_state(self, controls: Controls, timestep_s: float) -> None:
        """Advance the point-mass vehicle state by one chronological timestep."""

        if not isinstance(controls, Controls):
            raise TypeError("controls must be Controls")
        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.validate()

        initial_speed_mps = self.speed_mps
        aero_forces = self.aero.update_state(
            initial_speed_mps,
            self.air_density_kgpm3,
            timestep_s,
        )
        rolling_force_n = self.rolling_resistance_coefficient * (
            self.mass_kg * self.gravity_mps2 + aero_forces.downforce_n
        )
        resistance_force_n = aero_forces.drag_n + rolling_force_n

        requested_curvature_per_m = tan(controls.steering_angle_rad) / (
            self.chassis.wheelbase_m
        )
        tire_normal_loads = self.chassis.tire_normal_loads_n(
            self.mass_kg,
            self.gravity_mps2,
            aero_forces,
            self.longitudinal_acceleration_mps2,
        )
        lateral_capacity_n = sum(
            self.tire.lateral_force_capacity_n(normal_load_n)
            for normal_load_n in tire_normal_loads.all_n
        )
        requested_lateral_force_n = (
            self.mass_kg
            * initial_speed_mps**2
            * abs(requested_curvature_per_m)
        )
        lateral_force_n = min(requested_lateral_force_n, lateral_capacity_n)
        if initial_speed_mps > 0:
            curvature_sign = 1.0 if requested_curvature_per_m >= 0 else -1.0
            effective_curvature_per_m = (
                curvature_sign
                * lateral_force_n
                / (self.mass_kg * initial_speed_mps**2)
            )
        else:
            effective_curvature_per_m = requested_curvature_per_m

        available_motor_torque_nm = self.drivetrain.available_motor_torque_nm(
            initial_speed_mps,
            self.battery,
        )
        applied_motor_torque_nm = min(
            controls.motor_torque_request_nm,
            available_motor_torque_nm,
        )
        requested_drive_force_n = (
            self.drivetrain.wheel_force_from_motor_torque_n(
                applied_motor_torque_nm
            )
        )

        friction_braking_force_n = 0.0
        if controls.friction_brake_force_request_n > 0:
            maximum_deceleration_mps2 = (
                self.brakes.maximum_deceleration_mps2(
                    self,
                    initial_speed_mps,
                    effective_curvature_per_m,
                    self.gravity_mps2,
                    self.air_density_kgpm3,
                )
            )
            maximum_friction_braking_force_n = max(
                0.0,
                self.effective_longitudinal_mass_kg
                * maximum_deceleration_mps2
                - resistance_force_n,
            )
            friction_braking_force_n = min(
                controls.friction_brake_force_request_n,
                maximum_friction_braking_force_n,
            )

        longitudinal_acceleration_mps2 = self.longitudinal_acceleration_mps2
        drive_force_n = 0.0
        for _ in range(10):
            tire_normal_loads = self.chassis.tire_normal_loads_n(
                self.mass_kg,
                self.gravity_mps2,
                aero_forces,
                longitudinal_acceleration_mps2,
            )
            rear_lateral_force_n = (
                (1.0 - self.chassis.static_front_weight_fraction)
                * lateral_force_n
            )
            rear_lateral_force_per_tire_n = rear_lateral_force_n / 2.0
            rear_drive_capacity_n = sum(
                self.tire.combined_longitudinal_force_capacity_n(
                    normal_load_n,
                    rear_lateral_force_per_tire_n,
                )
                for normal_load_n in tire_normal_loads.rear_n
            )
            drive_force_n = min(requested_drive_force_n, rear_drive_capacity_n)
            maximum_acceleration_to_speed_limit_mps2 = max(
                0.0,
                (
                    self.drivetrain.vehicle_speed_limit_mps
                    - initial_speed_mps
                )
                / timestep_s,
            )
            speed_limited_drive_force_n = (
                self.effective_longitudinal_mass_kg
                * maximum_acceleration_to_speed_limit_mps2
                + friction_braking_force_n
                + resistance_force_n
            )
            drive_force_n = min(drive_force_n, speed_limited_drive_force_n)
            longitudinal_acceleration_mps2 = (
                drive_force_n
                - friction_braking_force_n
                - resistance_force_n
            ) / self.effective_longitudinal_mass_kg

        if initial_speed_mps == 0 and longitudinal_acceleration_mps2 < 0:
            longitudinal_acceleration_mps2 = 0.0

        unconstrained_final_speed_mps = (
            initial_speed_mps + longitudinal_acceleration_mps2 * timestep_s
        )
        if unconstrained_final_speed_mps >= 0:
            final_speed_mps = unconstrained_final_speed_mps
            distance_step_m = (
                initial_speed_mps * timestep_s
                + 0.5 * longitudinal_acceleration_mps2 * timestep_s**2
            )
        else:
            stopping_time_s = (
                -initial_speed_mps / longitudinal_acceleration_mps2
            )
            final_speed_mps = 0.0
            distance_step_m = (
                initial_speed_mps * stopping_time_s
                + 0.5
                * longitudinal_acceleration_mps2
                * stopping_time_s**2
            )

        initial_heading_rad = self.heading_rad
        heading_change_rad = effective_curvature_per_m * distance_step_m
        final_heading_rad = initial_heading_rad + heading_change_rad
        if abs(effective_curvature_per_m) > 1e-12:
            self.x_m += (
                sin(final_heading_rad) - sin(initial_heading_rad)
            ) / effective_curvature_per_m
            self.y_m += -(
                cos(final_heading_rad) - cos(initial_heading_rad)
            ) / effective_curvature_per_m
        else:
            self.x_m += distance_step_m * cos(initial_heading_rad)
            self.y_m += distance_step_m * sin(initial_heading_rad)

        average_speed_mps = 0.5 * (initial_speed_mps + final_speed_mps)
        actual_motor_torque_nm = (
            self.drivetrain.motor_torque_for_wheel_force_nm(drive_force_n)
        )
        battery_power_w = self.drivetrain.positive_battery_power_w(
            drive_force_n,
            average_speed_mps,
            self.battery,
        )
        self.battery.update_state(battery_power_w, timestep_s)
        self.drivetrain.update_state(
            actual_motor_torque_nm,
            self.drivetrain.motor_speed_rpm(final_speed_mps),
            drive_force_n,
            timestep_s,
        )
        self.brakes.update_state(
            controls.friction_brake_force_request_n,
            friction_braking_force_n,
            timestep_s,
        )
        self.chassis.update_state(tire_normal_loads, timestep_s)
        self.tire.update_state(
            drive_force_n - friction_braking_force_n,
            lateral_force_n,
            timestep_s,
        )

        self.time_s += timestep_s
        self.distance_m += distance_step_m
        self.speed_mps = final_speed_mps
        self.heading_rad = final_heading_rad
        self.longitudinal_acceleration_mps2 = longitudinal_acceleration_mps2
        self.lateral_acceleration_mps2 = (
            average_speed_mps**2 * effective_curvature_per_m
        )
        self.curvature_per_m = effective_curvature_per_m
        self.current_controls = controls
