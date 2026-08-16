"""Vehicle parameters and physical limits."""

from dataclasses import dataclass, field
from math import cos, isfinite, sin, sqrt, tan

from scipy.optimize import brentq

from lapsim.controls import Controls
from utils.units import pounds_to_kilograms

from .aero import Aero
from .electrical import RCTheveninBattery
from .environment import STANDARD_AIR_DENSITY_KGPM3, STANDARD_GRAVITY_MPS2
from .interfaces import (
    AeroModel,
    BatteryModel,
    BrakeModel,
    ChassisModel,
    ComponentModel,
    DrivetrainModel,
    SuspensionModel,
    TireModel,
)
from .mech import Brakes, Chassis, Suspension, Tire, WheelSlip
from .powertrain import Drivetrain

DEFAULT_MASS_LB = 675.0
DEFAULT_MASS_KG = pounds_to_kilograms(DEFAULT_MASS_LB)
DEFAULT_ROLLING_RESISTANCE_COEFFICIENT = 0.012


@dataclass(slots=True)
class Vehicle:
    """Parameters for the initial rear-wheel-drive TR21-based vehicle model."""

    mass_kg: float = DEFAULT_MASS_KG
    tire: TireModel = field(default_factory=Tire)
    drivetrain: DrivetrainModel = field(default_factory=Drivetrain)
    aero: AeroModel = field(default_factory=Aero)
    battery: BatteryModel = field(default_factory=RCTheveninBattery)
    brakes: BrakeModel = field(default_factory=Brakes)
    chassis: ChassisModel = field(default_factory=Chassis)
    suspension: SuspensionModel = field(default_factory=Suspension)
    wheel_slip: WheelSlip = field(default_factory=WheelSlip)

    # Environment used by the distance-domain update_state simulation.
    gravity_mps2: float = STANDARD_GRAVITY_MPS2
    air_density_kgpm3: float = STANDARD_AIR_DENSITY_KGPM3

    # Initial conditions restored by reset_state
    initial_speed_mps: float = 0.0
    initial_x_m: float = 0.0
    initial_y_m: float = 0.0
    initial_heading_rad: float = 0.0

    # Vehicle state.  Distance is the integration coordinate; elapsed time is
    # derived internally for every spatial cell.
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

    # Retained force-balance diagnostics exposed through telemetry.
    requested_curvature_per_m: float = field(init=False, default=0.0)
    requested_lateral_force_n: float = field(init=False, default=0.0)
    lateral_force_capacity_n: float = field(init=False, default=0.0)
    current_lateral_force_n: float = field(init=False, default=0.0)
    available_motor_torque_nm: float = field(init=False, default=0.0)
    envelope_limited_motor_torque_nm: float = field(init=False, default=0.0)
    requested_drive_force_n: float = field(init=False, default=0.0)
    rear_drive_capacity_n: float = field(init=False, default=0.0)
    speed_limited_drive_force_n: float = field(init=False, default=0.0)
    current_drive_force_n: float = field(init=False, default=0.0)
    current_friction_braking_force_n: float = field(init=False, default=0.0)
    maximum_friction_braking_force_n: float = field(init=False, default=0.0)
    current_rolling_resistance_force_n: float = field(init=False, default=0.0)
    current_cornering_drag_force_n: float = field(init=False, default=0.0)
    current_resistance_force_n: float = field(init=False, default=0.0)

    # Road resistance
    rolling_resistance_coefficient: float = DEFAULT_ROLLING_RESISTANCE_COEFFICIENT
    cornering_drag_coefficient: float = 0.0

    def __post_init__(self) -> None:
        self.validate()
        self.reset_state()

    def validate(self) -> None:
        """Validate the vehicle and all mutable component models."""

        # The vehicle owns one tire model. Drivetrain kinematic conversions
        # reference that same object rather than duplicating wheel geometry.
        self.drivetrain.tire = self.tire
        for name, component, protocol in (
            ("tire", self.tire, TireModel),
            ("drivetrain", self.drivetrain, DrivetrainModel),
            ("aero", self.aero, AeroModel),
            ("battery", self.battery, BatteryModel),
            ("brakes", self.brakes, BrakeModel),
            ("chassis", self.chassis, ChassisModel),
            ("suspension", self.suspension, SuspensionModel),
            ("wheel_slip", self.wheel_slip, ComponentModel),
        ):
            if not isinstance(component, protocol):
                raise TypeError(f"{name} does not satisfy {protocol.__name__}")
            component.validate()

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
            "cornering_drag_coefficient": self.cornering_drag_coefficient,
            "initial_speed_mps": self.initial_speed_mps,
        }
        for name, value in nonnegative_parameters.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def components(self) -> tuple[ComponentModel, ...]:
        """Return the independently resettable component models."""

        return (
            self.tire,
            self.drivetrain,
            self.aero,
            self.battery,
            self.brakes,
            self.chassis,
            self.suspension,
            self.wheel_slip,
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
        self.requested_curvature_per_m = 0.0
        self.requested_lateral_force_n = 0.0
        self.lateral_force_capacity_n = 0.0
        self.current_lateral_force_n = 0.0
        self.available_motor_torque_nm = 0.0
        self.envelope_limited_motor_torque_nm = 0.0
        self.requested_drive_force_n = 0.0
        self.rear_drive_capacity_n = 0.0
        self.speed_limited_drive_force_n = 0.0
        self.current_drive_force_n = 0.0
        self.current_friction_braking_force_n = 0.0
        self.maximum_friction_braking_force_n = 0.0
        self.current_rolling_resistance_force_n = 0.0
        self.current_cornering_drag_force_n = 0.0
        self.current_resistance_force_n = 0.0

    def update_state(self, controls: Controls, distance_step_m: float) -> None:
        """Advance exactly one positive spatial cell.

        ``distance_step_m`` is the independent integration coordinate.  The
        elapsed timestep is solved from the constant-acceleration relation for
        the cell and is then supplied to time-domain component models.

        A spatial cell that the present controls cannot traverse is rejected
        rather than silently committing less distance than requested.
        """

        if not isinstance(controls, Controls):
            raise TypeError("controls must be Controls")
        if not isfinite(distance_step_m) or distance_step_m <= 0.0:
            raise ValueError("distance_step_m must be finite and positive")

        initial_speed_mps = self.speed_mps
        if initial_speed_mps < 0.0:
            raise RuntimeError("vehicle speed cannot be negative")

        aero_forces = self.aero.forces_n(
            initial_speed_mps,
            self.air_density_kgpm3,
        )
        rolling_force_n = self.rolling_resistance_coefficient * (
            self.mass_kg * self.gravity_mps2 + aero_forces.downforce_n
        )
        requested_curvature_per_m = tan(controls.steering_angle_rad) / (
            self.chassis.wheelbase_m
        )
        initial_tire_normal_loads = self.suspension.tire_normal_loads_n(
            self.mass_kg,
            self.gravity_mps2,
            aero_forces,
            self.chassis,
            longitudinal_acceleration_mps2=self.longitudinal_acceleration_mps2,
            lateral_acceleration_mps2=self.lateral_acceleration_mps2,
        )
        lateral_capacity_n = sum(
            self.tire.lateral_force_capacity_n(normal_load_n)
            for normal_load_n in initial_tire_normal_loads.all_n
        )
        requested_lateral_force_n = (
            self.mass_kg * initial_speed_mps**2 * abs(requested_curvature_per_m)
        )
        lateral_force_n = min(requested_lateral_force_n, lateral_capacity_n)
        total_normal_force_n = (
            self.mass_kg * self.gravity_mps2 + aero_forces.downforce_n
        )
        cornering_drag_force_n = (
            self.cornering_drag_coefficient * lateral_force_n**2 / total_normal_force_n
        )
        resistance_force_n = (
            aero_forces.drag_n + rolling_force_n + cornering_drag_force_n
        )
        if initial_speed_mps > 0.0:
            curvature_sign = 1.0 if requested_curvature_per_m >= 0.0 else -1.0
            effective_curvature_per_m = (
                curvature_sign * lateral_force_n / (self.mass_kg * initial_speed_mps**2)
            )
        else:
            effective_curvature_per_m = requested_curvature_per_m

        available_motor_torque_nm = self.drivetrain.available_motor_torque_nm(
            initial_speed_mps, self.battery
        )
        applied_motor_torque_nm = min(
            controls.motor_torque_request_nm, available_motor_torque_nm
        )
        requested_drive_force_n = self.drivetrain.wheel_force_from_motor_torque_n(
            applied_motor_torque_nm
        )
        (
            front_friction_brake_force_request_n,
            rear_friction_brake_force_request_n,
        ) = self.brakes.axle_force_requests_from_pressures_n(
            controls.front_brake_pressure_psi,
            controls.rear_brake_pressure_psi,
            self.tire.rolling_radius_m,
        )
        front_brake_force_request_n = front_friction_brake_force_request_n
        rear_brake_force_request_n = (
            rear_friction_brake_force_request_n
            + controls.rear_regenerative_brake_force_request_n
        )
        total_brake_force_request_n = (
            front_brake_force_request_n + rear_brake_force_request_n
        )

        def operating_point(timestep_s: float) -> tuple[float, ...]:
            def forces_at_acceleration(
                assumed_acceleration_mps2: float,
            ) -> tuple[float, ...]:
                tire_normal_loads = self.suspension.tire_normal_loads_n(
                    self.mass_kg,
                    self.gravity_mps2,
                    aero_forces,
                    self.chassis,
                    longitudinal_acceleration_mps2=assumed_acceleration_mps2,
                    lateral_acceleration_mps2=lateral_force_n / self.mass_kg,
                )
                rear_lateral_force_per_tire_n = (
                    (1.0 - self.chassis.static_front_weight_fraction)
                    * lateral_force_n
                    / 2.0
                )
                rear_drive_capacity_n = sum(
                    self.tire.combined_longitudinal_force_capacity_n(
                        normal_load_n, rear_lateral_force_per_tire_n
                    )
                    for normal_load_n in tire_normal_loads.rear_n
                )
                (
                    front_friction_force_n,
                    rear_friction_force_n,
                    front_braking_capacity_n,
                    rear_braking_capacity_n,
                ) = self.brakes.axle_friction_forces_n(
                    self,
                    0.0,
                    tire_normal_loads,
                    lateral_force_n,
                    additional_front_force_request_n=(
                        front_friction_brake_force_request_n
                    ),
                    additional_rear_force_request_n=(
                        rear_friction_brake_force_request_n
                        + controls.rear_regenerative_brake_force_request_n
                    ),
                )
                (
                    front_friction_force_n,
                    rear_friction_force_n,
                    front_braking_slip_ratio,
                    rear_braking_slip_ratio,
                ) = self.brakes.slip_limited_axle_forces_n(
                    front_friction_force_n,
                    rear_friction_force_n,
                    front_braking_capacity_n,
                    rear_braking_capacity_n,
                    initial_speed_mps,
                    timestep_s,
                )
                friction_braking_force_n = (
                    front_friction_force_n + rear_friction_force_n
                )
                maximum_friction_braking_force_n = (
                    front_braking_capacity_n + rear_braking_capacity_n
                )
                rear_drive_capacity_n = max(
                    0.0, rear_drive_capacity_n - rear_friction_force_n
                )
                drive_force_n = min(requested_drive_force_n, rear_drive_capacity_n)
                maximum_acceleration_to_speed_limit_mps2 = max(
                    0.0,
                    (self.drivetrain.vehicle_speed_limit_mps**2 - initial_speed_mps**2)
                    / (2.0 * distance_step_m),
                )
                speed_limited_drive_force_n = (
                    self.effective_longitudinal_mass_kg
                    * maximum_acceleration_to_speed_limit_mps2
                    + friction_braking_force_n
                    + resistance_force_n
                )
                drive_force_n = min(drive_force_n, speed_limited_drive_force_n)
                force_balance_acceleration_mps2 = (
                    drive_force_n - friction_braking_force_n - resistance_force_n
                ) / self.effective_longitudinal_mass_kg
                return (
                    force_balance_acceleration_mps2,
                    drive_force_n,
                    rear_drive_capacity_n,
                    speed_limited_drive_force_n,
                    friction_braking_force_n,
                    maximum_friction_braking_force_n,
                    front_friction_force_n,
                    rear_friction_force_n,
                    front_braking_slip_ratio,
                    rear_braking_slip_ratio,
                    tire_normal_loads,
                )

            lower_acceleration_mps2 = -100.0
            upper_acceleration_mps2 = 100.0

            def force_balance_residual(assumed_acceleration_mps2: float) -> float:
                return (
                    assumed_acceleration_mps2
                    - forces_at_acceleration(assumed_acceleration_mps2)[0]
                )

            lower_residual = force_balance_residual(lower_acceleration_mps2)
            upper_residual = force_balance_residual(upper_acceleration_mps2)
            if lower_residual > 0.0 or upper_residual < 0.0:
                raise RuntimeError("vehicle force balance root is not bracketed")
            acceleration_mps2 = brentq(
                force_balance_residual,
                lower_acceleration_mps2,
                upper_acceleration_mps2,
                xtol=1e-8,
            )
            return forces_at_acceleration(acceleration_mps2)

        timestep_s = (
            distance_step_m / initial_speed_mps
            if initial_speed_mps > 1e-6
            else sqrt(2.0 * distance_step_m)
        )
        converged = False
        for _ in range(50):
            operating_values = operating_point(timestep_s)
            longitudinal_acceleration_mps2 = operating_values[0]
            if initial_speed_mps == 0.0 and longitudinal_acceleration_mps2 <= 0.0:
                raise ValueError(
                    "cannot traverse distance_step_m from rest without positive "
                    "longitudinal acceleration"
                )
            final_speed_squared_mps2 = (
                initial_speed_mps**2
                + 2.0 * longitudinal_acceleration_mps2 * distance_step_m
            )
            if final_speed_squared_mps2 < -1e-9:
                raise ValueError(
                    "cannot traverse distance_step_m: vehicle stops before the "
                    "end of the cell"
                )
            final_speed_mps = sqrt(max(0.0, final_speed_squared_mps2))
            next_timestep_s = (
                2.0 * distance_step_m / (initial_speed_mps + final_speed_mps)
            )
            if abs(next_timestep_s - timestep_s) <= 1e-10 * max(1.0, next_timestep_s):
                timestep_s = next_timestep_s
                converged = True
                break
            timestep_s = next_timestep_s
        if not converged:
            raise RuntimeError(
                "distance-domain vehicle update did not converge on an internal "
                "timestep"
            )

        # Evaluate once more at the converged dt, then commit all component
        # states with that same physical elapsed time.
        (
            longitudinal_acceleration_mps2,
            drive_force_n,
            rear_drive_capacity_n,
            speed_limited_drive_force_n,
            friction_braking_force_n,
            maximum_friction_braking_force_n,
            front_friction_braking_force_n,
            rear_friction_braking_force_n,
            front_braking_slip_ratio,
            rear_braking_slip_ratio,
            tire_normal_loads,
        ) = operating_point(timestep_s)
        aero_forces = self.aero.update_state(
            initial_speed_mps, self.air_density_kgpm3, timestep_s
        )

        initial_heading_rad = self.heading_rad
        heading_change_rad = effective_curvature_per_m * distance_step_m
        final_heading_rad = initial_heading_rad + heading_change_rad
        if abs(effective_curvature_per_m) > 1e-12:
            self.x_m += (
                sin(final_heading_rad) - sin(initial_heading_rad)
            ) / effective_curvature_per_m
            self.y_m += (
                -(cos(final_heading_rad) - cos(initial_heading_rad))
                / effective_curvature_per_m
            )
        else:
            self.x_m += distance_step_m * cos(initial_heading_rad)
            self.y_m += distance_step_m * sin(initial_heading_rad)

        average_speed_mps = 0.5 * (initial_speed_mps + final_speed_mps)
        self.wheel_slip.update_state(
            average_speed_mps,
            drive_force_n,
            rear_drive_capacity_n,
            timestep_s,
        )
        wheel_surface_speed_mps = self.wheel_slip.current_wheel_surface_speed_mps
        actual_motor_torque_nm = self.drivetrain.motor_torque_for_wheel_force_nm(
            drive_force_n
        )
        motor_speed_rpm = self.drivetrain.motor_speed_rpm(wheel_surface_speed_mps)
        battery_power_w = self.drivetrain.positive_battery_power_w(
            drive_force_n,
            wheel_surface_speed_mps,
            self.battery,
        )
        self.battery.update_state(battery_power_w, timestep_s)
        self.drivetrain.update_state(
            actual_motor_torque_nm,
            motor_speed_rpm,
            drive_force_n,
            timestep_s,
            battery_power_w,
        )
        self.brakes.update_state(
            total_brake_force_request_n,
            friction_braking_force_n,
            timestep_s,
            front_pressure_psi=controls.front_brake_pressure_psi,
            rear_pressure_psi=controls.rear_brake_pressure_psi,
            front_friction_force_n=front_friction_braking_force_n,
            rear_friction_force_n=rear_friction_braking_force_n,
            front_braking_slip_ratio=front_braking_slip_ratio,
            rear_braking_slip_ratio=rear_braking_slip_ratio,
            front_force_request_n=front_brake_force_request_n,
            rear_force_request_n=rear_brake_force_request_n,
        )
        self.suspension.update_state(tire_normal_loads, timestep_s)
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
        self.requested_curvature_per_m = requested_curvature_per_m
        self.requested_lateral_force_n = requested_lateral_force_n
        self.lateral_force_capacity_n = lateral_capacity_n
        self.current_lateral_force_n = lateral_force_n
        self.available_motor_torque_nm = available_motor_torque_nm
        self.envelope_limited_motor_torque_nm = applied_motor_torque_nm
        self.requested_drive_force_n = requested_drive_force_n
        self.rear_drive_capacity_n = rear_drive_capacity_n
        self.speed_limited_drive_force_n = speed_limited_drive_force_n
        self.current_drive_force_n = drive_force_n
        self.current_friction_braking_force_n = friction_braking_force_n
        self.maximum_friction_braking_force_n = maximum_friction_braking_force_n
        self.current_rolling_resistance_force_n = rolling_force_n
        self.current_cornering_drag_force_n = cornering_drag_force_n
        self.current_resistance_force_n = resistance_force_n

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        """Write vehicle state, controls, force balance, and active limits."""

        lateral_utilization = 0.0
        if self.lateral_force_capacity_n > 0.0:
            lateral_utilization = min(
                self.current_lateral_force_n / self.lateral_force_capacity_n,
                1.0,
            )
        telemetry.update(
            {
                "vehicle.time_s": self.time_s,
                "vehicle.distance_m": self.distance_m,
                "vehicle.x_m": self.x_m,
                "vehicle.y_m": self.y_m,
                "vehicle.heading_rad": self.heading_rad,
                "vehicle.speed_mps": self.speed_mps,
                "vehicle.longitudinal_acceleration_mps2": (
                    self.longitudinal_acceleration_mps2
                ),
                "vehicle.lateral_acceleration_mps2": (self.lateral_acceleration_mps2),
                "vehicle.requested_curvature_per_m": (self.requested_curvature_per_m),
                "vehicle.achieved_curvature_per_m": self.curvature_per_m,
                "vehicle.requested_lateral_force_n": (self.requested_lateral_force_n),
                "vehicle.lateral_force_capacity_n": (self.lateral_force_capacity_n),
                "vehicle.lateral_force_n": self.current_lateral_force_n,
                "vehicle.lateral_utilization": lateral_utilization,
                "vehicle.available_motor_torque_nm": (self.available_motor_torque_nm),
                "vehicle.envelope_limited_motor_torque_nm": (
                    self.envelope_limited_motor_torque_nm
                ),
                "vehicle.requested_drive_force_n": self.requested_drive_force_n,
                "vehicle.rear_drive_capacity_n": self.rear_drive_capacity_n,
                "vehicle.speed_limited_drive_force_n": (
                    self.speed_limited_drive_force_n
                ),
                "vehicle.drive_force_n": self.current_drive_force_n,
                "vehicle.friction_braking_force_n": (
                    self.current_friction_braking_force_n
                ),
                "vehicle.maximum_friction_braking_force_n": (
                    self.maximum_friction_braking_force_n
                ),
                "vehicle.rolling_resistance_force_n": (
                    self.current_rolling_resistance_force_n
                ),
                "vehicle.cornering_drag_force_n": (self.current_cornering_drag_force_n),
                "vehicle.total_resistance_force_n": self.current_resistance_force_n,
                "controls.motor_torque_request_nm": (
                    self.current_controls.motor_torque_request_nm
                ),
                "controls.front_brake_pressure_psi": (
                    self.current_controls.front_brake_pressure_psi
                ),
                "controls.rear_brake_pressure_psi": (
                    self.current_controls.rear_brake_pressure_psi
                ),
                "controls.rear_regenerative_brake_force_request_n": (
                    self.current_controls.rear_regenerative_brake_force_request_n
                ),
                "controls.steering_angle_rad": (
                    self.current_controls.steering_angle_rad
                ),
                "limits.motor_envelope_active": float(
                    self.envelope_limited_motor_torque_nm
                    < self.current_controls.motor_torque_request_nm - 1e-9
                ),
                "limits.traction_active": float(
                    self.current_drive_force_n < self.requested_drive_force_n - 1e-9
                ),
                "limits.lateral_saturated": float(
                    self.current_lateral_force_n >= self.lateral_force_capacity_n - 1e-9
                    and self.requested_lateral_force_n > 0.0
                ),
                "limits.brake_grip_active": float(
                    self.current_friction_braking_force_n
                    < self.brakes.current_force_request_n - 1e-9
                ),
                "limits.speed_active": float(
                    self.current_drive_force_n
                    >= self.speed_limited_drive_force_n - 1e-9
                    and self.requested_drive_force_n > self.speed_limited_drive_force_n
                ),
            }
        )
        for component in self.components:
            component.update_telemetry(telemetry)

    def telemetry_snapshot(self) -> dict[str, float]:
        """Collect one namespaced scalar snapshot from the whole vehicle."""

        telemetry: dict[str, float] = {}
        self.update_telemetry(telemetry)
        return telemetry
