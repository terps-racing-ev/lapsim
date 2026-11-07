"""Vehicle configuration, public state, and time-step coordination.

Force balance lives in dynamics.py, kinematics in motion.py, and energy
accounting in powertrain/power_flow.py. Components own their parameters and
history.
"""

from dataclasses import dataclass, field
from math import isfinite

from lapsim.core.controls import Controls
from utils.units import pounds_to_kilograms

from .aero import Aero, AeroForces
from .dynamics import ForceBalanceSolver, ForceRequests, OperatingPoint
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
    WheelAlignmentModel,
)
from .mech import (
    Brakes,
    Chassis,
    Suspension,
    Tire,
    WheelAlignment,
    WheelAlignmentState,
)
from .motion import MotionStep, integrate_motion
from .powertrain.power_flow import (
    PowerFlow,
    maximum_regenerative_brake_force_n,
    regenerative_battery_power_w,
    regenerative_efficiency,
)
from .powertrain import Drivetrain

DEFAULT_MASS_LB = 630.0
DEFAULT_MASS_KG = pounds_to_kilograms(DEFAULT_MASS_LB)
DEFAULT_ROLLING_RESISTANCE_COEFFICIENT = 0.012
DEFAULT_CORNERING_DRAG_COEFFICIENT = 0.036


@dataclass(slots=True)
class Vehicle:
    """Rear-wheel-drive vehicle assembled from replaceable subsystems.

    Configure physical parameters on their owning component. ``update_state``
    coordinates one time interval; the fields below retain public motion and
    force diagnostics for solvers, analysis scripts, and telemetry.
    """

    mass_kg: float = DEFAULT_MASS_KG
    tire: TireModel = field(default_factory=Tire)
    drivetrain: DrivetrainModel = field(default_factory=Drivetrain)
    aero: AeroModel = field(default_factory=Aero)
    battery: BatteryModel = field(default_factory=RCTheveninBattery)
    brakes: BrakeModel = field(default_factory=Brakes)
    chassis: ChassisModel = field(default_factory=Chassis)
    suspension: SuspensionModel = field(default_factory=Suspension)
    alignment: WheelAlignmentModel = field(default_factory=WheelAlignment)

    # Environment used by the time-domain update_state simulation.
    gravity_mps2: float = STANDARD_GRAVITY_MPS2
    air_density_kgpm3: float = STANDARD_AIR_DENSITY_KGPM3

    # Initial conditions restored by reset_state
    initial_speed_mps: float = 0.0
    initial_x_m: float = 0.0
    initial_y_m: float = 0.0
    initial_heading_rad: float = 0.0

    # Vehicle state. Time is the integration coordinate; distance is integrated
    # from the speed over every update interval.
    time_s: float = field(init=False, default=0.0)
    distance_m: float = field(init=False, default=0.0)
    speed_mps: float = field(init=False, default=0.0)
    x_m: float = field(init=False, default=0.0)
    y_m: float = field(init=False, default=0.0)
    heading_rad: float = field(init=False, default=0.0)
    longitudinal_acceleration_mps2: float = field(init=False, default=0.0)
    lateral_acceleration_mps2: float = field(init=False, default=0.0)
    curvature_per_m: float = field(init=False, default=0.0)
    body_slip_angle_rad: float = field(init=False, default=0.0)
    yaw_rate_radps: float = field(init=False, default=0.0)
    yaw_moment_nm: float = field(init=False, default=0.0)
    current_controls: Controls = field(init=False, default_factory=Controls)

    # Retained force-balance diagnostics exposed through telemetry.
    lateral_force_capacity_n: float = field(init=False, default=0.0)
    current_lateral_force_n: float = field(init=False, default=0.0)
    available_motor_torque_nm: float = field(init=False, default=0.0)
    envelope_limited_motor_torque_nm: float = field(init=False, default=0.0)
    requested_drive_force_n: float = field(init=False, default=0.0)
    rear_drive_capacity_n: float = field(init=False, default=0.0)
    speed_limited_drive_force_n: float = field(init=False, default=0.0)
    current_drive_force_n: float = field(init=False, default=0.0)
    current_friction_braking_force_n: float = field(init=False, default=0.0)
    current_regenerative_braking_force_n: float = field(init=False, default=0.0)
    current_regenerative_power_w: float = field(init=False, default=0.0)
    maximum_friction_braking_force_n: float = field(init=False, default=0.0)
    current_rolling_resistance_force_n: float = field(init=False, default=0.0)
    current_cornering_drag_force_n: float = field(init=False, default=0.0)
    current_resistance_force_n: float = field(init=False, default=0.0)

    # Road resistance
    rolling_resistance_coefficient: float = DEFAULT_ROLLING_RESISTANCE_COEFFICIENT
    cornering_drag_coefficient: float = DEFAULT_CORNERING_DRAG_COEFFICIENT

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
            ("alignment", self.alignment, WheelAlignmentModel),
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
            self.alignment,
        )

    @property
    def effective_longitudinal_mass_kg(self) -> float:
        """Mass accelerated longitudinally, including rotating inertia."""

        return self.mass_kg + self.drivetrain.equivalent_rotating_mass_kg

    def aero_forces_n(
        self,
        speed_mps: float,
        lateral_acceleration_mps2: float = 0.0,
        air_density_kgpm3: float | None = None,
        curvature_per_m: float | None = None,
    ) -> AeroForces:
        """Return aero forces at the quasi-static body roll for this state."""

        selected_curvature_per_m = curvature_per_m
        if selected_curvature_per_m is None:
            selected_curvature_per_m = (
                lateral_acceleration_mps2 / speed_mps**2 if speed_mps > 1e-12 else 0.0
            )
        self.aero.select_configuration(selected_curvature_per_m)
        body_roll_angle_rad = self.suspension.body_roll_angle_rad(
            self.mass_kg,
            self.chassis,
            lateral_acceleration_mps2,
        )
        return self.aero.forces_n(
            speed_mps,
            (
                self.air_density_kgpm3
                if air_density_kgpm3 is None
                else air_density_kgpm3
            ),
            body_roll_angle_rad,
        )

    def wheel_alignment_state(
        self,
        steering_angle_rad: float,
        lateral_acceleration_mps2: float,
    ) -> WheelAlignmentState:
        """Return per-wheel steer, camber, and travel for an operating point."""

        body_roll_angle_rad = self.suspension.body_roll_angle_rad(
            self.mass_kg,
            self.chassis,
            lateral_acceleration_mps2,
        )
        suspension_travel_m = self.suspension.wheel_travel_m(
            self.chassis,
            body_roll_angle_rad,
        )
        return self.alignment.calculate(
            self.chassis,
            steering_angle_rad,
            body_roll_angle_rad,
            suspension_travel_m,
        )

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
        self.body_slip_angle_rad = 0.0
        self.yaw_rate_radps = 0.0
        self.yaw_moment_nm = 0.0
        self.current_controls = Controls()
        self.lateral_force_capacity_n = 0.0
        self.current_lateral_force_n = 0.0
        self.available_motor_torque_nm = 0.0
        self.envelope_limited_motor_torque_nm = 0.0
        self.requested_drive_force_n = 0.0
        self.rear_drive_capacity_n = 0.0
        self.speed_limited_drive_force_n = 0.0
        self.current_drive_force_n = 0.0
        self.current_friction_braking_force_n = 0.0
        self.current_regenerative_braking_force_n = 0.0
        self.current_regenerative_power_w = 0.0
        self.maximum_friction_braking_force_n = 0.0
        self.current_rolling_resistance_force_n = 0.0
        self.current_cornering_drag_force_n = 0.0
        self.current_resistance_force_n = 0.0

    @property
    def regenerative_efficiency(self) -> float:
        """Wheel-to-pack efficiency from the drivetrain's regen model."""

        return regenerative_efficiency(self.drivetrain)

    def maximum_regenerative_brake_force_n(self, vehicle_speed_mps: float) -> float:
        """Query drivetrain regen capacity before tire grip and driver limits."""

        return maximum_regenerative_brake_force_n(
            self.drivetrain, self.battery, vehicle_speed_mps
        )

    def regenerative_battery_power_w(
        self,
        regenerative_brake_force_n: float,
        wheel_surface_speed_mps: float,
    ) -> float:
        """Return nonnegative recovered pack power through drivetrain losses."""

        return regenerative_battery_power_w(
            self.drivetrain,
            self.battery,
            regenerative_brake_force_n,
            wheel_surface_speed_mps,
        )

    def update_state(self, controls: Controls, timestep_s: float) -> None:
        """Solve forces, integrate motion, and advance components by one step.

        Forces use the starting speed and previous component history. Motion
        assumes constant acceleration over the interval and stops at zero
        speed if braking brings the car to rest before the interval ends.
        See docs/update_state.md for the equations and update order.
        """

        if not isinstance(controls, Controls):
            raise TypeError("controls must be Controls")
        if not isfinite(timestep_s) or timestep_s <= 0.0:
            raise ValueError("timestep_s must be finite and positive")
        if self.speed_mps < 0.0:
            raise RuntimeError("vehicle speed cannot be negative")

        requests = ForceRequests.from_controls(self, controls)
        operating_point = ForceBalanceSolver(
            self, controls, requests, timestep_s
        ).solve()
        motion = integrate_motion(
            initial_speed_mps=self.speed_mps,
            initial_heading_rad=self.heading_rad,
            longitudinal_acceleration_mps2=operating_point.acceleration_mps2,
            lateral_acceleration_mps2=operating_point.lateral_acceleration_mps2,
            body_slip_angle_rad=operating_point.body_slip_angle_rad,
            speed_limit_mps=self.drivetrain.vehicle_speed_limit_mps,
            timestep_s=timestep_s,
        )
        self.aero.update_state(
            self.speed_mps,
            self.air_density_kgpm3,
            timestep_s,
            operating_point.aero_forces.body_roll_angle_rad,
        )
        self.x_m += motion.x_displacement_m
        self.y_m += motion.y_displacement_m
        power_flow = PowerFlow.calculate(
            self.drivetrain,
            self.battery,
            controls,
            operating_point.tire_states,
            motion.average_speed_mps(timestep_s),
        )
        self._update_components(
            controls, timestep_s, requests, operating_point, power_flow
        )
        self._record_step(
            controls, timestep_s, requests, operating_point, motion, power_flow
        )

    def _update_components(
        self,
        controls: Controls,
        timestep_s: float,
        requests: ForceRequests,
        operating_point: OperatingPoint,
        power_flow: PowerFlow,
    ) -> None:
        """Commit the solved step to each component in dependency order.

        Aero is updated before power accounting in ``update_state``. Pack
        power here is net of regen; drivetrain power is propulsion only.
        Every history-dependent component receives the same full time step.
        """

        tire_states = operating_point.tire_states
        self.battery.update_state(power_flow.net_battery_power_w, timestep_s)
        self.drivetrain.update_state(
            power_flow.motor_torque_nm,
            power_flow.motor_speed_rpm,
            tire_states.drive_force_n,
            timestep_s,
            power_flow.propulsion_battery_power_w,
        )
        self.brakes.update_state(
            requests.total_brake_force_n,
            power_flow.friction_braking_force_n,
            timestep_s,
            front_pressure_psi=controls.front_brake_pressure_psi,
            rear_pressure_psi=controls.rear_brake_pressure_psi,
            front_friction_force_n=tire_states.front_braking_force_n,
            rear_friction_force_n=max(
                tire_states.rear_braking_force_n
                - power_flow.regenerative_braking_force_n,
                0.0,
            ),
            front_force_request_n=requests.front_brake_force_n,
            rear_force_request_n=requests.rear_brake_force_n,
        )
        self.suspension.update_state(operating_point.normal_loads_n, timestep_s)
        self.alignment.update_state(operating_point.alignment_state, timestep_s)
        self.tire.update_state(tire_states, timestep_s)

    def _record_step(
        self,
        controls: Controls,
        timestep_s: float,
        requests: ForceRequests,
        operating_point: OperatingPoint,
        motion: MotionStep,
        power_flow: PowerFlow,
    ) -> None:
        """Retain motion and diagnostics without recalculating any physics."""

        self.time_s += timestep_s
        self.distance_m += motion.distance_m
        self.speed_mps = motion.final_speed_mps
        self.heading_rad = motion.final_heading_rad
        self.longitudinal_acceleration_mps2 = operating_point.acceleration_mps2
        self.lateral_acceleration_mps2 = operating_point.lateral_acceleration_mps2
        self.curvature_per_m = motion.curvature_per_m
        self.body_slip_angle_rad = operating_point.body_slip_angle_rad
        self.yaw_rate_radps = motion.yaw_rate_radps
        self.yaw_moment_nm = operating_point.yaw_moment_nm
        self.current_controls = controls

        self.lateral_force_capacity_n = operating_point.lateral_capacity_n
        self.current_lateral_force_n = (
            operating_point.lateral_acceleration_mps2 * self.mass_kg
        )
        self.available_motor_torque_nm = requests.available_motor_torque_nm
        self.envelope_limited_motor_torque_nm = requests.applied_motor_torque_nm
        self.requested_drive_force_n = requests.drive_force_n
        self.rear_drive_capacity_n = operating_point.rear_drive_capacity_n
        self.speed_limited_drive_force_n = operating_point.speed_limited_drive_force_n
        self.current_drive_force_n = operating_point.tire_states.drive_force_n
        self.current_friction_braking_force_n = power_flow.friction_braking_force_n
        self.current_regenerative_braking_force_n = (
            power_flow.regenerative_braking_force_n
        )
        self.current_regenerative_power_w = power_flow.regenerative_power_w
        self.maximum_friction_braking_force_n = (
            operating_point.tire_states.longitudinal_capacity_n
        )
        self.current_rolling_resistance_force_n = (
            operating_point.resistance_force_n
            - operating_point.aero_forces.drag_n
            - operating_point.cornering_drag_force_n
        )
        self.current_cornering_drag_force_n = operating_point.cornering_drag_force_n
        self.current_resistance_force_n = operating_point.resistance_force_n

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        """Write vehicle state, controls, force balance, and active limits."""

        lateral_utilization = 0.0
        if self.lateral_force_capacity_n > 0.0:
            lateral_utilization = min(
                abs(self.current_lateral_force_n) / self.lateral_force_capacity_n,
                1.0,
            )
        telemetry.update(
            {
                "vehicle.time_s": self.time_s,
                "vehicle.distance_m": self.distance_m,
                "vehicle.x_m": self.x_m,
                "vehicle.y_m": self.y_m,
                "vehicle.heading_rad": self.heading_rad,
                "vehicle.body_slip_angle_rad": self.body_slip_angle_rad,
                "vehicle.yaw_rate_radps": self.yaw_rate_radps,
                "vehicle.yaw_moment_nm": self.yaw_moment_nm,
                "vehicle.speed_mps": self.speed_mps,
                "vehicle.longitudinal_acceleration_mps2": (
                    self.longitudinal_acceleration_mps2
                ),
                "vehicle.lateral_acceleration_mps2": (self.lateral_acceleration_mps2),
                "vehicle.achieved_curvature_per_m": self.curvature_per_m,
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
                "vehicle.regenerative_braking_force_n": (
                    self.current_regenerative_braking_force_n
                ),
                "vehicle.regenerative_power_w": self.current_regenerative_power_w,
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
                "limits.lateral_saturated": float(lateral_utilization >= 0.95),
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
