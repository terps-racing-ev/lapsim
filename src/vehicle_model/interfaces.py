"""Stable interfaces between the vehicle coordinator and component models.

Solvers should depend on these protocols and the small result dataclasses in
this package, not on a particular baseline implementation. A more detailed
model can therefore replace one component without inheriting from the simple
model or changing the solver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .aero.model import AeroForces
    from .mech.loads import TireForces, TireNormalLoads
    from .mech.tire import TireStates
    from .vehicle import Vehicle


@runtime_checkable
class ComponentModel(Protocol):
    """Minimum lifecycle shared by every mutable component model."""

    def validate(self) -> None: ...

    def reset_state(self) -> None: ...

    def update_telemetry(self, telemetry: dict[str, float]) -> None: ...


class BatteryModel(ComponentModel, Protocol):
    """DC pack-terminal power source/sink seen by the inverter."""

    current_power_w: float

    @property
    def discharge_power_limit_w(self) -> float: ...

    @property
    def charge_power_limit_w(self) -> float: ...

    def limit_discharge_power_w(self, requested_power_w: float) -> float: ...

    def limit_charge_power_w(self, requested_power_w: float) -> float: ...

    def update_state(self, power_w: float, timestep_s: float) -> None: ...


class MotorModel(ComponentModel, Protocol):
    """Motor-shaft torque envelope and electrical conversion model."""

    efficiency: float
    rotor_inertia_kgm2: float
    peak_power_w: float
    continuous_power_w: float
    max_speed_rpm: float
    current_torque_nm: float
    current_speed_rpm: float

    @property
    def maximum_torque_nm(self) -> float: ...

    def torque_limit_nm(self, motor_speed_rpm: float) -> float: ...

    def continuous_torque_limit_nm(self, motor_speed_rpm: float) -> float: ...

    def mechanical_power_limit_w(
        self,
        electrical_input_limit_w: float,
    ) -> float: ...

    def electrical_power_for_mechanical_output_w(
        self,
        mechanical_output_power_w: float,
        motor_speed_rpm: float,
        motor_torque_nm: float,
    ) -> float: ...

    def update_state(
        self,
        motor_torque_nm: float,
        motor_speed_rpm: float,
        timestep_s: float,
    ) -> None: ...


class InverterModel(ComponentModel, Protocol):
    """DC-to-motor electrical power conversion model."""

    efficiency: float
    current_dc_input_power_w: float
    current_motor_output_power_w: float

    def motor_electrical_power_limit_w(
        self,
        battery_discharge_limit_w: float,
    ) -> float: ...

    def dc_power_for_motor_electrical_power_w(
        self,
        motor_electrical_power_w: float,
    ) -> float: ...

    def update_state(
        self,
        dc_input_power_w: float,
        motor_output_power_w: float,
        timestep_s: float,
    ) -> None: ...


class ChainDriveModel(ComponentModel, Protocol):
    """Sprocket-and-chain transmission between motor shaft and wheels."""

    ratio: float
    efficiency: float
    input_inertia_kgm2: float
    output_inertia_kgm2: float

    def wheel_torque_from_motor_torque_nm(
        self,
        motor_torque_nm: float,
    ) -> float: ...

    def motor_torque_from_wheel_torque_nm(
        self,
        wheel_torque_nm: float,
    ) -> float: ...

    def wheel_power_for_motor_power_w(
        self,
        motor_power_w: float,
    ) -> float: ...

    def motor_power_for_wheel_power_w(
        self,
        wheel_power_w: float,
    ) -> float: ...


# Compatibility alias for the earlier generic name.
FinalDriveModel = ChainDriveModel


class DrivetrainModel(ComponentModel, Protocol):
    """Powertrain interface consumed by vehicle and lap solvers."""

    motor: MotorModel
    inverter: InverterModel
    chain_drive: ChainDriveModel
    tire: TireModel
    rolling_radius_m: float
    driven_wheel_inertia_kgm2: float
    current_motor_torque_nm: float
    current_motor_speed_rpm: float
    current_wheel_force_n: float

    @property
    def equivalent_rotating_mass_kg(self) -> float: ...

    @property
    def vehicle_speed_limit_mps(self) -> float: ...

    @property
    def max_motor_torque_nm(self) -> float: ...

    def available_motor_torque_nm(
        self,
        vehicle_speed_mps: float,
        battery: BatteryModel,
    ) -> float: ...

    def available_wheel_force_n(
        self,
        vehicle_speed_mps: float,
        battery: BatteryModel,
    ) -> float: ...

    def wheel_force_from_motor_torque_n(self, motor_torque_nm: float) -> float: ...

    def motor_torque_for_wheel_force_nm(self, wheel_force_n: float) -> float: ...

    def motor_speed_rpm(self, vehicle_speed_mps: float) -> float: ...

    def positive_battery_power_w(
        self,
        wheel_force_n: float,
        vehicle_speed_mps: float,
        battery: BatteryModel,
    ) -> float: ...

    def update_state(
        self,
        motor_torque_nm: float,
        motor_speed_rpm: float,
        wheel_force_n: float,
        timestep_s: float,
        battery_power_w: float = 0.0,
    ) -> None: ...


class AeroModel(ComponentModel, Protocol):
    """Aerodynamic force model with explicit front/rear downforce output."""

    frontal_area_m2: float
    drag_coefficient: float
    lift_coefficient: float

    def forces_n(
        self,
        vehicle_speed_mps: float,
        air_density_kgpm3: float,
        body_roll_angle_rad: float = 0.0,
    ) -> AeroForces: ...

    def update_state(
        self,
        vehicle_speed_mps: float,
        air_density_kgpm3: float,
        timestep_s: float,
        body_roll_angle_rad: float = 0.0,
    ) -> AeroForces: ...


class TireModel(ComponentModel, Protocol):
    """Four-contact-patch force, capacity, and slip model."""

    rolling_radius_m: float

    @property
    def maximum_longitudinal_coefficient(self) -> float: ...

    def lateral_force_capacity_n(self, normal_load_n: float) -> float: ...

    def longitudinal_force_capacity_n(self, normal_load_n: float) -> float: ...

    def combined_longitudinal_force_capacity_n(
        self,
        normal_load_n: float,
        lateral_force_n: float,
    ) -> float: ...

    def lateral_forces_n(
        self,
        normal_loads_n: TireNormalLoads,
        total_lateral_force_n: float,
    ) -> TireForces: ...

    def calculate_forces(
        self,
        normal_loads_n: TireNormalLoads,
        total_lateral_force_n: float,
        drive_force_request_n: float,
        front_brake_force_request_n: float,
        rear_brake_force_request_n: float,
        vehicle_speed_mps: float,
        timestep_s: float,
    ) -> TireStates: ...

    def update_state(
        self,
        states: TireStates,
        timestep_s: float,
    ) -> None: ...


class ChassisModel(ComponentModel, Protocol):
    """Vehicle geometry and static mass distribution."""

    wheelbase_m: float
    cg_height_m: float
    front_track_width_m: float
    rear_track_width_m: float
    front_axle_height_m: float
    rear_axle_height_m: float
    front_roll_axis_height_m: float
    rear_roll_axis_height_m: float
    static_front_weight_fraction: float


class SuspensionModel(ComponentModel, Protocol):
    """Normal-load distribution model consumed by tire calculations."""

    current_tire_normal_loads_n: TireNormalLoads

    def body_roll_angle_rad(
        self,
        mass_kg: float,
        chassis: ChassisModel,
        lateral_acceleration_mps2: float,
    ) -> float: ...

    def tire_normal_loads_n(
        self,
        mass_kg: float,
        gravity_mps2: float,
        aero_forces: AeroForces,
        chassis: ChassisModel,
        longitudinal_acceleration_mps2: float = 0.0,
        lateral_acceleration_mps2: float = 0.0,
    ) -> TireNormalLoads: ...

    def update_state(
        self,
        tire_normal_loads_n: TireNormalLoads,
        timestep_s: float,
    ) -> None: ...


class BrakeModel(ComponentModel, Protocol):
    """Friction-brake limit and operating-state model."""

    maximum_pressure_psi: float
    current_force_request_n: float
    current_friction_force_n: float

    def axle_force_requests_from_pressures_n(
        self,
        front_pressure_psi: float,
        rear_pressure_psi: float,
        rolling_radius_m: float,
    ) -> tuple[float, float]: ...

    def axle_pressures_for_force_requests_psi(
        self,
        front_force_n: float,
        rear_force_n: float,
        rolling_radius_m: float,
    ) -> tuple[float, float]: ...

    def maximum_deceleration_mps2(
        self,
        vehicle: Vehicle,
        speed_mps: float,
        curvature_per_m: float,
        gravity_mps2: float,
        air_density_kgpm3: float,
    ) -> float: ...

    def update_state(
        self,
        force_request_n: float,
        friction_force_n: float,
        timestep_s: float,
        **kwargs: float,
    ) -> None: ...
