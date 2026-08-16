"""Integration tests for replaceable vehicle component models."""

from unittest import TestCase

from lapsim import Controls
from vehicle_model import (
    AeroForces,
    Battery,
    Drivetrain,
    ChainDrive,
    Inverter,
    Motor,
    TireNormalLoads,
    Vehicle,
)


class AsymmetricSuspension:
    """Small structural-protocol implementation used only by this test."""

    def __init__(self) -> None:
        self.call_count = 0
        self.current_tire_normal_loads_n = TireNormalLoads(
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def validate(self) -> None:
        pass

    def reset_state(self) -> None:
        self.current_tire_normal_loads_n = TireNormalLoads(
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry["suspension.test_call_count"] = float(self.call_count)

    def tire_normal_loads_n(
        self,
        mass_kg: float,
        gravity_mps2: float,
        aero_forces: AeroForces,
        chassis,
        longitudinal_acceleration_mps2: float = 0.0,
        lateral_acceleration_mps2: float = 0.0,
    ) -> TireNormalLoads:
        self.call_count += 1
        total_n = mass_kg * gravity_mps2 + aero_forces.downforce_n
        return TireNormalLoads(
            front_left_n=0.30 * total_n,
            front_right_n=0.20 * total_n,
            rear_left_n=0.275 * total_n,
            rear_right_n=0.225 * total_n,
        )

    def update_state(
        self,
        tire_normal_loads_n: TireNormalLoads,
        timestep_s: float,
    ) -> None:
        self.current_tire_normal_loads_n = tire_normal_loads_n


class TrackingChainDrive(ChainDrive):
    """Confirm that drivetrain power flow uses the component seam."""

    def __init__(self) -> None:
        super().__init__()
        self.inverse_power_conversion_called = False

    def motor_power_for_wheel_power_w(self, wheel_power_w: float) -> float:
        self.inverse_power_conversion_called = True
        return super().motor_power_for_wheel_power_w(wheel_power_w)


class ComponentExtensibilityTests(TestCase):
    def test_vehicle_accepts_structural_suspension_replacement(self) -> None:
        suspension = AsymmetricSuspension()
        vehicle = Vehicle(suspension=suspension, initial_speed_mps=10.0)

        vehicle.update_state(Controls(steering_angle_rad=0.05), 0.5)

        self.assertGreater(suspension.call_count, 0)
        loads = suspension.current_tire_normal_loads_n
        self.assertNotEqual(loads.front_left_n, loads.front_right_n)

    def test_nested_powertrain_models_are_independently_configurable(self) -> None:
        drivetrain = Drivetrain(
            motor=Motor(peak_power_w=70_000.0),
            inverter=Inverter(efficiency=0.96),
            chain_drive=ChainDrive(ratio=4.0, efficiency=0.94),
        )
        vehicle = Vehicle(
            drivetrain=drivetrain,
            battery=Battery(max_discharge_power_w=50_000.0),
        )

        self.assertEqual(vehicle.drivetrain.motor.peak_power_w, 70_000.0)
        self.assertEqual(vehicle.drivetrain.inverter.efficiency, 0.96)
        self.assertEqual(vehicle.drivetrain.chain_drive.ratio, 4.0)
        self.assertLess(
            drivetrain.max_motor_mechanical_power_w(vehicle.battery),
            50_000.0,
        )

    def test_drivetrain_uses_chain_drive_power_conversion_interface(self) -> None:
        chain_drive = TrackingChainDrive()
        drivetrain = Drivetrain(chain_drive=chain_drive)

        drivetrain.positive_battery_power_w(
            wheel_force_n=1_000.0,
            vehicle_speed_mps=10.0,
            battery=Battery(),
        )

        self.assertTrue(chain_drive.inverse_power_conversion_called)
