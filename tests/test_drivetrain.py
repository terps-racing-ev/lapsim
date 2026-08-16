"""Tests for the configured drivetrain envelope."""

from math import isclose
from unittest import TestCase

from utils.units import miles_per_hour_to_meters_per_second
from vehicle_model import Vehicle
from vehicle_model.electrical import Battery
from vehicle_model.mech import Tire
from vehicle_model.powertrain import ChainDrive, Drivetrain, FinalDrive, Motor


class DrivetrainTests(TestCase):
    def test_endurance_calibration_is_the_default_propulsion_setup(self) -> None:
        drivetrain = Drivetrain()

        self.assertTrue(isclose(drivetrain.tire.rolling_radius_m, 0.2032))
        self.assertTrue(isclose(drivetrain.rolling_radius_m, 0.2032))
        self.assertTrue(isclose(drivetrain.chain_drive.ratio, 3.455))
        self.assertTrue(isclose(drivetrain.motor.efficiency, 0.95963664851588))
        self.assertTrue(isclose(drivetrain.chain_drive.efficiency, 0.776852813358272))

    def test_legacy_final_drive_names_alias_chain_drive(self) -> None:
        drivetrain = Drivetrain()

        self.assertIs(FinalDrive, ChainDrive)
        self.assertIs(drivetrain.final_drive, drivetrain.chain_drive)

    def test_peak_torque_curve_interpolation(self) -> None:
        drivetrain = Drivetrain()
        expected_torque_by_rpm = {
            0.0: 230.0,
            1_000.0: 230.0,
            2_000.0: 230.0,
            2_500.0: 227.0,
            3_000.0: 224.0,
            3_500.0: 221.0,
            4_000.0: 218.0,
            4_500.0: 212.0,
            5_000.0: 206.0,
            7_000.0: 206.0,
        }

        for speed_rpm, expected_torque_nm in expected_torque_by_rpm.items():
            self.assertTrue(
                isclose(
                    drivetrain.motor_torque_limit_nm(speed_rpm),
                    expected_torque_nm,
                )
            )

    def test_motor_peak_power_is_independent_of_battery_limit(self) -> None:
        drivetrain = Drivetrain()
        high_power_battery = Battery(max_discharge_power_w=200_000.0)

        self.assertEqual(
            drivetrain.max_motor_mechanical_power_w(high_power_battery),
            80_000.0,
        )

    def test_configured_speed_cap_is_100_mph(self) -> None:
        drivetrain = Drivetrain()

        self.assertTrue(
            isclose(
                drivetrain.configured_speed_limit_mps,
                miles_per_hour_to_meters_per_second(100.0),
            )
        )

    def test_rotational_inertia_is_reflected_to_vehicle_mass(self) -> None:
        drivetrain = Drivetrain(
            tire=Tire(rolling_radius_m=0.2),
            motor=Motor(rotor_inertia_kgm2=0.01),
            chain_drive=ChainDrive(
                ratio=4.0,
                input_inertia_kgm2=0.0025,
                output_inertia_kgm2=0.04,
            ),
            driven_wheel_inertia_kgm2=0.0,
        )

        self.assertTrue(
            isclose(
                drivetrain.wheel_referenced_rotational_inertia_kgm2,
                0.24,
            )
        )
        self.assertTrue(isclose(drivetrain.equivalent_rotating_mass_kg, 6.0))

    def test_vehicle_drivetrain_uses_the_vehicle_tire_radius(self) -> None:
        tire = Tire(rolling_radius_m=0.19)
        drivetrain = Drivetrain()

        vehicle = Vehicle(tire=tire, drivetrain=drivetrain)

        self.assertIs(vehicle.drivetrain.tire, vehicle.tire)
        self.assertEqual(vehicle.drivetrain.rolling_radius_m, 0.19)

    def test_rotational_inertia_can_be_disabled(self) -> None:
        drivetrain = Drivetrain(
            motor=Motor(rotor_inertia_kgm2=0.0),
            chain_drive=ChainDrive(
                input_inertia_kgm2=0.0,
                output_inertia_kgm2=0.0,
            ),
            driven_wheel_inertia_kgm2=0.0,
        )

        self.assertEqual(drivetrain.equivalent_rotating_mass_kg, 0.0)
