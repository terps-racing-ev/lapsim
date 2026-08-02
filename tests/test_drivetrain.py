"""Tests for the configured drivetrain envelope."""

from math import isclose
from unittest import TestCase

from lapsim.battery import Battery
from lapsim.drivetrain import Drivetrain
from lapsim.utils.units import miles_per_hour_to_meters_per_second


class DrivetrainTests(TestCase):
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
            rolling_radius_m=0.2,
            final_drive_ratio=4.0,
            motor_rotor_inertia_kgm2=0.01,
            final_drive_input_inertia_kgm2=0.0025,
            final_drive_output_inertia_kgm2=0.04,
            driven_wheel_inertia_kgm2=0.0,
        )

        self.assertTrue(
            isclose(
                drivetrain.wheel_referenced_rotational_inertia_kgm2,
                0.24,
            )
        )
        self.assertTrue(
            isclose(drivetrain.equivalent_rotating_mass_kg, 6.0)
        )

    def test_rotational_inertia_can_be_disabled(self) -> None:
        drivetrain = Drivetrain(
            motor_rotor_inertia_kgm2=0.0,
            final_drive_input_inertia_kgm2=0.0,
            final_drive_output_inertia_kgm2=0.0,
            driven_wheel_inertia_kgm2=0.0,
        )

        self.assertEqual(drivetrain.equivalent_rotating_mass_kg, 0.0)
