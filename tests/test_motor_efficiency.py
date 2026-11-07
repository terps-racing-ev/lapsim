"""Source-data, electrical-budget, and telemetry checks for motor efficiency."""

from copy import deepcopy
import csv
from math import pi
from unittest import TestCase

from vehicle_model import Battery, Drivetrain, Motor, MotorEfficiencyMap, Vehicle
from vehicle_model.powertrain.motor_efficiency import DEFAULT_EFFICIENCY_MAP_PATH


class MotorEfficiencyTests(TestCase):
    def test_default_motor_uses_published_band_edges(self) -> None:
        motor = Vehicle().drivetrain.motor
        self.assertIsNotNone(motor.efficiency_map)
        self.assertEqual(motor.efficiency_at(1000.0, 50.0), 0.86)
        self.assertEqual(motor.efficiency_at(3000.0, 100.0), 0.96)
        self.assertAlmostEqual(
            motor.electrical_power_for_mechanical_output_w(10_000.0, 1000.0, 50.0),
            10_000.0 / 0.86,
        )

    def test_all_source_nodes_preserve_bands_and_missing_values(self) -> None:
        efficiency_map = Motor().efficiency_map
        count = 0
        with DEFAULT_EFFICIENCY_MAP_PATH.open(newline="") as stream:
            for row in csv.DictReader(stream):
                actual = efficiency_map.lookup_fraction(float(row["speed_rpm"]), float(row["torque_nm"]))
                if row["status"] == "digitized_band":
                    self.assertEqual(actual, float(row["efficiency_min_fraction"]))
                    count += 1
                else:
                    self.assertIsNone(actual)
        self.assertEqual(count, 1911)
        self.assertIs(deepcopy(efficiency_map), efficiency_map)

    def test_bilinear_interpolation_requires_only_contributing_nodes(self) -> None:
        efficiency_map = MotorEfficiencyMap(
            (1000.0, 2000.0), (50.0, 100.0), ((0.8, 0.9), (0.9, 1.0))
        )
        self.assertAlmostEqual(efficiency_map.lookup_fraction(1500.0, 75.0), 0.9)
        missing = MotorEfficiencyMap(
            (1000.0, 2000.0), (50.0, 100.0), ((0.8, None), (0.9, 1.0))
        )
        self.assertIsNone(missing.lookup_fraction(1500.0, 75.0))
        self.assertEqual(missing.lookup_fraction(1000.0, 50.0), 0.8)
        self.assertAlmostEqual(missing.lookup_fraction(1000.0, 75.0), 0.85)

    def test_unpublished_and_generating_points_use_explicit_fallback(self) -> None:
        motor = Motor()
        for speed, torque in ((0.0, 50.0), (3000.0, 0.0), (6500.0, 50.0), (3000.0, -50.0)):
            with self.subTest(speed=speed, torque=torque):
                self.assertIsNone(motor.efficiency_map.lookup_fraction(speed, torque))
                self.assertEqual(motor.efficiency_at(speed, torque), motor.efficiency)
        motor.efficiency_map = None
        motor.efficiency = 0.8
        self.assertEqual(motor.efficiency_at(3000.0, 100.0), 0.8)
        self.assertEqual(motor.electrical_power_for_mechanical_output_w(8000.0, 3000.0, 100.0), 10000.0)
        self.assertEqual(motor.mechanical_power_limit_w(10000.0, 3000.0), 8000.0)

    def test_power_limit_and_demand_use_the_same_map(self) -> None:
        motor = Motor(efficiency_map=MotorEfficiencyMap(
            (1000.0, 2000.0), (50.0, 100.0), ((0.8, 0.8), (0.8, 0.8))
        ))
        output = motor.mechanical_power_limit_w(10_000.0, 1500.0)
        self.assertAlmostEqual(output, 8000.0, places=6)
        torque = output / (1500.0 * 2 * pi / 60)
        self.assertLessEqual(
            motor.electrical_power_for_mechanical_output_w(output, 1500.0, torque), 10_000.0
        )

    def test_drivetrain_never_requests_more_than_its_electrical_budget(self) -> None:
        drivetrain = Drivetrain()
        for rpm in (300.0, 1000.0, 3000.0, 5400.0, 6000.0):
            for cap in (5000.0, 25000.0, 80000.0):
                with self.subTest(rpm=rpm, cap=cap):
                    battery = Battery(max_discharge_power_w=cap)
                    torque = drivetrain.available_motor_torque_at_speed_rpm(rpm, battery)
                    mechanical = torque * rpm * 2 * pi / 60
                    electrical = drivetrain.motor.electrical_power_for_mechanical_output_w(mechanical, rpm, torque)
                    dc = drivetrain.inverter.dc_power_for_motor_electrical_power_w(electrical)
                    self.assertLessEqual(dc, cap + 1e-6)
                    self.assertLessEqual(mechanical, drivetrain.motor.peak_power_w + 1e-6)
                    self.assertLessEqual(torque, drivetrain.motor.torque_limit_nm(rpm))

    def test_wheel_to_pack_power_includes_chain_map_and_inverter_once(self) -> None:
        drivetrain = Drivetrain()
        rpm, torque = 1000.0, 50.0
        speed = rpm * 2 * pi / 60 * drivetrain.tire.rolling_radius_m / drivetrain.chain_drive.ratio
        force = drivetrain.wheel_force_from_motor_torque_n(torque)
        power = drivetrain.positive_battery_power_w(force, speed, Battery())
        self.assertAlmostEqual(power, torque * rpm * 2 * pi / 60 / 0.86 / 0.97)

    def test_telemetry_reports_map_usage_and_electrical_loss(self) -> None:
        motor = Motor()
        for rpm, torque, valid in ((1000.0, 50.0, 1.0), (6500.0, 50.0, 0.0)):
            motor.update_state(torque, rpm, 0.01)
            telemetry = {}
            motor.update_telemetry(telemetry)
            self.assertEqual(telemetry["motor.efficiency_map_valid"], valid)
            self.assertEqual(telemetry["motor.efficiency_fallback_active"], 1.0 - valid)
            self.assertAlmostEqual(
                telemetry["motor.electrical_input_power_w"] - telemetry["motor.mechanical_power_w"],
                telemetry["motor.loss_power_w"],
            )
            self.assertGreaterEqual(telemetry["motor.loss_power_w"], 0.0)

    def test_map_rejects_invalid_axes_values_and_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            MotorEfficiencyMap((1000.0, 1000.0), (50.0, 100.0), ((0.9, 0.9), (0.9, 0.9)))
        with self.assertRaises(ValueError):
            MotorEfficiencyMap((1000.0, 2000.0), (50.0, 100.0), ((0.0, 0.9), (0.9, 0.9)))
        with self.assertRaises(ValueError):
            Motor().efficiency_at(float("nan"), 50.0)
