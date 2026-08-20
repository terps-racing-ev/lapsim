"""Tests for component-owned, schema-free telemetry."""

from math import isnan
from unittest import TestCase

from lapsim import Controls, TelemetryRecorder, replay_controls
from vehicle_model import Vehicle


class TelemetryRecorderTests(TestCase):
    def test_optional_component_channels_are_aligned_with_nan(self) -> None:
        recorder = TelemetryRecorder()

        recorder.record({"component.a": 1.0})
        recorder.record({"component.a": 2.0, "component.b": 3.0})
        recorder.record({"component.b": 4.0})
        telemetry = recorder.freeze()

        self.assertEqual(telemetry["component.a"][:2], (1.0, 2.0))
        self.assertTrue(isnan(telemetry["component.a"][2]))
        self.assertTrue(isnan(telemetry["component.b"][0]))
        self.assertEqual(telemetry["component.b"][1:], (3.0, 4.0))

    def test_recorder_integrates_positive_battery_energy(self) -> None:
        recorder = TelemetryRecorder()

        recorder.record({"battery.power_w": 1_000.0}, timestep_s=2.0)
        recorder.record({"battery.power_w": -500.0}, timestep_s=3.0)
        telemetry = recorder.freeze()

        self.assertEqual(telemetry.cumulative_energy_j, (2_000.0, 2_000.0))


class ComponentTelemetryTests(TestCase):
    def test_vehicle_snapshot_collects_every_component_namespace(self) -> None:
        vehicle = Vehicle(initial_speed_mps=10.0)
        vehicle.update_state(Controls(motor_torque_request_nm=40.0), 0.5)

        snapshot = vehicle.telemetry_snapshot()

        expected_prefixes = {
            "vehicle",
            "controls",
            "limits",
            "aero",
            "drivetrain",
            "motor",
            "inverter",
            "chain_drive",
            "battery",
            "brakes",
            "chassis",
            "suspension",
            "tire",
        }
        self.assertTrue(
            expected_prefixes.issubset(
                {name.split(".", maxsplit=1)[0] for name in snapshot}
            )
        )
        self.assertEqual(snapshot["tire.rolling_radius_m"], 0.2032)
        self.assertEqual(
            snapshot["drivetrain.rolling_radius_m"],
            snapshot["tire.rolling_radius_m"],
        )
        for position in ("front_left", "front_right", "rear_left", "rear_right"):
            self.assertIn(f"tire.{position}.normal_load_n", snapshot)
            self.assertIn(f"tire.{position}.longitudinal_force_n", snapshot)
            self.assertIn(f"tire.{position}.lateral_force_n", snapshot)
            self.assertIn(f"tire.{position}.slip_ratio", snapshot)

    def test_corner_grip_power_reduction_is_explained_by_limit_channels(self) -> None:
        vehicle = Vehicle(initial_speed_mps=20.0)

        vehicle.update_state(
            Controls(
                motor_torque_request_nm=100.0,
                steering_angle_rad=0.3,
            ),
            0.5,
        )
        snapshot = vehicle.telemetry_snapshot()

        self.assertEqual(snapshot["limits.lateral_saturated"], 1.0)
        self.assertEqual(snapshot["limits.traction_active"], 1.0)
        self.assertAlmostEqual(
            snapshot["drivetrain.wheel_force_n"],
            snapshot["vehicle.rear_drive_capacity_n"],
        )
        self.assertLess(
            snapshot["drivetrain.wheel_force_n"],
            snapshot["vehicle.requested_drive_force_n"],
        )
        self.assertEqual(snapshot["battery.power_w"], 0.0)

    def test_replay_returns_mapping_with_component_channels(self) -> None:
        telemetry = replay_controls(
            Vehicle(),
            (Controls(motor_torque_request_nm=20.0),) * 3,
            0.5,
        )

        self.assertEqual(telemetry.sample_count, 3)
        self.assertIn("vehicle.speed_mps", telemetry)
        self.assertIn("battery.terminal_voltage_v", telemetry)
        self.assertIn("limits.traction_active", telemetry)
        self.assertEqual(telemetry.speed_mps, telemetry["vehicle.speed_mps"])
