"""Tests for recorded controls and distance-indexed replay."""

from math import isclose
from unittest import TestCase

from lapsim.controls import Controls
from lapsim.recorded_lap import RecordedLap
from lapsim.replay import replay_controls
from vehicle_model.vehicle import Vehicle


class RecordedLapTests(TestCase):
    def test_recorded_channels_convert_to_controls(self) -> None:
        lap = RecordedLap(
            time_s=(0.0, 0.1),
            x_m=(0.0, 1.0),
            y_m=(0.0, 0.0),
            speed_mps=(5.0, 5.1),
            distance_trip_m=(0.0, 0.5),
            motor_speed_rpm=(1_000.0, 1_010.0),
            motor_torque_command_nm=(20.0, -5.0),
            motor_torque_feedback_nm=(19.0, -4.0),
            inverter_torque_feedback_nm=(19.0, -4.0),
            inverter_motor_speed_rpm=(1_000.0, 1_010.0),
            accelerator_percent=(10.0, 0.0),
            brake_pressure_psi=(0.0, 20.0),
            pack_power_w=(2_000.0, -100.0),
        )

        controls = lap.controls(
            (0.0, 0.1),
            wheelbase_m=1.5,
        )

        self.assertEqual(controls[0].motor_torque_request_nm, 20.0)
        self.assertEqual(controls[1].motor_torque_request_nm, 0.0)
        self.assertEqual(controls[1].front_brake_pressure_psi, 20.0)
        self.assertEqual(controls[1].rear_brake_pressure_psi, 20.0)
        self.assertGreater(controls[1].steering_angle_rad, 0.0)


class ReplayTests(TestCase):
    def test_replay_requires_positive_distance_steps(self) -> None:
        controls = (Controls(), Controls())

        with self.assertRaisesRegex(
            ValueError, "distance_step_m must be finite and positive"
        ):
            replay_controls(Vehicle(), controls, 0.0)
        with self.assertRaisesRegex(
            ValueError, "distance_step_m must contain one value per control"
        ):
            replay_controls(Vehicle(), controls, (1.0,))
        with self.assertRaisesRegex(
            ValueError, "Every distance step must be finite and positive"
        ):
            replay_controls(Vehicle(), controls, (1.0, -1.0))

    def test_replay_retains_one_state_per_control(self) -> None:
        vehicle = Vehicle(initial_speed_mps=10.0)
        telemetry = replay_controls(
            vehicle,
            (Controls(motor_torque_request_nm=20.0),) * 3,
            1.0,
        )

        self.assertEqual(len(telemetry.time_s), 3)
        self.assertGreater(telemetry.time_s[-1], 0.0)
        self.assertTrue(isclose(telemetry.distance_m[-1], 3.0))
        self.assertGreater(telemetry.speed_mps[-1], 10.0)
        self.assertGreater(telemetry.battery_power_w[-1], 0.0)
