"""Tests for the simple driven-wheel slip model."""

from unittest import TestCase

from lapsim import Controls
from vehicle_model import Vehicle, WheelSlip


class WheelSlipTests(TestCase):
    def test_slip_increases_with_force_utilization(self) -> None:
        model = WheelSlip(peak_longitudinal_slip_ratio=0.10)

        model.update_state(10.0, 500.0, 1_000.0, 0.01)

        self.assertAlmostEqual(model.current_slip_ratio, 0.05)
        self.assertAlmostEqual(model.current_wheel_surface_speed_mps, 10.5)

    def test_vehicle_slip_increases_motor_speed_and_power(self) -> None:
        slipping = Vehicle(initial_speed_mps=10.0)
        no_slip = Vehicle(
            initial_speed_mps=10.0,
            wheel_slip=WheelSlip(peak_longitudinal_slip_ratio=0.0),
        )
        controls = Controls(motor_torque_request_nm=100.0)

        slipping.update_state(controls, 0.01)
        no_slip.update_state(controls, 0.01)

        self.assertGreater(slipping.wheel_slip.current_slip_ratio, 0.0)
        self.assertGreater(
            slipping.drivetrain.current_motor_speed_rpm,
            no_slip.drivetrain.current_motor_speed_rpm,
        )
        self.assertGreater(
            slipping.battery.current_power_w,
            no_slip.battery.current_power_w,
        )

    def test_zero_longitudinal_force_has_zero_slip(self) -> None:
        model = WheelSlip()

        model.update_state(10.0, 0.0, 1_000.0, 0.01)

        self.assertEqual(model.current_slip_ratio, 0.0)
        self.assertEqual(model.current_wheel_surface_speed_mps, 10.0)
