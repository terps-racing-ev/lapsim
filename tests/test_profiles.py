"""Tests for generic controls profiles shared by every event."""

from unittest import TestCase

from lapsim import (
    ConstantControlsProfile,
    Controls,
    PiecewiseLinearControlsProfile,
)


class ControlsProfileTests(TestCase):
    def test_constant_profile_returns_the_supplied_controls(self) -> None:
        controls = Controls(motor_torque_request_nm=42.0)
        profile = ConstantControlsProfile(controls)

        self.assertEqual(profile.controls_at(0.0), controls)
        self.assertEqual(profile.controls_at(100.0), controls)

    def test_piecewise_profile_interpolates_and_wraps_periodically(self) -> None:
        profile = PiecewiseLinearControlsProfile(
            distance_m=(0.0, 50.0),
            controls=(
                Controls(motor_torque_request_nm=0.0),
                Controls(motor_torque_request_nm=100.0),
            ),
            period_m=100.0,
        )

        self.assertAlmostEqual(profile.controls_at(25.0).motor_torque_request_nm, 50.0)
        self.assertAlmostEqual(profile.controls_at(75.0).motor_torque_request_nm, 50.0)
        self.assertAlmostEqual(profile.controls_at(125.0).motor_torque_request_nm, 50.0)
