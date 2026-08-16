"""Safety-contract tests for recorded control conversion."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.common import (
    RecordedControlAdapter,
    UnmodeledRecordedControlError,
)
from vehicle_model import Vehicle


class RecordedControlAdapterTests(unittest.TestCase):
    def test_active_brake_pressure_stays_a_pressure_control(self) -> None:
        adapter = RecordedControlAdapter()
        adapter.validate_trace(np.array([20.0, 0.0]), np.array([0.0, 50.0]))
        controls = adapter.controls(
            motor_torque_nm=0.0,
            brake_pressure_psi=50.0,
            curvature_per_m=0.0,
            wheelbase_m=1.55,
        )
        self.assertEqual(controls.front_brake_pressure_psi, 50.0)
        self.assertEqual(controls.rear_brake_pressure_psi, 50.0)

    def test_negative_torque_is_rejected_by_default(self) -> None:
        adapter = RecordedControlAdapter(brake_force_per_psi_n=0.0)
        with self.assertRaisesRegex(
            UnmodeledRecordedControlError, "negative torque/regen"
        ):
            adapter.validate_trace(np.array([20.0, -5.0]), np.zeros(2))

    def test_approximations_must_be_selected_explicitly(self) -> None:
        adapter = RecordedControlAdapter(
            brake_force_per_psi_n=0.0,
            negative_torque_policy="clip",
        )
        adapter.validate_trace(np.array([20.0, -5.0]), np.array([0.0, 50.0]))
        vehicle = Vehicle(initial_speed_mps=10.0)
        adapter.configure_vehicle(vehicle)
        controls = adapter.controls(
            motor_torque_nm=-5.0,
            brake_pressure_psi=50.0,
            curvature_per_m=0.02,
            wheelbase_m=1.55,
        )
        self.assertEqual(controls.motor_torque_request_nm, 0.0)
        self.assertEqual(controls.front_brake_pressure_psi, 50.0)
        self.assertNotEqual(controls.steering_angle_rad, 0.0)
        vehicle.update_state(controls, 0.01)
        self.assertEqual(vehicle.brakes.current_force_request_n, 0.0)

    def test_calibration_is_applied_by_vehicle_brake_model(self) -> None:
        adapter = RecordedControlAdapter(
            brake_force_per_psi_n=8.0,
            brake_deadband_psi=10.0,
        )
        vehicle = Vehicle(initial_speed_mps=10.0)
        adapter.configure_vehicle(vehicle)
        controls = adapter.controls(
            motor_torque_nm=0.0,
            brake_pressure_psi=60.0,
            curvature_per_m=0.0,
            wheelbase_m=1.55,
        )
        vehicle.update_state(controls, 0.01)
        self.assertAlmostEqual(vehicle.brakes.current_force_request_n, 400.0)


if __name__ == "__main__":
    unittest.main()
