"""Tests for chronological vehicle state updates."""

from math import isclose
from unittest import TestCase

from lapsim import Controls
from lapsim.vehicle import Vehicle


class VehicleStateUpdateTests(TestCase):
    def test_acceleration_then_braking_and_reset(self) -> None:
        vehicle = Vehicle()

        for _ in range(100):
            vehicle.update_state(
                Controls(motor_torque_request_nm=230.0),
                0.01,
            )

        accelerated_speed_mps = vehicle.speed_mps
        self.assertGreater(accelerated_speed_mps, 0.0)
        self.assertGreater(vehicle.distance_m, 0.0)
        self.assertGreater(vehicle.battery.current_power_w, 0.0)

        for _ in range(100):
            vehicle.update_state(
                Controls(friction_brake_force_request_n=1_000_000.0),
                0.01,
            )

        self.assertLess(vehicle.speed_mps, accelerated_speed_mps)
        self.assertGreaterEqual(vehicle.speed_mps, 0.0)

        vehicle.reset_state()
        self.assertEqual(vehicle.time_s, 0.0)
        self.assertEqual(vehicle.distance_m, 0.0)
        self.assertEqual(vehicle.speed_mps, vehicle.initial_speed_mps)
        self.assertEqual(vehicle.battery.current_power_w, 0.0)

    def test_steering_advances_heading_and_lateral_position(self) -> None:
        vehicle = Vehicle(initial_speed_mps=10.0)
        vehicle.update_state(Controls(steering_angle_rad=0.1), 0.1)

        self.assertGreater(vehicle.heading_rad, 0.0)
        self.assertGreater(vehicle.y_m, 0.0)
        self.assertGreater(vehicle.lateral_acceleration_mps2, 0.0)

    def test_timestep_is_accumulated(self) -> None:
        vehicle = Vehicle()
        controls = Controls(motor_torque_request_nm=10.0)
        vehicle.update_state(controls, 0.125)

        self.assertTrue(isclose(vehicle.time_s, 0.125))
        self.assertIs(vehicle.current_controls, controls)

    def test_speed_limiter_holds_speed_without_cutting_all_torque(self) -> None:
        vehicle = Vehicle(initial_speed_mps=5.0)
        vehicle.drivetrain.configured_speed_limit_mps = 5.0

        vehicle.update_state(Controls(motor_torque_request_nm=230.0), 0.01)

        self.assertTrue(isclose(vehicle.speed_mps, 5.0, abs_tol=1e-9))
        self.assertGreater(vehicle.drivetrain.current_motor_torque_nm, 0.0)

    def test_friction_brake_force_request_is_applied_and_clamped(self) -> None:
        vehicle = Vehicle(initial_speed_mps=10.0)
        requested_force_n = 500.0

        vehicle.update_state(
            Controls(friction_brake_force_request_n=requested_force_n),
            0.01,
        )

        self.assertEqual(
            vehicle.brakes.current_force_request_n,
            requested_force_n,
        )
        self.assertEqual(
            vehicle.brakes.current_friction_force_n,
            requested_force_n,
        )

    def test_negative_timestep_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Vehicle().update_state(Controls(), -0.1)

    def test_rotational_inertia_reduces_acceleration(self) -> None:
        vehicle_with_inertia = Vehicle()
        vehicle_without_inertia = Vehicle()
        vehicle_without_inertia.drivetrain.motor_rotor_inertia_kgm2 = 0.0
        vehicle_without_inertia.drivetrain.final_drive_input_inertia_kgm2 = 0.0
        vehicle_without_inertia.drivetrain.final_drive_output_inertia_kgm2 = 0.0
        vehicle_without_inertia.drivetrain.driven_wheel_inertia_kgm2 = 0.0
        controls = Controls(motor_torque_request_nm=150.0)

        vehicle_with_inertia.update_state(controls, 0.01)
        vehicle_without_inertia.update_state(controls, 0.01)

        self.assertLess(
            vehicle_with_inertia.longitudinal_acceleration_mps2,
            vehicle_without_inertia.longitudinal_acceleration_mps2,
        )

    def test_rotational_inertia_reduces_braking_deceleration(self) -> None:
        vehicle_with_inertia = Vehicle(initial_speed_mps=10.0)
        vehicle_without_inertia = Vehicle(initial_speed_mps=10.0)
        vehicle_without_inertia.drivetrain.motor_rotor_inertia_kgm2 = 0.0
        vehicle_without_inertia.drivetrain.final_drive_input_inertia_kgm2 = 0.0
        vehicle_without_inertia.drivetrain.final_drive_output_inertia_kgm2 = 0.0

        deceleration_with_inertia = (
            vehicle_with_inertia.brakes.maximum_deceleration_mps2(
                vehicle_with_inertia,
                10.0,
                0.0,
                vehicle_with_inertia.gravity_mps2,
                vehicle_with_inertia.air_density_kgpm3,
            )
        )
        deceleration_without_inertia = (
            vehicle_without_inertia.brakes.maximum_deceleration_mps2(
                vehicle_without_inertia,
                10.0,
                0.0,
                vehicle_without_inertia.gravity_mps2,
                vehicle_without_inertia.air_density_kgpm3,
            )
        )

        self.assertLess(
            deceleration_with_inertia,
            deceleration_without_inertia,
        )
