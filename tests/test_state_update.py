"""Tests for distance-domain vehicle state updates."""

from math import isclose
from unittest import TestCase

from lapsim import Controls
from vehicle_model.vehicle import Vehicle


class VehicleStateUpdateTests(TestCase):
    def test_spatial_braking_converges_independently_of_previous_acceleration(
        self,
    ) -> None:
        controls = Controls(
            front_brake_pressure_psi=1.0e6,
            rear_brake_pressure_psi=1.0e6,
            steering_angle_rad=0.06,
        )
        accelerating = Vehicle(initial_speed_mps=20.0)
        braking = Vehicle(initial_speed_mps=20.0)
        accelerating.longitudinal_acceleration_mps2 = 5.0
        braking.longitudinal_acceleration_mps2 = -5.0

        accelerating.update_state(controls, 0.5)
        braking.update_state(controls, 0.5)

        self.assertAlmostEqual(accelerating.speed_mps, braking.speed_mps, places=8)

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

        vehicle.update_state(
            Controls(
                front_brake_pressure_psi=1_000_000.0,
                rear_brake_pressure_psi=1_000_000.0,
            ),
            0.001,
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

    def test_distance_step_is_accumulated_and_timestep_is_internal(self) -> None:
        vehicle = Vehicle(initial_speed_mps=10.0)
        controls = Controls(motor_torque_request_nm=10.0)
        distance_step_m = 0.125
        vehicle.update_state(controls, distance_step_m)

        self.assertTrue(isclose(vehicle.distance_m, distance_step_m))
        self.assertTrue(
            isclose(
                vehicle.time_s,
                2.0 * distance_step_m / (10.0 + vehicle.speed_mps),
            )
        )
        self.assertIs(vehicle.current_controls, controls)

    def test_speed_limiter_holds_speed_without_cutting_all_torque(self) -> None:
        vehicle = Vehicle(initial_speed_mps=5.0)
        vehicle.drivetrain.configured_speed_limit_mps = 5.0

        vehicle.update_state(Controls(motor_torque_request_nm=230.0), 0.01)

        self.assertTrue(isclose(vehicle.speed_mps, 5.0, abs_tol=1e-9))
        self.assertGreater(vehicle.drivetrain.current_motor_torque_nm, 0.0)

    def test_brake_pressure_is_converted_to_force_and_clamped(self) -> None:
        vehicle = Vehicle(initial_speed_mps=10.0)
        pressure_psi = 50.0
        front_force_n, rear_force_n = (
            vehicle.brakes.axle_force_requests_from_pressures_n(
                pressure_psi,
                pressure_psi,
                vehicle.tire.rolling_radius_m,
            )
        )
        requested_force_n = front_force_n + rear_force_n

        vehicle.update_state(
            Controls(
                front_brake_pressure_psi=pressure_psi,
                rear_brake_pressure_psi=pressure_psi,
            ),
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

    def test_independent_axle_and_regen_brake_requests_are_retained(self) -> None:
        vehicle = Vehicle(initial_speed_mps=10.0)
        front_pressure_psi, rear_pressure_psi = (
            vehicle.brakes.axle_pressures_for_force_requests_psi(
                400.0,
                200.0,
                vehicle.tire.rolling_radius_m,
            )
        )

        vehicle.update_state(
            Controls(
                front_brake_pressure_psi=front_pressure_psi,
                rear_brake_pressure_psi=rear_pressure_psi,
                rear_regenerative_brake_force_request_n=100.0,
            ),
            0.01,
        )

        self.assertEqual(vehicle.brakes.current_force_request_n, 700.0)
        self.assertEqual(vehicle.brakes.current_front_force_request_n, 400.0)
        self.assertEqual(vehicle.brakes.current_rear_force_request_n, 300.0)

    def test_cornering_drag_adds_speed_dependent_tire_scrub_loss(self) -> None:
        baseline = Vehicle(initial_speed_mps=15.0)
        with_cornering_drag = Vehicle(
            initial_speed_mps=15.0,
            cornering_drag_coefficient=0.1,
        )
        controls = Controls(steering_angle_rad=0.1)

        baseline.update_state(controls, 0.1)
        with_cornering_drag.update_state(controls, 0.1)

        self.assertGreater(with_cornering_drag.current_cornering_drag_force_n, 0.0)
        self.assertLess(with_cornering_drag.speed_mps, baseline.speed_mps)

    def test_nonpositive_distance_step_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Vehicle().update_state(Controls(), -0.1)

    def test_rest_without_drive_cannot_traverse_a_spatial_cell(self) -> None:
        vehicle = Vehicle()

        with self.assertRaisesRegex(ValueError, "cannot traverse"):
            vehicle.update_state(Controls(), 0.1)

        self.assertEqual(vehicle.distance_m, 0.0)
        self.assertEqual(vehicle.time_s, 0.0)

    def test_stopping_before_cell_end_is_rejected_without_partial_commit(self) -> None:
        vehicle = Vehicle(initial_speed_mps=1.0)

        with self.assertRaisesRegex(ValueError, "stops before"):
            vehicle.update_state(
                Controls(
                    front_brake_pressure_psi=1_000_000.0,
                    rear_brake_pressure_psi=1_000_000.0,
                ),
                10.0,
            )

        self.assertEqual(vehicle.distance_m, 0.0)
        self.assertEqual(vehicle.time_s, 0.0)

    def test_rotational_inertia_reduces_acceleration(self) -> None:
        vehicle_with_inertia = Vehicle()
        vehicle_without_inertia = Vehicle()
        vehicle_without_inertia.drivetrain.motor.rotor_inertia_kgm2 = 0.0
        vehicle_without_inertia.drivetrain.chain_drive.input_inertia_kgm2 = 0.0
        vehicle_without_inertia.drivetrain.chain_drive.output_inertia_kgm2 = 0.0
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
        vehicle_without_inertia.drivetrain.motor.rotor_inertia_kgm2 = 0.0
        vehicle_without_inertia.drivetrain.chain_drive.input_inertia_kgm2 = 0.0
        vehicle_without_inertia.drivetrain.chain_drive.output_inertia_kgm2 = 0.0

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
