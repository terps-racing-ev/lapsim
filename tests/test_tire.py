"""Tests for the default lateral and longitudinal tire models."""

from math import radians
from unittest import TestCase

from lapsim import Controls
from vehicle_model import Pacejka61LateralModel, Tire, TireNormalLoads, Vehicle


class PacejkaLateralTests(TestCase):
    def test_matlab_reference_values_at_nominal_conditions(self) -> None:
        model = Pacejka61LateralModel()

        self.assertAlmostEqual(model.nominal_load_n, 1080.0)
        self.assertAlmostEqual(model.nominal_pressure_pa, 98_000.0)
        self.assertAlmostEqual(
            model.force_n(1080.0, radians(-10.0)),
            2396.101601414008,
        )
        self.assertAlmostEqual(
            model.force_n(1080.0, radians(10.0)),
            -2383.393727434875,
        )
        self.assertAlmostEqual(model.peak_force_n(1080.0), 2581.2)

    def test_force_is_zero_when_tire_is_unloaded(self) -> None:
        model = Pacejka61LateralModel()

        self.assertEqual(model.force_n(0.0, radians(10.0)), 0.0)
        self.assertEqual(model.peak_force_n(-1.0), 0.0)

    def test_matlab_reference_with_pressure_and_camber(self) -> None:
        model = Pacejka61LateralModel()

        self.assertAlmostEqual(
            model.force_n(
                800.0,
                radians(7.0),
                camber_angle_rad=radians(-2.0),
                inflation_pressure_pa=90_000.0,
            ),
            -1839.778929099007,
        )

    def test_peak_coefficient_decreases_with_load(self) -> None:
        model = Pacejka61LateralModel()

        low_load_mu = model.peak_force_n(500.0) / 500.0
        high_load_mu = model.peak_force_n(1000.0) / 1000.0

        self.assertGreater(low_load_mu, high_load_mu)

    def test_default_lateral_force_uses_longitudinal_coefficients(self) -> None:
        tire = Tire()

        self.assertIsNone(tire.pacejka_lateral)
        self.assertAlmostEqual(
            tire.lateral_coefficient(1080.0),
            tire.longitudinal_coefficient(1080.0),
        )
        self.assertAlmostEqual(tire.lateral_force_capacity_n(1080.0), 1546.782848766)

    def test_pacejka_lateral_model_remains_available_explicitly(self) -> None:
        tire = Tire(pacejka_lateral=Pacejka61LateralModel())

        self.assertAlmostEqual(tire.lateral_force_capacity_n(1080.0), 2581.2)
        self.assertAlmostEqual(
            tire.pure_lateral_force_n(1080.0, radians(10.0)),
            -2383.393727434875,
        )
        self.assertAlmostEqual(tire.longitudinal_coefficient(1080.0), 1.43220634145)

    def test_default_lateral_lookup_uses_load_sensitivity(self) -> None:
        tire = Tire()

        self.assertAlmostEqual(tire.lateral_coefficient(1080.0), 1.43220634145)


class FourCornerTireTests(TestCase):
    loads = TireNormalLoads(500.0, 900.0, 700.0, 1_100.0)

    def test_each_tire_resolves_its_own_load_force_capacity_and_slip(self) -> None:
        tire = Tire()
        states = tire.calculate_forces(
            self.loads,
            total_lateral_force_n=1_200.0,
            drive_force_request_n=1_000.0,
            front_brake_force_request_n=300.0,
            rear_brake_force_request_n=200.0,
            vehicle_speed_mps=10.0,
            timestep_s=0.01,
        )

        self.assertEqual(
            tuple(state.normal_load_n for state in states.all),
            self.loads.all_n,
        )
        self.assertAlmostEqual(states.lateral_force_n, 1_200.0)
        self.assertLessEqual(states.drive_force_n, 1_000.0)
        self.assertLessEqual(states.braking_force_n, 500.0)
        self.assertTrue(all(state.force_utilization <= 1.0 for state in states.all))
        self.assertNotEqual(states.rear_left.slip_ratio, states.rear_right.slip_ratio)

    def test_braking_relaxation_is_owned_and_applied_by_each_tire(self) -> None:
        tire = Tire(longitudinal_slip_relaxation_length_m=2.0)
        states = tire.calculate_forces(
            self.loads,
            total_lateral_force_n=0.0,
            drive_force_request_n=0.0,
            front_brake_force_request_n=1_000.0,
            rear_brake_force_request_n=500.0,
            vehicle_speed_mps=10.0,
            timestep_s=0.01,
        )

        self.assertLess(states.front_braking_force_n, 1_000.0)
        self.assertLess(states.rear_braking_force_n, 500.0)
        self.assertTrue(all(state.slip_ratio < 0.0 for state in states.all))

    def test_each_brake_force_is_clipped_by_its_own_contact_patch(self) -> None:
        states = Tire().calculate_forces(
            self.loads,
            total_lateral_force_n=0.0,
            drive_force_request_n=0.0,
            front_brake_force_request_n=10_000.0,
            rear_brake_force_request_n=10_000.0,
            vehicle_speed_mps=10.0,
            timestep_s=0.01,
        )

        self.assertTrue(
            all(
                state.braking_force_n == state.longitudinal_capacity_n
                for state in states.all
            )
        )
        self.assertNotEqual(
            states.front_left.braking_force_n,
            states.front_right.braking_force_n,
        )

    def test_tire_slip_sets_vehicle_motor_speed_and_power(self) -> None:
        slipping = Vehicle(initial_speed_mps=10.0)
        no_slip = Vehicle(
            initial_speed_mps=10.0,
            tire=Tire(peak_longitudinal_slip_ratio=0.0),
        )
        controls = Controls(motor_torque_request_nm=100.0)

        slipping.update_state(controls, 0.01)
        no_slip.update_state(controls, 0.01)

        self.assertGreater(slipping.tire.current_states.driven_slip_ratio, 0.0)
        self.assertGreater(
            slipping.drivetrain.current_motor_speed_rpm,
            no_slip.drivetrain.current_motor_speed_rpm,
        )
        self.assertGreater(
            slipping.battery.current_power_w,
            no_slip.battery.current_power_w,
        )

    def test_vehicle_force_balance_is_the_sum_of_all_tires(self) -> None:
        vehicle = Vehicle(initial_speed_mps=12.0)
        vehicle.update_state(
            Controls(motor_torque_request_nm=80.0, steering_angle_rad=0.08),
            0.1,
        )
        states = vehicle.tire.current_states

        self.assertAlmostEqual(vehicle.current_drive_force_n, states.drive_force_n)
        self.assertAlmostEqual(
            vehicle.current_friction_braking_force_n,
            states.braking_force_n,
        )
        self.assertAlmostEqual(
            states.longitudinal_force_n,
            sum(state.longitudinal_force_n for state in states.all),
        )
        self.assertAlmostEqual(
            states.lateral_force_n,
            sum(state.lateral_force_n for state in states.all),
        )
        self.assertAlmostEqual(
            vehicle.current_lateral_force_n,
            states.lateral_force_n,
        )

    def test_turn_direction_changes_the_loaded_side(self) -> None:
        left = Vehicle(initial_speed_mps=12.0)
        right = Vehicle(initial_speed_mps=12.0)
        left.update_state(Controls(steering_angle_rad=0.08), 0.1)
        right.update_state(Controls(steering_angle_rad=-0.08), 0.1)

        self.assertGreater(
            left.tire.current_states.front_right.normal_load_n,
            left.tire.current_states.front_left.normal_load_n,
        )
        self.assertGreater(
            right.tire.current_states.front_left.normal_load_n,
            right.tire.current_states.front_right.normal_load_n,
        )
        self.assertGreater(left.current_lateral_force_n, 0.0)
        self.assertLess(right.current_lateral_force_n, 0.0)
