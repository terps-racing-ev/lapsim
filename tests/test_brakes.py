"""Tests for brake pressure-to-torque parameters."""

from unittest import TestCase

from vehicle_model import Brakes, Tire, Vehicle


class BrakeParameterTests(TestCase):
    def test_axle_gains_convert_to_equivalent_vehicle_force(self) -> None:
        brakes = Brakes()

        self.assertAlmostEqual(
            brakes.front_torque_per_pressure_nm_per_psi,
            1.14436624424632,
        )
        self.assertAlmostEqual(
            brakes.rear_torque_per_pressure_nm_per_psi,
            0.609098162019589,
        )
        self.assertAlmostEqual(
            brakes.equivalent_vehicle_force_per_pressure_n_per_psi(0.2032),
            8.62925396784403,
        )
        self.assertAlmostEqual(brakes.front_brake_force_fraction, 0.6526315790)

    def test_axle_pressure_force_mapping_round_trips(self) -> None:
        brakes = Brakes(pressure_deadband_psi=5.0)
        radius_m = 0.2032

        requested_forces_n = brakes.axle_force_requests_from_pressures_n(
            55.0,
            35.0,
            radius_m,
        )
        recovered_pressures_psi = brakes.axle_pressures_for_force_requests_psi(
            *requested_forces_n,
            radius_m,
        )

        self.assertAlmostEqual(recovered_pressures_psi[0], 55.0)
        self.assertAlmostEqual(recovered_pressures_psi[1], 35.0)

    def test_braking_slip_relaxation_delays_force_buildup(self) -> None:
        brakes = Brakes(braking_slip_relaxation_length_m=2.0)

        front_force, rear_force, front_slip, rear_slip = (
            brakes.slip_limited_axle_forces_n(
                1_000.0,
                500.0,
                1_000.0,
                500.0,
                10.0,
                0.01,
            )
        )

        self.assertLess(front_force, 1_000.0)
        self.assertLess(rear_force, 500.0)
        self.assertGreater(front_slip, 0.0)
        self.assertGreater(rear_slip, 0.0)

    def test_constant_tire_mu_is_load_independent(self) -> None:
        tire = Tire(constant_friction_coefficient=1.8)

        self.assertEqual(tire.longitudinal_coefficient(100.0), 1.8)
        self.assertEqual(tire.longitudinal_coefficient(1_000.0), 1.8)
        self.assertEqual(tire.lateral_coefficient(500.0), 1.8)

    def test_constant_mu_braking_solver_has_strict_root_bracket(self) -> None:
        vehicle = Vehicle(tire=Tire(constant_friction_coefficient=1.8))

        deceleration_mps2 = vehicle.brakes.maximum_deceleration_mps2(
            vehicle,
            speed_mps=20.0,
            curvature_per_m=0.0,
            gravity_mps2=vehicle.gravity_mps2,
            air_density_kgpm3=vehicle.air_density_kgpm3,
        )

        self.assertGreater(deceleration_mps2, vehicle.gravity_mps2)
