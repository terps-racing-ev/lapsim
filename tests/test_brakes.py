"""Tests for brake pressure-to-torque parameters."""

from unittest import TestCase

from vehicle_model import Brakes, Tire, Vehicle


class BrakeParameterTests(TestCase):
    def test_default_pressure_limit_is_applied_per_axle(self) -> None:
        brakes = Brakes()
        radius_m = 0.2032

        self.assertEqual(brakes.maximum_pressure_psi, 300.0)
        self.assertEqual(
            brakes.axle_force_requests_from_pressures_n(301.0, 1_000.0, radius_m),
            brakes.axle_force_requests_from_pressures_n(300.0, 300.0, radius_m),
        )

    def test_rejects_invalid_pressure_limit(self) -> None:
        for invalid_limit in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(invalid_limit=invalid_limit):
                with self.assertRaises(ValueError):
                    Brakes(maximum_pressure_psi=invalid_limit)

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

    def test_pressure_inversion_stops_at_hardware_limit(self) -> None:
        brakes = Brakes(pressure_force_model="firmware-force-map")

        self.assertEqual(
            brakes.axle_pressures_for_force_requests_psi(1.0e6, 1.0e6, 0.2032),
            (300.0, 300.0),
        )

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
