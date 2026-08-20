"""Tests for explicit aerodynamic reference area and coefficients."""

from math import isclose, radians
from unittest import TestCase

from lapsim import Controls
from vehicle_model import Aero, Vehicle


class AeroTests(TestCase):
    def test_default_geometry_and_coefficients_match_supplied_values(self) -> None:
        aero = Aero()

        self.assertTrue(isclose(aero.frontal_area_m2, 0.65799997432))
        self.assertTrue(isclose(aero.drag_coefficient, 3.62 / 2.946))
        self.assertEqual(aero.lift_coefficient, -3.62)
        self.assertTrue(isclose(aero.front_downforce_fraction, 0.5269293255))
        self.assertTrue(isclose(aero.drag_area_m2, 0.808540362199049))
        self.assertTrue(isclose(aero.downforce_area_m2, 2.3819599070384))

    def test_negative_lift_coefficient_produces_positive_downforce(self) -> None:
        aero = Aero()
        air_density_kgpm3 = 1.2
        speed_mps = 10.0
        dynamic_pressure_pa = 0.5 * air_density_kgpm3 * speed_mps**2

        forces = aero.forces_n(speed_mps, air_density_kgpm3)

        self.assertTrue(isclose(forces.drag_n, dynamic_pressure_pa * aero.drag_area_m2))
        self.assertTrue(
            isclose(forces.downforce_n, dynamic_pressure_pa * aero.downforce_area_m2)
        )
        self.assertGreater(forces.downforce_n, 0.0)

    def test_coefficient_area_aliases_update_coefficients(self) -> None:
        aero = Aero(frontal_area_m2=0.5)

        aero.drag_area_m2 = 0.75
        aero.downforce_area_m2 = 1.25

        self.assertEqual(aero.drag_coefficient, 1.5)
        self.assertEqual(aero.lift_coefficient, -2.5)

    def test_body_roll_linearly_reduces_then_clamps_downforce(self) -> None:
        aero = Aero()
        speed_mps = 20.0
        air_density_kgpm3 = 1.2
        level = aero.forces_n(speed_mps, air_density_kgpm3)
        half_degree = aero.forces_n(
            speed_mps, air_density_kgpm3, radians(0.5)
        )
        one_degree = aero.forces_n(
            speed_mps, air_density_kgpm3, radians(-1.0)
        )
        above_limit = aero.forces_n(
            speed_mps, air_density_kgpm3, radians(2.0)
        )

        self.assertEqual(level.downforce_multiplier, 1.0)
        self.assertAlmostEqual(half_degree.downforce_multiplier, 0.75)
        self.assertAlmostEqual(one_degree.downforce_multiplier, 0.50)
        self.assertAlmostEqual(above_limit.downforce_multiplier, 0.50)
        self.assertAlmostEqual(half_degree.downforce_n, 0.75 * level.downforce_n)
        self.assertAlmostEqual(one_degree.downforce_n, 0.50 * level.downforce_n)
        self.assertEqual(above_limit.drag_n, level.drag_n)

    def test_vehicle_aero_uses_suspension_body_roll(self) -> None:
        vehicle = Vehicle()
        roll_angle_rad = radians(1.0)
        elastic_roll_arm_m = vehicle.suspension.elastic_roll_arm_m(vehicle.chassis)
        lateral_acceleration_mps2 = (
            roll_angle_rad
            * vehicle.suspension.total_roll_stiffness_nm_per_rad
            / (vehicle.mass_kg * elastic_roll_arm_m)
        )

        forces = vehicle.aero_forces_n(20.0, lateral_acceleration_mps2)

        self.assertAlmostEqual(forces.body_roll_angle_rad, roll_angle_rad)
        self.assertAlmostEqual(forces.downforce_multiplier, 0.50)

    def test_vehicle_update_commits_matching_aero_and_suspension_roll(self) -> None:
        vehicle = Vehicle(initial_speed_mps=20.0)

        vehicle.update_state(Controls(steering_angle_rad=0.08), 0.1)
        telemetry = vehicle.telemetry_snapshot()

        self.assertAlmostEqual(
            telemetry["aero.body_roll_angle_rad"],
            telemetry["suspension.body_roll_angle_rad"],
        )
        self.assertEqual(telemetry["aero.roll_limit_deg"], 1.0)
        self.assertLess(telemetry["aero.downforce_multiplier"], 1.0)
