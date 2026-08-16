"""Tests for explicit aerodynamic reference area and coefficients."""

from math import isclose
from unittest import TestCase

from vehicle_model import Aero


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
