"""Regression tests for the coastdown road-load fit."""

from unittest import TestCase, skipUnless

import numpy as np

from vehicle_model import Vehicle

try:
    from analysis.testing.coastdown.plot_vcu_speed import (
        apparent_drag_from_quadratic_coefficient,
        fit_road_load_acceleration,
    )
except ImportError:  # asammdf is an optional dependency
    apparent_drag_from_quadratic_coefficient = None
    fit_road_load_acceleration = None


@skipUnless(fit_road_load_acceleration is not None, "asammdf is not installed")
class CoastdownFitTests(TestCase):
    def test_fits_constant_linear_and_quadratic_terms(self) -> None:
        speed = np.linspace(1.0, 16.0, 100)
        acceleration = -0.3 - 0.04 * speed - 0.002 * speed**2

        fitted, intercept, linear, quadratic, r_squared = fit_road_load_acceleration(
            speed, acceleration, fixed_intercept_mps2=-0.1
        )

        np.testing.assert_allclose(fitted, acceleration, atol=1e-12)
        self.assertAlmostEqual(intercept, -0.3)
        self.assertAlmostEqual(linear, -0.04)
        self.assertAlmostEqual(quadratic, -0.002)
        self.assertAlmostEqual(r_squared, 1.0)

    def test_rejects_insufficient_speed_variation(self) -> None:
        with self.assertRaisesRegex(ValueError, "insufficient variation"):
            fit_road_load_acceleration(
                np.ones(10), np.full(10, -0.5)
            )

    def test_quadratic_conversion_removes_downforce_rolling_loss(self) -> None:
        vehicle = Vehicle()
        expected_drag_area = (
            vehicle.aero.drag_coefficient * vehicle.aero.frontal_area_m2
        )
        combined_area = (
            expected_drag_area
            + vehicle.rolling_resistance_coefficient
            * -vehicle.aero.lift_coefficient
            * vehicle.aero.frontal_area_m2
        )
        quadratic = (
            -0.5
            * vehicle.air_density_kgpm3
            * combined_area
            / vehicle.effective_longitudinal_mass_kg
        )

        drag_area, drag_coefficient = apparent_drag_from_quadratic_coefficient(
            quadratic
        )

        self.assertAlmostEqual(drag_area, expected_drag_area)
        self.assertAlmostEqual(drag_coefficient, vehicle.aero.drag_coefficient)
