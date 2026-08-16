"""Tests for the default lateral and longitudinal tire models."""

from math import radians
from unittest import TestCase

from vehicle_model import Pacejka61LateralModel, Tire


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

    def test_tire_uses_pacejka_only_for_lateral_force(self) -> None:
        tire = Tire()

        self.assertAlmostEqual(tire.lateral_force_capacity_n(1080.0), 2581.2)
        self.assertAlmostEqual(
            tire.pure_lateral_force_n(1080.0, radians(10.0)),
            -2383.393727434875,
        )
        self.assertAlmostEqual(tire.longitudinal_coefficient(1080.0), 1.43220634145)

    def test_legacy_lateral_lookup_remains_available_explicitly(self) -> None:
        tire = Tire(pacejka_lateral=None)

        self.assertAlmostEqual(tire.lateral_coefficient(1080.0), 1.43220634145)
