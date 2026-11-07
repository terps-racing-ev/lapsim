"""Physical invariants for the quasi-static capacity calculation."""

from pathlib import Path
import sys
from unittest import TestCase

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.performance.plot_gg_envelope import CapacityEnvelope
from vehicle_model import Vehicle


class GGEnvelopeTests(TestCase):
    def ideal_vehicle(self) -> Vehicle:
        vehicle = Vehicle()
        vehicle.tire.constant_friction_coefficient = 1.2
        vehicle.chassis.static_front_weight_fraction = 0.5
        vehicle.chassis.cg_height_m = 1e-9
        vehicle.aero.drag_coefficient = 0.0
        vehicle.aero.lift_coefficient = 0.0
        vehicle.rolling_resistance_coefficient = 0.0
        vehicle.cornering_drag_coefficient = 0.0
        vehicle.brakes.maximum_pressure_psi = 100_000.0
        vehicle.battery.initial_state_of_charge = 0.8
        vehicle.battery.max_charge_power_w = 80_000.0
        vehicle.reset_state()
        return vehicle

    def test_ideal_braking_matches_analytic_friction_ellipse(self) -> None:
        vehicle = self.ideal_vehicle()
        envelope = CapacityEnvelope(vehicle, 40.0, False)
        mu_g = 1.2 * vehicle.gravity_mps2
        for fraction_y in (0.0, 0.3, 0.6, 0.9):
            ay = fraction_y * mu_g
            ax = -(1.0 - fraction_y**2)**0.5 * mu_g * (
                vehicle.mass_kg / vehicle.effective_longitudinal_mass_kg
            )
            self.assertAlmostEqual(min(envelope.constraints(ax, ay)), 0.0, delta=1e-4)
            self.assertLess(min(envelope.constraints(ax * 1.01, ay)), 0.0)

    def test_regen_cannot_exceed_saturated_tire_grip(self) -> None:
        vehicle = self.ideal_vehicle()
        without = CapacityEnvelope(vehicle, 40.0, False).boundary(21)
        with_regen = CapacityEnvelope(vehicle, 40.0, True).boundary(21)
        np.testing.assert_allclose(without, with_regen, atol=1e-7)

    def test_zero_charge_headroom_disables_regen(self) -> None:
        vehicle = self.ideal_vehicle()
        vehicle.battery.max_charge_power_w = 0.0
        without = CapacityEnvelope(vehicle, 40.0, False)
        with_regen = CapacityEnvelope(vehicle, 40.0, True)
        self.assertEqual(with_regen.regen, 0.0)
        np.testing.assert_allclose(without.boundary(21), with_regen.boundary(21))

    def test_regen_boundary_contains_no_regen_boundary(self) -> None:
        vehicle = Vehicle()
        # A restricted hydraulic system leaves rear grip for regen to use.
        vehicle.brakes.maximum_pressure_psi = 300.0
        vehicle.battery.initial_state_of_charge = 0.8
        vehicle.battery.max_charge_power_w = 80_000.0
        vehicle.reset_state()
        without = CapacityEnvelope(vehicle, 40.0, False).boundary(21)
        enabled = CapacityEnvelope(vehicle, 40.0, True)
        for ay, ax in without:
            self.assertGreaterEqual(min(enabled.constraints(
                ax * vehicle.gravity_mps2, ay * vehicle.gravity_mps2,
            )), -0.01)
        with_regen = enabled.boundary(21)
        np.testing.assert_allclose(without[0], with_regen[0], atol=1e-7)
        self.assertLess(with_regen[10, 1], without[10, 1])

    def test_regen_cannot_expand_800_psi_envelope(self) -> None:
        vehicle = Vehicle()
        self.assertEqual(vehicle.brakes.maximum_pressure_psi, 800.0)
        vehicle.battery.initial_state_of_charge = 0.8
        vehicle.battery.max_charge_power_w = 80_000.0
        vehicle.reset_state()
        for speed in range(10, 71, 10):
            with self.subTest(speed_mph=speed):
                without = CapacityEnvelope(vehicle, speed, False).boundary(21)
                with_regen = CapacityEnvelope(vehicle, speed, True).boundary(21)
                np.testing.assert_allclose(without, with_regen, atol=1e-7)
