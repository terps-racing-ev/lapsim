"""Tests for explicit aerodynamic reference area and coefficients."""

from math import isclose, radians
from unittest import TestCase

from lapsim import Controls
from vehicle_model import ActiveAero, Aero, Vehicle


class AeroTests(TestCase):
    def test_default_geometry_and_coefficients_match_supplied_values(self) -> None:
        aero = Aero()

        self.assertTrue(isclose(aero.frontal_area_m2, 0.983996414))
        self.assertTrue(isclose(aero.drag_coefficient, 1.6048838348388528))
        self.assertTrue(isclose(aero.lift_coefficient, -2.4206997842152696))
        self.assertTrue(isclose(aero.front_downforce_fraction, 0.5269293255))
        self.assertTrue(isclose(aero.drag_area_m2, 1.579199938368))
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

    def test_active_aero_automatically_reduces_drag_and_downforce_on_straights(
        self,
    ) -> None:
        baseline = Vehicle()
        active = Vehicle.with_active_aero()

        baseline_forces = baseline.aero_forces_n(
            20.0,
            curvature_per_m=0.0,
        )
        straight_forces = active.aero_forces_n(
            20.0,
            curvature_per_m=0.0,
        )
        corner_forces = active.aero_forces_n(
            20.0,
            lateral_acceleration_mps2=20.0**2 * 0.01,
            curvature_per_m=0.01,
        )

        self.assertAlmostEqual(straight_forces.drag_n, 0.70 * baseline_forces.drag_n)
        self.assertAlmostEqual(
            straight_forces.downforce_n,
            0.70 * baseline_forces.downforce_n,
        )
        self.assertTrue(straight_forces.low_drag_mode_active)
        self.assertAlmostEqual(corner_forces.drag_n, baseline_forces.drag_n)
        self.assertFalse(corner_forces.low_drag_mode_active)

    def test_active_aero_configuration_can_be_forced(self) -> None:
        aero = ActiveAero()

        aero.set_deployment_mode("low_drag")
        forced_low_drag = aero.select_configuration(0.1)
        aero.set_deployment_mode("high_downforce")
        forced_high_downforce = aero.select_configuration(0.0)

        self.assertTrue(forced_low_drag)
        self.assertFalse(forced_high_downforce)

    def test_active_aero_vehicle_factory_adds_three_pounds(self) -> None:
        baseline = Vehicle()
        active = Vehicle.with_active_aero()

        self.assertIsInstance(active.aero, ActiveAero)
        self.assertAlmostEqual(active.mass_kg - baseline.mass_kg, 1.36077711)

    def test_active_aero_state_is_reported_in_vehicle_telemetry(self) -> None:
        vehicle = Vehicle.with_active_aero(initial_speed_mps=20.0)

        vehicle.update_state(Controls(), 0.1)
        telemetry = vehicle.telemetry_snapshot()

        self.assertEqual(telemetry["aero.active_aero.enabled"], 1.0)
        self.assertEqual(
            telemetry["aero.active_aero.low_drag_mode_active"],
            1.0,
        )
        self.assertAlmostEqual(
            telemetry["aero.effective_drag_coefficient"],
            1.123418684387197,
        )
