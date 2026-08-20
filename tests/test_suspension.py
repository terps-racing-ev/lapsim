"""Tests for quasi-static longitudinal and lateral load transfer."""

from math import isclose
from unittest import TestCase

from vehicle_model import AeroForces, Chassis, Suspension


class SuspensionLoadTransferTests(TestCase):
    def setUp(self) -> None:
        self.suspension = Suspension()
        self.chassis = Chassis()
        self.aero_forces = AeroForces(0.0, 0.0, 0.0, 0.0)

    def loads(self, lateral_acceleration_mps2: float):
        return self.suspension.tire_normal_loads_n(
            mass_kg=300.0,
            gravity_mps2=9.81,
            aero_forces=self.aero_forces,
            chassis=self.chassis,
            lateral_acceleration_mps2=lateral_acceleration_mps2,
        )

    def test_zero_lateral_acceleration_keeps_left_and_right_equal(self) -> None:
        loads = self.loads(0.0)

        self.assertEqual(loads.front_left_n, loads.front_right_n)
        self.assertEqual(loads.rear_left_n, loads.rear_right_n)

    def test_positive_lateral_acceleration_loads_right_side(self) -> None:
        loads = self.loads(9.81)

        self.assertGreater(loads.front_right_n, loads.front_left_n)
        self.assertGreater(loads.rear_right_n, loads.rear_left_n)
        self.assertAlmostEqual(loads.front_axle_n, 0.47 * 300.0 * 9.81)
        self.assertAlmostEqual(loads.rear_axle_n, 0.53 * 300.0 * 9.81)

    def test_roll_moment_distribution_matches_55_percent_front_tlltd(self) -> None:
        loads = self.loads(9.81)
        front_transfer_n = 0.5 * (
            loads.front_right_n - loads.front_left_n
        )
        rear_transfer_n = 0.5 * (
            loads.rear_right_n - loads.rear_left_n
        )
        self.suspension.update_state(loads, timestep_s=0.01)
        self.assertAlmostEqual(
            front_transfer_n / (front_transfer_n + rear_transfer_n),
            0.55,
        )
        self.assertAlmostEqual(
            self.suspension.current_front_roll_stiffness_fraction,
            0.6106209447618399,
        )

    def test_roll_stiffness_sets_body_roll_angle(self) -> None:
        loads = self.loads(9.81)
        self.suspension.update_state(loads, timestep_s=0.01)
        expected_roll_angle_rad = (
            300.0
            * 9.81
            * (
                self.chassis.cg_height_m
                - self.chassis.front_roll_axis_height_m
                - (1.0 - self.chassis.static_front_weight_fraction)
                * (
                    self.chassis.rear_roll_axis_height_m
                    - self.chassis.front_roll_axis_height_m
                )
            )
            / self.suspension.total_roll_stiffness_nm_per_rad
        )

        self.assertTrue(
            isclose(
                self.suspension.current_body_roll_angle_rad,
                expected_roll_angle_rad,
                rel_tol=1e-12,
            )
        )

    def test_negative_lateral_acceleration_loads_left_side(self) -> None:
        loads = self.loads(-9.81)

        self.assertGreater(loads.front_left_n, loads.front_right_n)
        self.assertGreater(loads.rear_left_n, loads.rear_right_n)

    def test_transfer_clamps_at_wheel_lift_without_negative_loads(self) -> None:
        loads = self.loads(100.0)

        self.assertGreaterEqual(min(loads.all_n), 0.0)
        self.assertAlmostEqual(loads.front_axle_n, 0.47 * 300.0 * 9.81)
        self.assertAlmostEqual(loads.rear_axle_n, 0.53 * 300.0 * 9.81)
