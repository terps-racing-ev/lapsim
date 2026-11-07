"""Maximum steady circle speed and force-balance regression coverage."""

from math import pi
from unittest import TestCase

from lapsim import (
    ConstantControlsProfile, Controls, SkidpadConfig, SpatialTrack,
    simulate_skidpad,
)
from lapsim.solvers.skidpad import (
    _solve_controls_at_speed, solve_skidpad_operating_point,
)
from vehicle_model import Vehicle


class SkidpadTests(TestCase):
    def test_search_reaches_known_feasible_branch_and_brackets_limit(self):
        vehicle = Vehicle()
        curvature = 1.0 / 9.125
        point = solve_skidpad_operating_point(vehicle, curvature, initial_speed_mps=9.0)

        self.assertTrue(point.feasible)
        # The previous nested control search stopped at 10.51 m/s even though
        # a steady force/yaw balance exists above 10.75 m/s with these tires.
        self.assertGreater(point.speed_mps, 10.75)
        self.assertAlmostEqual(point.vehicle.curvature_per_m, curvature, places=6)
        self.assertLess(abs(point.longitudinal_residual_mps2), 1e-4)
        above_limit = _solve_controls_at_speed(
            vehicle, point.speed_mps + 0.02, curvature, point.controls,
            point.vehicle.body_slip_angle_rad,
        )
        self.assertFalse(above_limit.feasible)
        self.assertEqual(vehicle.speed_mps, 0.0)
        self.assertEqual(vehicle.time_s, 0.0)

    def test_timing_is_circumference_over_speed_for_both_turns(self):
        radius = 9.125
        circumference = 2.0 * pi * radius
        for turn in (-1.0, 1.0):
            with self.subTest(turn=turn):
                track = SpatialTrack.from_cells(
                    cell_length_m=(circumference / 4.0,) * 4,
                    curvature_per_m=(turn / radius,) * 4,
                    closed=True,
                )
                result = simulate_skidpad(
                    Vehicle(), track, ConstantControlsProfile(Controls()),
                    config=SkidpadConfig(warmup_laps=0, starting_speed_mps=9.0),
                )
                self.assertTrue(result.completed, result.failure_reason)
                speed = result.point_breakdown["steady_speed_mps"]
                self.assertAlmostEqual(result.scoring_time_s, circumference / speed)
                self.assertAlmostEqual(result.point_breakdown["steady_radius_m"], radius)
                self.assertTrue(all(
                    abs(k - turn / radius) < 1e-6
                    for k in result.telemetry["vehicle.achieved_curvature_per_m"]
                ))
