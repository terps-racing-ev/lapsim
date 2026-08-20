"""Small end-to-end test across track, vehicle, and both lap solvers."""

from math import pi
from unittest import TestCase

from lapsim import LapTimeSolver, SpeedLimitSolver
from lapsim.courses.track import Curve, Straight, Track
from vehicle_model import Vehicle


class SolverIntegrationTests(TestCase):
    def test_minimum_time_profile_runs_with_composed_vehicle(self) -> None:
        quarter_turn = Curve(radius_m=10.0, span_rad=pi / 2.0)
        track = Track.from_segments(
            [
                Straight(length_m=20.0),
                quarter_turn,
                Straight(length_m=20.0),
                quarter_turn,
                Straight(length_m=20.0),
                quarter_turn,
                Straight(length_m=20.0),
                quarter_turn,
            ]
        )
        vehicle = Vehicle()

        speed_limits = SpeedLimitSolver(vehicle, max_step_m=2.0).solve(track)
        lap = LapTimeSolver(vehicle).solve(
            speed_limits,
            starting_speed_mps=5.0,
        )
        self.assertGreater(lap.lap_time_s, 0.0)
        self.assertEqual(len(lap.speed_mps), len(speed_limits.distance_m))
        self.assertTrue(
            all(
                speed_mps <= limit_mps
                for speed_mps, limit_mps in zip(
                    lap.speed_mps,
                    speed_limits.speed_limit_mps,
                    strict=True,
                )
            )
        )
