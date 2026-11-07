"""Analytic braking-distance checks for the path entry-speed search."""

from math import sqrt
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from lapsim.courses.spatial_track import SpatialTrack
from lapsim.solvers.path_constraints import PathConstraintSolver


class PathConstraintTests(TestCase):
    def test_braking_bound_propagates_across_lap_boundary(self) -> None:
        track = SpatialTrack.from_cells(
            cell_length_m=(1.0,) * 4,
            curvature_per_m=(0.0, 0.0, 0.1, 0.0),
            closed=True,
        )
        vehicle = SimpleNamespace(
            chassis=SimpleNamespace(wheelbase_m=1.5), validate=lambda: None
        )

        def advance(candidate, controls, distance_m):
            candidate.speed_mps = sqrt(max(0.0, candidate.speed_mps**2 - 10 * distance_m))

        with patch(
            "lapsim.solvers.path_constraints.advance_vehicle_by_distance", advance
        ), patch.object(
            PathConstraintSolver, "_steady_state_speed_limit",
            side_effect=lambda car, curvature: 5.0 if curvature else 30.0,
        ):
            result = PathConstraintSolver().solve(track, vehicle)
        for actual, expected in zip(
            result.braking_speed_ceiling_mps, (sqrt(45), sqrt(35), 5.0, sqrt(55))
        ):
            self.assertLessEqual(actual, expected + 1e-10)
            self.assertAlmostEqual(actual, expected, places=6)

    def test_entry_speed_matches_constant_deceleration_distance(self) -> None:
        vehicle = SimpleNamespace(chassis=SimpleNamespace(wheelbase_m=1.5))
        deceleration_mps2 = 5.0

        def advance(candidate, controls, distance_m):
            candidate.speed_mps = sqrt(
                max(0.0, candidate.speed_mps**2 - 2 * deceleration_mps2 * distance_m)
            )

        with patch(
            "lapsim.solvers.path_constraints.advance_vehicle_by_distance", advance
        ):
            for next_speed, local_limit, distance in (
                (10.0, 20.0, 5.0),
                (10.0, 11.0, 5.0),
                (10.0, 8.0, 5.0),
                (3.0, 25.0, 30.0),
            ):
                with self.subTest(next_speed=next_speed, local_limit=local_limit):
                    actual = PathConstraintSolver()._maximum_entry_speed_mps(
                        vehicle=vehicle,
                        next_speed_mps=next_speed,
                        local_speed_limit_mps=local_limit,
                        curvature_per_m=0.0,
                        cell_length_m=distance,
                    )
                    expected = min(
                        local_limit, sqrt(next_speed**2 + 2 * deceleration_mps2 * distance)
                    )
                    self.assertLessEqual(actual, expected + 1e-10)
                    self.assertAlmostEqual(actual, expected, places=6)
