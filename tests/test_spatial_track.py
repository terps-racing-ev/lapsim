"""Tests for generic vehicle-independent spatial tracks."""

from math import pi
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from lapsim.courses.spatial_track import SpatialTrack
from lapsim.courses.track import Curve, Straight, Track


class SpatialTrackTests(TestCase):
    def test_discretizes_arbitrary_segment_track_and_closes_geometry(self) -> None:
        track = Track.from_segments(
            [
                Straight(20.0),
                Curve(10.0, pi),
                Straight(20.0),
                Curve(10.0, pi),
            ]
        )

        spatial = SpatialTrack.from_track(track, maximum_cell_length_m=2.0)

        self.assertAlmostEqual(spatial.length_m, track.total_length_m)
        self.assertAlmostEqual(spatial.x_m[0], spatial.x_m[-1], places=9)
        self.assertAlmostEqual(spatial.y_m[0], spatial.y_m[-1], places=9)
        self.assertEqual(len(spatial.curvature_per_m), len(spatial.distance_m) - 1)

    def test_closed_distance_wrap_is_periodic(self) -> None:
        spatial = SpatialTrack.from_cells(
            cell_length_m=(10.0, 10.0),
            curvature_per_m=(0.0, 0.0),
        )

        self.assertAlmostEqual(spatial.wrap_distance_m(25.0), 5.0)

    def test_csv_round_trip_preserves_solver_geometry(self) -> None:
        spatial = SpatialTrack(
            distance_m=(0.0, 1.0, 2.0),
            x_m=(0.0, 1.0, 0.0),
            y_m=(0.0, 0.0, 0.0),
            curvature_per_m=(0.1, -0.1),
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "track.csv"
            spatial.to_csv(path)
            loaded = SpatialTrack.from_csv(path)

        self.assertEqual(loaded, spatial)

    def test_can_convert_cells_to_legacy_segment_track(self) -> None:
        spatial = SpatialTrack(
            distance_m=(0.0, 2.0, 5.0),
            x_m=(0.0, 2.0, 5.0),
            y_m=(0.0, 0.0, 0.0),
            curvature_per_m=(0.0, 0.2),
        )

        legacy = spatial.to_track()

        self.assertAlmostEqual(legacy.total_length_m, spatial.length_m)
        self.assertIsInstance(legacy.segments[0], Straight)
        self.assertIsInstance(legacy.segments[1], Curve)
