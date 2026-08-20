"""Tests for track representation and discretization."""

from math import isclose, pi
from unittest import TestCase

from lapsim.courses.track import Curve, Straight, Track


class TrackTests(TestCase):
    def test_track_can_be_built_directly_from_segments(self) -> None:
        track = Track.from_segments(
            [
                Straight(length_m=10.0),
                Curve(radius_m=5.0, span_rad=pi / 2.0),
            ]
        )

        self.assertEqual(len(track.segments), 2)
        self.assertTrue(
            isclose(track.total_length_m, 10.0 + 5.0 * pi / 2.0)
        )

    def test_direct_track_requires_at_least_one_segment(self) -> None:
        with self.assertRaises(ValueError):
            Track.from_segments([])
