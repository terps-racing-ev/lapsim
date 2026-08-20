"""Tests for distance-indexed recorded channels."""

from unittest import TestCase

import numpy as np

from lapsim.courses.spatial import SpatialCoordinate


class SpatialCoordinateTests(TestCase):
    def test_zero_order_hold_changes_at_recorded_station(self) -> None:
        station = SpatialCoordinate.from_samples((10.0, 11.0, 13.0))

        values = station.zero_order_hold(
            (20.0, 40.0, 60.0), np.array([0.0, 0.999, 1.0, 2.5])
        )

        np.testing.assert_allclose(values, (20.0, 20.0, 40.0, 40.0))

    def test_curvature_interpolates_linearly(self) -> None:
        station = SpatialCoordinate.from_samples((5.0, 7.0, 9.0))

        self.assertAlmostEqual(float(station.interpolate((0.0, 0.2, 0.4), 1.0)), 0.1)

    def test_repeated_and_small_backward_samples_keep_latest_value(self) -> None:
        station = SpatialCoordinate.from_samples((20.0, 20.0, 21.0, 20.99, 22.0))

        np.testing.assert_allclose(station.distance_m, (0.0, 1.0, 2.0))
        np.testing.assert_allclose(
            station.values((1.0, 2.0, 3.0, 4.0, 5.0)), (2.0, 4.0, 5.0)
        )

    def test_material_distance_reversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "materially decreases"):
            SpatialCoordinate.from_samples((0.0, 1.0, 0.5, 2.0))

    def test_queries_outside_recorded_distance_are_rejected(self) -> None:
        station = SpatialCoordinate.from_samples((0.0, 1.0, 2.0))

        with self.assertRaisesRegex(ValueError, "outside"):
            station.interpolate((0.0, 1.0, 2.0), 2.1)
        with self.assertRaisesRegex(ValueError, "outside"):
            station.zero_order_hold((0.0, 1.0, 2.0), -0.1)

    def test_initial_and_final_station_are_valid_queries(self) -> None:
        station = SpatialCoordinate.from_samples((4.0, 5.0, 6.0))

        np.testing.assert_allclose(
            station.interpolate((10.0, 20.0, 30.0), (0.0, 2.0)),
            (10.0, 30.0),
        )
        np.testing.assert_allclose(
            station.zero_order_hold((10.0, 20.0, 30.0), (0.0, 2.0)),
            (10.0, 30.0),
        )
