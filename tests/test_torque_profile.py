"""Tests for vehicle-independent normalized torque profiles."""

from unittest import TestCase

from lapsim.courses.spatial_track import SpatialTrack
from lapsim.optimization.torque_profile import UniformPeriodicTorqueParameterization


class TorqueProfileTests(TestCase):
    def test_uniform_parameterization_interpolates_across_periodic_seam(self) -> None:
        track = SpatialTrack.from_cells(
            cell_length_m=(25.0, 25.0, 25.0, 25.0),
            curvature_per_m=(0.0, 0.0, 0.0, 0.0),
        )
        profile = UniformPeriodicTorqueParameterization(4).build(
            (0.0, 1.0, 0.0, 1.0), track
        )

        self.assertAlmostEqual(profile.request_fraction(12.5), 0.5)
        self.assertAlmostEqual(profile.request_fraction(87.5), 0.5)
        self.assertAlmostEqual(profile.request_fraction(112.5), 0.5)
