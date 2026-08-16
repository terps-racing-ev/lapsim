"""Tiny synthetic smoke test for the generic optimizer composition."""

from unittest import TestCase

from lapsim.endurance import EnduranceRunConfig
from lapsim.optimizer import (
    EnduranceTorqueOptimizer,
    LocalPolishConfig,
    OptimizationConfig,
)
from lapsim.scoring import FSAEEnduranceEfficiencyScoring
from lapsim.spatial_track import SpatialTrack
from lapsim.torque_profile import UniformPeriodicTorqueParameterization
from vehicle_model import Vehicle


class EnduranceOptimizerTests(TestCase):
    def test_optimizes_injected_track_vehicle_factory_and_scoring(self) -> None:
        # A torque-only optimizer deliberately has no automatic brake driver.
        # Use a straight closed smoke-test path so torque alone is sufficient.
        track = SpatialTrack.from_cells(
            cell_length_m=(20.0, 20.0, 20.0, 20.0),
            curvature_per_m=(0.0, 0.0, 0.0, 0.0),
        )
        scoring = FSAEEnduranceEfficiencyScoring(
            event_laps=2,
            endurance_minimum_time_s=10.0,
            endurance_maximum_time_s=14.5,
            fastest_average_lap_time_s=5.0,
            minimum_adjusted_energy_per_lap_kg=0.005,
            maximum_adjusted_energy_per_lap_kg=0.2,
            efficiency_factor_minimum=0.1,
            efficiency_factor_maximum=0.9,
            driver_change_lap=1,
        )
        optimizer = EnduranceTorqueOptimizer(
            vehicle_factory=Vehicle,
            track=track,
            scoring_model=scoring,
            run_config=EnduranceRunConfig(laps=2, starting_speed_mps=5.0),
            parameterization=UniformPeriodicTorqueParameterization(2),
        )

        result = optimizer.optimize(
            OptimizationConfig(
                maximum_iterations=1,
                population_size_multiplier=2,
                polish=False,
            )
        )

        self.assertEqual(len(result.variables), 2)
        self.assertEqual(result.run.completed_laps, 2)
        self.assertGreater(result.objective_evaluations, 0)
        self.assertIsNotNone(result.run.telemetry)

        polished = optimizer.polish_from(
            result.variables,
            LocalPolishConfig(maximum_evaluations=8),
        )
        self.assertEqual(len(polished.variables), 2)
        self.assertGreater(polished.objective_evaluations, 0)
        self.assertIsNotNone(polished.run.telemetry)
