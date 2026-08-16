"""Official-result regression tests for pluggable endurance scoring."""

from dataclasses import dataclass
from unittest import TestCase

from lapsim.scoring import FSAE_2026_MI6_SCORING


@dataclass(frozen=True)
class Run:
    completed_laps: int
    driving_time_s: float
    pack_energy_kwh: float
    failure_reason: str | None = None


class EnduranceScoringTests(TestCase):
    def test_oregon_result_reproduces_2026_mi6_points(self) -> None:
        score = FSAE_2026_MI6_SCORING.score(Run(22, 1312.281, 5.302))

        self.assertAlmostEqual(score.endurance_points, 275.0, places=6)
        self.assertAlmostEqual(score.efficiency_points, 48.6, delta=0.3)

    def test_pittsburgh_result_reproduces_efficiency_win(self) -> None:
        score = FSAE_2026_MI6_SCORING.score(Run(22, 65.192 * 22.0, 3.263))

        self.assertAlmostEqual(score.efficiency_factor, 0.797, delta=0.002)
        self.assertAlmostEqual(score.efficiency_points, 100.0, delta=0.3)

    def test_driver_change_lap_points_and_partial_efficiency(self) -> None:
        score = FSAE_2026_MI6_SCORING.score(
            Run(11, 75.0 * 11.0, 1.5, failure_reason="DNF after lap 11")
        )

        self.assertEqual(score.endurance_lap_points, 14.0)
        self.assertTrue(score.efficiency_completion_eligible)
        self.assertGreater(score.efficiency_points, 0.0)

    def test_time_or_energy_over_event_limit_gets_zero_efficiency(self) -> None:
        slow = FSAE_2026_MI6_SCORING.score(Run(22, 86.6 * 22.0, 3.0))
        inefficient = FSAE_2026_MI6_SCORING.score(Run(22, 70.0 * 22.0, 7.0))

        self.assertEqual(slow.efficiency_points, 0.0)
        self.assertEqual(inefficient.efficiency_points, 0.0)
