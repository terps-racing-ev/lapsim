"""End-to-end coverage for the points-producing event API."""

from math import pi
from unittest import TestCase

from lapsim import (
    AccelerationConfig,
    ConstantControlsProfile,
    Controls,
    EnduranceRunConfig,
    FSAEEnduranceEfficiencyScoring,
    FSAE_2026_MI_ACCELERATION_SCORING,
    FSAE_2026_MI_SKIDPAD_SCORING,
    SkidpadConfig,
    SpatialTrack,
    simulate_acceleration,
    simulate_endurance,
    simulate_skidpad,
)
from lapsim.courses.track import Curve, Track
from vehicle_model import Vehicle


def open_straight(length_m: float = 75.0) -> SpatialTrack:
    cell_count = 15
    return SpatialTrack.from_cells(
        cell_length_m=(length_m / cell_count,) * cell_count,
        curvature_per_m=(0.0,) * cell_count,
        closed=False,
    )


def closed_circle(radius_m: float = 25.0) -> SpatialTrack:
    return SpatialTrack.from_track(
        Track.from_segments([Curve(radius_m, 2.0 * pi)]),
        maximum_cell_length_m=5.0,
    )


class TimedEventScoringTests(TestCase):
    def test_reproduces_2026_michigan_acceleration_result(self) -> None:
        score = FSAE_2026_MI_ACCELERATION_SCORING.score(3.716, completed=True)

        self.assertAlmostEqual(score.points, 98.54, places=2)
        self.assertEqual(
            FSAE_2026_MI_ACCELERATION_SCORING.score(5.6, completed=True).points,
            4.5,
        )
        self.assertEqual(
            FSAE_2026_MI_ACCELERATION_SCORING.score(None, completed=False).points,
            0.0,
        )

    def test_reproduces_2026_michigan_skidpad_result(self) -> None:
        score = FSAE_2026_MI_SKIDPAD_SCORING.score(4.851, completed=True)

        self.assertAlmostEqual(score.points, 69.39, places=2)
        self.assertEqual(
            FSAE_2026_MI_SKIDPAD_SCORING.score(6.0, completed=True).points,
            3.5,
        )


class EventSimulationTests(TestCase):
    def test_acceleration_returns_points_metrics_and_component_telemetry(self) -> None:
        result = simulate_acceleration(
            Vehicle(),
            open_straight(),
            ConstantControlsProfile(Controls(motor_torque_request_nm=230.0)),
            config=AccelerationConfig(starting_speed_mps=0.0),
        )

        self.assertTrue(result.completed, result.failure_reason)
        self.assertEqual(result.event, "acceleration")
        self.assertGreater(result.points, 4.5)
        self.assertEqual(result.completed_laps, 1)
        self.assertGreater(result.energy_kwh, 0.0)
        self.assertGreater(result.telemetry.sample_count, 0)
        self.assertIn("motor.speed_rpm", result.telemetry)
        self.assertIn("battery.terminal_voltage_v", result.telemetry)
        self.assertIn("event.cumulative_net_energy_j", result.telemetry)

    def test_skidpad_scores_only_after_warmup_and_uses_track_steering(self) -> None:
        track = closed_circle()
        result = simulate_skidpad(
            Vehicle(),
            track,
            ConstantControlsProfile(Controls(motor_torque_request_nm=5.0)),
            config=SkidpadConfig(
                warmup_laps=1,
                scored_laps=1,
                starting_speed_mps=5.0,
            ),
        )

        self.assertTrue(result.completed, result.failure_reason)
        self.assertEqual(result.event, "skidpad")
        self.assertEqual(result.completed_laps, 2)
        self.assertEqual(len(result.lap_times_s), 2)
        self.assertAlmostEqual(result.scoring_time_s, result.lap_times_s[-1])
        self.assertGreater(result.telemetry.sample_count, 0)
        requested_curvature = result.telemetry["vehicle.requested_curvature_per_m"]
        self.assertTrue(all(value > 0.0 for value in requested_curvature))

    def test_endurance_wraps_existing_physics_in_common_event_result(self) -> None:
        track = closed_circle()
        scoring = FSAEEnduranceEfficiencyScoring(
            event_laps=1,
            endurance_minimum_time_s=10.0,
            endurance_maximum_time_s=100.0,
            fastest_average_lap_time_s=20.0,
            minimum_adjusted_energy_per_lap_kg=0.001,
            maximum_adjusted_energy_per_lap_kg=1.0,
            efficiency_factor_minimum=0.001,
            efficiency_factor_maximum=1.0,
            driver_change_lap=1,
        )
        result = simulate_endurance(
            Vehicle(),
            track,
            ConstantControlsProfile(Controls(motor_torque_request_nm=5.0)),
            config=EnduranceRunConfig(
                laps=1,
                starting_speed_mps=5.0,
                maximum_brake_pressure_psi=100.0,
            ),
            scoring=scoring,
        )

        self.assertTrue(result.completed, result.failure_reason)
        self.assertEqual(result.event, "endurance")
        self.assertGreater(result.points, 0.0)
        self.assertIn("endurance_points", result.point_breakdown)
        self.assertIn("efficiency_points", result.point_breakdown)
        self.assertGreater(result.telemetry.sample_count, 0)
        self.assertIn("energy.cumulative_net_j", result.telemetry)
        self.assertIn("event.lap_index", result.telemetry)
        self.assertIn("event.cumulative_net_energy_j", result.telemetry)

    def test_failed_event_still_exposes_a_telemetry_object(self) -> None:
        result = simulate_acceleration(
            Vehicle(),
            open_straight(10.0),
            ConstantControlsProfile(Controls()),
        )

        self.assertFalse(result.completed)
        self.assertEqual(result.points, 0.0)
        self.assertIsNotNone(result.failure_reason)
        self.assertEqual(result.telemetry.sample_count, 0)
