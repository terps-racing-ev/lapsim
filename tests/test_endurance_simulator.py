"""End-to-end tests for prescribed-path stateful endurance simulation."""

from math import atan, pi
from unittest import TestCase

from lapsim.controls import Controls
from lapsim.endurance import EnduranceRunConfig, EnduranceSimulator
from lapsim.path_constraints import PathConstraintSolver
from lapsim.spatial_track import SpatialTrack
from lapsim.track import Curve, Track
from lapsim.torque_profile import UniformPeriodicTorqueParameterization
from vehicle_model import Vehicle


def small_closed_track() -> SpatialTrack:
    return SpatialTrack.from_track(
        Track.from_segments([Curve(25.0, 2.0 * pi)]),
        maximum_cell_length_m=5.0,
    )


class ConstantDriverControls:
    def controls_at(self, lap_distance_m: float):
        del lap_distance_m
        return Controls(
            motor_torque_request_nm=5.0,
            front_brake_pressure_psi=1.0,
            rear_brake_pressure_psi=1.0,
            steering_angle_rad=atan(1.0 / 25.0 * 1.55),
        )


class ConstantUnsafeControls:
    def controls_at(self, lap_distance_m: float):
        del lap_distance_m
        return Controls(
            motor_torque_request_nm=1_000.0,
            front_brake_pressure_psi=0.0,
            rear_brake_pressure_psi=0.0,
            steering_angle_rad=atan(1.0 / 25.0 * 1.55),
        )


class EnduranceSimulatorTests(TestCase):
    def test_preserves_battery_state_and_records_component_telemetry(self) -> None:
        track = small_closed_track()
        vehicle = Vehicle()
        constraints = PathConstraintSolver().solve(track, vehicle)

        result = EnduranceSimulator().run(
            vehicle,
            constraints,
            ConstantDriverControls(),
            EnduranceRunConfig(laps=2, starting_speed_mps=8.0),
            record_telemetry=True,
        )

        self.assertTrue(result.completed)
        self.assertEqual(result.completed_laps, 2)
        self.assertEqual(len(result.lap_times_s), 2)
        self.assertAlmostEqual(vehicle.distance_m, 2.0 * track.length_m)
        self.assertLess(result.final_state_of_charge, 1.0)
        self.assertGreater(result.pack_energy_kwh, 0.0)
        self.assertIsNotNone(result.telemetry)
        assert result.telemetry is not None
        self.assertIn("motor.torque_nm", result.telemetry)
        self.assertIn("endurance.path_speed_ceiling_mps", result.telemetry)
        self.assertIn("controls.front_brake_pressure_psi", result.telemetry)
        self.assertIn("controls.rear_brake_pressure_psi", result.telemetry)
        self.assertTrue(
            any(
                pressure_psi > 0.0
                for pressure_psi in result.telemetry[
                    "controls.front_brake_pressure_psi"
                ]
            )
        )
        self.assertTrue(
            any(
                force_n > 0.0 for force_n in result.telemetry["brakes.friction_force_n"]
            )
        )
        self.assertAlmostEqual(
            result.telemetry["energy.cumulative_net_j"][-1] / 3_600_000.0,
            result.pack_energy_kwh,
        )

    def test_lateral_saturation_fails_instead_of_relaxing_path_curvature(self) -> None:
        track = small_closed_track()
        vehicle = Vehicle()
        constraints = PathConstraintSolver().solve(track, vehicle)
        profile = UniformPeriodicTorqueParameterization(2).build((0.0, 0.0), track)

        # Deliberately corrupt the supplied ceiling so the simulation enters
        # the 10 m-radius curve above its physical lateral-force limit.
        overspeed_mps = max(constraints.local_corner_speed_mps) * 2.0
        unsafe_constraints = type(constraints)(
            track=track,
            local_corner_speed_mps=constraints.local_corner_speed_mps,
            braking_speed_ceiling_mps=(overspeed_mps,) * track.cell_count,
            passes=constraints.passes,
        )

        result = EnduranceSimulator().run(
            vehicle,
            unsafe_constraints,
            profile,
            EnduranceRunConfig(laps=1, starting_speed_mps=overspeed_mps),
        )

        self.assertFalse(result.completed)
        self.assertIsNotNone(result.failure_reason)
        assert result.failure_reason is not None
        self.assertIn("Car would spin out", result.failure_reason)
        self.assertIn("local corner-speed limit", result.failure_reason)

    def test_validates_supplied_controls_without_acting_as_speed_limiter(self) -> None:
        track = small_closed_track()
        vehicle = Vehicle()
        constraints = PathConstraintSolver().solve(track, vehicle)
        lower_ceiling_mps = min(constraints.braking_speed_ceiling_mps) - 1.0
        strict_constraints = type(constraints)(
            track=track,
            local_corner_speed_mps=constraints.local_corner_speed_mps,
            braking_speed_ceiling_mps=(lower_ceiling_mps,) * track.cell_count,
            passes=constraints.passes,
        )
        supplied_controls = ConstantUnsafeControls().controls_at(0.0)

        result = EnduranceSimulator().run(
            vehicle,
            strict_constraints,
            ConstantUnsafeControls(),
            EnduranceRunConfig(
                laps=1, starting_speed_mps=lower_ceiling_mps,
            ),
        )

        self.assertFalse(result.completed)
        self.assertIsNotNone(result.failure_reason)
        assert result.failure_reason is not None
        self.assertIn("supplied controls exceeded the path ceiling", result.failure_reason)
        self.assertEqual(vehicle.current_controls, supplied_controls)

    def test_does_not_clamp_an_unsafe_starting_speed(self) -> None:
        track = small_closed_track()
        vehicle = Vehicle()
        constraints = PathConstraintSolver().solve(track, vehicle)
        requested_speed_mps = constraints.local_corner_speed_mps[0] + 1.0

        result = EnduranceSimulator().run(
            vehicle,
            constraints,
            ConstantDriverControls(),
            EnduranceRunConfig(laps=1, starting_speed_mps=requested_speed_mps),
        )

        self.assertFalse(result.completed)
        self.assertAlmostEqual(vehicle.speed_mps, requested_speed_mps)
        self.assertEqual(vehicle.distance_m, 0.0)
