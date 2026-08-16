"""Regression tests for distance-indexed straight control replay."""

import unittest

import numpy as np

from analysis.accel.analyze_acceleration import (
    Straight,
    simulate_straight,
)
from analysis.common import RecordedControlAdapter


class StraightSpatialReplayTests(unittest.TestCase):
    def _sample(self, time_s: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "time_s": time_s,
            "distance_from_straight_start_m": np.array([0.0, 4.99, 5.0, 10.0]),
            "gnss_speed_mps": np.full(4, 10.0),
            "imu_longitudinal_accel_mps2": np.zeros(4),
            "imu_lateral_accel_mps2": np.zeros(4),
            "battery_soc_percent": np.full(4, 95.0),
            "torque_feedback_nm": np.array([80.0, 80.0, 0.0, 0.0]),
            "torque_command_nm": np.array([80.0, 80.0, 0.0, 0.0]),
            "brake_pressure_psi": np.zeros(4),
        }

    def _simulate(self, time_s: np.ndarray) -> dict[str, np.ndarray]:
        return simulate_straight(
            Straight(1, 0.0, 10.0, slice(0, 1)),
            self._sample(time_s),
            torque_source="feedback",
            control_adapter=RecordedControlAdapter(brake_force_per_psi_n=0.0),
            spatial_step_m=0.05,
        )

    def test_torque_step_is_applied_at_recorded_distance(self) -> None:
        result = self._simulate(np.array([0.0, 0.1, 10.0, 20.0]))
        before_step = np.interp(
            4.8, result["sim_distance_m"], result["sim_motor_torque_nm"]
        )
        after_step = np.interp(
            5.2, result["sim_distance_m"], result["sim_motor_torque_nm"]
        )
        self.assertGreater(before_step, 70.0)
        self.assertLess(after_step, 1.0)

    def test_recorded_timestamps_do_not_change_spatial_replay(self) -> None:
        fast_log = self._simulate(np.array([0.0, 0.1, 0.2, 0.3]))
        slow_log = self._simulate(np.array([0.0, 4.0, 11.0, 30.0]))
        comparison_distance = np.linspace(0.0, 9.5, 40)
        fast_torque = np.interp(
            comparison_distance,
            fast_log["sim_distance_m"],
            fast_log["sim_motor_torque_nm"],
        )
        slow_torque = np.interp(
            comparison_distance,
            slow_log["sim_distance_m"],
            slow_log["sim_motor_torque_nm"],
        )
        np.testing.assert_allclose(fast_torque, slow_torque, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
