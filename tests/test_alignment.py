"""Behavior tests for per-wheel steering and camber geometry."""

from math import radians
from unittest import TestCase

from lapsim import Controls
from vehicle_model import (
    Chassis,
    Tire,
    TireKinematics,
    TireNormalLoads,
    Vehicle,
    WheelAlignment,
    WheelKinematics,
)


class WheelAlignmentTests(TestCase):
    def setUp(self) -> None:
        self.chassis = Chassis()
        self.zero_travel = (0.0, 0.0, 0.0, 0.0)

    def test_default_front_caster_is_7p3_degrees(self) -> None:
        self.assertAlmostEqual(
            WheelAlignment().front_caster_angle_rad,
            radians(7.3),
        )

    def test_full_anti_ackermann_gives_outside_wheel_more_steer(self) -> None:
        state = WheelAlignment(anti_ackermann_fraction=1.0).calculate(
            self.chassis,
            steering_angle_rad=0.15,
            body_roll_angle_rad=0.0,
            suspension_travel_m=self.zero_travel,
        )

        self.assertGreater(
            state.front_right.steering_angle_rad,
            state.front_left.steering_angle_rad,
        )

    def test_zero_anti_ackermann_is_parallel_steer(self) -> None:
        state = WheelAlignment(anti_ackermann_fraction=0.0).calculate(
            self.chassis,
            steering_angle_rad=-0.15,
            body_roll_angle_rad=0.0,
            suspension_travel_m=self.zero_travel,
        )

        self.assertEqual(
            state.front_left.steering_angle_rad,
            state.front_right.steering_angle_rad,
        )

    def test_positive_toe_setting_points_each_wheel_inward(self) -> None:
        state = WheelAlignment(
            front_toe_in_angle_rad=0.01,
            rear_toe_in_angle_rad=0.02,
        ).calculate(
            self.chassis,
            steering_angle_rad=0.0,
            body_roll_angle_rad=0.0,
            suspension_travel_m=self.zero_travel,
        )

        self.assertEqual(
            state.steering_angles_rad,
            (-0.01, 0.01, -0.02, 0.02),
        )

    def test_positive_caster_produces_opposite_local_front_camber(self) -> None:
        state = WheelAlignment(
            front_caster_angle_rad=radians(8.0),
        ).calculate(
            self.chassis,
            steering_angle_rad=0.2,
            body_roll_angle_rad=0.0,
            suspension_travel_m=self.zero_travel,
        )

        self.assertGreater(state.front_left.camber_angle_rad, 0.0)
        self.assertLess(state.front_right.camber_angle_rad, 0.0)
        self.assertEqual(state.rear_left.camber_angle_rad, 0.0)
        self.assertEqual(state.rear_right.camber_angle_rad, 0.0)

    def test_measured_camber_gain_matches_full_bump_data(self) -> None:
        state = WheelAlignment().calculate(
            self.chassis,
            steering_angle_rad=0.0,
            body_roll_angle_rad=0.0,
            suspension_travel_m=(0.0254, 0.0254, 0.0254, 0.0254),
        )

        self.assertAlmostEqual(state.front_left.camber_angle_rad, radians(-1.3))
        self.assertAlmostEqual(state.front_right.camber_angle_rad, radians(-1.3))
        self.assertAlmostEqual(state.rear_left.camber_angle_rad, radians(-2.7))
        self.assertAlmostEqual(state.rear_right.camber_angle_rad, radians(-2.7))

    def test_measured_rear_curves_remain_independent_in_full_droop(self) -> None:
        state = WheelAlignment().calculate(
            self.chassis,
            steering_angle_rad=0.0,
            body_roll_angle_rad=0.0,
            suspension_travel_m=(-0.0254, -0.0254, -0.0254, -0.0254),
        )

        self.assertAlmostEqual(state.front_left.camber_angle_rad, radians(1.8))
        self.assertAlmostEqual(state.front_right.camber_angle_rad, radians(1.8))
        self.assertAlmostEqual(state.rear_left.camber_angle_rad, radians(2.3))
        self.assertAlmostEqual(state.rear_right.camber_angle_rad, radians(2.1))

    def test_measured_camber_gain_interpolates_between_sweep_points(self) -> None:
        state = WheelAlignment().calculate(
            self.chassis,
            steering_angle_rad=0.0,
            body_roll_angle_rad=0.0,
            suspension_travel_m=(0.009525, 0.009525, 0.009525, 0.009525),
        )

        self.assertAlmostEqual(state.front_left.camber_angle_rad, radians(-0.5))
        self.assertAlmostEqual(state.rear_left.camber_angle_rad, radians(-0.9))

    def test_measured_camber_gain_clamps_outside_sweep_range(self) -> None:
        state = WheelAlignment().calculate(
            self.chassis,
            steering_angle_rad=0.0,
            body_roll_angle_rad=0.0,
            suspension_travel_m=(0.1, -0.1, 0.1, -0.1),
        )

        self.assertAlmostEqual(state.front_left.camber_angle_rad, radians(-1.3))
        self.assertAlmostEqual(state.front_right.camber_angle_rad, radians(1.8))
        self.assertAlmostEqual(state.rear_left.camber_angle_rad, radians(-2.7))
        self.assertAlmostEqual(state.rear_right.camber_angle_rad, radians(2.1))

    def test_vehicle_commits_alignment_to_each_tire_state(self) -> None:
        vehicle = Vehicle(
            initial_speed_mps=12.0,
            alignment=WheelAlignment(
                front_static_camber_angle_rad=radians(-1.0),
                rear_static_camber_angle_rad=radians(-0.5),
                front_caster_angle_rad=radians(7.0),
            ),
        )

        vehicle.update_state(Controls(steering_angle_rad=0.12), 0.01)

        self.assertEqual(
            tuple(state.camber_angle_rad for state in vehicle.tire.current_states.all),
            vehicle.alignment.current_state.camber_angles_rad,
        )
        self.assertNotEqual(
            vehicle.tire.current_states.front_left.camber_angle_rad,
            vehicle.tire.current_states.front_right.camber_angle_rad,
        )

    def test_toe_scrub_increases_straight_line_deceleration(self) -> None:
        baseline = Vehicle(initial_speed_mps=15.0)
        with_toe = Vehicle(
            initial_speed_mps=15.0,
            alignment=WheelAlignment(
                front_toe_in_angle_rad=0.02,
                rear_toe_in_angle_rad=0.02,
            ),
        )

        baseline.update_state(Controls(), 0.01)
        with_toe.update_state(Controls(), 0.01)

        self.assertLess(
            with_toe.longitudinal_acceleration_mps2,
            baseline.longitudinal_acceleration_mps2,
        )


class PerWheelCamberForceTests(TestCase):
    def test_pacejka_uses_each_wheels_camber(self) -> None:
        kinematics = TireKinematics(
            WheelKinematics(0.10, 10.0, 0.0, 0.0),
            WheelKinematics(0.10, 10.0, 0.0, 0.0),
            WheelKinematics(0.10, 10.0, 0.0, radians(-4.0)),
            WheelKinematics(0.10, 10.0, 0.0, radians(-4.0)),
        )
        states = Tire().calculate_forces(
            TireNormalLoads(800.0, 800.0, 800.0, 800.0),
            kinematics,
            drive_force_request_n=0.0,
            front_brake_force_request_n=0.0,
            rear_brake_force_request_n=0.0,
            timestep_s=0.01,
        )

        self.assertNotEqual(
            states.front_left.lateral_force_n,
            states.rear_left.lateral_force_n,
        )
        self.assertEqual(states.rear_left.camber_angle_rad, radians(-4.0))
