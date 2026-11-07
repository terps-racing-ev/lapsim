"""Check physical invariants at the extracted subsystem boundaries."""

from math import cos, sin
from unittest import TestCase

from vehicle_model import AeroForces, Brakes, Chassis, Suspension
from vehicle_model.electrical.battery_circuit import TerminalCircuit
from vehicle_model.motion import integrate_motion


class TerminalCircuitTests(TestCase):
    def test_current_reconstructs_discharge_and_charge_power(self) -> None:
        circuit = TerminalCircuit(400.0, 0.2, 300.0, 450.0)
        for power_w in (0.0, 1e-6, 50_000.0, -20_000.0):
            with self.subTest(power_w=power_w):
                current_a = circuit.current_for_power_a(power_w)
                terminal_voltage_v = 400.0 - current_a * 0.2
                self.assertAlmostEqual(current_a * terminal_voltage_v, power_w)

    def test_voltage_bounds_determine_power_limits(self) -> None:
        circuit = TerminalCircuit(400.0, 0.2, 300.0, 450.0)
        self.assertEqual(circuit.discharge_power_limit_w, 500.0 * 300.0)
        self.assertEqual(circuit.charge_power_limit_w, 250.0 * 450.0)
        self.assertEqual(
            TerminalCircuit(290.0, 0.2, 300.0, 450.0).discharge_power_limit_w,
            0.0,
        )


class MotionIntegrationTests(TestCase):
    def test_slip_rotates_course_without_changing_distance_or_body_heading(
        self,
    ) -> None:
        motion = integrate_motion(
            initial_speed_mps=10.0,
            initial_heading_rad=0.3,
            longitudinal_acceleration_mps2=0.0,
            lateral_acceleration_mps2=0.0,
            body_slip_angle_rad=0.2,
            speed_limit_mps=50.0,
            timestep_s=0.5,
        )
        self.assertEqual(motion.distance_m, 5.0)
        self.assertEqual(motion.final_heading_rad, 0.3)
        self.assertAlmostEqual(motion.x_displacement_m, 5.0 * cos(0.5))
        self.assertAlmostEqual(motion.y_displacement_m, 5.0 * sin(0.5))

    def test_stopping_distance_is_independent_of_remaining_stationary_time(
        self,
    ) -> None:
        arguments = dict(
            initial_speed_mps=10.0,
            initial_heading_rad=0.0,
            longitudinal_acceleration_mps2=-5.0,
            lateral_acceleration_mps2=0.0,
            body_slip_angle_rad=0.0,
            speed_limit_mps=50.0,
        )
        for timestep_s in (2.0, 4.0):
            motion = integrate_motion(**arguments, timestep_s=timestep_s)
            self.assertEqual(motion.distance_m, 10.0)
            self.assertEqual(motion.final_speed_mps, 0.0)
            self.assertEqual(motion.average_speed_mps(timestep_s), 10.0 / timestep_s)


class MechanicalBoundaryTests(TestCase):
    def test_pressure_map_reads_mutated_hardware_gains(self) -> None:
        brakes = Brakes()
        front_before, rear_before = brakes.axle_force_requests_from_pressures_n(
            50.0, 50.0, 0.2
        )
        brakes.front_torque_per_pressure_lbfin_per_psi *= 2.0
        front_after, rear_after = brakes.axle_force_requests_from_pressures_n(
            50.0, 50.0, 0.2
        )
        self.assertEqual(front_after, 2.0 * front_before)
        self.assertEqual(rear_after, rear_before)

    def test_candidate_loads_do_not_commit_suspension_telemetry(self) -> None:
        suspension = Suspension()
        loads = suspension.tire_normal_loads_n(
            300.0,
            9.81,
            AeroForces(0.0, 0.0, 0.0, 0.0),
            Chassis(),
            lateral_acceleration_mps2=9.81,
        )
        self.assertEqual(suspension.current_body_roll_angle_rad, 0.0)
        self.assertEqual(suspension.current_tire_normal_loads_n.total_n, 0.0)
        suspension.update_state(loads, 0.01)
        self.assertEqual(suspension.current_tire_normal_loads_n, loads)
        self.assertGreater(suspension.current_body_roll_angle_rad, 0.0)

    def test_suspension_variant_keeps_its_body_roll_law(self) -> None:
        class FixedRollSuspension(Suspension):
            def body_roll_angle_rad(self, mass_kg, chassis, lateral_acceleration_mps2):
                return 0.02

        suspension = FixedRollSuspension()
        loads = suspension.tire_normal_loads_n(
            300.0,
            9.81,
            AeroForces(0.0, 0.0, 0.0, 0.0),
            Chassis(),
            lateral_acceleration_mps2=9.81,
        )
        suspension.update_state(loads, 0.01)
        self.assertEqual(suspension.current_body_roll_angle_rad, 0.02)
