"""Numerical references captured before separating vehicle responsibilities."""

import json
from math import isclose
from pathlib import Path
from unittest import TestCase

from lapsim import Controls
from vehicle_model import ActiveAero, Battery, Pacejka61LateralModel, Tire, Vehicle


def reference_vehicle(**kwargs):
    """Keep the pre-refactor tire and motor setup explicit as defaults evolve."""

    vehicle = Vehicle(
        tire=Tire(
            pacejka_lateral=Pacejka61LateralModel(correlation_factor=0.8),
            pacejka_longitudinal=None,
        ),
        **kwargs,
    )
    vehicle.drivetrain.motor.efficiency_map = None
    return vehicle


def reference_scenarios():
    """Exercise both solver branches and the stateful subsystem boundaries."""

    yield "launch", reference_vehicle(), [(Controls(motor_torque_request_nm=230), 0.01)] * 12
    yield "corner", reference_vehicle(initial_speed_mps=15), [
        (Controls(motor_torque_request_nm=50, steering_angle_rad=0.08), 0.05)
    ] * 4
    yield "brake", reference_vehicle(initial_speed_mps=20), [
        (
            Controls(
                front_brake_pressure_psi=70,
                rear_brake_pressure_psi=50,
                steering_angle_rad=-0.06,
            ),
            0.05,
        )
    ] * 4
    yield "stop", reference_vehicle(initial_speed_mps=1), [
        (Controls(front_brake_pressure_psi=1e6, rear_brake_pressure_psi=1e6), 10)
    ]
    speed_limited = reference_vehicle(initial_speed_mps=5)
    speed_limited.drivetrain.configured_speed_limit_mps = 5
    yield "speed_limit", speed_limited, [
        (Controls(motor_torque_request_nm=230), 0.01)
    ] * 4
    yield "regen", reference_vehicle(
        initial_speed_mps=10, battery=Battery(max_charge_power_w=80_000)
    ), [(Controls(rear_regenerative_brake_force_request_n=500), 0.1)] * 4
    yield "active_aero", reference_vehicle(initial_speed_mps=15, aero=ActiveAero()), [
        (Controls(motor_torque_request_nm=30), 0.05),
        (Controls(steering_angle_rad=0.08), 0.05),
    ] * 2
    yield "rest", reference_vehicle(), [(Controls(), 0.1)] * 2


class VehicleRegressionTests(TestCase):
    def test_motion_forces_and_energy_match_before_refactor(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "vehicle_refactor_reference.json"
        reference = json.loads(fixture.read_text())
        for name, vehicle, steps in reference_scenarios():
            for controls, timestep_s in steps:
                vehicle.update_state(controls, timestep_s)
            snapshot = vehicle.telemetry_snapshot()
            for channel, expected in reference[name].items():
                with self.subTest(scenario=name, channel=channel):
                    self.assertTrue(
                        isclose(
                            snapshot[channel], expected, rel_tol=1e-8, abs_tol=1e-8
                        ),
                        f"{snapshot[channel]} != {expected}",
                    )
