"""Tests for the default lateral and longitudinal tire models."""

from dataclasses import replace
from math import radians
from unittest import TestCase

from lapsim import Controls
from vehicle_model import (
    Pacejka52UpcR20LateralModel,
    Pacejka52UpcR20LongitudinalModel,
    Pacejka61LateralModel,
    Tire,
    TireKinematics,
    TireNormalLoads,
    Vehicle,
    WheelKinematics,
)


class PacejkaLateralTests(TestCase):
    def test_matlab_reference_values_at_nominal_conditions(self) -> None:
        model = Pacejka61LateralModel(correlation_factor=1.0)

        self.assertAlmostEqual(model.nominal_load_n, 1080.0)
        self.assertAlmostEqual(model.nominal_pressure_pa, 98_000.0)
        self.assertAlmostEqual(
            model.force_n(1080.0, radians(-10.0)),
            2396.101601414008,
        )
        self.assertAlmostEqual(
            model.force_n(1080.0, radians(10.0)),
            -2383.393727434875,
        )
        self.assertAlmostEqual(model.peak_force_n(1080.0), 2581.2)

    def test_force_is_zero_when_tire_is_unloaded(self) -> None:
        model = Pacejka61LateralModel()

        self.assertEqual(model.force_n(0.0, radians(10.0)), 0.0)
        self.assertEqual(model.peak_force_n(-1.0), 0.0)

    def test_matlab_reference_with_pressure_and_camber(self) -> None:
        model = Pacejka61LateralModel(correlation_factor=1.0)

        self.assertAlmostEqual(
            model.force_n(
                800.0,
                radians(7.0),
                camber_angle_rad=radians(-2.0),
                inflation_pressure_pa=90_000.0,
            ),
            -1839.778929099007,
        )

    def test_upc_r20_published_lateral_model_is_available(self) -> None:
        model = Pacejka52UpcR20LateralModel(correlation_factor=1.0)

        model.validate()
        self.assertAlmostEqual(model.nominal_load_n, 1_000.0)
        self.assertAlmostEqual(model.nominal_pressure_pa, 82_737.0)
        self.assertAlmostEqual(model.pcy1, 1.477553032924075)
        self.assertAlmostEqual(model.pdy1, 2.419662634889429)
        self.assertEqual(model.horizontal_shift_scale, 0.0)
        self.assertAlmostEqual(
            model.force_n(
                800.0,
                radians(5.0),
                inflation_pressure_pa=model.nominal_pressure_pa,
            ),
            -1512.8752072737998,
        )

        tire = Tire(
            pacejka_lateral=model,
            inflation_pressure_pa=model.nominal_pressure_pa,
        )
        self.assertGreater(tire.lateral_force_capacity_n(800.0), 0.0)

    def test_upc_r20_published_longitudinal_model_is_available(self) -> None:
        model = Pacejka52UpcR20LongitudinalModel(correlation_factor=1.0)
        pressure_pa = 10.0 * 6_894.757293168
        camber_rad = radians(-1.0)

        model.validate()
        self.assertAlmostEqual(model.nominal_load_n, 1_000.0)
        self.assertAlmostEqual(model.nominal_pressure_pa, 82_737.0)
        self.assertAlmostEqual(model.pdx1, 2.85110165444411)
        self.assertAlmostEqual(
            model.peak_force_n(
                800.0,
                camber_angle_rad=camber_rad,
                inflation_pressure_pa=pressure_pa,
            ),
            1_852.9587197960823,
        )
        tire = Tire(
            pacejka_longitudinal=model,
            inflation_pressure_pa=pressure_pa,
        )
        self.assertAlmostEqual(
            tire.longitudinal_force_capacity_n(800.0, camber_rad),
            model.peak_force_n(
                800.0,
                camber_angle_rad=-camber_rad,
                inflation_pressure_pa=pressure_pa,
            ),
        )
        self.assertAlmostEqual(
            tire.pure_longitudinal_force_n(800.0, 0.1, camber_angle_rad=camber_rad),
            model.force_n(800.0, 0.1, camber_angle_rad=-camber_rad,
                          inflation_pressure_pa=pressure_pa),
        )
        telemetry: dict[str, float] = {}
        tire.update_telemetry(telemetry)
        self.assertAlmostEqual(
            telemetry["tire.pacejka_longitudinal_active"],
            1.0,
        )

    def test_peak_coefficient_decreases_with_load(self) -> None:
        model = Pacejka61LateralModel()

        low_load_mu = model.peak_force_n(500.0) / 500.0
        high_load_mu = model.peak_force_n(1000.0) / 1000.0

        self.assertGreater(low_load_mu, high_load_mu)

    def test_pacejka_lateral_model_remains_available_explicitly(self) -> None:
        tire = Tire(
            pacejka_lateral=Pacejka61LateralModel(), pacejka_longitudinal=None
        )

        self.assertAlmostEqual(tire.lateral_force_capacity_n(1080.0), 2064.96)
        self.assertAlmostEqual(
            tire.pure_lateral_force_n(1080.0, radians(10.0)),
            -1906.7149819479,
        )
        self.assertAlmostEqual(tire.longitudinal_coefficient(1080.0), 1.43220634145)

    def test_default_tire_uses_upc_models_at_80_percent(self) -> None:
        tire = Tire()

        self.assertIsInstance(tire.pacejka_lateral, Pacejka52UpcR20LateralModel)
        self.assertIsInstance(
            tire.pacejka_longitudinal, Pacejka52UpcR20LongitudinalModel
        )
        for model in (tire.pacejka_lateral, tire.pacejka_longitudinal):
            self.assertEqual(model.correlation_factor, 0.80)
            published = replace(model, correlation_factor=1.0)
            for load_n in (500.0, 800.0, 1200.0):
                for camber_deg in (-2.0, 0.0, 2.0):
                    for pressure_pa in (68_947.57293168, 98_000.0):
                        kwargs = dict(
                            camber_angle_rad=radians(camber_deg),
                            inflation_pressure_pa=pressure_pa,
                        )
                        with self.subTest(
                            model=type(model).__name__, load=load_n,
                            camber=camber_deg, pressure=pressure_pa,
                        ):
                            self.assertAlmostEqual(
                                model.peak_force_n(load_n, **kwargs),
                                0.80 * published.peak_force_n(load_n, **kwargs),
                            )
                            for slip in (-0.1, 0.0, 0.1):
                                self.assertAlmostEqual(
                                    model.force_n(load_n, slip, **kwargs),
                                    0.80 * published.force_n(load_n, slip, **kwargs),
                                )

        telemetry: dict[str, float] = {}
        tire.update_telemetry(telemetry)
        self.assertEqual(telemetry["tire.lateral_correlation_factor"], 0.80)
        self.assertEqual(telemetry["tire.longitudinal_correlation_factor"], 0.80)

    def test_upc_models_reject_invalid_correlation_factors(self) -> None:
        for model_class in (
            Pacejka52UpcR20LateralModel, Pacejka52UpcR20LongitudinalModel
        ):
            for factor in (0.0, -0.1, 1.01, float("nan"), float("inf")):
                with self.subTest(model=model_class.__name__, factor=factor):
                    with self.assertRaises(ValueError):
                        model_class(correlation_factor=factor).validate()


class FourCornerTireTests(TestCase):
    loads = TireNormalLoads(500.0, 900.0, 700.0, 1_100.0)

    @staticmethod
    def straight_kinematics(speed_mps: float = 10.0) -> TireKinematics:
        return TireKinematics(
            *(WheelKinematics(longitudinal_speed_mps=speed_mps) for _ in range(4))
        )

    def test_camber_thrust_changes_toward_physical_tire_lean_on_both_sides(self) -> None:
        # Compare with zero camber to remove the fit's residual force offset.
        # Body +y is left: inward lean pushes left tires right and right tires left.
        for model in (Pacejka61LateralModel(), Pacejka52UpcR20LateralModel()):
            for load_n in (800.0, 1200.0):
                tire = Tire(pacejka_lateral=model)
                loads = TireNormalLoads(*(load_n for _ in range(4)))

                def states_at(camber_deg):
                    kin = TireKinematics(*(
                        WheelKinematics(longitudinal_speed_mps=10.0,
                                        camber_angle_rad=radians(camber_deg))
                        for _ in range(4)
                    ))
                    return tire.calculate_forces(loads, kin, 0.0, 0.0, 0.0, 0.01)

                baseline = states_at(0.0)
                for camber_deg in (-2.0, 2.0):
                    with self.subTest(model=type(model).__name__, load=load_n, camber=camber_deg):
                        states = states_at(camber_deg)
                        for i, (state, zero) in enumerate(zip(states.all, baseline.all, strict=True)):
                            side_sign = 1.0 if i % 2 == 0 else -1.0
                            force_change = state.body_lateral_force_n - zero.body_lateral_force_n
                            self.assertGreater(force_change * camber_deg * side_sign, 0.0)
                        self.assertAlmostEqual(states.body_lateral_force_n, 0.0)

    def test_vehicle_camber_is_converted_for_both_lateral_force_and_capacity(self) -> None:
        for model in (Pacejka61LateralModel(), Pacejka52UpcR20LateralModel()):
            tire = Tire(pacejka_lateral=model, inflation_pressure_pa=90_000.0)
            for camber_deg in (-2.0, 2.0):
                with self.subTest(model=type(model).__name__, camber=camber_deg):
                    camber = radians(camber_deg)
                    raw_force = model.force_n(1200.0, radians(5.0),
                                              camber_angle_rad=-camber,
                                              inflation_pressure_pa=tire.inflation_pressure_pa)
                    self.assertAlmostEqual(tire.pure_lateral_force_n(
                        1200.0, radians(5.0), camber_angle_rad=camber), raw_force)
                    self.assertAlmostEqual(tire.lateral_force_capacity_n(1200.0, camber),
                                           model.peak_force_n(1200.0, camber_angle_rad=-camber,
                                                             inflation_pressure_pa=tire.inflation_pressure_pa))
                    self.assertLessEqual(abs(tire.lateral_force_from_slip_angle_n(
                        1200.0, radians(5.0), camber)), tire.lateral_force_capacity_n(1200.0, camber))

    def test_fixed_axle_split_does_not_transfer_demand_to_stronger_axle(self) -> None:
        tire = Tire(constant_friction_coefficient=1.0)
        front_limited_loads = TireNormalLoads(200.0, 200.0, 800.0, 800.0)

        forces = tire.lateral_forces_n(
            front_limited_loads,
            total_lateral_force_n=1_000.0,
            front_lateral_force_fraction=0.47,
        )

        self.assertAlmostEqual(forces.front_axle_n, 400.0)
        self.assertAlmostEqual(forces.rear_axle_n, 400.0 * 0.53 / 0.47)
        self.assertAlmostEqual(
            forces.front_axle_n / forces.total_n,
            0.47,
        )
        self.assertLess(forces.total_n, 1_000.0)

    def test_mirrored_tires_cancel_fitted_force_offset_on_a_straight(self) -> None:
        equal_loads = TireNormalLoads(800.0, 800.0, 800.0, 800.0)

        states = Tire().calculate_forces(
            equal_loads,
            self.straight_kinematics(),
            drive_force_request_n=0.0,
            front_brake_force_request_n=0.0,
            rear_brake_force_request_n=0.0,
            timestep_s=0.01,
        )

        self.assertAlmostEqual(states.body_lateral_force_n, 0.0)

    def test_each_tire_resolves_its_own_load_force_capacity_and_slip(self) -> None:
        tire = Tire()
        kinematics = TireKinematics(
            WheelKinematics(0.06, 10.0, 0.0),
            WheelKinematics(0.05, 10.0, 0.0),
            WheelKinematics(0.0, 10.0, -0.1),
            WheelKinematics(0.0, 10.0, -0.2),
        )
        states = tire.calculate_forces(
            self.loads,
            kinematics,
            drive_force_request_n=1_000.0,
            front_brake_force_request_n=300.0,
            rear_brake_force_request_n=200.0,
            timestep_s=0.01,
        )

        self.assertEqual(
            tuple(state.normal_load_n for state in states.all),
            self.loads.all_n,
        )
        self.assertGreater(states.body_lateral_force_n, 0.0)
        self.assertEqual(
            tuple(state.steering_angle_rad for state in states.all),
            tuple(wheel.steering_angle_rad for wheel in kinematics.all),
        )
        self.assertEqual(
            tuple(state.slip_angle_rad for state in states.all),
            tuple(wheel.slip_angle_rad for wheel in kinematics.all),
        )
        self.assertLessEqual(states.drive_force_n, 1_000.0 + 1e-9)
        self.assertLessEqual(states.braking_force_n, 500.0)
        self.assertTrue(all(state.force_utilization <= 1.0 for state in states.all))
        self.assertNotEqual(states.rear_left.slip_ratio, states.rear_right.slip_ratio)

    def test_braking_relaxation_is_owned_and_applied_by_each_tire(self) -> None:
        tire = Tire(longitudinal_slip_relaxation_length_m=2.0)
        states = tire.calculate_forces(
            self.loads,
            self.straight_kinematics(),
            drive_force_request_n=0.0,
            front_brake_force_request_n=1_000.0,
            rear_brake_force_request_n=500.0,
            timestep_s=0.01,
        )

        self.assertLess(states.front_braking_force_n, 1_000.0)
        self.assertLess(states.rear_braking_force_n, 500.0)
        self.assertTrue(all(state.slip_ratio < 0.0 for state in states.all))

    def test_each_brake_force_is_clipped_by_its_own_contact_patch(self) -> None:
        states = Tire().calculate_forces(
            self.loads,
            self.straight_kinematics(),
            drive_force_request_n=0.0,
            front_brake_force_request_n=10_000.0,
            rear_brake_force_request_n=10_000.0,
            timestep_s=0.01,
        )

        self.assertTrue(
            all(
                state.braking_force_n == state.longitudinal_capacity_n
                for state in states.all
            )
        )
        self.assertNotEqual(
            states.front_left.braking_force_n,
            states.front_right.braking_force_n,
        )

    def test_tire_slip_sets_vehicle_motor_speed_and_power(self) -> None:
        slipping = Vehicle(initial_speed_mps=10.0)
        no_slip = Vehicle(
            initial_speed_mps=10.0,
            tire=Tire(peak_longitudinal_slip_ratio=0.0),
        )
        controls = Controls(motor_torque_request_nm=100.0)

        slipping.update_state(controls, 0.01)
        no_slip.update_state(controls, 0.01)

        self.assertGreater(slipping.tire.current_states.driven_slip_ratio, 0.0)
        self.assertGreater(
            slipping.drivetrain.motor.current_speed_rpm,
            no_slip.drivetrain.motor.current_speed_rpm,
        )
        self.assertGreater(
            slipping.battery.current_power_w,
            no_slip.battery.current_power_w,
        )

    def test_vehicle_force_balance_is_the_sum_of_all_tires(self) -> None:
        vehicle = Vehicle(initial_speed_mps=12.0)
        vehicle.update_state(
            Controls(motor_torque_request_nm=80.0, steering_angle_rad=0.08),
            0.1,
        )
        states = vehicle.tire.current_states

        self.assertAlmostEqual(vehicle.current_drive_force_n, states.drive_force_n)
        self.assertAlmostEqual(
            vehicle.current_friction_braking_force_n,
            states.braking_force_n,
        )
        self.assertAlmostEqual(
            states.longitudinal_force_n,
            sum(state.longitudinal_force_n for state in states.all),
        )
        self.assertAlmostEqual(
            states.lateral_force_n,
            sum(state.lateral_force_n for state in states.all),
        )
        self.assertAlmostEqual(
            states.body_lateral_force_n,
            sum(state.body_lateral_force_n for state in states.all),
        )
        self.assertAlmostEqual(
            vehicle.current_lateral_force_n,
            vehicle.mass_kg * vehicle.lateral_acceleration_mps2,
        )

    def test_turn_direction_changes_the_loaded_side(self) -> None:
        left = Vehicle(initial_speed_mps=12.0)
        right = Vehicle(initial_speed_mps=12.0)
        left.update_state(Controls(steering_angle_rad=0.08), 0.1)
        right.update_state(Controls(steering_angle_rad=-0.08), 0.1)

        self.assertGreater(
            left.tire.current_states.front_right.normal_load_n,
            left.tire.current_states.front_left.normal_load_n,
        )
        self.assertGreater(
            right.tire.current_states.front_left.normal_load_n,
            right.tire.current_states.front_right.normal_load_n,
        )
        self.assertGreater(left.current_lateral_force_n, 0.0)
        self.assertLess(right.current_lateral_force_n, 0.0)
