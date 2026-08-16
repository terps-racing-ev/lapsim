"""Tests for the static and one-RC OCV-based 108s3p pack models."""

from math import exp, isclose
from unittest import TestCase

from lapsim import Controls
from vehicle_model import (
    BatteryModel,
    OCVPackBattery,
    RCTheveninBattery,
    Vehicle,
)


class OCVPackBatteryTests(TestCase):
    def test_default_pack_configuration_and_ocv_lookup_endpoints(self) -> None:
        battery = OCVPackBattery()

        self.assertIsInstance(battery, BatteryModel)
        self.assertTrue(isclose(battery.pack_capacity_ah, 15.4))
        self.assertEqual(battery.series_cells, 108)
        self.assertEqual(battery.parallel_cells, 3)
        self.assertTrue(isclose(battery.internal_resistance_ohm, 0.348617379))
        self.assertTrue(isclose(battery.cell_ocv_voltage_v(0.0), 2.811))
        self.assertTrue(isclose(battery.cell_ocv_voltage_v(1.0), 4.182))
        self.assertTrue(isclose(battery.open_circuit_voltage_v, 451.656))

    def test_terminal_voltage_follows_ohmic_relation_and_soc_counts_down(self) -> None:
        battery = OCVPackBattery()
        initial_soc = battery.state_of_charge

        battery.update_state(50_000.0, 1.0)

        self.assertGreater(battery.current_a, 0.0)
        self.assertTrue(
            isclose(
                battery.terminal_voltage_v,
                battery.operating_open_circuit_voltage_v
                - battery.current_a * battery.internal_resistance_ohm,
                rel_tol=1e-12,
            )
        )
        self.assertTrue(
            isclose(
                battery.terminal_voltage_v * battery.current_a,
                battery.current_power_w,
                rel_tol=1e-12,
            )
        )
        self.assertLess(battery.state_of_charge, initial_soc)
        self.assertGreater(battery.net_charge_removed_ah, 0.0)

    def test_low_soc_voltage_bound_derates_discharge_power(self) -> None:
        battery = OCVPackBattery(initial_state_of_charge=0.0)

        self.assertEqual(battery.discharge_power_limit_w, 0.0)
        self.assertEqual(
            battery.limit_discharge_power_w(battery.max_discharge_power_w),
            battery.discharge_power_limit_w,
        )
        with self.assertRaises(ValueError):
            battery.update_state(
                battery.discharge_power_limit_w + 1.0,
                0.1,
            )

    def test_full_pack_cannot_accept_more_charge(self) -> None:
        battery = OCVPackBattery(max_charge_power_w=10_000.0)

        self.assertEqual(battery.charge_power_limit_w, 0.0)
        self.assertEqual(battery.limit_charge_power_w(1_000.0), 0.0)

    def test_charging_uses_negative_current_and_increases_soc(self) -> None:
        battery = OCVPackBattery(
            initial_state_of_charge=0.5,
            max_charge_power_w=10_000.0,
        )
        initial_soc = battery.state_of_charge

        battery.update_state(-1_000.0, 1.0)

        self.assertLess(battery.current_a, 0.0)
        self.assertGreater(battery.terminal_voltage_v, battery.open_circuit_voltage_v)
        self.assertGreater(battery.state_of_charge, initial_soc)
        self.assertLess(battery.net_charge_removed_ah, 0.0)

    def test_reset_restores_the_configured_soc(self) -> None:
        battery = OCVPackBattery(initial_state_of_charge=0.75)
        battery.update_state(10_000.0, 1.0)

        battery.reset_state()

        self.assertTrue(isclose(battery.state_of_charge, 0.75))
        self.assertEqual(battery.current_power_w, 0.0)
        self.assertEqual(battery.current_a, 0.0)
        self.assertEqual(battery.net_charge_removed_ah, 0.0)

    def test_vehicle_accepts_the_pack_model_via_battery_protocol(self) -> None:
        vehicle = Vehicle(battery=OCVPackBattery())

        vehicle.update_state(Controls(motor_torque_request_nm=100.0), 0.01)

        self.assertGreater(vehicle.battery.current_power_w, 0.0)
        self.assertGreater(vehicle.battery.current_a, 0.0)

    def test_vehicle_uses_the_rc_pack_model_by_default(self) -> None:
        self.assertIsInstance(Vehicle().battery, RCTheveninBattery)


class RCTheveninBatteryTests(TestCase):
    def test_default_fitted_pack_parameters(self) -> None:
        battery = RCTheveninBattery()

        self.assertIsInstance(battery, BatteryModel)
        self.assertTrue(
            isclose(battery.internal_resistance_ohm, 0.121472656)
        )
        self.assertTrue(
            isclose(battery.polarization_resistance_ohm, 0.431980951)
        )
        self.assertTrue(
            isclose(battery.polarization_capacitance_f, 10.478534332)
        )
        self.assertTrue(
            isclose(battery.polarization_time_constant_s, 4.52652723)
        )

    def test_terminal_voltage_and_exact_rc_state_step(self) -> None:
        battery = RCTheveninBattery(initial_state_of_charge=0.9)
        timestep_s = 0.2

        battery.update_state(30_000.0, timestep_s)

        expected_terminal_voltage_v = (
            battery.operating_open_circuit_voltage_v
            - battery.operating_polarization_voltage_v
            - battery.current_a * battery.internal_resistance_ohm
        )
        self.assertTrue(
            isclose(
                battery.terminal_voltage_v,
                expected_terminal_voltage_v,
                rel_tol=1e-12,
            )
        )
        self.assertTrue(
            isclose(
                battery.current_a * battery.terminal_voltage_v,
                battery.current_power_w,
                rel_tol=1e-12,
            )
        )
        decay = exp(-timestep_s / battery.polarization_time_constant_s)
        expected_polarization_voltage_v = (
            battery.polarization_resistance_ohm
            * (1.0 - decay)
            * battery.current_a
        )
        self.assertTrue(
            isclose(
                battery.polarization_voltage_v,
                expected_polarization_voltage_v,
                rel_tol=1e-12,
            )
        )

    def test_zero_current_relaxes_the_polarization_state(self) -> None:
        battery = RCTheveninBattery(
            initial_state_of_charge=0.9,
            initial_polarization_voltage_v=12.0,
        )
        timestep_s = battery.polarization_time_constant_s

        battery.update_state(0.0, timestep_s)

        self.assertTrue(
            isclose(
                battery.polarization_voltage_v,
                12.0 / exp(1.0),
                rel_tol=1e-12,
            )
        )
        self.assertTrue(
            isclose(
                battery.terminal_voltage_v,
                battery.operating_open_circuit_voltage_v - 12.0,
                rel_tol=1e-12,
            )
        )

    def test_reset_restores_initial_polarization(self) -> None:
        battery = RCTheveninBattery(
            initial_state_of_charge=0.8,
            initial_polarization_voltage_v=2.0,
        )
        battery.update_state(10_000.0, 0.1)

        battery.reset_state()

        self.assertEqual(battery.polarization_voltage_v, 2.0)
        self.assertEqual(battery.operating_polarization_voltage_v, 2.0)
        self.assertTrue(isclose(battery.state_of_charge, 0.8))
