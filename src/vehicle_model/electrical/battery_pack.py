"""Coulomb-counted pack models: OCV plus ohmic loss, optionally one RC branch.

The base model owns SOC, pack scaling, and terminal limits. The RC model adds
only polarization history; both use TerminalCircuit for instantaneous math.
"""

from bisect import bisect_left
from dataclasses import dataclass, field
from math import exp, isfinite

from .battery_circuit import TerminalCircuit
from .battery_ocv import CELL_OCV_TABLE
from .battery_parameters import (
    DEFAULT_MAX_DISCHARGE_POWER_W,
    DEFAULT_MAX_CHARGE_POWER_W,
    DEFAULT_PACK_SERIES_CELLS,
    DEFAULT_PACK_PARALLEL_CELLS,
    DEFAULT_CELL_CAPACITY_AH,
    DEFAULT_CELL_INTERNAL_RESISTANCE_OHM,
    DEFAULT_MINIMUM_CELL_VOLTAGE_V,
    DEFAULT_MAXIMUM_CELL_VOLTAGE_V,
    DEFAULT_RC_CELL_OHMIC_RESISTANCE_OHM,
    DEFAULT_RC_CELL_POLARIZATION_RESISTANCE_OHM,
    DEFAULT_RC_CELL_POLARIZATION_CAPACITANCE_F,
)


@dataclass(slots=True)
class OCVPackBattery:
    """Coulomb-counted 108s3p pack with a cell OCV lookup and ohmic loss.

    The nominal capacity is 15.4 Ah at the pack terminals (5.133... Ah per
    cell times three parallel cells). Positive current and positive terminal
    power mean discharge. The pack terminal model is Vout = Voc - I * R.
    """

    cell_capacity_ah: float = DEFAULT_CELL_CAPACITY_AH
    max_discharge_power_w: float = DEFAULT_MAX_DISCHARGE_POWER_W
    max_charge_power_w: float = DEFAULT_MAX_CHARGE_POWER_W
    initial_state_of_charge: float = 1.0
    series_cells: int = DEFAULT_PACK_SERIES_CELLS
    parallel_cells: int = DEFAULT_PACK_PARALLEL_CELLS
    cell_internal_resistance_ohm: float = DEFAULT_CELL_INTERNAL_RESISTANCE_OHM
    minimum_cell_voltage_v: float = DEFAULT_MINIMUM_CELL_VOLTAGE_V
    maximum_cell_voltage_v: float = DEFAULT_MAXIMUM_CELL_VOLTAGE_V
    cell_ocv_table: tuple[tuple[float, float], ...] = CELL_OCV_TABLE
    current_power_w: float = field(init=False, default=0.0)
    current_a: float = field(init=False, default=0.0)
    operating_open_circuit_voltage_v: float = field(init=False, default=0.0)
    terminal_voltage_v: float = field(init=False, default=0.0)
    state_of_charge: float = field(init=False, default=0.0)
    net_charge_removed_ah: float = field(init=False, default=0.0)
    coulomb_throughput_ah: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()
        self.reset_state()

    def validate(self) -> None:
        """Validate model parameters and lookup-table ordering and coverage."""

        if self.cell_capacity_ah <= 0:
            raise ValueError("cell_capacity_ah must be positive")
        if self.max_discharge_power_w <= 0:
            raise ValueError("max_discharge_power_w must be positive")
        if self.max_charge_power_w < 0:
            raise ValueError("max_charge_power_w cannot be negative")
        if not 0.0 <= self.initial_state_of_charge <= 1.0:
            raise ValueError("initial_state_of_charge must be between 0 and 1")
        if self.series_cells <= 0 or self.parallel_cells <= 0:
            raise ValueError("series_cells and parallel_cells must be positive")
        if self.cell_internal_resistance_ohm <= 0:
            raise ValueError("cell_internal_resistance_ohm must be positive")
        if self.minimum_cell_voltage_v <= 0:
            raise ValueError("minimum_cell_voltage_v must be positive")
        if self.maximum_cell_voltage_v <= self.minimum_cell_voltage_v:
            raise ValueError(
                "maximum_cell_voltage_v must exceed minimum_cell_voltage_v"
            )
        if len(self.cell_ocv_table) < 2:
            raise ValueError("cell_ocv_table must contain at least two points")

        previous_soc = -1.0
        for state_of_charge, voltage_v in self.cell_ocv_table:
            if not 0.0 <= state_of_charge <= 1.0:
                raise ValueError("cell_ocv_table SOC values must be between 0 and 1")
            if state_of_charge <= previous_soc:
                raise ValueError(
                    "cell_ocv_table SOC values must be strictly increasing"
                )
            if voltage_v <= 0:
                raise ValueError("cell_ocv_table voltages must be positive")
            previous_soc = state_of_charge
        if self.cell_ocv_table[0][0] != 0.0 or self.cell_ocv_table[-1][0] != 1.0:
            raise ValueError("cell_ocv_table must cover SOC endpoints 0.0 and 1.0")

    def reset_state(self) -> None:
        """Restore initial SOC and clear all present operating-point state."""

        self.state_of_charge = self.initial_state_of_charge
        self.current_power_w = 0.0
        self.current_a = 0.0
        self.operating_open_circuit_voltage_v = self.open_circuit_voltage_v
        self.terminal_voltage_v = self.operating_open_circuit_voltage_v
        self.net_charge_removed_ah = 0.0
        self.coulomb_throughput_ah = 0.0

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry.update(
            {
                "battery.power_w": self.current_power_w,
                "battery.current_a": self.current_a,
                "battery.terminal_voltage_v": self.terminal_voltage_v,
                "battery.open_circuit_voltage_v": (
                    self.operating_open_circuit_voltage_v
                ),
                "battery.state_of_charge": self.state_of_charge,
                "battery.net_charge_removed_ah": self.net_charge_removed_ah,
                "battery.coulomb_throughput_ah": self.coulomb_throughput_ah,
                "battery.discharge_power_limit_w": self.discharge_power_limit_w,
                "battery.charge_power_limit_w": self.charge_power_limit_w,
                "battery.ohmic_loss_power_w": (
                    self.current_a**2 * self.internal_resistance_ohm
                ),
            }
        )

    @property
    def pack_capacity_ah(self) -> float:
        """Nominal pack capacity in Ah: cell capacity times parallel count."""

        return self.cell_capacity_ah * self.parallel_cells

    @property
    def internal_resistance_ohm(self) -> float:
        """Equivalent pack resistance in ohms for series strings in parallel."""

        return (
            self.series_cells * self.cell_internal_resistance_ohm / self.parallel_cells
        )

    @property
    def open_circuit_voltage_v(self) -> float:
        """Present pack OCV in volts from the interpolated cell lookup."""

        return self.series_cells * self.cell_ocv_voltage_v(self.state_of_charge)

    @property
    def minimum_terminal_voltage_v(self) -> float:
        """Pack low-voltage bound in volts."""

        return self.series_cells * self.minimum_cell_voltage_v

    @property
    def maximum_terminal_voltage_v(self) -> float:
        """Pack high-voltage bound in volts."""

        return self.series_cells * self.maximum_cell_voltage_v

    def cell_ocv_voltage_v(self, state_of_charge: float) -> float:
        """Linearly interpolate cell OCV (V) for an SOC fraction."""

        bounded_soc = min(max(state_of_charge, 0.0), 1.0)
        index = bisect_left(
            self.cell_ocv_table,
            bounded_soc,
            key=lambda point: point[0],
        )
        if index == 0:
            return self.cell_ocv_table[0][1]
        if index == len(self.cell_ocv_table):
            return self.cell_ocv_table[-1][1]

        lower_soc, lower_voltage_v = self.cell_ocv_table[index - 1]
        upper_soc, upper_voltage_v = self.cell_ocv_table[index]
        fraction = (bounded_soc - lower_soc) / (upper_soc - lower_soc)
        return lower_voltage_v + fraction * (upper_voltage_v - lower_voltage_v)

    @property
    def effective_source_voltage_v(self) -> float:
        """Circuit source voltage; RC models subtract polarization here."""

        return self.open_circuit_voltage_v

    @property
    def _terminal_circuit(self) -> TerminalCircuit:
        return TerminalCircuit(
            source_voltage_v=self.effective_source_voltage_v,
            resistance_ohm=self.internal_resistance_ohm,
            minimum_voltage_v=self.minimum_terminal_voltage_v,
            maximum_voltage_v=self.maximum_terminal_voltage_v,
        )

    @property
    def electrical_discharge_power_limit_w(self) -> float:
        """Voltage-limited discharge ceiling at the current SOC and RC state."""

        if self.state_of_charge <= 0.0:
            return 0.0
        return self._terminal_circuit.discharge_power_limit_w

    @property
    def electrical_charge_power_limit_w(self) -> float:
        """Voltage-limited charge magnitude at the current SOC and RC state."""

        if self.state_of_charge >= 1.0:
            return 0.0
        return self._terminal_circuit.charge_power_limit_w

    @property
    def discharge_power_limit_w(self) -> float:
        """Present DC-terminal discharge ceiling in watts."""

        return min(
            self.max_discharge_power_w,
            self.electrical_discharge_power_limit_w,
        )

    @property
    def charge_power_limit_w(self) -> float:
        """Present DC-terminal charge ceiling magnitude in watts."""

        return min(self.max_charge_power_w, self.electrical_charge_power_limit_w)

    def limit_discharge_power_w(self, requested_power_w: float) -> float:
        """Limit a nonnegative terminal-discharge request in watts."""

        if requested_power_w < 0:
            raise ValueError("requested_power_w cannot be negative")
        return min(requested_power_w, self.discharge_power_limit_w)

    def limit_charge_power_w(self, requested_power_w: float) -> float:
        """Limit a nonnegative terminal-charge request magnitude in watts."""

        if requested_power_w < 0:
            raise ValueError("requested_power_w cannot be negative")
        return min(requested_power_w, self.charge_power_limit_w)

    def current_for_terminal_power_a(self, power_w: float) -> float:
        """Return physical pack current for terminal power using the low-I root."""

        if power_w > self.discharge_power_limit_w:
            raise ValueError("power_w exceeds present discharge_power_limit_w")
        if -power_w > self.charge_power_limit_w:
            raise ValueError("charging power exceeds present charge_power_limit_w")
        return self._terminal_circuit.current_for_power_a(power_w)

    def update_state(self, power_w: float, timestep_s: float) -> None:
        """Apply terminal power and Coulomb-count SOC for one positive timestep."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        operating_open_circuit_voltage_v = self.open_circuit_voltage_v
        current_a = self.current_for_terminal_power_a(power_w)
        self.current_power_w = power_w
        self.current_a = current_a
        self.operating_open_circuit_voltage_v = operating_open_circuit_voltage_v
        self.terminal_voltage_v = (
            operating_open_circuit_voltage_v - current_a * self.internal_resistance_ohm
        )
        self._advance_charge(current_a, timestep_s)

    def _advance_charge(self, current_a: float, timestep_s: float) -> None:
        """Coulomb-count charge removal and throughput, then clamp SOC."""

        charge_delta_ah = current_a * timestep_s / 3600.0
        self.net_charge_removed_ah += charge_delta_ah
        self.coulomb_throughput_ah += abs(charge_delta_ah)
        self.state_of_charge = min(
            1.0,
            max(0.0, self.state_of_charge - charge_delta_ah / self.pack_capacity_ah),
        )


@dataclass(slots=True)
class RCTheveninBattery(OCVPackBattery):
    """OCV pack with instantaneous and first-order polarization losses.

    The terminal model is::

        V_terminal = V_oc(SOC) - I * R0 - V_p
        dV_p/dt = (I * R1 - V_p) / (R1 * C1)

    Positive current means discharge. The RC state is advanced with the exact
    zero-order-hold solution, so its time constant does not depend on the
    caller's simulation timestep.
    """

    cell_internal_resistance_ohm: float = DEFAULT_RC_CELL_OHMIC_RESISTANCE_OHM
    cell_polarization_resistance_ohm: float = (
        DEFAULT_RC_CELL_POLARIZATION_RESISTANCE_OHM
    )
    cell_polarization_capacitance_f: float = DEFAULT_RC_CELL_POLARIZATION_CAPACITANCE_F
    initial_polarization_voltage_v: float = 0.0
    polarization_voltage_v: float = field(init=False, default=0.0)
    operating_polarization_voltage_v: float = field(init=False, default=0.0)

    def validate(self) -> None:
        """Validate the base pack and RC-branch parameters."""

        OCVPackBattery.validate(self)
        if self.cell_polarization_resistance_ohm <= 0:
            raise ValueError("cell_polarization_resistance_ohm must be positive")
        if self.cell_polarization_capacitance_f <= 0:
            raise ValueError("cell_polarization_capacitance_f must be positive")
        if not isfinite(self.initial_polarization_voltage_v):
            raise ValueError("initial_polarization_voltage_v must be finite")

    def reset_state(self) -> None:
        """Restore SOC and the configured initial polarization state."""

        OCVPackBattery.reset_state(self)
        self.polarization_voltage_v = self.initial_polarization_voltage_v
        self.operating_polarization_voltage_v = self.initial_polarization_voltage_v
        self.terminal_voltage_v = (
            self.operating_open_circuit_voltage_v - self.initial_polarization_voltage_v
        )

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        OCVPackBattery.update_telemetry(self, telemetry)
        telemetry.update(
            {
                "battery.polarization_voltage_v": (
                    self.operating_polarization_voltage_v
                ),
                "battery.next_polarization_voltage_v": (self.polarization_voltage_v),
                "battery.polarization_time_constant_s": (
                    self.polarization_time_constant_s
                ),
                "battery.polarization_power_w": (
                    self.current_a * self.operating_polarization_voltage_v
                ),
            }
        )

    @property
    def polarization_resistance_ohm(self) -> float:
        """Equivalent pack polarization resistance R1 in ohms."""

        return (
            self.series_cells
            * self.cell_polarization_resistance_ohm
            / self.parallel_cells
        )

    @property
    def polarization_capacitance_f(self) -> float:
        """Equivalent pack polarization capacitance C1 in farads."""

        return (
            self.parallel_cells
            * self.cell_polarization_capacitance_f
            / self.series_cells
        )

    @property
    def polarization_time_constant_s(self) -> float:
        """Polarization time constant R1*C1 in seconds."""

        return self.polarization_resistance_ohm * self.polarization_capacitance_f

    @property
    def effective_source_voltage_v(self) -> float:
        """OCV minus the present polarization voltage."""

        return self.open_circuit_voltage_v - self.polarization_voltage_v

    def update_state(self, power_w: float, timestep_s: float) -> None:
        """Apply terminal power, then advance SOC and polarization state."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")

        operating_open_circuit_voltage_v = self.open_circuit_voltage_v
        operating_polarization_voltage_v = self.polarization_voltage_v
        current_a = self.current_for_terminal_power_a(power_w)
        self.current_power_w = power_w
        self.current_a = current_a
        self.operating_open_circuit_voltage_v = operating_open_circuit_voltage_v
        self.operating_polarization_voltage_v = operating_polarization_voltage_v
        self.terminal_voltage_v = (
            operating_open_circuit_voltage_v
            - operating_polarization_voltage_v
            - current_a * self.internal_resistance_ohm
        )

        self._advance_charge(current_a, timestep_s)

        decay = exp(-timestep_s / self.polarization_time_constant_s)
        self.polarization_voltage_v = (
            decay * operating_polarization_voltage_v
            + self.polarization_resistance_ohm * (1.0 - decay) * current_a
        )
