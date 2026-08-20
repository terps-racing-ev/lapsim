"""Electrical-subteam ideal, OCV, and one-RC battery models."""

from bisect import bisect_left
from dataclasses import dataclass, field
from math import exp, isfinite

from .battery_ocv import CELL_OCV_TABLE

DEFAULT_MAX_DISCHARGE_POWER_W = 80_000.0
DEFAULT_MAX_CHARGE_POWER_W = 0.0


# 108s3p HVC pack defaults. Safe-voltage values are from Core/Inc/Config/cells.h
# in the same HVC-Firmware branch as the OCV table.
DEFAULT_PACK_SERIES_CELLS = 108
DEFAULT_PACK_PARALLEL_CELLS = 3
DEFAULT_PACK_CAPACITY_AH = 15.4
DEFAULT_CELL_CAPACITY_AH = DEFAULT_PACK_CAPACITY_AH / DEFAULT_PACK_PARALLEL_CELLS
# The default pack resistance is a discharge-only, trusted-HVC-SOC calibration.
# Express it as cell resistance so the series/parallel scaling remains explicit.
DEFAULT_PACK_INTERNAL_RESISTANCE_OHM = 0.348617379
DEFAULT_CELL_INTERNAL_RESISTANCE_OHM = (
    DEFAULT_PACK_INTERNAL_RESISTANCE_OHM
    * DEFAULT_PACK_PARALLEL_CELLS
    / DEFAULT_PACK_SERIES_CELLS
)
DEFAULT_MINIMUM_CELL_VOLTAGE_V = 2.8
DEFAULT_MAXIMUM_CELL_VOLTAGE_V = 4.2

# One-RC Thevenin calibration from the 6.20 endurance HVC log. The fit uses
# trusted HVC SOC directly and has no voltage or SOC offset. R0 is the
# instantaneous ohmic resistance; R1-C1 describes the slower polarization
# voltage. Pack values are converted to equivalent per-cell parameters so
# series/parallel scaling remains explicit and mutable.
DEFAULT_RC_PACK_OHMIC_RESISTANCE_OHM = 0.121472656
DEFAULT_RC_PACK_POLARIZATION_RESISTANCE_OHM = 0.431980951
DEFAULT_RC_PACK_POLARIZATION_CAPACITANCE_F = 10.478534332
DEFAULT_RC_CELL_OHMIC_RESISTANCE_OHM = (
    DEFAULT_RC_PACK_OHMIC_RESISTANCE_OHM
    * DEFAULT_PACK_PARALLEL_CELLS
    / DEFAULT_PACK_SERIES_CELLS
)
DEFAULT_RC_CELL_POLARIZATION_RESISTANCE_OHM = (
    DEFAULT_RC_PACK_POLARIZATION_RESISTANCE_OHM
    * DEFAULT_PACK_PARALLEL_CELLS
    / DEFAULT_PACK_SERIES_CELLS
)
DEFAULT_RC_CELL_POLARIZATION_CAPACITANCE_F = (
    DEFAULT_RC_PACK_POLARIZATION_CAPACITANCE_F
    * DEFAULT_PACK_SERIES_CELLS
    / DEFAULT_PACK_PARALLEL_CELLS
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
    def electrical_discharge_power_limit_w(self) -> float:
        """Maximum safe terminal discharge power at present OCV, in watts."""

        if self.state_of_charge <= 0.0:
            return 0.0
        open_circuit_voltage_v = self.open_circuit_voltage_v
        resistance_ohm = self.internal_resistance_ohm
        current_for_low_voltage_bound_a = max(
            0.0,
            (open_circuit_voltage_v - self.minimum_terminal_voltage_v) / resistance_ohm,
        )
        current_for_peak_power_a = open_circuit_voltage_v / (2.0 * resistance_ohm)
        allowed_current_a = min(
            current_for_low_voltage_bound_a,
            current_for_peak_power_a,
        )
        return allowed_current_a * (
            open_circuit_voltage_v - allowed_current_a * resistance_ohm
        )

    @property
    def electrical_charge_power_limit_w(self) -> float:
        """Maximum safe terminal charge-power magnitude at present OCV, in W."""

        if self.state_of_charge >= 1.0:
            return 0.0
        charge_current_a = max(
            0.0,
            (self.maximum_terminal_voltage_v - self.open_circuit_voltage_v)
            / self.internal_resistance_ohm,
        )
        return charge_current_a * (
            self.open_circuit_voltage_v
            + charge_current_a * self.internal_resistance_ohm
        )

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
        if power_w == 0.0:
            return 0.0

        open_circuit_voltage_v = self.open_circuit_voltage_v
        discriminant = (
            open_circuit_voltage_v**2 - 4.0 * self.internal_resistance_ohm * power_w
        )
        if discriminant < 0.0:
            raise ValueError("power_w has no physical terminal-current solution")
        # This rearrangement of the quadratic low-current root avoids loss of
        # precision for ordinary low-current operating points.
        return 2.0 * power_w / (open_circuit_voltage_v + discriminant**0.5)

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

    @property
    def electrical_discharge_power_limit_w(self) -> float:
        """Maximum safe terminal discharge power at the present RC state."""

        if self.state_of_charge <= 0.0:
            return 0.0
        source_voltage_v = self.effective_source_voltage_v
        resistance_ohm = self.internal_resistance_ohm
        current_for_low_voltage_bound_a = max(
            0.0,
            (source_voltage_v - self.minimum_terminal_voltage_v) / resistance_ohm,
        )
        current_for_peak_power_a = max(
            0.0,
            source_voltage_v / (2.0 * resistance_ohm),
        )
        allowed_current_a = min(
            current_for_low_voltage_bound_a,
            current_for_peak_power_a,
        )
        return allowed_current_a * (
            source_voltage_v - allowed_current_a * resistance_ohm
        )

    @property
    def electrical_charge_power_limit_w(self) -> float:
        """Maximum safe terminal charge magnitude at the present RC state."""

        if self.state_of_charge >= 1.0:
            return 0.0
        charge_current_a = max(
            0.0,
            (self.maximum_terminal_voltage_v - self.effective_source_voltage_v)
            / self.internal_resistance_ohm,
        )
        return charge_current_a * (
            self.effective_source_voltage_v
            + charge_current_a * self.internal_resistance_ohm
        )

    def current_for_terminal_power_a(self, power_w: float) -> float:
        """Return current for terminal power at the present RC state."""

        if power_w > self.discharge_power_limit_w:
            raise ValueError("power_w exceeds present discharge_power_limit_w")
        if -power_w > self.charge_power_limit_w:
            raise ValueError("charging power exceeds present charge_power_limit_w")
        if power_w == 0.0:
            return 0.0

        source_voltage_v = self.effective_source_voltage_v
        discriminant = (
            source_voltage_v**2 - 4.0 * self.internal_resistance_ohm * power_w
        )
        if discriminant < 0.0:
            raise ValueError("power_w has no physical terminal-current solution")
        return 2.0 * power_w / (source_voltage_v + discriminant**0.5)

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

        charge_delta_ah = current_a * timestep_s / 3600.0
        self.net_charge_removed_ah += charge_delta_ah
        self.coulomb_throughput_ah += abs(charge_delta_ah)
        self.state_of_charge = min(
            1.0,
            max(0.0, self.state_of_charge - charge_delta_ah / self.pack_capacity_ah),
        )

        decay = exp(-timestep_s / self.polarization_time_constant_s)
        self.polarization_voltage_v = (
            decay * operating_polarization_voltage_v
            + self.polarization_resistance_ohm * (1.0 - decay) * current_a
        )


@dataclass(slots=True)
class Battery:
    """Ideal constant-power source with no voltage or SOC state."""

    max_discharge_power_w: float = DEFAULT_MAX_DISCHARGE_POWER_W
    max_charge_power_w: float = DEFAULT_MAX_CHARGE_POWER_W
    current_power_w: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable battery parameters."""

        if self.max_discharge_power_w <= 0:
            raise ValueError("max_discharge_power_w must be positive")
        if self.max_charge_power_w < 0:
            raise ValueError("max_charge_power_w cannot be negative")

    def reset_state(self) -> None:
        """Clear the battery's current operating power."""

        self.current_power_w = 0.0

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry.update(
            {
                "battery.power_w": self.current_power_w,
                "battery.discharge_power_limit_w": self.discharge_power_limit_w,
                "battery.charge_power_limit_w": self.charge_power_limit_w,
            }
        )

    @property
    def discharge_power_limit_w(self) -> float:
        """Present DC-terminal discharge limit exposed to the inverter."""

        return self.max_discharge_power_w

    @property
    def charge_power_limit_w(self) -> float:
        """Present DC-terminal charge limit exposed to regenerative systems."""

        return self.max_charge_power_w

    def update_state(self, power_w: float, timestep_s: float) -> None:
        """Retain terminal power; positive discharges and negative charges."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        if power_w > self.max_discharge_power_w:
            raise ValueError("power_w exceeds max_discharge_power_w")
        if -power_w > self.max_charge_power_w:
            raise ValueError("charging power exceeds max_charge_power_w")
        self.current_power_w = power_w

    def limit_discharge_power_w(self, requested_power_w: float) -> float:
        """Apply the battery's present discharge-power limit."""

        if requested_power_w < 0:
            raise ValueError("requested_power_w cannot be negative")
        return min(requested_power_w, self.max_discharge_power_w)

    def limit_charge_power_w(self, requested_power_w: float) -> float:
        """Apply the battery's charge-power limit to a positive magnitude."""

        if requested_power_w < 0:
            raise ValueError("requested_power_w cannot be negative")
        return min(requested_power_w, self.max_charge_power_w)
