"""Terminal voltage, power limits, and current for a resistive pack circuit.

The source voltage may be OCV alone or OCV minus polarization. SOC and RC
history belong to the battery models, outside this instantaneous circuit.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TerminalCircuit:
    """One source and series resistance, with safe terminal-voltage bounds."""

    source_voltage_v: float
    resistance_ohm: float
    minimum_voltage_v: float
    maximum_voltage_v: float

    @property
    def discharge_power_limit_w(self) -> float:
        current_for_low_voltage_bound_a = max(
            0.0,
            (self.source_voltage_v - self.minimum_voltage_v) / self.resistance_ohm,
        )
        current_for_peak_power_a = max(
            0.0, self.source_voltage_v / (2.0 * self.resistance_ohm)
        )
        allowed_current_a = min(
            current_for_low_voltage_bound_a, current_for_peak_power_a
        )
        return allowed_current_a * (
            self.source_voltage_v - allowed_current_a * self.resistance_ohm
        )

    @property
    def charge_power_limit_w(self) -> float:
        charge_current_a = max(
            0.0,
            (self.maximum_voltage_v - self.source_voltage_v) / self.resistance_ohm,
        )
        return charge_current_a * (
            self.source_voltage_v + charge_current_a * self.resistance_ohm
        )

    def current_for_power_a(self, power_w: float) -> float:
        """Return the low-current root: positive discharge, negative charge.

        The battery checks SOC and configured power limits before calling.
        This form of the quadratic root retains precision at low power.
        """

        if power_w == 0.0:
            return 0.0
        discriminant = self.source_voltage_v**2 - 4.0 * self.resistance_ohm * power_w
        if discriminant < 0.0:
            raise ValueError("power_w has no physical terminal-current solution")
        return 2.0 * power_w / (self.source_voltage_v + discriminant**0.5)
