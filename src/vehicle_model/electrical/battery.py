"""Compatibility imports for battery models and their existing default names.

Use battery_ideal.py for an ideal source, battery_pack.py for OCV/RC history,
and battery_circuit.py for the instantaneous terminal equations.
"""

from .battery_ideal import Battery
from .battery_pack import OCVPackBattery, RCTheveninBattery
from .battery_ocv import CELL_OCV_TABLE
from .battery_parameters import (
    DEFAULT_MAX_DISCHARGE_POWER_W,
    DEFAULT_MAX_CHARGE_POWER_W,
    DEFAULT_PACK_SERIES_CELLS,
    DEFAULT_PACK_PARALLEL_CELLS,
    DEFAULT_PACK_CAPACITY_AH,
    DEFAULT_CELL_CAPACITY_AH,
    DEFAULT_PACK_INTERNAL_RESISTANCE_OHM,
    DEFAULT_CELL_INTERNAL_RESISTANCE_OHM,
    DEFAULT_MINIMUM_CELL_VOLTAGE_V,
    DEFAULT_MAXIMUM_CELL_VOLTAGE_V,
    DEFAULT_RC_PACK_OHMIC_RESISTANCE_OHM,
    DEFAULT_RC_PACK_POLARIZATION_RESISTANCE_OHM,
    DEFAULT_RC_PACK_POLARIZATION_CAPACITANCE_F,
    DEFAULT_RC_CELL_OHMIC_RESISTANCE_OHM,
    DEFAULT_RC_CELL_POLARIZATION_RESISTANCE_OHM,
    DEFAULT_RC_CELL_POLARIZATION_CAPACITANCE_F,
)

__all__ = ["Battery", "OCVPackBattery", "RCTheveninBattery"]
