"""Pack layout, voltage bounds, and calibrated battery defaults.

Cell parameters scale to pack values through series/parallel counts. OCV data
is maintained separately in battery_ocv.py.
"""

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
