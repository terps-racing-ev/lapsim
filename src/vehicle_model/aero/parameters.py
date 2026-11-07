"""Reference-area conversion, baseline coefficients, and deployment defaults."""

from math import radians
from typing import Literal

from utils.units import square_inches_to_square_meters

LEGACY_FRONTAL_AREA_IN2 = 1019.902
LEGACY_FRONTAL_AREA_M2 = square_inches_to_square_meters(LEGACY_FRONTAL_AREA_IN2)
DEFAULT_FRONTAL_AREA_M2 = 0.983996414
REFERENCE_AREA_COEFFICIENT_SCALE = LEGACY_FRONTAL_AREA_M2 / DEFAULT_FRONTAL_AREA_M2
# The supplied coefficients used the legacy 1019.902 in^2 reference area.
# Rescale them to the real package area while preserving CdA and ClA, then
# convert positive downforce to this model's SAE signed-lift convention.
SUPPLIED_DOWNFORCE_COEFFICIENT = 3.62
SUPPLIED_DRAG_COEFFICIENT = 2.4
DEFAULT_DOWNFORCE_COEFFICIENT = (
    SUPPLIED_DOWNFORCE_COEFFICIENT * REFERENCE_AREA_COEFFICIENT_SCALE
)
DEFAULT_LIFT_TO_DRAG_RATIO = 2.946
DEFAULT_DRAG_COEFFICIENT = SUPPLIED_DRAG_COEFFICIENT * REFERENCE_AREA_COEFFICIENT_SCALE
DEFAULT_LIFT_COEFFICIENT = -DEFAULT_DOWNFORCE_COEFFICIENT
DEFAULT_FRONT_DOWNFORCE_FRACTION = 0.5269293255
DEFAULT_ROLL_LIMIT_RAD = radians(1.0)
DEFAULT_DOWNFORCE_RETENTION_AT_ROLL_LIMIT = 0.50
DEFAULT_ACTIVE_AERO_DRAG_REDUCTION_FRACTION = 0.30
DEFAULT_ACTIVE_AERO_DOWNFORCE_REDUCTION_FRACTION = 0.30
DEFAULT_ACTIVE_AERO_STRAIGHT_CURVATURE_THRESHOLD_PER_M = 0.005

ActiveAeroMode = Literal["automatic", "low_drag", "high_downforce"]
ACTIVE_AERO_MODES: tuple[ActiveAeroMode, ...] = (
    "automatic",
    "low_drag",
    "high_downforce",
)
