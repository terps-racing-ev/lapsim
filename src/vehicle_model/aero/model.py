"""Compatibility imports for aero models, results, and baseline parameters.

Passive force calculation lives in passive.py; active.py owns deployment
policy. forces.py defines the shared immutable result.
"""

from .active import ActiveAero
from .forces import AeroForces
from .passive import Aero
from .parameters import (
    LEGACY_FRONTAL_AREA_IN2,
    LEGACY_FRONTAL_AREA_M2,
    DEFAULT_FRONTAL_AREA_M2,
    REFERENCE_AREA_COEFFICIENT_SCALE,
    SUPPLIED_DOWNFORCE_COEFFICIENT,
    SUPPLIED_DRAG_COEFFICIENT,
    DEFAULT_DOWNFORCE_COEFFICIENT,
    DEFAULT_LIFT_TO_DRAG_RATIO,
    DEFAULT_DRAG_COEFFICIENT,
    DEFAULT_LIFT_COEFFICIENT,
    DEFAULT_FRONT_DOWNFORCE_FRACTION,
    DEFAULT_ROLL_LIMIT_RAD,
    DEFAULT_DOWNFORCE_RETENTION_AT_ROLL_LIMIT,
    DEFAULT_ACTIVE_AERO_DRAG_REDUCTION_FRACTION,
    DEFAULT_ACTIVE_AERO_DOWNFORCE_REDUCTION_FRACTION,
    DEFAULT_ACTIVE_AERO_STRAIGHT_CURVATURE_THRESHOLD_PER_M,
    ActiveAeroMode,
    ACTIVE_AERO_MODES,
)

__all__ = ["ActiveAero", "ActiveAeroMode", "Aero", "AeroForces"]
