"""Public import facade for the supported Pacejka coefficient models.

Equations and coefficients live together in a module for each generation and
force direction. Tire adapts their raw inclination convention to vehicle camber.
"""

from .pacejka61 import Pacejka61LateralModel
from .pacejka52_lateral import Pacejka52UpcR20LateralModel
from .pacejka52_longitudinal import Pacejka52UpcR20LongitudinalModel

__all__ = [
    "Pacejka61LateralModel",
    "Pacejka52UpcR20LateralModel",
    "Pacejka52UpcR20LongitudinalModel",
]
