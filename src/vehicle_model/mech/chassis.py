"""Mechanical-subteam chassis geometry and mass distribution."""

from dataclasses import dataclass

from utils.units import inches_to_meters

from .loads import TireNormalLoads

DEFAULT_WHEELBASE_IN = 61.0
DEFAULT_CG_HEIGHT_IN = 11.0
DEFAULT_WHEELBASE_M = inches_to_meters(DEFAULT_WHEELBASE_IN)
DEFAULT_CG_HEIGHT_M = inches_to_meters(DEFAULT_CG_HEIGHT_IN)
DEFAULT_STATIC_FRONT_WEIGHT_FRACTION = 0.47


@dataclass(slots=True)
class Chassis:
    """Geometry consumed by steering and suspension models."""

    wheelbase_m: float = DEFAULT_WHEELBASE_M
    cg_height_m: float = DEFAULT_CG_HEIGHT_M
    static_front_weight_fraction: float = DEFAULT_STATIC_FRONT_WEIGHT_FRACTION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable chassis parameters."""

        if self.wheelbase_m <= 0:
            raise ValueError("wheelbase_m must be positive")
        if self.cg_height_m <= 0:
            raise ValueError("cg_height_m must be positive")
        if not 0 <= self.static_front_weight_fraction <= 1:
            raise ValueError("static_front_weight_fraction must be between 0 and 1")

    def reset_state(self) -> None:
        """The baseline chassis has no dynamic state."""

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry.update(
            {
                "chassis.wheelbase_m": self.wheelbase_m,
                "chassis.cg_height_m": self.cg_height_m,
                "chassis.static_front_weight_fraction": (
                    self.static_front_weight_fraction
                ),
            }
        )


__all__ = ["Chassis", "TireNormalLoads"]
