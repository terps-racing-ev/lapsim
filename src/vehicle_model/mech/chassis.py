"""Mechanical-subteam chassis geometry and mass distribution."""

from dataclasses import dataclass

from utils.units import inches_to_meters

from .loads import TireNormalLoads

DEFAULT_WHEELBASE_IN = 61.0
DEFAULT_CG_HEIGHT_IN = 11.0
DEFAULT_FRONT_TRACK_WIDTH_IN = 48.0
DEFAULT_REAR_TRACK_WIDTH_IN = 46.0
DEFAULT_FRONT_AXLE_HEIGHT_IN = 8.0
DEFAULT_REAR_AXLE_HEIGHT_IN = 8.0
DEFAULT_FRONT_ROLL_AXIS_HEIGHT_IN = 2.088
DEFAULT_REAR_ROLL_AXIS_HEIGHT_IN = 2.884
DEFAULT_WHEELBASE_M = inches_to_meters(DEFAULT_WHEELBASE_IN)
DEFAULT_CG_HEIGHT_M = inches_to_meters(DEFAULT_CG_HEIGHT_IN)
DEFAULT_FRONT_TRACK_WIDTH_M = inches_to_meters(DEFAULT_FRONT_TRACK_WIDTH_IN)
DEFAULT_REAR_TRACK_WIDTH_M = inches_to_meters(DEFAULT_REAR_TRACK_WIDTH_IN)
DEFAULT_FRONT_AXLE_HEIGHT_M = inches_to_meters(DEFAULT_FRONT_AXLE_HEIGHT_IN)
DEFAULT_REAR_AXLE_HEIGHT_M = inches_to_meters(DEFAULT_REAR_AXLE_HEIGHT_IN)
DEFAULT_FRONT_ROLL_AXIS_HEIGHT_M = inches_to_meters(
    DEFAULT_FRONT_ROLL_AXIS_HEIGHT_IN
)
DEFAULT_REAR_ROLL_AXIS_HEIGHT_M = inches_to_meters(DEFAULT_REAR_ROLL_AXIS_HEIGHT_IN)
DEFAULT_STATIC_FRONT_WEIGHT_FRACTION = 0.47


@dataclass(slots=True)
class Chassis:
    """Geometry consumed by steering and suspension models."""

    wheelbase_m: float = DEFAULT_WHEELBASE_M
    cg_height_m: float = DEFAULT_CG_HEIGHT_M
    front_track_width_m: float = DEFAULT_FRONT_TRACK_WIDTH_M
    rear_track_width_m: float = DEFAULT_REAR_TRACK_WIDTH_M
    front_axle_height_m: float = DEFAULT_FRONT_AXLE_HEIGHT_M
    rear_axle_height_m: float = DEFAULT_REAR_AXLE_HEIGHT_M
    front_roll_axis_height_m: float = DEFAULT_FRONT_ROLL_AXIS_HEIGHT_M
    rear_roll_axis_height_m: float = DEFAULT_REAR_ROLL_AXIS_HEIGHT_M
    static_front_weight_fraction: float = DEFAULT_STATIC_FRONT_WEIGHT_FRACTION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable chassis parameters."""

        if self.wheelbase_m <= 0:
            raise ValueError("wheelbase_m must be positive")
        if self.cg_height_m <= 0:
            raise ValueError("cg_height_m must be positive")
        if self.front_track_width_m <= 0:
            raise ValueError("front_track_width_m must be positive")
        if self.rear_track_width_m <= 0:
            raise ValueError("rear_track_width_m must be positive")
        if self.front_axle_height_m < 0:
            raise ValueError("front_axle_height_m cannot be negative")
        if self.rear_axle_height_m < 0:
            raise ValueError("rear_axle_height_m cannot be negative")
        if self.front_roll_axis_height_m < 0:
            raise ValueError("front_roll_axis_height_m cannot be negative")
        if self.rear_roll_axis_height_m < 0:
            raise ValueError("rear_roll_axis_height_m cannot be negative")
        if not 0 <= self.static_front_weight_fraction <= 1:
            raise ValueError("static_front_weight_fraction must be between 0 and 1")

    def reset_state(self) -> None:
        """The baseline chassis has no dynamic state."""

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry.update(
            {
                "chassis.wheelbase_m": self.wheelbase_m,
                "chassis.cg_height_m": self.cg_height_m,
                "chassis.front_track_width_m": self.front_track_width_m,
                "chassis.rear_track_width_m": self.rear_track_width_m,
                "chassis.front_axle_height_m": self.front_axle_height_m,
                "chassis.rear_axle_height_m": self.rear_axle_height_m,
                "chassis.front_roll_axis_height_m": (
                    self.front_roll_axis_height_m
                ),
                "chassis.rear_roll_axis_height_m": self.rear_roll_axis_height_m,
                "chassis.static_front_weight_fraction": (
                    self.static_front_weight_fraction
                ),
            }
        )


__all__ = ["Chassis", "TireNormalLoads"]
