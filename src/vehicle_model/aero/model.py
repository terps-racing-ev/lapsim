"""Aero-subteam parameters and force calculations."""

from dataclasses import dataclass, field
from math import degrees, isfinite, radians
from typing import Literal

from utils.units import square_inches_to_square_meters

LEGACY_FRONTAL_AREA_IN2 = 1019.902
LEGACY_FRONTAL_AREA_M2 = square_inches_to_square_meters(LEGACY_FRONTAL_AREA_IN2)
DEFAULT_FRONTAL_AREA_M2 = 0.983996414
REFERENCE_AREA_COEFFICIENT_SCALE = (
    LEGACY_FRONTAL_AREA_M2 / DEFAULT_FRONTAL_AREA_M2
)
# The supplied coefficients used the legacy 1019.902 in^2 reference area.
# Rescale them to the real package area while preserving CdA and ClA, then
# convert positive downforce to this model's SAE signed-lift convention.
SUPPLIED_DOWNFORCE_COEFFICIENT = 3.62
SUPPLIED_DRAG_COEFFICIENT = 2.4
DEFAULT_DOWNFORCE_COEFFICIENT = (
    SUPPLIED_DOWNFORCE_COEFFICIENT * REFERENCE_AREA_COEFFICIENT_SCALE
)
DEFAULT_LIFT_TO_DRAG_RATIO = 2.946
DEFAULT_DRAG_COEFFICIENT = (
    SUPPLIED_DRAG_COEFFICIENT * REFERENCE_AREA_COEFFICIENT_SCALE
)
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


@dataclass(frozen=True, slots=True)
class AeroForces:
    """Aerodynamic forces at one vehicle speed."""

    drag_n: float
    downforce_n: float
    front_downforce_n: float
    rear_downforce_n: float
    downforce_multiplier: float = 1.0
    body_roll_angle_rad: float = 0.0
    drag_coefficient_multiplier: float = 1.0
    downforce_coefficient_multiplier: float = 1.0
    low_drag_mode_active: bool = False


@dataclass(slots=True)
class Aero:
    """Speed-squared drag and lift model with explicit reference area."""

    frontal_area_m2: float = DEFAULT_FRONTAL_AREA_M2
    drag_coefficient: float = DEFAULT_DRAG_COEFFICIENT
    lift_coefficient: float = DEFAULT_LIFT_COEFFICIENT
    front_downforce_fraction: float = DEFAULT_FRONT_DOWNFORCE_FRACTION
    roll_limit_rad: float = DEFAULT_ROLL_LIMIT_RAD
    downforce_retention_at_roll_limit: float = (
        DEFAULT_DOWNFORCE_RETENTION_AT_ROLL_LIMIT
    )
    current_forces_n: AeroForces = field(
        init=False,
        default_factory=lambda: AeroForces(0.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable aerodynamic parameters."""

        if not isfinite(self.frontal_area_m2) or self.frontal_area_m2 <= 0:
            raise ValueError("frontal_area_m2 must be finite and positive")
        if not isfinite(self.drag_coefficient) or self.drag_coefficient < 0:
            raise ValueError("drag_coefficient must be finite and nonnegative")
        if not isfinite(self.lift_coefficient):
            raise ValueError("lift_coefficient must be finite")
        if not 0 <= self.front_downforce_fraction <= 1:
            raise ValueError("front_downforce_fraction must be between 0 and 1")
        if not isfinite(self.roll_limit_rad) or self.roll_limit_rad <= 0.0:
            raise ValueError("roll_limit_rad must be finite and positive")
        if not 0.0 <= self.downforce_retention_at_roll_limit <= 1.0:
            raise ValueError(
                "downforce_retention_at_roll_limit must be between 0 and 1"
            )

    @property
    def drag_area_m2(self) -> float:
        """Return CdA for compatibility with coefficient-area consumers."""

        return self.drag_coefficient * self.frontal_area_m2

    @drag_area_m2.setter
    def drag_area_m2(self, value: float) -> None:
        if value < 0:
            raise ValueError("drag_area_m2 cannot be negative")
        self.drag_coefficient = value / self.frontal_area_m2

    @property
    def downforce_area_m2(self) -> float:
        """Return positive ClA magnitude when lift coefficient is negative."""

        return -self.lift_coefficient * self.frontal_area_m2

    @downforce_area_m2.setter
    def downforce_area_m2(self, value: float) -> None:
        if value < 0:
            raise ValueError("downforce_area_m2 cannot be negative")
        self.lift_coefficient = -value / self.frontal_area_m2

    def reset_state(self) -> None:
        """Clear the most recently calculated aerodynamic forces."""

        self.current_forces_n = AeroForces(0.0, 0.0, 0.0, 0.0)

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        forces = self.current_forces_n
        telemetry.update(
            {
                "aero.drag_n": forces.drag_n,
                "aero.downforce_n": forces.downforce_n,
                "aero.front_downforce_n": forces.front_downforce_n,
                "aero.rear_downforce_n": forces.rear_downforce_n,
                "aero.frontal_area_m2": self.frontal_area_m2,
                "aero.drag_coefficient": self.drag_coefficient,
                "aero.lift_coefficient": self.lift_coefficient,
                "aero.drag_area_m2": self.drag_area_m2,
                "aero.downforce_area_m2": self.downforce_area_m2,
                "aero.body_roll_angle_rad": forces.body_roll_angle_rad,
                "aero.body_roll_angle_deg": degrees(forces.body_roll_angle_rad),
                "aero.downforce_multiplier": forces.downforce_multiplier,
                "aero.roll_limit_rad": self.roll_limit_rad,
                "aero.roll_limit_deg": degrees(self.roll_limit_rad),
                "aero.downforce_retention_at_roll_limit": (
                    self.downforce_retention_at_roll_limit
                ),
                "aero.effective_drag_coefficient": (
                    self.drag_coefficient * forces.drag_coefficient_multiplier
                ),
                "aero.effective_lift_coefficient": (
                    self.lift_coefficient * forces.downforce_coefficient_multiplier
                ),
                "aero.low_drag_mode_active": float(forces.low_drag_mode_active),
            }
        )

    def update_state(
        self,
        vehicle_speed_mps: float,
        air_density_kgpm3: float,
        timestep_s: float,
        body_roll_angle_rad: float = 0.0,
    ) -> AeroForces:
        """Calculate and retain the aerodynamic forces for one timestep."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.current_forces_n = self.forces_n(
            vehicle_speed_mps,
            air_density_kgpm3,
            body_roll_angle_rad,
        )
        return self.current_forces_n

    def forces_n(
        self,
        vehicle_speed_mps: float,
        air_density_kgpm3: float,
        body_roll_angle_rad: float = 0.0,
    ) -> AeroForces:
        """Calculate drag and axle downforce at the requested speed."""

        if vehicle_speed_mps < 0:
            raise ValueError("vehicle_speed_mps cannot be negative")
        if air_density_kgpm3 <= 0:
            raise ValueError("air_density_kgpm3 must be positive")
        if not isfinite(body_roll_angle_rad):
            raise ValueError("body_roll_angle_rad must be finite")

        dynamic_pressure_pa = 0.5 * air_density_kgpm3 * vehicle_speed_mps**2
        drag_n = dynamic_pressure_pa * self.drag_coefficient * self.frontal_area_m2
        downforce_multiplier = self.downforce_multiplier(body_roll_angle_rad)
        downforce_n = downforce_multiplier * (
            -dynamic_pressure_pa * self.lift_coefficient * self.frontal_area_m2
        )
        front_downforce_n = self.front_downforce_fraction * downforce_n
        rear_downforce_n = downforce_n - front_downforce_n
        return AeroForces(
            drag_n=drag_n,
            downforce_n=downforce_n,
            front_downforce_n=front_downforce_n,
            rear_downforce_n=rear_downforce_n,
            downforce_multiplier=downforce_multiplier,
            body_roll_angle_rad=body_roll_angle_rad,
        )

    def downforce_multiplier(self, body_roll_angle_rad: float) -> float:
        """Linearly reduce downforce to the retained fraction at the roll limit."""

        if not isfinite(body_roll_angle_rad):
            raise ValueError("body_roll_angle_rad must be finite")
        roll_fraction = min(abs(body_roll_angle_rad) / self.roll_limit_rad, 1.0)
        return 1.0 - roll_fraction * (
            1.0 - self.downforce_retention_at_roll_limit
        )


@dataclass(slots=True)
class ActiveAero(Aero):
    """Two-position aero package with automatic straight-line deployment.

    ``automatic`` selects the low-drag configuration when absolute path
    curvature is no greater than ``straight_curvature_threshold_per_m``.
    The two forced modes make switching explicit for component tests, driver
    strategy studies, and fail-safe analysis.
    """

    drag_reduction_fraction: float = DEFAULT_ACTIVE_AERO_DRAG_REDUCTION_FRACTION
    downforce_reduction_fraction: float = (
        DEFAULT_ACTIVE_AERO_DOWNFORCE_REDUCTION_FRACTION
    )
    straight_curvature_threshold_per_m: float = (
        DEFAULT_ACTIVE_AERO_STRAIGHT_CURVATURE_THRESHOLD_PER_M
    )
    deployment_mode: ActiveAeroMode = "automatic"
    current_path_curvature_per_m: float = field(init=False, default=0.0)
    low_drag_mode_active: bool = field(init=False, default=False)

    def validate(self) -> None:
        super(ActiveAero, self).validate()
        for name, value in (
            ("drag_reduction_fraction", self.drag_reduction_fraction),
            ("downforce_reduction_fraction", self.downforce_reduction_fraction),
        ):
            if not isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if (
            not isfinite(self.straight_curvature_threshold_per_m)
            or self.straight_curvature_threshold_per_m < 0.0
        ):
            raise ValueError(
                "straight_curvature_threshold_per_m must be finite and nonnegative"
            )
        if self.deployment_mode not in ACTIVE_AERO_MODES:
            raise ValueError(
                f"deployment_mode must be one of {', '.join(ACTIVE_AERO_MODES)}"
            )

    def reset_state(self) -> None:
        super(ActiveAero, self).reset_state()
        self.current_path_curvature_per_m = 0.0
        self.low_drag_mode_active = False

    def set_deployment_mode(self, mode: ActiveAeroMode) -> None:
        """Select automatic operation or force either physical configuration."""

        if mode not in ACTIVE_AERO_MODES:
            raise ValueError(f"mode must be one of {', '.join(ACTIVE_AERO_MODES)}")
        self.deployment_mode = mode
        self.select_configuration(self.current_path_curvature_per_m)

    def select_configuration(self, curvature_per_m: float) -> bool:
        """Select and return low-drag state for a path operating point."""

        if not isfinite(curvature_per_m):
            raise ValueError("curvature_per_m must be finite")
        self.current_path_curvature_per_m = curvature_per_m
        if self.deployment_mode == "low_drag":
            self.low_drag_mode_active = True
        elif self.deployment_mode == "high_downforce":
            self.low_drag_mode_active = False
        else:
            self.low_drag_mode_active = (
                abs(curvature_per_m)
                <= self.straight_curvature_threshold_per_m
            )
        return self.low_drag_mode_active

    def forces_n(
        self,
        vehicle_speed_mps: float,
        air_density_kgpm3: float,
        body_roll_angle_rad: float = 0.0,
    ) -> AeroForces:
        """Calculate forces in the currently selected physical configuration."""

        forces = super(ActiveAero, self).forces_n(
            vehicle_speed_mps,
            air_density_kgpm3,
            body_roll_angle_rad,
        )
        if not self.low_drag_mode_active:
            return forces

        drag_multiplier = 1.0 - self.drag_reduction_fraction
        downforce_multiplier = 1.0 - self.downforce_reduction_fraction
        downforce_n = forces.downforce_n * downforce_multiplier
        front_downforce_n = forces.front_downforce_n * downforce_multiplier
        return AeroForces(
            drag_n=forces.drag_n * drag_multiplier,
            downforce_n=downforce_n,
            front_downforce_n=front_downforce_n,
            rear_downforce_n=downforce_n - front_downforce_n,
            downforce_multiplier=forces.downforce_multiplier,
            body_roll_angle_rad=forces.body_roll_angle_rad,
            drag_coefficient_multiplier=drag_multiplier,
            downforce_coefficient_multiplier=downforce_multiplier,
            low_drag_mode_active=True,
        )

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        super(ActiveAero, self).update_telemetry(telemetry)
        telemetry.update(
            {
                "aero.active_aero.enabled": 1.0,
                "aero.active_aero.low_drag_mode_active": float(
                    self.low_drag_mode_active
                ),
                "aero.active_aero.path_curvature_per_m": (
                    self.current_path_curvature_per_m
                ),
                "aero.active_aero.drag_reduction_fraction": (
                    self.drag_reduction_fraction
                ),
                "aero.active_aero.downforce_reduction_fraction": (
                    self.downforce_reduction_fraction
                ),
                "aero.active_aero.straight_curvature_threshold_per_m": (
                    self.straight_curvature_threshold_per_m
                ),
            }
        )
