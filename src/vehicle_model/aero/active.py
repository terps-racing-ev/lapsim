"""Two-position aero deployment policy layered over the passive force law."""

from dataclasses import dataclass, field
from math import isfinite

from .forces import AeroForces
from .passive import Aero
from .parameters import (
    DEFAULT_ACTIVE_AERO_DRAG_REDUCTION_FRACTION,
    DEFAULT_ACTIVE_AERO_DOWNFORCE_REDUCTION_FRACTION,
    DEFAULT_ACTIVE_AERO_STRAIGHT_CURVATURE_THRESHOLD_PER_M,
    ActiveAeroMode,
    ACTIVE_AERO_MODES,
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
                abs(curvature_per_m) <= self.straight_curvature_threshold_per_m
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
