"""Quasi-static longitudinal and lateral tire-load distribution."""

from dataclasses import dataclass, field
from math import isfinite, pi

from utils.units import pound_force_feet_to_newton_meters

from ..aero import AeroForces
from ..interfaces import ChassisModel
from .loads import TireNormalLoads
from .load_transfer import (
    LoadTransferResult,
    calculate_load_transfer,
    elastic_roll_arm_m,
    wheel_travel_m,
)

DEFAULT_TOTAL_ROLL_STIFFNESS_LBFT_PER_DEG = 950.2
DEFAULT_TOTAL_ROLL_STIFFNESS_NM_PER_DEG = pound_force_feet_to_newton_meters(
    DEFAULT_TOTAL_ROLL_STIFFNESS_LBFT_PER_DEG
)
DEFAULT_TOTAL_ROLL_STIFFNESS_NM_PER_RAD = (
    DEFAULT_TOTAL_ROLL_STIFFNESS_NM_PER_DEG * 180.0 / pi
)
DEFAULT_FRONT_LATERAL_LOAD_TRANSFER_FRACTION = 0.55


@dataclass(slots=True)
class Suspension:
    """Quasi-static load transfer with an elastic roll-stiffness split.

    The baseline includes roll-axis geometric transfer and elastic sprung
    transfer, and neglects unsprung-mass transfer. Total roll moment is
    ``mass * lateral_acceleration * cg_height``. Roll
    stiffness determines body roll angle.  The elastic front/rear stiffness
    split is solved from the requested total lateral load-transfer
    distribution after including roll-axis geometric transfer.  Positive
    lateral acceleration denotes a left turn and therefore transfers normal
    load from the left tires to the right tires.
    """

    total_roll_stiffness_nm_per_rad: float = DEFAULT_TOTAL_ROLL_STIFFNESS_NM_PER_RAD
    front_lateral_load_transfer_fraction: float = (
        DEFAULT_FRONT_LATERAL_LOAD_TRANSFER_FRACTION
    )

    current_tire_normal_loads_n: TireNormalLoads = field(
        init=False,
        default_factory=lambda: TireNormalLoads(0.0, 0.0, 0.0, 0.0),
    )
    current_body_roll_angle_rad: float = field(init=False, default=0.0)
    current_total_roll_moment_nm: float = field(init=False, default=0.0)
    current_elastic_roll_moment_nm: float = field(init=False, default=0.0)
    current_front_roll_stiffness_fraction: float = field(init=False, default=0.0)
    current_front_lateral_load_transfer_n: float = field(init=False, default=0.0)
    current_rear_lateral_load_transfer_n: float = field(init=False, default=0.0)
    _pending_load_transfer: LoadTransferResult = field(
        init=False, default_factory=LoadTransferResult
    )

    def validate(self) -> None:
        """Validate roll stiffness and lateral load-transfer distribution."""

        if (
            not isfinite(self.total_roll_stiffness_nm_per_rad)
            or self.total_roll_stiffness_nm_per_rad <= 0.0
        ):
            raise ValueError("total_roll_stiffness_nm_per_rad must be positive")
        if (
            not isfinite(self.front_lateral_load_transfer_fraction)
            or not 0.0 <= self.front_lateral_load_transfer_fraction <= 1.0
        ):
            raise ValueError(
                "front_lateral_load_transfer_fraction must be between 0 and 1"
            )

    def reset_state(self) -> None:
        self.current_tire_normal_loads_n = TireNormalLoads(0.0, 0.0, 0.0, 0.0)
        self.current_body_roll_angle_rad = 0.0
        self.current_total_roll_moment_nm = 0.0
        self.current_elastic_roll_moment_nm = 0.0
        self.current_front_roll_stiffness_fraction = 0.0
        self.current_front_lateral_load_transfer_n = 0.0
        self.current_rear_lateral_load_transfer_n = 0.0
        self._pending_load_transfer = LoadTransferResult()

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        loads = self.current_tire_normal_loads_n
        telemetry.update(
            {
                "suspension.front_left_normal_load_n": loads.front_left_n,
                "suspension.front_right_normal_load_n": loads.front_right_n,
                "suspension.rear_left_normal_load_n": loads.rear_left_n,
                "suspension.rear_right_normal_load_n": loads.rear_right_n,
                "suspension.front_axle_normal_load_n": loads.front_axle_n,
                "suspension.rear_axle_normal_load_n": loads.rear_axle_n,
                "suspension.total_roll_stiffness_nm_per_rad": (
                    self.total_roll_stiffness_nm_per_rad
                ),
                "suspension.total_roll_stiffness_nm_per_deg": (
                    self.total_roll_stiffness_nm_per_rad * pi / 180.0
                ),
                "suspension.front_lateral_load_transfer_fraction": (
                    self.front_lateral_load_transfer_fraction
                ),
                "suspension.body_roll_angle_rad": self.current_body_roll_angle_rad,
                "suspension.body_roll_angle_deg": (
                    self.current_body_roll_angle_rad * 180.0 / pi
                ),
                "suspension.total_roll_moment_nm": (self.current_total_roll_moment_nm),
                "suspension.elastic_roll_moment_nm": (
                    self.current_elastic_roll_moment_nm
                ),
                "suspension.front_roll_stiffness_fraction": (
                    self.current_front_roll_stiffness_fraction
                ),
                "suspension.front_lateral_load_transfer_n": (
                    self.current_front_lateral_load_transfer_n
                ),
                "suspension.rear_lateral_load_transfer_n": (
                    self.current_rear_lateral_load_transfer_n
                ),
            }
        )

    def update_state(
        self,
        tire_normal_loads_n: TireNormalLoads,
        timestep_s: float,
    ) -> None:
        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.current_tire_normal_loads_n = tire_normal_loads_n
        self.current_body_roll_angle_rad = (
            self._pending_load_transfer.body_roll_angle_rad
        )
        self.current_total_roll_moment_nm = (
            self._pending_load_transfer.total_roll_moment_nm
        )
        self.current_elastic_roll_moment_nm = (
            self._pending_load_transfer.elastic_roll_moment_nm
        )
        self.current_front_roll_stiffness_fraction = (
            self._pending_load_transfer.front_roll_stiffness_fraction
        )
        self.current_front_lateral_load_transfer_n = (
            self._pending_load_transfer.front_lateral_load_transfer_n
        )
        self.current_rear_lateral_load_transfer_n = (
            self._pending_load_transfer.rear_lateral_load_transfer_n
        )

    @staticmethod
    def elastic_roll_arm_m(chassis: ChassisModel) -> float:
        """Height of the sprung CG above the roll axis, clamped at zero."""

        return elastic_roll_arm_m(chassis)

    def body_roll_angle_rad(
        self,
        mass_kg: float,
        chassis: ChassisModel,
        lateral_acceleration_mps2: float,
    ) -> float:
        """Return candidate body roll without advancing suspension state."""

        return (
            mass_kg
            * lateral_acceleration_mps2
            * self.elastic_roll_arm_m(chassis)
            / self.total_roll_stiffness_nm_per_rad
        )

    def wheel_travel_m(
        self,
        chassis: ChassisModel,
        body_roll_angle_rad: float,
    ) -> tuple[float, float, float, float]:
        """Return roll-induced wheel travel: positive bump, negative droop."""

        return wheel_travel_m(chassis, body_roll_angle_rad)

    def tire_normal_loads_n(
        self,
        mass_kg: float,
        gravity_mps2: float,
        aero_forces: AeroForces,
        chassis: ChassisModel,
        longitudinal_acceleration_mps2: float = 0.0,
        lateral_acceleration_mps2: float = 0.0,
    ) -> TireNormalLoads:
        """Calculate candidate loads and retain their roll diagnostics.

        Repeated force-solver evaluations replace the pending result. Public
        component state changes only when update_state commits the final one.
        """

        self._pending_load_transfer = calculate_load_transfer(
            mass_kg,
            gravity_mps2,
            aero_forces,
            chassis,
            self.total_roll_stiffness_nm_per_rad,
            self.front_lateral_load_transfer_fraction,
            longitudinal_acceleration_mps2,
            lateral_acceleration_mps2,
            roll_arm_m=self.elastic_roll_arm_m(chassis),
            roll_angle_rad=self.body_roll_angle_rad(
                mass_kg, chassis, lateral_acceleration_mps2
            ),
        )
        return self._pending_load_transfer.normal_loads_n
