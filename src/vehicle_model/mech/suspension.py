"""Quasi-static longitudinal and lateral tire-load distribution."""

from dataclasses import dataclass, field
from math import isfinite, pi

from ..aero import AeroForces
from ..interfaces import ChassisModel
from .loads import TireNormalLoads


DEFAULT_TOTAL_ROLL_STIFFNESS_NM_PER_DEG = 900.0
DEFAULT_TOTAL_ROLL_STIFFNESS_NM_PER_RAD = (
    DEFAULT_TOTAL_ROLL_STIFFNESS_NM_PER_DEG * 180.0 / pi
)
DEFAULT_FRONT_LATERAL_LOAD_TRANSFER_FRACTION = 0.55


@dataclass(slots=True)
class Suspension:
    """Quasi-static load transfer with an elastic roll-stiffness split.

    The baseline neglects geometric and unsprung-mass load transfer.  Total
    sprung roll moment is ``mass * lateral_acceleration * cg_height``.  Roll
    stiffness determines body roll angle.  The elastic front/rear stiffness
    split is solved from the requested total lateral load-transfer
    distribution after including roll-axis geometric transfer.  Positive
    lateral acceleration denotes a left turn and therefore transfers normal
    load from the left tires to the right tires.
    """

    total_roll_stiffness_nm_per_rad: float = (
        DEFAULT_TOTAL_ROLL_STIFFNESS_NM_PER_RAD
    )
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
    _calculated_body_roll_angle_rad: float = field(init=False, default=0.0)
    _calculated_total_roll_moment_nm: float = field(init=False, default=0.0)
    _calculated_elastic_roll_moment_nm: float = field(init=False, default=0.0)
    _calculated_front_roll_stiffness_fraction: float = field(
        init=False, default=0.0
    )
    _calculated_front_lateral_load_transfer_n: float = field(
        init=False, default=0.0
    )
    _calculated_rear_lateral_load_transfer_n: float = field(
        init=False, default=0.0
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
        self._calculated_body_roll_angle_rad = 0.0
        self._calculated_total_roll_moment_nm = 0.0
        self._calculated_elastic_roll_moment_nm = 0.0
        self._calculated_front_roll_stiffness_fraction = 0.0
        self._calculated_front_lateral_load_transfer_n = 0.0
        self._calculated_rear_lateral_load_transfer_n = 0.0

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
                "suspension.total_roll_moment_nm": (
                    self.current_total_roll_moment_nm
                ),
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
        self.current_body_roll_angle_rad = self._calculated_body_roll_angle_rad
        self.current_total_roll_moment_nm = self._calculated_total_roll_moment_nm
        self.current_elastic_roll_moment_nm = (
            self._calculated_elastic_roll_moment_nm
        )
        self.current_front_roll_stiffness_fraction = (
            self._calculated_front_roll_stiffness_fraction
        )
        self.current_front_lateral_load_transfer_n = (
            self._calculated_front_lateral_load_transfer_n
        )
        self.current_rear_lateral_load_transfer_n = (
            self._calculated_rear_lateral_load_transfer_n
        )

    @staticmethod
    def elastic_roll_arm_m(chassis: ChassisModel) -> float:
        cg_from_front_fraction = 1.0 - chassis.static_front_weight_fraction
        roll_axis_height_at_cg_m = (
            chassis.front_roll_axis_height_m
            + cg_from_front_fraction
            * (
                chassis.rear_roll_axis_height_m
                - chassis.front_roll_axis_height_m
            )
        )
        return max(chassis.cg_height_m - roll_axis_height_at_cg_m, 0.0)

    def body_roll_angle_rad(
        self,
        mass_kg: float,
        chassis: ChassisModel,
        lateral_acceleration_mps2: float,
    ) -> float:
        """Return quasi-static sprung-body roll for a lateral acceleration."""

        return (
            mass_kg
            * lateral_acceleration_mps2
            * self.elastic_roll_arm_m(chassis)
            / self.total_roll_stiffness_nm_per_rad
        )

    def tire_normal_loads_n(
        self,
        mass_kg: float,
        gravity_mps2: float,
        aero_forces: AeroForces,
        chassis: ChassisModel,
        longitudinal_acceleration_mps2: float = 0.0,
        lateral_acceleration_mps2: float = 0.0,
    ) -> TireNormalLoads:
        """Calculate per-tire loads from pitch and quasi-static body roll."""

        static_front_axle_n = (
            chassis.static_front_weight_fraction * mass_kg * gravity_mps2
        )
        static_rear_axle_n = (
            (1.0 - chassis.static_front_weight_fraction) * mass_kg * gravity_mps2
        )
        front_without_transfer_n = static_front_axle_n + aero_forces.front_downforce_n
        rear_without_transfer_n = static_rear_axle_n + aero_forces.rear_downforce_n
        rearward_transfer_n = (
            mass_kg
            * longitudinal_acceleration_mps2
            * chassis.cg_height_m
            / chassis.wheelbase_m
        )
        rearward_transfer_n = min(
            max(rearward_transfer_n, -rear_without_transfer_n),
            front_without_transfer_n,
        )
        front_axle_n = front_without_transfer_n - rearward_transfer_n
        rear_axle_n = rear_without_transfer_n + rearward_transfer_n

        total_lateral_force_n = mass_kg * lateral_acceleration_mps2
        elastic_roll_arm_m = self.elastic_roll_arm_m(chassis)
        total_roll_moment_nm = total_lateral_force_n * chassis.cg_height_m
        elastic_roll_moment_nm = total_lateral_force_n * elastic_roll_arm_m
        front_geometric_roll_moment_nm = (
            chassis.static_front_weight_fraction
            * total_lateral_force_n
            * chassis.front_roll_axis_height_m
        )
        rear_geometric_roll_moment_nm = (
            (1.0 - chassis.static_front_weight_fraction)
            * total_lateral_force_n
            * chassis.rear_roll_axis_height_m
        )
        body_roll_angle_rad = self.body_roll_angle_rad(
            mass_kg,
            chassis,
            lateral_acceleration_mps2,
        )

        target_front_fraction = self.front_lateral_load_transfer_fraction
        target_ratio = (
            target_front_fraction / (1.0 - target_front_fraction)
            if target_front_fraction < 1.0
            else float("inf")
        )
        if abs(elastic_roll_moment_nm) <= 1e-12:
            front_roll_stiffness_fraction = target_front_fraction
        elif target_front_fraction >= 1.0:
            front_roll_stiffness_fraction = 1.0
        elif target_front_fraction <= 0.0:
            front_roll_stiffness_fraction = 0.0
        else:
            numerator = (
                target_ratio
                * chassis.front_track_width_m
                * (rear_geometric_roll_moment_nm + elastic_roll_moment_nm)
                - chassis.rear_track_width_m * front_geometric_roll_moment_nm
            )
            denominator = elastic_roll_moment_nm * (
                chassis.rear_track_width_m
                + target_ratio * chassis.front_track_width_m
            )
            front_roll_stiffness_fraction = min(
                max(numerator / denominator, 0.0),
                1.0,
            )
        front_roll_stiffness_nm_per_rad = (
            front_roll_stiffness_fraction
            * self.total_roll_stiffness_nm_per_rad
        )
        rear_roll_stiffness_nm_per_rad = (
            self.total_roll_stiffness_nm_per_rad
            - front_roll_stiffness_nm_per_rad
        )
        front_transfer_n = (
            front_geometric_roll_moment_nm
            + front_roll_stiffness_nm_per_rad * body_roll_angle_rad
        ) / chassis.front_track_width_m
        rear_transfer_n = (
            rear_geometric_roll_moment_nm
            + rear_roll_stiffness_nm_per_rad * body_roll_angle_rad
        ) / chassis.rear_track_width_m
        front_transfer_n = min(
            max(front_transfer_n, -front_axle_n / 2.0),
            front_axle_n / 2.0,
        )
        rear_transfer_n = min(
            max(rear_transfer_n, -rear_axle_n / 2.0),
            rear_axle_n / 2.0,
        )

        self._calculated_body_roll_angle_rad = body_roll_angle_rad
        self._calculated_total_roll_moment_nm = total_roll_moment_nm
        self._calculated_elastic_roll_moment_nm = elastic_roll_moment_nm
        self._calculated_front_roll_stiffness_fraction = (
            front_roll_stiffness_fraction
        )
        self._calculated_front_lateral_load_transfer_n = front_transfer_n
        self._calculated_rear_lateral_load_transfer_n = rear_transfer_n
        return TireNormalLoads(
            front_left_n=front_axle_n / 2.0 - front_transfer_n,
            front_right_n=front_axle_n / 2.0 + front_transfer_n,
            rear_left_n=rear_axle_n / 2.0 - rear_transfer_n,
            rear_right_n=rear_axle_n / 2.0 + rear_transfer_n,
        )
