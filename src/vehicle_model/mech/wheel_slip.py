"""Simple quasi-static driven-wheel longitudinal slip model."""

from dataclasses import dataclass, field
from math import isfinite


DEFAULT_PEAK_LONGITUDINAL_SLIP_RATIO = 0.10


@dataclass(slots=True)
class WheelSlip:
    """Estimate driven-wheel slip from longitudinal tire-force utilization.

    The baseline assumes the stable, rising side of a tire force-slip curve:
    zero longitudinal force produces zero slip and the configured peak slip is
    reached at the available rear-tire force. It is intentionally algebraic;
    wheel-hop, relaxation length, and post-peak wheelspin are not represented.
    """

    peak_longitudinal_slip_ratio: float = DEFAULT_PEAK_LONGITUDINAL_SLIP_RATIO
    current_slip_ratio: float = field(init=False, default=0.0)
    current_force_utilization: float = field(init=False, default=0.0)
    current_vehicle_speed_mps: float = field(init=False, default=0.0)
    current_wheel_surface_speed_mps: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (
            not isfinite(self.peak_longitudinal_slip_ratio)
            or self.peak_longitudinal_slip_ratio < 0.0
            or self.peak_longitudinal_slip_ratio >= 1.0
        ):
            raise ValueError(
                "peak_longitudinal_slip_ratio must be finite and in [0, 1)"
            )

    def reset_state(self) -> None:
        self.current_slip_ratio = 0.0
        self.current_force_utilization = 0.0
        self.current_vehicle_speed_mps = 0.0
        self.current_wheel_surface_speed_mps = 0.0

    def update_state(
        self,
        vehicle_speed_mps: float,
        longitudinal_force_n: float,
        force_capacity_n: float,
        timestep_s: float,
    ) -> None:
        if vehicle_speed_mps < 0.0:
            raise ValueError("vehicle_speed_mps cannot be negative")
        if longitudinal_force_n < 0.0:
            raise ValueError("longitudinal_force_n cannot be negative")
        if force_capacity_n < 0.0:
            raise ValueError("force_capacity_n cannot be negative")
        if timestep_s <= 0.0:
            raise ValueError("timestep_s must be positive")

        utilization = 0.0
        if force_capacity_n > 0.0:
            utilization = min(longitudinal_force_n / force_capacity_n, 1.0)
        self.current_force_utilization = utilization
        self.current_slip_ratio = self.peak_longitudinal_slip_ratio * utilization
        self.current_vehicle_speed_mps = vehicle_speed_mps
        self.current_wheel_surface_speed_mps = vehicle_speed_mps * (
            1.0 + self.current_slip_ratio
        )
    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry.update(
            {
                "wheel_slip.ratio": self.current_slip_ratio,
                "wheel_slip.percent": 100.0 * self.current_slip_ratio,
                "wheel_slip.force_utilization": self.current_force_utilization,
                "wheel_slip.vehicle_speed_mps": self.current_vehicle_speed_mps,
                "wheel_slip.wheel_surface_speed_mps": (
                    self.current_wheel_surface_speed_mps
                ),
                "wheel_slip.slip_speed_mps": (
                    self.current_wheel_surface_speed_mps
                    - self.current_vehicle_speed_mps
                ),
            }
        )
