"""Electrical-subteam constant-efficiency inverter model."""

from dataclasses import dataclass, field

DEFAULT_EFFICIENCY = 0.97


@dataclass(slots=True)
class Inverter:
    """Constant-efficiency DC-to-motor power conversion.

    A map-based inverter can replace this class while retaining the same
    interface and can own temperature, voltage, and current state.
    """

    efficiency: float = DEFAULT_EFFICIENCY
    current_dc_input_power_w: float = field(init=False, default=0.0)
    current_motor_output_power_w: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be greater than 0 and at most 1")

    def reset_state(self) -> None:
        self.current_dc_input_power_w = 0.0
        self.current_motor_output_power_w = 0.0

    def motor_electrical_power_limit_w(
        self,
        battery_discharge_limit_w: float,
    ) -> float:
        if battery_discharge_limit_w < 0:
            raise ValueError("battery_discharge_limit_w cannot be negative")
        return battery_discharge_limit_w * self.efficiency

    def dc_power_for_motor_electrical_power_w(
        self,
        motor_electrical_power_w: float,
    ) -> float:
        if motor_electrical_power_w < 0:
            raise ValueError("motor_electrical_power_w cannot be negative")
        return motor_electrical_power_w / self.efficiency

    def update_state(
        self,
        dc_input_power_w: float,
        motor_output_power_w: float,
        timestep_s: float,
    ) -> None:
        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        if dc_input_power_w < 0 or motor_output_power_w < 0:
            raise ValueError("Propulsion power cannot be negative")
        self.current_dc_input_power_w = dc_input_power_w
        self.current_motor_output_power_w = motor_output_power_w

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry.update(
            {
                "inverter.dc_input_power_w": self.current_dc_input_power_w,
                "inverter.motor_output_power_w": (self.current_motor_output_power_w),
                "inverter.loss_power_w": max(
                    self.current_dc_input_power_w - self.current_motor_output_power_w,
                    0.0,
                ),
            }
        )
