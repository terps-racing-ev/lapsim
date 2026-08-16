"""Powertrain-subteam sprocket-and-chain reduction model."""

from dataclasses import dataclass

DEFAULT_RATIO = 3.455
# Constant motor-shaft-to-wheel efficiency inferred from brake-free,
# positive-acceleration samples on five straights of the first endurance lap.
DEFAULT_EFFICIENCY = 0.776852813358272
DEFAULT_INPUT_INERTIA_KGM2 = 0.00005
DEFAULT_OUTPUT_INERTIA_KGM2 = 0.003


@dataclass(slots=True)
class ChainDrive:
    """Fixed-ratio sprocket-and-chain drive with constant efficiency."""

    ratio: float = DEFAULT_RATIO
    efficiency: float = DEFAULT_EFFICIENCY
    input_inertia_kgm2: float = DEFAULT_INPUT_INERTIA_KGM2
    output_inertia_kgm2: float = DEFAULT_OUTPUT_INERTIA_KGM2

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.ratio <= 0:
            raise ValueError("ratio must be positive")
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be greater than 0 and at most 1")
        if self.input_inertia_kgm2 < 0 or self.output_inertia_kgm2 < 0:
            raise ValueError("Chain-drive inertias cannot be negative")

    def reset_state(self) -> None:
        """The baseline chain drive has no dynamic state."""

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        telemetry.update(
            {
                "chain_drive.ratio": self.ratio,
                "chain_drive.efficiency": self.efficiency,
                "chain_drive.input_inertia_kgm2": self.input_inertia_kgm2,
                "chain_drive.output_inertia_kgm2": self.output_inertia_kgm2,
            }
        )

    def wheel_torque_from_motor_torque_nm(self, motor_torque_nm: float) -> float:
        return motor_torque_nm * self.ratio * self.efficiency

    def motor_torque_from_wheel_torque_nm(self, wheel_torque_nm: float) -> float:
        if wheel_torque_nm < 0:
            raise ValueError("wheel_torque_nm cannot be negative without regen")
        return wheel_torque_nm / (self.ratio * self.efficiency)

    def wheel_power_for_motor_power_w(self, motor_power_w: float) -> float:
        """Convert motor-shaft input power to driven-wheel output power."""

        if motor_power_w < 0:
            raise ValueError("motor_power_w cannot be negative without regen")
        return motor_power_w * self.efficiency

    def motor_power_for_wheel_power_w(self, wheel_power_w: float) -> float:
        """Return motor-shaft power needed for driven-wheel output power."""

        if wheel_power_w < 0:
            raise ValueError("wheel_power_w cannot be negative without regen")
        return wheel_power_w / self.efficiency


# Compatibility name for external scripts written before the terminology was
# made hardware-specific. New code should import and construct ChainDrive.
FinalDrive = ChainDrive

__all__ = ["ChainDrive", "FinalDrive"]
