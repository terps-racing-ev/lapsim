"""Battery parameters and electrical power limits."""

from dataclasses import dataclass, field


DEFAULT_MAX_DISCHARGE_POWER_W = 80_000.0
DEFAULT_MAX_CHARGE_POWER_W = 0.0


@dataclass(slots=True)
class Battery:
    """Simple constant-power battery model prepared for future extensions."""

    max_discharge_power_w: float = DEFAULT_MAX_DISCHARGE_POWER_W
    max_charge_power_w: float = DEFAULT_MAX_CHARGE_POWER_W
    current_power_w: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the current mutable battery parameters."""

        if self.max_discharge_power_w <= 0:
            raise ValueError("max_discharge_power_w must be positive")
        if self.max_charge_power_w < 0:
            raise ValueError("max_charge_power_w cannot be negative")

    def reset_state(self) -> None:
        """Clear the battery's current operating power."""

        self.current_power_w = 0.0

    def update_state(self, power_w: float, timestep_s: float) -> None:
        """Retain terminal power; positive discharges and negative charges."""

        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        if power_w > self.max_discharge_power_w:
            raise ValueError("power_w exceeds max_discharge_power_w")
        if -power_w > self.max_charge_power_w:
            raise ValueError("charging power exceeds max_charge_power_w")
        self.current_power_w = power_w


    def limit_discharge_power_w(self, requested_power_w: float) -> float:
        """Apply the battery's present discharge-power limit."""

        if requested_power_w < 0:
            raise ValueError("requested_power_w cannot be negative")
        return min(requested_power_w, self.max_discharge_power_w)

    def limit_charge_power_w(self, requested_power_w: float) -> float:
        """Apply the battery's charge-power limit to a positive magnitude."""

        if requested_power_w < 0:
            raise ValueError("requested_power_w cannot be negative")
        return min(requested_power_w, self.max_charge_power_w)
