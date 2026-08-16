"""Powertrain-subteam electric-motor model."""

from dataclasses import dataclass, field
from math import pi

DEFAULT_TORQUE_CURVE_RPM = (0.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0)
DEFAULT_TORQUE_CURVE_NM = (230.0, 230.0, 224.0, 218.0, 206.0)
DEFAULT_CONTINUOUS_TORQUE_CURVE_RPM = (
    0.0,
    5_500.0,
    6_000.0,
    6_500.0,
    7_000.0,
)
DEFAULT_CONTINUOUS_TORQUE_CURVE_NM = (82.0, 82.0, 82.0, 75.0, 70.0)
DEFAULT_PEAK_POWER_W = 80_000.0
DEFAULT_CONTINUOUS_POWER_W = 50_000.0
DEFAULT_MAX_SPEED_RPM = 7_000.0
# Constant-efficiency baseline fitted from the first 2025 endurance lap.
DEFAULT_EFFICIENCY = 0.95963664851588
DEFAULT_ROTOR_INERTIA_KGM2 = 0.01215


@dataclass(slots=True)
class Motor:
    """Torque-curve motor with constant electrical-to-shaft efficiency.

    Replace this object with any ``MotorModel`` implementation to add an
    efficiency map, thermal derating, voltage dependence, or richer state.
    """

    torque_curve_rpm: list[float] = field(
        default_factory=lambda: list(DEFAULT_TORQUE_CURVE_RPM)
    )
    torque_curve_nm: list[float] = field(
        default_factory=lambda: list(DEFAULT_TORQUE_CURVE_NM)
    )
    continuous_torque_curve_rpm: list[float] = field(
        default_factory=lambda: list(DEFAULT_CONTINUOUS_TORQUE_CURVE_RPM)
    )
    continuous_torque_curve_nm: list[float] = field(
        default_factory=lambda: list(DEFAULT_CONTINUOUS_TORQUE_CURVE_NM)
    )
    peak_power_w: float = DEFAULT_PEAK_POWER_W
    continuous_power_w: float = DEFAULT_CONTINUOUS_POWER_W
    max_speed_rpm: float = DEFAULT_MAX_SPEED_RPM
    efficiency: float = DEFAULT_EFFICIENCY
    rotor_inertia_kgm2: float = DEFAULT_ROTOR_INERTIA_KGM2
    current_torque_nm: float = field(init=False, default=0.0)
    current_speed_rpm: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.peak_power_w <= 0 or self.continuous_power_w <= 0:
            raise ValueError("Motor power limits must be positive")
        if self.continuous_power_w > self.peak_power_w:
            raise ValueError("continuous_power_w cannot exceed peak_power_w")
        if self.max_speed_rpm <= 0:
            raise ValueError("max_speed_rpm must be positive")
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be greater than 0 and at most 1")
        if self.rotor_inertia_kgm2 < 0:
            raise ValueError("rotor_inertia_kgm2 cannot be negative")
        self._validate_curve(
            self.torque_curve_rpm,
            self.torque_curve_nm,
            "torque_curve",
        )
        self._validate_curve(
            self.continuous_torque_curve_rpm,
            self.continuous_torque_curve_nm,
            "continuous_torque_curve",
        )

    @staticmethod
    def _validate_curve(
        speed_rpm: list[float],
        torque_nm: list[float],
        name: str,
    ) -> None:
        if len(speed_rpm) < 2 or len(torque_nm) != len(speed_rpm):
            raise ValueError(f"{name} requires matching arrays of at least 2 points")
        if any(value < 0 for value in speed_rpm):
            raise ValueError(f"{name} RPM values cannot be negative")
        if any(upper <= lower for lower, upper in zip(speed_rpm, speed_rpm[1:])):
            raise ValueError(f"{name} RPM values must strictly increase")
        if any(value <= 0 for value in torque_nm):
            raise ValueError(f"{name} torque values must be positive")

    def reset_state(self) -> None:
        self.current_torque_nm = 0.0
        self.current_speed_rpm = 0.0

    def update_state(
        self,
        motor_torque_nm: float,
        motor_speed_rpm: float,
        timestep_s: float,
    ) -> None:
        if timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.current_torque_nm = motor_torque_nm
        self.current_speed_rpm = motor_speed_rpm

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        mechanical_power_w = (
            self.current_torque_nm * self.current_speed_rpm * 2.0 * pi / 60.0
        )
        telemetry.update(
            {
                "motor.torque_nm": self.current_torque_nm,
                "motor.speed_rpm": self.current_speed_rpm,
                "motor.mechanical_power_w": mechanical_power_w,
                "motor.peak_torque_limit_nm": self.torque_limit_nm(
                    self.current_speed_rpm
                ),
                "motor.continuous_torque_limit_nm": (
                    self.continuous_torque_limit_nm(self.current_speed_rpm)
                ),
            }
        )

    @property
    def maximum_torque_nm(self) -> float:
        return max(self.torque_curve_nm)

    def torque_limit_nm(self, motor_speed_rpm: float) -> float:
        return self._interpolate_curve(
            motor_speed_rpm,
            self.torque_curve_rpm,
            self.torque_curve_nm,
        )

    def continuous_torque_limit_nm(self, motor_speed_rpm: float) -> float:
        return self._interpolate_curve(
            motor_speed_rpm,
            self.continuous_torque_curve_rpm,
            self.continuous_torque_curve_nm,
        )

    @staticmethod
    def _interpolate_curve(
        motor_speed_rpm: float,
        curve_speed_rpm: list[float],
        curve_torque_nm: list[float],
    ) -> float:
        if motor_speed_rpm < 0:
            raise ValueError("motor_speed_rpm cannot be negative")
        clamped_rpm = min(
            max(motor_speed_rpm, curve_speed_rpm[0]),
            curve_speed_rpm[-1],
        )
        for index, upper_rpm in enumerate(curve_speed_rpm[1:], start=1):
            if clamped_rpm <= upper_rpm:
                lower_rpm = curve_speed_rpm[index - 1]
                fraction = (clamped_rpm - lower_rpm) / (upper_rpm - lower_rpm)
                return curve_torque_nm[index - 1] + fraction * (
                    curve_torque_nm[index] - curve_torque_nm[index - 1]
                )
        return curve_torque_nm[-1]

    def mechanical_power_limit_w(
        self,
        electrical_input_limit_w: float,
    ) -> float:
        if electrical_input_limit_w < 0:
            raise ValueError("electrical_input_limit_w cannot be negative")
        return min(electrical_input_limit_w * self.efficiency, self.peak_power_w)

    def electrical_power_for_mechanical_output_w(
        self,
        mechanical_output_power_w: float,
        motor_speed_rpm: float,
        motor_torque_nm: float,
    ) -> float:
        if mechanical_output_power_w < 0:
            raise ValueError("mechanical_output_power_w cannot be negative")
        if motor_speed_rpm < 0:
            raise ValueError("motor_speed_rpm cannot be negative")
        return mechanical_output_power_w / self.efficiency
