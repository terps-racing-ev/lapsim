"""Stable public facade for Formula SAE event simulation.

Implementation modules are grouped by responsibility.  Heavy vehicle and
solver exports stay lazy so package initialization remains acyclic.
"""

from .core.controls import Controls

__all__ = [
    "Brakes",
    "Chassis",
    "Controls",
    "Curve",
    "ControlsProfile",
    "ConstantControlsProfile",
    "PiecewiseLinearControlsProfile",
    "AccelerationConfig",
    "SkidpadConfig",
    "EventResult",
    "simulate_acceleration",
    "simulate_endurance",
    "simulate_skidpad",
    "EnduranceOptimizationResult",
    "EnduranceControlProfile",
    "EnduranceRunConfig",
    "EnduranceRunResult",
    "EnduranceSimulator",
    "EnduranceTorqueOptimizer",
    "FSAEEnduranceEfficiencyScoring",
    "FSAE_2026_MI_ACCELERATION_SCORING",
    "FSAE_2026_MI6_SCORING",
    "FSAE_2026_MI_SKIDPAD_SCORING",
    "LapResult",
    "LapTimeSolver",
    "RecordedLap",
    "ReplayTelemetry",
    "OptimizationConfig",
    "LocalPolishConfig",
    "PathConstraintSolver",
    "PathSpeedConstraints",
    "PeriodicPiecewiseLinearTorqueProfile",
    "ScoreBreakdown",
    "TimedEventScoreBreakdown",
    "TimedEventScoring",
    "SpeedLimitMap",
    "SpeedLimitSolver",
    "Telemetry",
    "TelemetryRecorder",
    "TireNormalLoads",
    "SpatialTrack",
    "SpatialCoordinate",
    "Straight",
    "Track",
    "TorqueProfile",
    "UniformPeriodicTorqueParameterization",
    "replay_controls",
]


def __getattr__(name: str):
    if name == "Brakes":
        from vehicle_model.mech import Brakes

        return Brakes
    if name in {"Chassis", "TireNormalLoads"}:
        from vehicle_model.mech import Chassis, TireNormalLoads

        return {"Chassis": Chassis, "TireNormalLoads": TireNormalLoads}[name]
    if name in {"LapResult", "LapTimeSolver"}:
        from .solvers.lap_time import LapResult, LapTimeSolver

        return {"LapResult": LapResult, "LapTimeSolver": LapTimeSolver}[name]
    if name in {"SpeedLimitMap", "SpeedLimitSolver"}:
        from .solvers.speed_limit import SpeedLimitMap, SpeedLimitSolver

        return {
            "SpeedLimitMap": SpeedLimitMap,
            "SpeedLimitSolver": SpeedLimitSolver,
        }[name]
    if name in {"Telemetry", "TelemetryRecorder"}:
        from .core.telemetry import Telemetry, TelemetryRecorder

        return {
            "Telemetry": Telemetry,
            "TelemetryRecorder": TelemetryRecorder,
        }[name]
    if name in {
        "ConstantControlsProfile",
        "ControlsProfile",
        "PiecewiseLinearControlsProfile",
    }:
        from .core.profiles import (
            ConstantControlsProfile,
            ControlsProfile,
            PiecewiseLinearControlsProfile,
        )

        return {
            "ConstantControlsProfile": ConstantControlsProfile,
            "ControlsProfile": ControlsProfile,
            "PiecewiseLinearControlsProfile": PiecewiseLinearControlsProfile,
        }[name]
    if name in {
        "AccelerationConfig",
        "EventResult",
        "SkidpadConfig",
        "simulate_acceleration",
        "simulate_endurance",
        "simulate_skidpad",
    }:
        from .events.api import (
            AccelerationConfig,
            EventResult,
            SkidpadConfig,
            simulate_acceleration,
            simulate_endurance,
            simulate_skidpad,
        )

        return {
            "AccelerationConfig": AccelerationConfig,
            "EventResult": EventResult,
            "SkidpadConfig": SkidpadConfig,
            "simulate_acceleration": simulate_acceleration,
            "simulate_endurance": simulate_endurance,
            "simulate_skidpad": simulate_skidpad,
        }[name]
    if name == "RecordedLap":
        from .data.recorded_lap import RecordedLap

        return RecordedLap
    if name in {"ReplayTelemetry", "replay_controls"}:
        from .data.replay import ReplayTelemetry, replay_controls

        return {
            "ReplayTelemetry": ReplayTelemetry,
            "replay_controls": replay_controls,
        }[name]
    if name in {"EnduranceRunConfig", "EnduranceRunResult", "EnduranceSimulator"}:
        from .events.endurance import (
            EnduranceRunConfig,
            EnduranceRunResult,
            EnduranceSimulator,
        )

        return {
            "EnduranceRunConfig": EnduranceRunConfig,
            "EnduranceRunResult": EnduranceRunResult,
            "EnduranceSimulator": EnduranceSimulator,
        }[name]
    if name in {"PathConstraintSolver", "PathSpeedConstraints"}:
        from .solvers.path_constraints import PathConstraintSolver, PathSpeedConstraints

        return {
            "PathConstraintSolver": PathConstraintSolver,
            "PathSpeedConstraints": PathSpeedConstraints,
        }[name]
    if name in {
        "FSAEEnduranceEfficiencyScoring",
        "FSAE_2026_MI_ACCELERATION_SCORING",
        "FSAE_2026_MI6_SCORING",
        "FSAE_2026_MI_SKIDPAD_SCORING",
        "ScoreBreakdown",
        "TimedEventScoreBreakdown",
        "TimedEventScoring",
    }:
        from .events.scoring import (
            FSAEEnduranceEfficiencyScoring,
            FSAE_2026_MI_ACCELERATION_SCORING,
            FSAE_2026_MI6_SCORING,
            FSAE_2026_MI_SKIDPAD_SCORING,
            ScoreBreakdown,
            TimedEventScoreBreakdown,
            TimedEventScoring,
        )

        return {
            "FSAEEnduranceEfficiencyScoring": FSAEEnduranceEfficiencyScoring,
            "FSAE_2026_MI_ACCELERATION_SCORING": (
                FSAE_2026_MI_ACCELERATION_SCORING
            ),
            "FSAE_2026_MI6_SCORING": FSAE_2026_MI6_SCORING,
            "FSAE_2026_MI_SKIDPAD_SCORING": FSAE_2026_MI_SKIDPAD_SCORING,
            "ScoreBreakdown": ScoreBreakdown,
            "TimedEventScoreBreakdown": TimedEventScoreBreakdown,
            "TimedEventScoring": TimedEventScoring,
        }[name]
    if name in {"Curve", "SpatialCoordinate", "SpatialTrack", "Straight", "Track"}:
        from .courses import Curve, SpatialCoordinate, SpatialTrack, Straight, Track

        return {
            "Curve": Curve,
            "SpatialCoordinate": SpatialCoordinate,
            "SpatialTrack": SpatialTrack,
            "Straight": Straight,
            "Track": Track,
        }[name]
    if name in {
        "EnduranceControlProfile",
        "PeriodicPiecewiseLinearTorqueProfile",
        "TorqueProfile",
        "UniformPeriodicTorqueParameterization",
    }:
        from .optimization.torque_profile import (
            EnduranceControlProfile,
            PeriodicPiecewiseLinearTorqueProfile,
            TorqueProfile,
            UniformPeriodicTorqueParameterization,
        )

        return {
            "EnduranceControlProfile": EnduranceControlProfile,
            "PeriodicPiecewiseLinearTorqueProfile": PeriodicPiecewiseLinearTorqueProfile,
            "TorqueProfile": TorqueProfile,
            "UniformPeriodicTorqueParameterization": UniformPeriodicTorqueParameterization,
        }[name]
    if name in {
        "EnduranceOptimizationResult",
        "EnduranceTorqueOptimizer",
        "LocalPolishConfig",
        "OptimizationConfig",
    }:
        from .optimization.optimizer import (
            EnduranceOptimizationResult,
            EnduranceTorqueOptimizer,
            LocalPolishConfig,
            OptimizationConfig,
        )

        return {
            "EnduranceOptimizationResult": EnduranceOptimizationResult,
            "EnduranceTorqueOptimizer": EnduranceTorqueOptimizer,
            "LocalPolishConfig": LocalPolishConfig,
            "OptimizationConfig": OptimizationConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
