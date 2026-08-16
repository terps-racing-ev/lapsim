"""Minimum-lap-time simulation package.

The vehicle model imports :mod:`lapsim.controls`, so the heavier vehicle and
solver exports are loaded lazily to keep package initialization acyclic.
"""

from .controls import Controls

__all__ = [
    "Brakes",
    "Chassis",
    "Controls",
    "EnduranceOptimizationResult",
    "EnduranceControlProfile",
    "EnduranceRunConfig",
    "EnduranceRunResult",
    "EnduranceSimulator",
    "EnduranceTorqueOptimizer",
    "FSAEEnduranceEfficiencyScoring",
    "FSAE_2026_MI6_SCORING",
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
    "SpeedLimitMap",
    "SpeedLimitSolver",
    "Telemetry",
    "TelemetryRecorder",
    "TireNormalLoads",
    "SpatialTrack",
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
        from .lap_time_solver import LapResult, LapTimeSolver

        return {"LapResult": LapResult, "LapTimeSolver": LapTimeSolver}[name]
    if name in {"SpeedLimitMap", "SpeedLimitSolver"}:
        from .speed_limit_solver import SpeedLimitMap, SpeedLimitSolver

        return {
            "SpeedLimitMap": SpeedLimitMap,
            "SpeedLimitSolver": SpeedLimitSolver,
        }[name]
    if name in {"Telemetry", "TelemetryRecorder"}:
        from .telemetry import Telemetry, TelemetryRecorder

        return {
            "Telemetry": Telemetry,
            "TelemetryRecorder": TelemetryRecorder,
        }[name]
    if name == "RecordedLap":
        from .recorded_lap import RecordedLap

        return RecordedLap
    if name in {"ReplayTelemetry", "replay_controls"}:
        from .replay import ReplayTelemetry, replay_controls

        return {
            "ReplayTelemetry": ReplayTelemetry,
            "replay_controls": replay_controls,
        }[name]
    if name in {"EnduranceRunConfig", "EnduranceRunResult", "EnduranceSimulator"}:
        from .endurance import (
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
        from .path_constraints import PathConstraintSolver, PathSpeedConstraints

        return {
            "PathConstraintSolver": PathConstraintSolver,
            "PathSpeedConstraints": PathSpeedConstraints,
        }[name]
    if name in {
        "FSAEEnduranceEfficiencyScoring",
        "FSAE_2026_MI6_SCORING",
        "ScoreBreakdown",
    }:
        from .scoring import (
            FSAEEnduranceEfficiencyScoring,
            FSAE_2026_MI6_SCORING,
            ScoreBreakdown,
        )

        return {
            "FSAEEnduranceEfficiencyScoring": FSAEEnduranceEfficiencyScoring,
            "FSAE_2026_MI6_SCORING": FSAE_2026_MI6_SCORING,
            "ScoreBreakdown": ScoreBreakdown,
        }[name]
    if name == "SpatialTrack":
        from .spatial_track import SpatialTrack

        return SpatialTrack
    if name in {
        "EnduranceControlProfile",
        "PeriodicPiecewiseLinearTorqueProfile",
        "UniformPeriodicTorqueParameterization",
    }:
        from .torque_profile import (
            EnduranceControlProfile,
            PeriodicPiecewiseLinearTorqueProfile,
            UniformPeriodicTorqueParameterization,
        )

        return {
            "EnduranceControlProfile": EnduranceControlProfile,
            "PeriodicPiecewiseLinearTorqueProfile": PeriodicPiecewiseLinearTorqueProfile,
            "UniformPeriodicTorqueParameterization": UniformPeriodicTorqueParameterization,
        }[name]
    if name in {
        "EnduranceOptimizationResult",
        "EnduranceTorqueOptimizer",
        "LocalPolishConfig",
        "OptimizationConfig",
    }:
        from .optimizer import (
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
