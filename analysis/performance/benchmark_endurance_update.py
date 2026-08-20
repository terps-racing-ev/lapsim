r"""Benchmark steady-state endurance-simulator cell updates.

Run from the ``python_lapsim`` repository root::

    $env:PYTHONPATH=(Resolve-Path src)
    .\.venv\Scripts\python.exe analysis\performance\benchmark_endurance_update.py

Path-constraint generation is measured as startup work and cached separately.
The latency samples cover one complete endurance cell, from one torque-profile
query to the next, including path control, the committed vehicle update, state
checks, and optional telemetry recording.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import sys
import time
from copy import deepcopy
from dataclasses import replace
from math import atan
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable
from unittest.mock import patch

import numpy as np

from lapsim import (
    Controls,
    EnduranceRunConfig,
    EnduranceSimulator,
    PathConstraintSolver,
    PathSpeedConstraints,
    SpatialTrack,
    TorqueProfile,
    UniformPeriodicTorqueParameterization,
)
from vehicle_model import Vehicle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACK = ROOT / "analysis/data/track/gnss_imu_endurance_track.csv"
DEFAULT_OUTPUT = ROOT / "analysis/performance/output/endurance_update_metrics.json"
DEFAULT_CONSTRAINT_CACHE = (
    ROOT / "analysis/performance/output/path_constraints_cache.json"
)


@dataclass
class CallTimer:
    calls: int = 0
    total_ns: int = 0

    def record(self, elapsed_ns: int) -> None:
        self.calls += 1
        self.total_ns += elapsed_ns


class CellTimingProfile:
    """Torque-profile proxy that marks successive endurance cell boundaries."""

    def __init__(self, profile: Any) -> None:
        self.profile = profile
        self.latencies_ns: list[int] = []
        self._cell_start_ns: int | None = None

    def request_fraction(self, lap_distance_m: float) -> float:
        now_ns = time.perf_counter_ns()
        if self._cell_start_ns is not None:
            self.latencies_ns.append(now_ns - self._cell_start_ns)
        self._cell_start_ns = now_ns
        return self.profile.request_fraction(lap_distance_m)

    def controls_at(self, lap_distance_m: float) -> Controls:
        now_ns = time.perf_counter_ns()
        if self._cell_start_ns is not None:
            self.latencies_ns.append(now_ns - self._cell_start_ns)
        self._cell_start_ns = now_ns
        return self.profile.controls_at(lap_distance_m)

    def finish(self) -> None:
        if self._cell_start_ns is not None:
            self.latencies_ns.append(time.perf_counter_ns() - self._cell_start_ns)
            self._cell_start_ns = None


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def summarize_ms(latencies_ns: list[int]) -> dict[str, float | int]:
    values_ms = [value / 1_000_000.0 for value in latencies_ns]
    return {
        "samples": len(values_ms),
        "mean_ms": fmean(values_ms),
        "median_ms": percentile(values_ms, 50),
        "p95_ms": percentile(values_ms, 95),
        "p99_ms": percentile(values_ms, 99),
        "maximum_ms": max(values_ms),
        "minimum_ms": min(values_ms),
        "standard_deviation_ms": float(np.std(values_ms)),
    }


def track_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    model_files = sorted((ROOT / "src/vehicle_model").rglob("*.py"))
    model_files.extend(
        [
            ROOT / "src/lapsim/core/controls.py",
            ROOT / "src/lapsim/solvers/path_constraints.py",
        ]
    )
    model_hash = hashlib.sha256()
    for model_path in model_files:
        model_hash.update(str(model_path.relative_to(ROOT)).encode("utf-8"))
        model_hash.update(model_path.read_bytes())
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
        "constraint_model_sha256": model_hash.hexdigest(),
    }


def load_or_solve_constraints(
    track: SpatialTrack,
    track_path: Path,
    cache_path: Path,
) -> tuple[PathSpeedConstraints, float, bool]:
    signature = track_signature(track_path)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("track_signature") == signature:
            return (
                PathSpeedConstraints(
                    track=track,
                    local_corner_speed_mps=tuple(cached["local_corner_speed_mps"]),
                    braking_speed_ceiling_mps=tuple(
                        cached["braking_speed_ceiling_mps"]
                    ),
                    passes=int(cached["passes"]),
                ),
                float(cached.get("generation_time_s", 0.0)),
                True,
            )

    start = time.perf_counter()
    constraints = PathConstraintSolver().solve(track, Vehicle())
    elapsed_s = time.perf_counter() - start
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "track_signature": signature,
                "local_corner_speed_mps": constraints.local_corner_speed_mps,
                "braking_speed_ceiling_mps": constraints.braking_speed_ceiling_mps,
                "passes": constraints.passes,
                "generation_time_s": elapsed_s,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return constraints, elapsed_s, False


def timed_wrapper(
    original: Callable[..., Any], timer: CallTimer
) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_ns = time.perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            timer.record(time.perf_counter_ns() - start_ns)

    return wrapper


@dataclass(frozen=True, slots=True)
class TabulatedControlProfile:
    """Explicit one-lap controls used as a repeatable benchmark fixture."""

    cell_center_distance_m: tuple[float, ...]
    controls: tuple[Controls, ...]

    def controls_at(self, lap_distance_m: float) -> Controls:
        index = int(np.searchsorted(self.cell_center_distance_m, lap_distance_m))
        return self.controls[min(index, len(self.controls) - 1)]


def build_reference_control_profile(
    constraints: PathSpeedConstraints,
    torque_fraction: float,
) -> tuple[TabulatedControlProfile, float]:
    """Generate explicit controls outside the timed endurance simulator.

    This fixture generator is intentionally separate from ``EnduranceSimulator``.
    It uses probe copies only once to produce a repeatable pressure schedule;
    the measured simulation merely applies that supplied schedule.
    """

    start = time.perf_counter()
    track = constraints.track
    vehicle = Vehicle()
    vehicle.speed_mps = constraints.braking_speed_ceiling_mps[0]
    controls_by_cell: list[Controls] = []
    for cell_index, distance_step_m in enumerate(track.cell_length_m):
        curvature_per_m = track.curvature_per_m[cell_index]
        target_speed_mps = constraints.braking_speed_ceiling_mps[
            (cell_index + 1) % track.cell_count
        ]
        motor_torque_nm = torque_fraction * vehicle.drivetrain.motor.torque_limit_nm(
            vehicle.drivetrain.motor_speed_rpm(vehicle.speed_mps)
        )
        base_controls = Controls(
            motor_torque_request_nm=motor_torque_nm,
            steering_angle_rad=atan(curvature_per_m * vehicle.chassis.wheelbase_m),
        )

        def final_speed_mps(pressure_psi: float) -> float:
            candidate = deepcopy(vehicle)
            try:
                candidate.update_state(
                    replace(
                        base_controls,
                        front_brake_pressure_psi=pressure_psi,
                        rear_brake_pressure_psi=pressure_psi,
                    ),
                    distance_step_m,
                )
            except ValueError as error:
                if "stops before the end of the cell" in str(error):
                    return 0.0
                raise
            return candidate.speed_mps

        pressure_psi = 0.0
        if final_speed_mps(0.0) > target_speed_mps:
            lower_psi = 0.0
            upper_psi = 10.0
            while final_speed_mps(upper_psi) > target_speed_mps:
                lower_psi = upper_psi
                upper_psi *= 2.0
                if upper_psi > 10_000.0:
                    raise RuntimeError("could not bracket reference brake pressure")
            for _ in range(32):
                candidate_psi = 0.5 * (lower_psi + upper_psi)
                if final_speed_mps(candidate_psi) <= target_speed_mps:
                    upper_psi = candidate_psi
                else:
                    lower_psi = candidate_psi
            pressure_psi = upper_psi

        controls = replace(
            base_controls,
            front_brake_pressure_psi=pressure_psi,
            rear_brake_pressure_psi=pressure_psi,
        )
        vehicle.update_state(controls, distance_step_m)
        controls_by_cell.append(controls)

    return (
        TabulatedControlProfile(
            cell_center_distance_m=track.cell_center_distance_m,
            controls=tuple(controls_by_cell),
        ),
        time.perf_counter() - start,
    )


def run_timed(
    constraints: PathSpeedConstraints,
    base_profile: Any,
    *,
    laps: int,
    record_telemetry: bool,
    collect_detail: bool,
) -> tuple[dict[str, Any], list[int]]:
    vehicle = Vehicle()
    timing_profile = CellTimingProfile(base_profile)
    committed_update_latencies_ns: list[int] = []
    committed_simulated_timesteps_s: list[float] = []

    original_update_state = Vehicle.update_state

    def vehicle_update_wrapper(
        update_vehicle: Vehicle, *args: Any, **kwargs: Any
    ) -> Any:
        start_ns = time.perf_counter_ns()
        initial_simulation_time_s = update_vehicle.time_s
        try:
            return original_update_state(update_vehicle, *args, **kwargs)
        finally:
            elapsed_ns = time.perf_counter_ns() - start_ns
            if update_vehicle is vehicle:
                committed_update_latencies_ns.append(elapsed_ns)
                committed_simulated_timesteps_s.append(
                    update_vehicle.time_s - initial_simulation_time_s
                )

    start_ns = time.perf_counter_ns()
    with ExitStack() as stack:
        if collect_detail:
            stack.enter_context(patch.object(Vehicle, "update_state", vehicle_update_wrapper))
        result = EnduranceSimulator().run(
            vehicle,
            constraints,
            timing_profile,
            EnduranceRunConfig(laps=laps),
            record_telemetry=record_telemetry,
        )
        timing_profile.finish()
    elapsed_ns = time.perf_counter_ns() - start_ns

    if not result.completed:
        raise RuntimeError(f"Benchmark endurance run failed: {result.failure_reason}")

    cell_count = len(timing_profile.latencies_ns)
    detail: dict[str, Any] = {
        "wall_time_s": elapsed_ns / 1_000_000_000.0,
        "simulated_driving_time_s": result.driving_time_s,
        "real_time_speedup": result.driving_time_s / (elapsed_ns / 1_000_000_000.0),
        "cell_latency": summarize_ms(timing_profile.latencies_ns),
        "cells_per_wall_second": cell_count / (elapsed_ns / 1_000_000_000.0),
        "mean_simulated_timestep_ms": 1_000.0 * result.driving_time_s / cell_count,
    }
    if collect_detail:
        detail["committed_vehicle_update"] = summarize_ms(
            committed_update_latencies_ns
        )
        simulated_timesteps_ns = [
            int(value * 1_000_000_000.0) for value in committed_simulated_timesteps_s
        ]
        detail["simulated_timestep"] = summarize_ms(simulated_timesteps_ns)
        detail["fraction_cells_slower_than_simulated_time"] = sum(
            latency_ns > timestep_ns
            for latency_ns, timestep_ns in zip(
                timing_profile.latencies_ns, simulated_timesteps_ns, strict=True
            )
        ) / cell_count
    return detail, timing_profile.latencies_ns


def profile_category(filename: str, function_name: str) -> str:
    normalized = filename.replace("\\", "/")
    if normalized.endswith("/lapsim/events/endurance.py"):
        return "endurance path controller"
    if normalized.endswith("/vehicle_model/vehicle.py"):
        return "vehicle force balance and spatial integration"
    if normalized.endswith("/vehicle_model/mech/suspension.py"):
        return "suspension/load transfer"
    if normalized.endswith("/vehicle_model/mech/tire.py"):
        return "tire force capacity"
    if normalized.endswith("/vehicle_model/mech/brakes.py"):
        return "brake force/slip"
    if normalized.endswith("/vehicle_model/aero/model.py"):
        return "aerodynamics"
    if "/vehicle_model/electrical/" in normalized:
        return "battery/electrical"
    if "/vehicle_model/powertrain/" in normalized:
        return "powertrain"
    if normalized.endswith("/lapsim/optimization/torque_profile.py"):
        return "torque-profile interpolation"
    return "Python/SciPy/runtime overhead"


def profile_one_lap(
    constraints: PathSpeedConstraints, profile: Any
) -> dict[str, Any]:
    profiler = cProfile.Profile()
    vehicle = Vehicle()
    profiler.enable()
    result = EnduranceSimulator().run(
        vehicle,
        constraints,
        profile,
        EnduranceRunConfig(laps=1),
        record_telemetry=False,
    )
    profiler.disable()
    if not result.completed:
        raise RuntimeError(f"Profile run failed: {result.failure_reason}")

    stats = pstats.Stats(profiler)
    category_seconds: dict[str, float] = defaultdict(float)
    functions: list[dict[str, Any]] = []
    for (filename, line, name), stat in stats.stats.items():
        primitive_calls, total_calls, self_s, cumulative_s, _callers = stat
        category_seconds[profile_category(filename, name)] += self_s
        functions.append(
            {
                "function": name,
                "file": str(Path(filename).name),
                "line": line,
                "calls": total_calls,
                "self_time_s": self_s,
                "cumulative_time_s": cumulative_s,
                "primitive_calls": primitive_calls,
            }
        )

    total_self_s = sum(category_seconds.values())
    return {
        "profiled_wall_time_s": stats.total_tt,
        "categories": [
            {
                "category": category,
                "self_time_s": seconds,
                "percent": 100.0 * seconds / total_self_s,
            }
            for category, seconds in sorted(
                category_seconds.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "top_functions_by_self_time": sorted(
            functions, key=lambda item: item["self_time_s"], reverse=True
        )[:20],
    }


def processor_name() -> str:
    value = platform.processor().strip()
    if value:
        return value
    return os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--constraint-cache", type=Path, default=DEFAULT_CONSTRAINT_CACHE)
    parser.add_argument("--warmup-laps", type=int, default=1)
    parser.add_argument("--measurement-laps", type=int, default=3)
    parser.add_argument("--torque-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_laps < 1 or args.measurement_laps < 1:
        raise ValueError("lap counts must be positive")
    if not 0.0 <= args.torque_fraction <= 1.0:
        raise ValueError("torque-fraction must be in [0, 1]")

    track = SpatialTrack.from_csv(args.track)
    constraints, startup_s, cache_used = load_or_solve_constraints(
        track, args.track, args.constraint_cache
    )
    base_profile, fixture_generation_s = build_reference_control_profile(
        constraints, args.torque_fraction
    )

    # Prime imports, caches, branch predictors, and allocator state.
    run_timed(
        constraints,
        base_profile,
        laps=args.warmup_laps,
        record_telemetry=False,
        collect_detail=False,
    )

    no_telemetry, raw_latencies_ns = run_timed(
        constraints,
        base_profile,
        laps=args.measurement_laps,
        record_telemetry=False,
        collect_detail=True,
    )
    with_telemetry, telemetry_latencies_ns = run_timed(
        constraints,
        base_profile,
        laps=args.measurement_laps,
        record_telemetry=True,
        collect_detail=False,
    )
    function_profile = profile_one_lap(constraints, base_profile)

    deadline_ms = 10.0
    metrics = {
        "benchmark": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version,
            "platform": platform.platform(),
            "processor": processor_name(),
            "logical_cpu_count": os.cpu_count(),
            "clock": "time.perf_counter_ns",
            "gc_behavior": "enabled (normal CPython behavior)",
            "warmup_laps": args.warmup_laps,
            "measurement_laps": args.measurement_laps,
            "torque_request_fraction": args.torque_fraction,
        },
        "track": {
            "path": str(args.track.resolve()),
            "cells_per_lap": track.cell_count,
            "length_m": track.length_m,
            "cell_length_min_m": min(track.cell_length_m),
            "cell_length_max_m": max(track.cell_length_m),
        },
        "startup": {
            "constraint_cache_used": cache_used,
            "constraint_generation_time_s": startup_s,
            "constraint_passes": constraints.passes,
            "cache_path": str(args.constraint_cache.resolve()),
            "reference_control_fixture_generation_time_s": fixture_generation_s,
        },
        "steady_state_without_telemetry": no_telemetry,
        "steady_state_with_telemetry": with_telemetry,
        "deadline_analysis": {
            "deadline_ms": deadline_ms,
            "without_telemetry_fraction_over_deadline": sum(
                value > deadline_ms * 1_000_000 for value in raw_latencies_ns
            )
            / len(raw_latencies_ns),
            "with_telemetry_fraction_over_deadline": sum(
                value > deadline_ms * 1_000_000 for value in telemetry_latencies_ns
            )
            / len(telemetry_latencies_ns),
        },
        "cprofile_one_lap_without_telemetry": function_profile,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
