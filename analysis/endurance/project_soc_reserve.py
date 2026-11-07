"""Check repeated quasi-static lap projections against a stateful SOC reserve.

Lap speed and requested power repeat without full vehicle dynamics. The RC
battery evolves continuously and candidates are rejected if either their power
request or SOC reserve is infeasible. This does not validate the full vehicle's
path controller, tire dynamics, or achievable competition pace.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.aero.sweep import ProjectedRun, capped_map, coarsen_track  # noqa: E402
from lapsim import (  # noqa: E402
    FSAE_2026_MI6_SCORING, LapTimeSolver, SpatialTrack, SpeedLimitSolver,
)
from vehicle_model import Vehicle  # noqa: E402


def model_source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate(speed_map, cap_mps: float, minimum_soc: float, initial_soc: float) -> dict:
    vehicle = Vehicle()
    vehicle.battery.initial_state_of_charge = initial_soc
    vehicle.reset_state()
    lap = LapTimeSolver(vehicle).solve(capped_map(speed_map, cap_mps), 10.0)
    battery = vehicle.battery
    elapsed_s = energy_kwh = distance_m = 0.0
    completed_laps = 0
    failure = None
    minimum_power_margin_w = float("inf")
    for lap_index in range(22):
        for length, speed, power in zip(
            lap.speed_limit_map.cell_length_m,
            lap.telemetry.speed_mps,
            lap.telemetry.battery_power_w,
            strict=True,
        ):
            margin_w = battery.discharge_power_limit_w - power
            minimum_power_margin_w = min(minimum_power_margin_w, margin_w)
            if margin_w < -1e-6:
                failure = "projected demand exceeds battery discharge power limit"
                break
            dt_s = length / speed
            battery.update_state(power, dt_s)
            elapsed_s += dt_s
            energy_kwh += power * dt_s / 3_600_000.0
            distance_m += length
            if battery.state_of_charge < minimum_soc:
                failure = "minimum SOC reserve reached"
                break
        if failure:
            break
        completed_laps += 1
    complete = completed_laps == 22 and failure is None
    score = FSAE_2026_MI6_SCORING.score(
        ProjectedRun(22, elapsed_s, energy_kwh)
    ) if complete else None
    result = {
        "speed_cap_mps": cap_mps,
        "speed_cap_mph": cap_mps / 0.44704,
        "projected_lap_time_s": lap.lap_time_s,
        "completed": complete,
        "completed_laps": completed_laps,
        "elapsed_time_s": elapsed_s,
        "distance_m": distance_m,
        "energy_kwh": energy_kwh,
        "final_soc": battery.state_of_charge,
        "minimum_power_margin_w": minimum_power_margin_w,
        "failure_reason": failure,
        "score": asdict(score) if score else None,
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-soc", type=float, default=0.2)
    parser.add_argument("--initial-soc", type=float, default=1.0)
    parser.add_argument("--cell-length-m", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/events/endurance_soc20")
    args = parser.parse_args()
    if not 0.0 <= args.minimum_soc < args.initial_soc <= 1.0:
        parser.error("require 0 <= minimum SOC < initial SOC <= 1")
    initial_source_hash = model_source_hash()
    track_path = ROOT / "analysis/data/track/gnss_imu_endurance_track.csv"
    track = coarsen_track(SpatialTrack.from_csv(track_path), args.cell_length_m)
    speed_map = SpeedLimitSolver(Vehicle(), max_step_m=args.cell_length_m).solve(track)
    results = [
        evaluate(speed_map, cap, args.minimum_soc, args.initial_soc)
        for cap in (14.0, 15.0, 16.0, 16.5, 17.0, 18.0)
    ]
    feasible = [row for row in results if row["completed"]]
    if feasible:
        lower = max(row["speed_cap_mps"] for row in feasible)
        upper = min(
            (row["speed_cap_mps"] for row in results if not row["completed"] and row["speed_cap_mps"] > lower),
            default=lower,
        )
        while upper - lower > 0.01:
            middle = 0.5 * (upper + lower)
            row = evaluate(speed_map, middle, args.minimum_soc, args.initial_soc)
            results.append(row)
            if row["completed"]:
                lower = middle
                feasible.append(row)
            else:
                upper = middle
        best = max(feasible, key=lambda row: row["score"]["combined_points"])
    else:
        best = None
    summary = {
        "method": __doc__,
        "minimum_soc": args.minimum_soc,
        "initial_soc": args.initial_soc,
        "laps": 22,
        "track": str(track_path),
        "cell_length_m": args.cell_length_m,
        "starting_speed_each_projected_lap_mps": 10.0,
        "vehicle": "Current Vehicle() defaults, no regenerative braking",
        "model_source_sha256": initial_source_hash,
        "model_sources_unchanged_during_run": initial_source_hash == model_source_hash(),
        "pace_search": "coarse speed-cap grid plus reserve-boundary bisection; not a global strategy optimum",
        "best_feasible_projection": best,
        "evaluations": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "soc_reserve_projection.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("BEST FEASIBLE:", json.dumps(best), flush=True)
    print(path, flush=True)
    if not summary["model_sources_unchanged_during_run"]:
        raise RuntimeError("Vehicle/solver sources changed during the projection; rerun for a consistent model snapshot")


if __name__ == "__main__":
    main()
