"""Optimize directional lateral-grip envelopes at a prescribed skidpad state.

Uses the vehicle's measured gain curves, roll, caster, load transfer and aero.
Each axle gets one symmetric static setting, maximizing its weaker left/right
pure-lateral envelope. Individual tire peaks are an upper bound; a separate
four-wheel force/yaw balance checks the requested 1.5 g operating point.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import atan, degrees, isfinite, pi, radians, sqrt
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_model import Vehicle  # noqa: E402
from vehicle_model.mech.alignment import WHEEL_POSITIONS, WHEEL_SIDE_SIGNS  # noqa: E402
from vehicle_model.mech.tire import TireKinematics  # noqa: E402
from vehicle_model.mech.pacejka import Pacejka52UpcR20LateralModel  # noqa: E402


def fixed_state(vehicle, radius_m, lateral_g, turn):
    speed = sqrt(lateral_g * vehicle.gravity_mps2 * radius_m)
    ay = turn * lateral_g * vehicle.gravity_mps2
    aero = vehicle.aero_forces_n(speed, ay, curvature_per_m=turn / radius_m)
    loads = vehicle.suspension.tire_normal_loads_n(
        vehicle.mass_kg, vehicle.gravity_mps2, aero, vehicle.chassis,
        longitudinal_acceleration_mps2=0.0, lateral_acceleration_mps2=ay,
    )
    return speed, ay, aero, loads


def solve_balance(vehicle, radius_m, lateral_g, turn):
    """Solve drive force, steer and sideslip at fixed speed/radius (no integration)."""
    speed, ay, aero, loads = fixed_state(vehicle, radius_m, lateral_g, turn)
    force_limit = vehicle.drivetrain.wheel_force_from_motor_torque_n(
        vehicle.drivetrain.available_motor_torque_nm(speed, vehicle.battery)
    )
    geometric_steer = atan(vehicle.chassis.wheelbase_m / radius_m)

    def evaluate(values):
        force, steer_magnitude, beta = values
        alignment = vehicle.wheel_alignment_state(turn * steer_magnitude, ay)
        kin = TireKinematics.from_vehicle_motion(
            speed, beta, turn / radius_m, alignment.steering_angles_rad,
            vehicle.chassis.wheel_positions_m, alignment.camber_angles_rad,
        )
        states = vehicle.tire.calculate_forces(loads, kin, force, 0.0, 0.0, 0.05)
        normal_force = states.normal_force_n(beta)
        resistance = (
            aero.drag_n
            + vehicle.rolling_resistance_coefficient * (vehicle.mass_kg * vehicle.gravity_mps2 + aero.downforce_n)
            + vehicle.cornering_drag_coefficient * normal_force**2
            / (vehicle.mass_kg * vehicle.gravity_mps2 + aero.downforce_n)
        )
        residual = np.array([
            (states.tangent_force_n(beta) - resistance) / vehicle.effective_longitudinal_mass_kg,
            normal_force / vehicle.mass_kg - ay,
            states.yaw_moment_nm(vehicle.chassis.wheel_positions_m)
            / (vehicle.mass_kg * vehicle.chassis.wheelbase_m),
        ])
        return residual, alignment, states

    candidates = []
    for beta_deg in (0.0, -2.0, 2.0):
        solution = least_squares(
            lambda values: evaluate(values)[0],
            (min(400.0, force_limit * 0.5), geometric_steer, turn * radians(beta_deg)),
            bounds=((0.0, 0.001, -0.4), (force_limit, 0.6, 0.4)),
            x_scale=(500.0, 0.2, 0.1), max_nfev=120,
            ftol=1e-9, xtol=1e-9, gtol=1e-9,
        )
        residual, alignment, states = evaluate(solution.x)
        candidates.append((np.linalg.norm(residual), solution, residual, alignment, states))
        if np.max(np.abs(residual)) < 1e-6:
            break
    _, solution, residual, alignment, states = min(candidates, key=lambda row: row[0])
    return {
        "feasible": bool(np.max(np.abs(residual)) < 0.02),
        "steering_deg": degrees(turn * solution.x[1]),
        "body_slip_deg": degrees(solution.x[2]),
        "drive_force_request_n": float(solution.x[0]),
        "residuals_mps2": dict(zip(("longitudinal", "lateral", "yaw_equivalent"), map(float, residual))),
        "wheels": {
            name: {
                "normal_load_n": float(state.normal_load_n),
                "travel_mm": corner.suspension_travel_m * 1000,
                "camber_gain_deg": degrees(corner.camber_gain_angle_rad),
                "dynamic_camber_deg": degrees(corner.camber_angle_rad),
                "slip_angle_deg": degrees(state.slip_angle_rad),
                "lateral_force_n": float(state.lateral_force_n),
            }
            for name, corner, state in zip(WHEEL_POSITIONS, alignment.all, states.all, strict=True)
        },
    }


def directional_peak(tire, load, camber, turn, orientation, max_slip_deg):
    """Search only restoring slip in the requested turn; preserve tire handedness."""
    def force(slip_deg):
        return turn * orientation * tire.lateral_force_from_slip_angle_n(
            load, -turn * orientation * radians(float(slip_deg)), radians(camber),
        )
    grid = np.linspace(0, max_slip_deg, 61)
    values = np.array([force(slip) for slip in grid])
    i = int(np.argmax(values))
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    result = minimize_scalar(lambda slip: -force(slip), bounds=(lo, hi), method="bounded")
    if -result.fun > values[i]:
        return float(-result.fun), float(result.x)
    return float(values[i]), float(grid[i])


def sweep_axle(vehicle, axle, settings, steering, radius, lateral_g, max_slip):
    indices = (0, 1) if axle == "front" else (2, 3)
    offsets, loads = {}, {}
    for turn in (1, -1):
        _, ay, _, normal = fixed_state(vehicle, radius, lateral_g, turn)
        alignment = vehicle.wheel_alignment_state(steering[turn], ay)
        static = degrees(getattr(vehicle.alignment, f"{axle}_static_camber_angle_rad"))
        offsets[turn] = [degrees(alignment.all[i].camber_angle_rad) - static for i in indices]
        loads[turn] = [normal.all_n[i] for i in indices]
    capacities = np.empty((2, len(settings)))
    for row, turn in enumerate((1, -1)):
        for col, setting in enumerate(settings):
            capacities[row, col] = sum(
                directional_peak(vehicle.tire, load, setting + offset, turn,
                                 WHEEL_SIDE_SIGNS[i], max_slip)[0]
                for i, load, offset in zip(indices, loads[turn], offsets[turn], strict=True)
            )
    return capacities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-m", type=float, default=9.125)
    parser.add_argument("--lateral-g", type=float, default=1.5)
    parser.add_argument("--camber-min-deg", type=float, default=-6.0)
    parser.add_argument("--camber-max-deg", type=float, default=6.0)
    parser.add_argument("--camber-step-deg", type=float, default=0.05)
    parser.add_argument("--max-slip-deg", type=float, default=15.0)
    parser.add_argument("--front-toe-deg", type=float, default=0.0)
    parser.add_argument("--tire-model", choices=("default", "upc"), default="default")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/skidpad/static_camber")
    args = parser.parse_args()
    numeric = (args.radius_m, args.lateral_g, args.camber_min_deg, args.camber_max_deg,
               args.camber_step_deg, args.max_slip_deg, args.front_toe_deg)
    if not all(isfinite(value) for value in numeric):
        parser.error("all numeric inputs must be finite")
    if min(args.radius_m, args.lateral_g, args.camber_step_deg, args.max_slip_deg) <= 0:
        parser.error("radius, lateral g, step and slip bound must be positive")
    if not -45 < args.camber_min_deg < args.camber_max_deg < 45 or not 0 <= args.front_toe_deg < 10:
        parser.error("invalid camber range or front toe (0 to <10 degrees)")
    vehicle = Vehicle()
    if args.tire_model == "upc":
        vehicle.tire.pacejka_lateral = Pacejka52UpcR20LateralModel()
    vehicle.alignment.front_toe_in_angle_rad = radians(args.front_toe_deg)
    vehicle.validate()
    speed = sqrt(args.lateral_g * vehicle.gravity_mps2 * args.radius_m)
    baseline = {turn: solve_balance(vehicle, args.radius_m, args.lateral_g, turn) for turn in (1, -1)}
    settings = np.linspace(args.camber_min_deg, args.camber_max_deg,
                           int(np.ceil((args.camber_max_deg - args.camber_min_deg) / args.camber_step_deg)) + 1)
    steering = {
        turn: radians(baseline[turn]["steering_deg"]) if baseline[turn]["feasible"]
        else turn * atan(vehicle.chassis.wheelbase_m / args.radius_m)
        for turn in (1, -1)
    }
    converged = False
    for iteration in range(8):
        sweeps, best = {}, {}
        for axle in ("front", "rear"):
            sweeps[axle] = sweep_axle(vehicle, axle, settings, steering, args.radius_m, args.lateral_g, args.max_slip_deg)
            i = int(np.argmax(sweeps[axle].min(axis=0)))
            best[axle] = {"static_camber_deg": float(settings[i]),
                          "left_peak_n": float(sweeps[axle][0, i]),
                          "right_peak_n": float(sweeps[axle][1, i]),
                          "at_search_boundary": i in (0, len(settings) - 1)}
            setattr(vehicle.alignment, f"{axle}_static_camber_angle_rad", radians(settings[i]))
        solved = {turn: solve_balance(vehicle, args.radius_m, args.lateral_g, turn) for turn in (1, -1)}
        new_steering = {turn: radians(solved[turn]["steering_deg"]) if solved[turn]["feasible"] else steering[turn]
                        for turn in (1, -1)}
        if max(abs(new_steering[turn] - steering[turn]) for turn in (1, -1)) < radians(0.001):
            converged = True
            break
        steering = new_steering
    for turn, balance in solved.items():
        for name, orientation in zip(WHEEL_POSITIONS, WHEEL_SIDE_SIGNS, strict=True):
            wheel = balance["wheels"][name]
            peak, slip = directional_peak(vehicle.tire, wheel["normal_load_n"],
                                          wheel["dynamic_camber_deg"], turn, orientation,
                                          args.max_slip_deg)
            wheel["directional_peak_n"] = peak
            wheel["peak_restoring_slip_deg"] = slip
            wheel["peak_at_slip_boundary"] = bool(slip >= args.max_slip_deg - 1e-3)
    for axle in ("front", "rear"):
        zero = sweep_axle(vehicle, axle, [0.0], steering, args.radius_m, args.lateral_g, args.max_slip_deg)
        best[axle]["gain_vs_zero_static_percent"] = 100 * (min(best[axle]["left_peak_n"], best[axle]["right_peak_n"]) / zero.min() - 1)
    roll = degrees(vehicle.suspension.body_roll_angle_rad(vehicle.mass_kg, vehicle.chassis, args.lateral_g * vehicle.gravity_mps2))
    summary = {
        "radius_m": args.radius_m, "lateral_g": args.lateral_g, "speed_mps": speed,
        "speed_kph": speed * 3.6, "circle_time_s": 2 * pi * args.radius_m / speed,
        "body_roll_deg": roll, "tire_model": type(vehicle.tire.pacejka_lateral).__name__,
        "tire_parameters": {"pressure_pa": vehicle.tire.inflation_pressure_pa,
                            "correlation_factor": vehicle.tire.pacejka_lateral.correlation_factor},
        "camber_convention": "Vehicle-local negative is inward; the Tire adapter maps coefficient inclination gamma = -camber. Slip and lateral force are mirrored separately for left/right wheels.",
        "vehicle_parameters": {"mass_kg": vehicle.mass_kg,
                               "front_weight_fraction": vehicle.chassis.static_front_weight_fraction,
                               "roll_stiffness_nm_per_rad": vehicle.suspension.total_roll_stiffness_nm_per_rad,
                               "front_load_transfer_fraction": vehicle.suspension.front_lateral_load_transfer_fraction,
                               "front_caster_deg": degrees(vehicle.alignment.front_caster_angle_rad),
                               "anti_ackermann_fraction": vehicle.alignment.anti_ackermann_fraction,
                               "front_toe_deg_per_wheel": args.front_toe_deg},
        "search": {"min_deg": args.camber_min_deg, "max_deg": args.camber_max_deg,
                   "step_deg": float(settings[1] - settings[0]), "max_slip_deg": args.max_slip_deg,
                   "steering_iterations": iteration + 1, "steering_converged": converged},
        "objective": "Maximize each axle's weaker left/right sum of directional individual-tire pure-lateral peaks at fixed 1.5 g loads.",
        "limitations": "Individual peak sums are an upper bound: wheel slips need not peak together. This is a fixed-g setup analysis, not a maximum-speed prediction. Travel is roll-only, with no aero heave/compliance. Optima depend on the configured tire fit and slip-angle bound. No vehicle setup defaults are changed.",
        "best": best,
        "baseline_balance": {("left" if turn == 1 else "right"): baseline[turn] for turn in (1, -1)},
        "optimized_balance": {("left" if turn == 1 else "right"): solved[turn] for turn in (1, -1)},
    }
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "camber_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("static_camber_deg", "front_left_turn_peak_n", "front_right_turn_peak_n", "rear_left_turn_peak_n", "rear_right_turn_peak_n"))
        writer.writerows(zip(settings, *sweeps["front"], *sweeps["rear"], strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, axle in zip(axes, ("front", "rear"), strict=True):
        for row, turn in enumerate(("Left turn", "Right turn")):
            axis.plot(settings, sweeps[axle][row], label=turn)
        axis.axvline(best[axle]["static_camber_deg"], color="0.3", linestyle="--",
                     label=f'Best: {best[axle]["static_camber_deg"]:+.2f} deg')
        axis.set(title=axle.capitalize(), xlabel="Static camber [deg]", ylabel="Directional axle peak envelope [N]")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"{args.radius_m:g} m skidpad at {args.lateral_g:g} g ({speed * 3.6:.1f} km/h)\n{summary['tire_model']} | roll {roll:.3f} deg | caster {summary['vehicle_parameters']['front_caster_deg']:.1f} deg")
    fig.savefig(out / "camber_sweep.png", dpi=180)
    plt.close(fig)
    lines = [f"# Static camber at {args.radius_m:g} m, {args.lateral_g:g} g", "",
             f"Speed: {speed:.3f} m/s ({speed * 3.6:.2f} km/h). Body roll: {roll:.3f} deg.", "",
             f"Tire model: `{summary['tire_model']}`. Negative camber leans toward the vehicle centerline.", "",
             summary["camber_convention"], "",
             f"Static camber search: {args.camber_min_deg:g} to {args.camber_max_deg:g} deg, {args.camber_step_deg:g} deg increments. Restoring slip envelope: 0 to {args.max_slip_deg:g} deg.", "",
             "| Axle | Best static camber | Gain in weaker-turn envelope vs 0 deg |",
             "| --- | ---: | ---: |"]
    for axle in ("front", "rear"):
        lines.append(f"| {axle.capitalize()} | {best[axle]['static_camber_deg']:+.2f} deg | {best[axle]['gain_vs_zero_static_percent']:+.2f}% |")
    lines.extend(["", summary["objective"], "", summary["limitations"], "",
                  "| Turn | Force/yaw balance feasible | FL camber | FR camber | RL camber | RR camber |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"])
    for turn, balance in summary["optimized_balance"].items():
        cambers = " | ".join(f"{wheel['dynamic_camber_deg']:+.2f} deg" for wheel in balance["wheels"].values())
        lines.append(f"| {turn.capitalize()} | {balance['feasible']} | {cambers} |")
    if any(wheel["peak_at_slip_boundary"] for balance in solved.values() for wheel in balance["wheels"].values()):
        lines.extend(["", "Some tire envelopes peak at the specified slip-angle limit. The optimum therefore depends on that limit; inspect `peak_restoring_slip_deg` in the JSON."])
    command = (
        "python analysis/skidpad/optimize_static_camber.py"
        f" --radius-m {args.radius_m:g} --lateral-g {args.lateral_g:g}"
        f" --camber-min-deg {args.camber_min_deg:g} --camber-max-deg {args.camber_max_deg:g}"
        f" --camber-step-deg {args.camber_step_deg:g} --max-slip-deg {args.max_slip_deg:g}"
        f" --front-toe-deg {args.front_toe_deg:g} --tire-model {args.tire_model}"
    )
    lines.extend(["", f"Steering iteration converged: {converged}. Full residuals, wheel loads/travel and baseline comparison are in `summary.json`.", "",
                  "Reproduce with:", "", f"```powershell\n{command}\n```", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"best": best, "balance_feasible": {turn: balance["feasible"] for turn, balance in summary["optimized_balance"].items()}, "output_dir": str(out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
