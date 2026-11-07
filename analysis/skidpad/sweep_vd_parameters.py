"""Compare one-at-a-time VD sensitivities at fixed skidpad radius and lateral g.

Exports one combined figure, all raw sweep rows and a ranked JSON/Markdown
summary. Capacity is a paired-axle pure-lateral envelope at the solved state,
not a maximum-speed prediction. The front/rear demand split is approximated
by the static weight fraction, as in the model's grip-envelope calculation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import dataclass
import json
from math import degrees, isfinite, pi, radians, sqrt
from pathlib import Path
import sys
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.skidpad.optimize_static_camber import fixed_state, solve_balance  # noqa: E402
from utils.units import pounds_to_kilograms, pound_force_feet_to_newton_meters  # noqa: E402
from vehicle_model import Vehicle  # noqa: E402
from vehicle_model.mech.alignment import CamberGainCurve, WHEEL_POSITIONS, WHEEL_SIDE_SIGNS  # noqa: E402

BLUE, ORANGE, TEAL = "#1c4e80", "#d9822b", "#19958d"


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    unit: str
    low: float
    high: float
    baseline: float
    apply: Callable[[Vehicle, float], None]


def set_path(vehicle, path, value):
    *owners, name = path.split(".")
    component = vehicle
    for owner in owners:
        component = getattr(component, owner)
    setattr(component, name, value)


def parameter_specs(base):
    specs = []

    def add(key, label, unit, low, high, baseline, path, convert=lambda x: x):
        specs.append(Parameter(key, label, unit, low, high, baseline,
                               lambda vehicle, value: set_path(vehicle, path, convert(value))))

    add("front_camber", "Front static camber", "deg", -5, 1,
        degrees(base.alignment.front_static_camber_angle_rad), "alignment.front_static_camber_angle_rad", radians)
    add("rear_camber", "Rear static camber", "deg", -5, 1,
        degrees(base.alignment.rear_static_camber_angle_rad), "alignment.rear_static_camber_angle_rad", radians)
    add("caster", "Front caster", "deg", 4, 12, 7.3, "alignment.front_caster_angle_rad", radians)
    add("anti_ackermann", "Anti-Ackermann fraction", "%", 0, 100, 100,
        "alignment.anti_ackermann_fraction", lambda x: x / 100)
    add("front_toe", "Front toe per wheel (+in)", "deg", -0.75, 0.75, 0,
        "alignment.front_toe_in_angle_rad", radians)
    add("rear_toe", "Rear toe per wheel (+in)", "deg", -0.5, 0.5, 0,
        "alignment.rear_toe_in_angle_rad", radians)
    roll_units = pound_force_feet_to_newton_meters(1) * 180 / pi
    add("roll_stiffness", "Total roll stiffness", "lb-ft/deg", 500, 1400,
        base.suspension.total_roll_stiffness_nm_per_rad / roll_units,
        "suspension.total_roll_stiffness_nm_per_rad", lambda x: x * roll_units)
    add("front_transfer", "Front load-transfer fraction", "%", 35, 70, 55,
        "suspension.front_lateral_load_transfer_fraction", lambda x: x / 100)
    add("cg_height", "CG height", "in", 8, 14, 11, "chassis.cg_height_m", lambda x: x * 0.0254)
    add("front_weight", "Static front weight fraction", "%", 40, 55, 47,
        "chassis.static_front_weight_fraction", lambda x: x / 100)
    add("front_track", "Front track width", "in", 44, 52, 48,
        "chassis.front_track_width_m", lambda x: x * 0.0254)
    add("rear_track", "Rear track width", "in", 42, 50, 46,
        "chassis.rear_track_width_m", lambda x: x * 0.0254)
    add("front_roll_axis", "Front roll-axis height", "in", 1, 4, 2.088,
        "chassis.front_roll_axis_height_m", lambda x: x * 0.0254)
    add("rear_roll_axis", "Rear roll-axis height", "in", 1, 5, 2.884,
        "chassis.rear_roll_axis_height_m", lambda x: x * 0.0254)
    add("mass", "Total vehicle mass", "lb", 550, 700, base.mass_kg / pounds_to_kilograms(1),
        "mass_kg", pounds_to_kilograms)
    add("pressure", "Tire model inflation pressure", "kPa", 80, 115, base.tire.inflation_pressure_pa / 1000,
        "tire.inflation_pressure_pa", lambda x: x * 1000)
    add("downforce", "Downforce coefficient scale", "x baseline", 0.7, 1.3, 1,
        "aero.lift_coefficient", lambda x: x * base.aero.lift_coefficient)
    add("aero_balance", "Front aero balance", "%", 40, 65, base.aero.front_downforce_fraction * 100,
        "aero.front_downforce_fraction", lambda x: x / 100)

    for axle in ("front", "rear"):
        def apply_gain(vehicle, value, selected_axle=axle):
            for side in ("left", "right"):
                attr = f"{selected_axle}_{side}_camber_gain_curve"
                curve = getattr(base.alignment, attr)
                setattr(vehicle.alignment, attr, CamberGainCurve(
                    curve.suspension_travel_m,
                    tuple(value * camber for camber in curve.camber_change_rad),
                ))
        specs.append(Parameter(f"{axle}_gain", f"{axle.capitalize()} camber-gain scale",
                               "x measured", 0.5, 1.5, 1, apply_gain))
    add("wheelbase", "Wheelbase", "in", 57, 65, 61, "chassis.wheelbase_m", lambda x: x * 0.0254)
    add("roll_retention", "Downforce retention at roll limit", "%", 35, 80,
        base.aero.downforce_retention_at_roll_limit * 100,
        "aero.downforce_retention_at_roll_limit", lambda x: x / 100)
    add("roll_breakpoint", "Aero roll breakpoint", "deg", 0.75, 1.5,
        degrees(base.aero.roll_limit_rad), "aero.roll_limit_rad", radians)
    add("drag", "Drag coefficient scale", "x baseline", 0.7, 1.3, 1,
        "aero.drag_coefficient", lambda x: x * base.aero.drag_coefficient)
    return specs


def paired_axle_peak(vehicle, balance, turn, axle, max_slip_deg):
    """Peak paired force with the solved inter-wheel slip split kept fixed.

    Only one shared slip shift is allowed per axle. Both tires must remain
    within the restoring slip range. This avoids independently selecting two
    incompatible tire peaks, and preserves toe/Ackermann effects on the split.
    Steering and camber are frozen at the balanced 1.5 g operating point.
    """
    indices = (0, 1) if axle == "front" else (2, 3)
    wheels = [balance["wheels"][WHEEL_POSITIONS[i]] for i in indices]
    slips = np.array([-turn * wheel["slip_angle_deg"] for wheel in wheels])
    offsets = slips - slips.mean()
    lo = max(0.0, float(-offsets.min()))
    hi = min(max_slip_deg, float(max_slip_deg - offsets.max()))
    if hi <= lo:
        raise ValueError("inter-wheel slip spread exceeds restoring-slip search range")
    ay = turn * vehicle.gravity_mps2 * balance["lateral_g"]
    alignment = vehicle.wheel_alignment_state(radians(balance["steering_deg"]), ay)
    beta = radians(balance["body_slip_deg"])
    projections = [np.cos(alignment.all[i].steering_angle_rad - beta) for i in indices]

    def force(common_slip):
        return sum(
            turn * WHEEL_SIDE_SIGNS[i] * vehicle.tire.lateral_force_from_slip_angle_n(
                wheel["normal_load_n"],
                -turn * WHEEL_SIDE_SIGNS[i] * radians(common_slip + offset),
                radians(wheel["dynamic_camber_deg"]),
            ) * projection
            for i, wheel, offset, projection in zip(indices, wheels, offsets, projections, strict=True)
        )
    grid = np.linspace(lo, hi, 81)
    values = np.array([force(slip) for slip in grid])
    i = int(np.argmax(values))
    refined = minimize_scalar(lambda slip: -force(slip), method="bounded",
                              bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]))
    peak = max(float(values[i]), float(-refined.fun))
    common = float(refined.x) if -refined.fun > values[i] else float(grid[i])
    return peak, bool(common >= hi - 0.001 or common <= lo + 0.001)


def evaluate(vehicle, radius, lateral_g, slip_limit):
    vehicle.validate()
    turn_results = {}
    for name, turn in (("left", 1), ("right", -1)):
        balance = solve_balance(vehicle, radius, lateral_g, turn)
        balance["lateral_g"] = lateral_g
        if not balance["feasible"]:
            return {"valid": False, "reason": f"{name} 1.5 g force/yaw balance infeasible",
                    "residuals": balance["residuals_mps2"]}
        front_peak, front_boundary = paired_axle_peak(vehicle, balance, turn, "front", slip_limit)
        rear_peak, rear_boundary = paired_axle_peak(vehicle, balance, turn, "rear", slip_limit)
        weight = vehicle.mass_kg * vehicle.gravity_mps2
        fraction = vehicle.chassis.static_front_weight_fraction
        front_index = front_peak / (fraction * weight)
        rear_index = rear_peak / ((1 - fraction) * weight)
        turn_results[name] = {
            "front_index": front_index, "rear_index": rear_index,
            "grip_index": min(front_index, rear_index),
            "slip_boundary": front_boundary or rear_boundary,
            "max_balance_residual_mps2": max(abs(v) for v in balance["residuals_mps2"].values()),
            "steering_deg": balance["steering_deg"],
            "body_slip_deg": balance["body_slip_deg"],
            "max_wheel_travel_mm": max(abs(w["travel_mm"]) for w in balance["wheels"].values()),
        }
    return {"valid": True, "turns": turn_results,
            "front_index": min(point["front_index"] for point in turn_results.values()),
            "rear_index": min(point["rear_index"] for point in turn_results.values()),
            "grip_index": min(point["grip_index"] for point in turn_results.values())}


def combined_figure(results, summary, path):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(22, 27), facecolor="white")
    outer = fig.add_gridspec(7, 4, height_ratios=[2.15, 1, 1, 1, 1, 1, 1],
                            left=0.065, right=0.985, top=0.925, bottom=0.065,
                            hspace=0.64, wspace=0.28)
    ranking = summary["ranking"]
    extrema = [value for item in ranking for value in (item["minimum_change_percent"], item["best_change_percent"])]
    rank_lim = max(abs(min(extrema)), abs(max(extrema))) * 1.24
    for col, chunk in enumerate((ranking[:12], ranking[12:])):
        axis = fig.add_subplot(outer[0, col * 2:col * 2 + 2])
        position = axis.get_position()
        axis.set_position([position.x0 + 0.13, position.y0,
                           position.width - 0.13, position.height])
        y = np.arange(len(chunk))
        negative = [min(item["minimum_change_percent"], 0) for item in chunk]
        positive = [max(item["best_change_percent"], 0) for item in chunk]
        axis.barh(y, negative, color="#d8b5a0", height=0.6, label="Loss within sweep range")
        axis.barh(y, positive, color=BLUE, height=0.6, label="Best gain within sweep range")
        axis.set_yticks(y, [f"{item['rank']:02d}  {item['label']}" for item in chunk])
        axis.invert_yaxis()
        axis.set_xlim(-rank_lim, rank_lim)
        axis.axvline(0, color="0.25", lw=0.8)
        axis.grid(axis="x", alpha=0.2)
        axis.set_xlabel("Vehicle grip-envelope change versus baseline [%]")
        axis.set_title("Ranked by full-range effect" if col == 0 else "Remaining parameters", loc="left", fontweight="bold")
        for i, item in enumerate(chunk):
            axis.text(positive[i] + rank_lim * 0.025, i,
                      f"+{max(item['best_change_percent'], 0):.2f}%", va="center", fontsize=10)
    overall_changes = [value for rows in results.values() for row in rows if row["valid"]
                       for value in (row["change_percent"], row["front_change_percent"], row["rear_change_percent"])]
    lower, upper = min(overall_changes), max(overall_changes)
    padding = max(1.0, (upper - lower) * 0.05)
    specs = summary["parameters"]
    for i, spec in enumerate(specs):
        axis = fig.add_subplot(outer[1 + i // 4, i % 4])
        rows = results[spec["key"]]
        x = [row["value"] for row in rows]
        for field, color, style, width, label in (
            ("front_change_percent", ORANGE, "--", 1.25, "Front axle"),
            ("rear_change_percent", TEAL, "--", 1.25, "Rear axle"),
            ("change_percent", BLUE, "-", 2.3, "Vehicle bottleneck"),
        ):
            axis.plot(x, [row.get(field, np.nan) for row in rows], color=color,
                      linestyle=style, lw=width, label=label)
        axis.axhline(0, color="0.7", lw=0.7)
        axis.axvline(spec["baseline"], color="0.5", linestyle=":", lw=1)
        axis.plot(spec["baseline"], 0, "o", color="0.15", markersize=4)
        best = next(item for item in ranking if item["key"] == spec["key"])
        axis.plot(best["best_value"], best["best_change_percent"], "D", color="#3a8c53", markersize=5)
        invalid = [row["value"] for row in rows if not row["valid"]]
        if invalid:
            axis.plot(invalid, [lower] * len(invalid), "x", color="#c64b43")
        axis.set_title(f"{i + 1:02d}  {spec['label']}", loc="left", fontweight="bold")
        axis.text(0.02, 0.96,
                  f"Best {best['best_change_percent']:+.2f}% at {best['best_value']:.3g} {spec['unit']}",
                  transform=axis.transAxes, va="top", fontsize=9)
        axis.set(xlabel=f"{spec['label']} [{spec['unit']}]", ylabel="Grip envelope change [%]",
                 ylim=(lower - padding, upper + padding), xlim=(spec["low"], spec["high"]))
        axis.grid(alpha=0.2)
        if i == 0:
            axis.legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("SKIDPAD | VEHICLE DYNAMICS SENSITIVITY", x=0.065, y=0.985,
                 ha="left", fontsize=25, fontweight="bold", color=BLUE)
    fig.text(0.065, 0.962,
             f"{summary['radius_m']:g} m radius  |  {summary['lateral_g']:g} g  |  {summary['speed_kph']:.2f} km/h  |  corrected MF 6.1 tire model\n"
             f"One parameter at a time; baseline static camber {summary['front_static_camber_deg']:.2f} deg front / {summary['rear_static_camber_deg']:.2f} deg rear.",
             fontsize=13, va="top")
    fig.text(0.065, 0.038,
             "READING THE PLOTS: solid blue = vehicle grip envelope; dashed orange/teal = axle envelopes; dotted vertical = baseline; green diamond = best sampled value.\n"
             "Ranking: total swept effect (loss to best gain); brown bars = losses, blue bars = gains. All parameter plots share one percentage scale; gains are not additive.\n"
             "Paired pure-lateral envelopes at fixed loads, steering and inter-wheel slip split; static-weight yaw split approximation. 0–15 deg restoring slip; no aero heave or compliance.\n"
             "This screens grip reserve, not maximum speed or lap-time gain. Baseline camber is held fixed unless swept; other parameters are not jointly optimized.",
             fontsize=11, linespacing=1.6, va="top")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def save_outputs(out, results, summary):
    summary["ranking"].sort(key=lambda point: point["full_range_effect_percent"], reverse=True)
    for rank, point in enumerate(summary["ranking"], 1):
        point["rank"] = rank
    summary["ranking_metric"] = "Largest full-range change (best minus worst) in vehicle grip envelope over the displayed screening range."
    summary["best_gain_ranking"] = [point["key"] for point in sorted(
        summary["ranking"], key=lambda point: point["best_change_percent"], reverse=True)]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "all_sweep_points.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out / "sweep_points.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["parameter", "value", "valid", "grip_index", "front_index", "rear_index",
                  "change_percent", "front_change_percent", "rear_change_percent", "reason"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rows in results.values():
            writer.writerows(rows)
    combined_figure(results, summary, out / "vd_sensitivity_all.png")
    lines = ["# Skidpad VD sensitivity", "",
             f"{summary['radius_m']:g} m radius, {summary['lateral_g']:g} g, {summary['speed_kph']:.2f} km/h.", "",
             f"Baseline camber: {summary['front_static_camber_deg']:g} deg front / {summary['rear_static_camber_deg']:g} deg rear. All other inputs use current Vehicle defaults.", "",
             summary["method"], "", summary["ranking_metric"], "",
             "| Rank | Parameter | Range | Best sampled value | Best gain | Full-range effect |",
             "| ---: | --- | --- | --- | ---: | ---: |"]
    for point in summary["ranking"]:
        lines.append(f"| {point['rank']} | {point['label']} | {point['range'][0]:g} to {point['range'][1]:g} {point['unit']} | {point['best_value']:.4g} {point['unit']} | {point['best_change_percent']:+.3f}% | {point['full_range_effect_percent']:.3f}% |")
    lines.extend(["", summary["limitations"], "", "![Combined sweeps](vd_sensitivity_all.png)", "",
                  "Run `python analysis/skidpad/sweep_vd_parameters.py` to reproduce the defaults.", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Combined figure: {(out / 'vd_sensitivity_all.png').resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-m", type=float, default=9.125)
    parser.add_argument("--lateral-g", type=float, default=1.5)
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--front-camber-deg", type=float, default=-1.98)
    parser.add_argument("--rear-camber-deg", type=float, default=-2.56)
    parser.add_argument("--max-slip-deg", type=float, default=15)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/skidpad/vd_sweep")
    parser.add_argument("--replot", action="store_true", help="Redraw existing JSON results without rerunning physics; other sweep inputs are ignored.")
    args = parser.parse_args()
    if args.replot:
        summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
        results = json.loads((args.output_dir / "all_sweep_points.json").read_text(encoding="utf-8"))
        save_outputs(args.output_dir, results, summary)
        return
    if args.points < 5 or not all(isfinite(x) for x in (args.radius_m, args.lateral_g, args.front_camber_deg, args.rear_camber_deg, args.max_slip_deg)):
        parser.error("use at least 5 points and finite inputs")
    if min(args.radius_m, args.lateral_g, args.max_slip_deg) <= 0:
        parser.error("radius, lateral g and slip limit must be positive")
    base = Vehicle()
    base.alignment.front_static_camber_angle_rad = radians(args.front_camber_deg)
    base.alignment.rear_static_camber_angle_rad = radians(args.rear_camber_deg)
    baseline = evaluate(base, args.radius_m, args.lateral_g, args.max_slip_deg)
    if not baseline["valid"]:
        raise RuntimeError(f"Baseline is infeasible: {baseline}")
    specs = parameter_specs(base)
    results, ranking = {}, []
    for index, spec in enumerate(specs):
        values = np.unique(np.append(np.linspace(spec.low, spec.high, args.points), spec.baseline))
        rows = []
        for value in values:
            if abs(value - spec.baseline) < 1e-10:
                point = deepcopy(baseline)
            else:
                candidate = deepcopy(base)
                spec.apply(candidate, float(value))
                point = evaluate(candidate, args.radius_m, args.lateral_g, args.max_slip_deg)
            row = {"parameter": spec.key, "value": float(value), **point}
            if point["valid"]:
                row.update({field: 100 * (point[key] / baseline["grip_index"] - 1)
                            for field, key in (("change_percent", "grip_index"),
                                               ("front_change_percent", "front_index"),
                                               ("rear_change_percent", "rear_index"))})
            rows.append(row)
        results[spec.key] = rows
        valid = [row for row in rows if row["valid"]]
        best, worst = max(valid, key=lambda row: row["grip_index"]), min(valid, key=lambda row: row["grip_index"])
        ranking.append({"key": spec.key, "label": spec.label, "unit": spec.unit,
                        "baseline": spec.baseline, "range": [spec.low, spec.high],
                        "best_value": best["value"], "best_change_percent": best["change_percent"],
                        "minimum_change_percent": worst["change_percent"],
                        "full_range_effect_percent": best["change_percent"] - worst["change_percent"],
                        "invalid_points": len(rows) - len(valid),
                        "best_at_range_boundary": best["value"] in (spec.low, spec.high)})
        print(f"[{index + 1:02d}/{len(specs)}] {spec.label}: best {best['change_percent']:+.3f}% at {best['value']:.4g} {spec.unit}; {len(rows) - len(valid)} infeasible", flush=True)
    ranking.sort(key=lambda point: point["best_change_percent"], reverse=True)
    for rank, point in enumerate(ranking, 1):
        point["rank"] = rank
    summary = {
        "radius_m": args.radius_m, "lateral_g": args.lateral_g,
        "speed_kph": sqrt(args.radius_m * args.lateral_g * base.gravity_mps2) * 3.6,
        "front_static_camber_deg": args.front_camber_deg, "rear_static_camber_deg": args.rear_camber_deg,
        "max_slip_deg": args.max_slip_deg, "samples_per_parameter": args.points,
        "baseline": baseline, "ranking": ranking,
        "parameters": [{"key": spec.key, "label": spec.label, "unit": spec.unit,
                        "low": spec.low, "high": spec.high, "baseline": spec.baseline} for spec in specs],
        "method": "Each parameter is varied independently, with other setup inputs held fixed. Solve four-wheel longitudinal/lateral/yaw balance at 1.5 g in both turns. At this geometry maximize paired-axle restoring force with the actual wheel-slip split fixed, including projection into the path-normal direction. Divide axle envelopes by their static-weight-fraction demand and use the weaker axle/turn as the vehicle grip index.",
        "limitations": "Screening ranges are assumptions, not verified hardware limits. Results are grip-envelope changes, not maximum-g or lap-time predictions. Envelope freezes loads/steer/camber at the requested 1.5 g state and approximates the yaw demand split by static weight fraction. Longitudinal friction-circle demand is included in operating-point feasibility but not subtracted from the pure-lateral reserve. Peaks can lie at the slip-angle bound. No aero heave/compliance, parameter interactions or camber re-optimization. Tire pressure is the coefficient-model input in kPa. Ranking depends on selected ranges.",
    }
    save_outputs(args.output_dir, results, summary)


if __name__ == "__main__":
    main()
