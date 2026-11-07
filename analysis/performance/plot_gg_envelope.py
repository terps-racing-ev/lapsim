"""Quasi-static G-G capacity envelopes from the current vehicle components.

Run directly; outputs include plots, boundary CSV, limits CSV, and assumptions.
This is a peak-grip capacity calculation, not a steering/slip equilibrium solve.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import atan, cos, isfinite, pi, sin, sqrt
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.units import miles_per_hour_to_meters_per_second  # noqa: E402
from vehicle_model import Vehicle  # noqa: E402


class CapacityEnvelope:
    """Retain speed-dependent actuators and evaluate coupled force constraints.

    Lateral axle split follows the existing capacity solvers' yaw-balanced
    static weight split, with equal lateral utilization within each axle.
    Hydraulic and regen requests share the rear contact patches. Hydraulic
    requests split equally left/right, matching Tire.calculate_forces.
    """

    def __init__(self, vehicle: Vehicle, speed_mph: float, regen: bool):
        self.vehicle = vehicle
        self.speed = miles_per_hour_to_meters_per_second(speed_mph)
        self.drive = vehicle.drivetrain.available_wheel_force_n(self.speed, vehicle.battery)
        self.front_brake, self.rear_brake = vehicle.brakes.axle_force_requests_from_pressures_n(
            vehicle.brakes.maximum_pressure_psi,
            vehicle.brakes.maximum_pressure_psi,
            vehicle.tire.rolling_radius_m,
        )
        self.regen = vehicle.maximum_regenerative_brake_force_n(self.speed) if regen else 0.0

    def constraints(self, ax: float, ay: float) -> np.ndarray:
        """Force margins [N]; all must be nonnegative for feasibility."""
        v = self.vehicle
        curvature = ay / self.speed**2
        aero = v.aero_forces_n(self.speed, ay, curvature_per_m=curvature)
        loads = v.suspension.tire_normal_loads_n(
            v.mass_kg, v.gravity_mps2, aero, v.chassis,
            longitudinal_acceleration_mps2=ax, lateral_acceleration_mps2=ay,
        )
        alignment = v.wheel_alignment_state(atan(curvature * v.chassis.wheelbase_m), ay)
        cy = np.array([
            v.tire.lateral_force_capacity_n(load, camber)
            for load, camber in zip(loads.all_n, alignment.camber_angles_rad, strict=True)
        ])
        cx = np.array([
            v.tire.longitudinal_force_capacity_n(load, camber)
            for load, camber in zip(loads.all_n, alignment.camber_angles_rad, strict=True)
        ])
        fraction = v.chassis.static_front_weight_fraction
        front_y = v.mass_kg * abs(ay) * fraction
        rear_y = v.mass_kg * abs(ay) * (1.0 - fraction)
        front_capacity, rear_capacity = sum(cy[:2]), sum(cy[2:])
        front_utilization = front_y / max(front_capacity, 1e-12)
        rear_utilization = rear_y / max(rear_capacity, 1e-12)
        remaining = cx * np.array([
            sqrt(max(0.0, 1.0 - front_utilization**2)),
            sqrt(max(0.0, 1.0 - front_utilization**2)),
            sqrt(max(0.0, 1.0 - rear_utilization**2)),
            sqrt(max(0.0, 1.0 - rear_utilization**2)),
        ])
        total_normal = v.mass_kg * v.gravity_mps2 + aero.downforce_n
        resistance = (
            aero.drag_n + v.rolling_resistance_coefficient * total_normal
            + v.cornering_drag_coefficient * (v.mass_kg * ay)**2 / total_normal
        )
        required_x = v.effective_longitudinal_mass_kg * ax + resistance
        max_drive = min(self.drive, sum(remaining[2:]))
        max_brake = sum(np.minimum(remaining, [
            self.front_brake / 2.0, self.front_brake / 2.0,
            (self.rear_brake + self.regen) / 2.0,
            (self.rear_brake + self.regen) / 2.0,
        ]))
        return np.array([
            front_capacity - front_y, rear_capacity - rear_y,
            max_drive - required_x, max_brake + required_x,
        ])

    def boundary(self, points: int = 361) -> np.ndarray:
        """Trace first feasible boundary along rays from the coast point.

        A coarse outward scan brackets the first constraint crossing, then
        Brent refines it. The connected envelope around coasting is retained.
        Coordinates returned are (lateral g, longitudinal g).
        """
        v = self.vehicle
        g = v.gravity_mps2
        aero = v.aero_forces_n(self.speed, 0.0, curvature_per_m=0.0)
        coast = -(aero.drag_n + v.rolling_resistance_coefficient * (
            v.mass_kg * g + aero.downforce_n
        )) / v.effective_longitudinal_mass_kg
        result = []
        for theta in np.linspace(0.0, 2.0 * pi, points):
            dx, dy = cos(theta), sin(theta)

            def margin(radius: float) -> float:
                return float(min(self.constraints(coast + radius * dx, radius * dy)))

            lower = 0.0
            upper = 0.25 * g
            while margin(upper) >= 0.0:
                lower, upper = upper, upper + 0.25 * g
                if upper > 10.0 * g:
                    raise RuntimeError("No capacity boundary found below 10 g")
            radius = brentq(margin, lower, upper, xtol=1e-7)
            ax, ay = coast + radius * dx, radius * dy
            if min(self.constraints(ax, ay)) < -0.01:
                raise RuntimeError("Envelope contains an infeasible point")
            result.append((ay / g, ax / g))
        return np.array(result)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.axhline(0, color="#8794a6", lw=0.8)
    axis.axvline(0, color="#8794a6", lw=0.8)
    axis.grid(alpha=0.20)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Lateral acceleration [g]")
    axis.set_ylabel("Longitudinal acceleration [g]")
    axis.spines[["top", "right"]].set_visible(False)


def generate(vehicle: Vehicle, speeds: list[float], points: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    curves = {}
    rows, limits = [], []
    for speed in speeds:
        for regen in (False, True):
            envelope = CapacityEnvelope(vehicle, speed, regen)
            curve = envelope.boundary(points)
            curves[speed, regen] = curve
            for ay, ax in curve:
                rows.append(dict(speed_mph=speed, regen_enabled=regen,
                                 lateral_g=ay, longitudinal_g=ax))
            limits.append(dict(
                speed_mph=speed, regen_enabled=regen,
                straight_acceleration_g=curve[0, 1],
                straight_braking_g=-curve[(points - 1) // 2, 1],
                peak_lateral_g=float(max(abs(curve[:, 0]))),
                zero_longitudinal_lateral_g=brentq(
                    lambda ay: float(min(envelope.constraints(0.0, ay))),
                    0.0, float(max(abs(curve[:, 0]))) * vehicle.gravity_mps2,
                ) / vehicle.gravity_mps2,
                available_regen_force_n=envelope.regen,
            ))
        print(f"Solved {speed:g} mph", flush=True)

    colors = plt.get_cmap("viridis")(np.linspace(0.06, 0.92, len(speeds)))
    values = np.concatenate(list(curves.values()))
    xlim = np.ceil(max(abs(values[:, 0])) * 2.0) / 2.0 + 0.1
    ymin = np.floor(min(values[:, 1]) * 2.0) / 2.0 - 0.1
    ymax = np.ceil(max(values[:, 1]) * 2.0) / 2.0 + 0.1
    note = (
        f"{vehicle.mass_kg:.1f} kg | {vehicle.brakes.maximum_pressure_psi:g} psi per axle | "
        f"SOC {vehicle.battery.state_of_charge:.0%} | regen charge cap "
        f"{vehicle.battery.max_charge_power_w / 1000:g} kW "
        f"(available {vehicle.battery.charge_power_limit_w / 1000:.1f} kW)\n"
        "Quasi-static peak-grip capacity; load transfer, camber, roll-sensitive aero, "
        "drag and rotating mass included.\n"
        "Ideal axle brake modulation; fixed yaw-balanced lateral split; "
        "steering/slip equilibrium and transient stability not solved."
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    for regen, axis in zip((False, True), axes, strict=True):
        for speed, color in zip(speeds, colors, strict=True):
            curve = curves[speed, regen]
            axis.plot(curve[:, 0], curve[:, 1], color=color, lw=2, label=f"{speed:g} mph")
        style_axis(axis)
        axis.set(xlim=(-xlim, xlim), ylim=(ymin, ymax),
                 title="Friction brakes + rear regen" if regen else "Friction brakes only")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92),
                  ncol=min(7, len(speeds)), fontsize=10, frameon=False)
    figure.suptitle("TR21 model | G-G tire-capacity bounds", fontsize=18, y=0.98)
    figure.text(0.5, 0.02, note, ha="center", fontsize=9, linespacing=1.5)
    figure.tight_layout(rect=(0, 0.15, 1, 0.85))
    for extension in ("png", "svg"):
        figure.savefig(output / f"gg_envelopes.{extension}", dpi=200, facecolor="white")
    plt.close(figure)

    columns = 4
    count_rows = int(np.ceil((len(speeds) + 1) / columns))
    figure, axes = plt.subplots(count_rows, columns, figsize=(16, 4 * count_rows + 1.6))
    flat = np.asarray(axes).ravel()
    for speed, axis in zip(speeds, flat):
        for regen, color, style in ((False, "#1e6091", "--"), (True, "#df6726", "-")):
            curve = curves[speed, regen]
            axis.plot(curve[:, 0], curve[:, 1], color=color, ls=style, lw=2)
        style_axis(axis)
        axis.set(title=f"{speed:g} mph", xlim=(-xlim, xlim), ylim=(ymin, ymax))
    for axis in flat[len(speeds):]:
        axis.axis("off")
    flat[-1].legend(handles=[
        Line2D([], [], color="#1e6091", ls="--", lw=2, label="Friction brakes only"),
        Line2D([], [], color="#df6726", lw=2, label="Friction + rear regen"),
    ], loc="center", frameon=False, fontsize=12)
    figure.suptitle("Regen comparison at each speed", fontsize=18, y=0.98)
    figure.text(0.5, 0.02, note, ha="center", fontsize=9, linespacing=1.5)
    figure.tight_layout(rect=(0, 0.15, 1, 0.94))
    figure.savefig(output / "gg_regen_comparison.png", dpi=180, facecolor="white")
    plt.close(figure)
    write_csv(output / "gg_boundary.csv", rows)
    write_csv(output / "gg_limits.csv", limits)
    summary = dict(
        mass_kg=vehicle.mass_kg, effective_longitudinal_mass_kg=vehicle.effective_longitudinal_mass_kg,
        state_of_charge=vehicle.battery.state_of_charge,
        regen_charge_cap_w=vehicle.battery.max_charge_power_w,
        available_charge_power_w=vehicle.battery.charge_power_limit_w,
        available_discharge_power_w=vehicle.battery.discharge_power_limit_w,
        maximum_brake_pressure_psi=vehicle.brakes.maximum_pressure_psi,
        tire_lateral_model=type(vehicle.tire.pacejka_lateral).__name__,
        tire_longitudinal_model="default load-sensitive coefficient table",
        speeds_mph=speeds, boundary_points=points, assumptions=note,
        regen_configuration="80 kW default matches analysis/endurance/project_comp_and_capped_scores.py; "
                            "Vehicle() otherwise defaults to disabled charging (0 kW).",
        limits=limits,
    )
    (output / "gg_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds-mph", type=float, nargs="+", default=list(range(10, 71, 10)))
    parser.add_argument("--soc", type=float, default=0.8)
    parser.add_argument("--regen-charge-kw", type=float, default=80.0)
    parser.add_argument("--points", type=int, default=361)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/performance/gg")
    args = parser.parse_args()
    if not isfinite(args.soc) or not 0 <= args.soc <= 1:
        parser.error("--soc must be between 0 and 1")
    if not isfinite(args.regen_charge_kw) or args.regen_charge_kw < 0:
        parser.error("--regen-charge-kw must be finite and nonnegative")
    if any(not isfinite(speed) or not 0 < speed <= 70 for speed in args.speeds_mph):
        parser.error("--speeds-mph values must be positive and at most 70 mph")
    if args.points < 17 or (args.points - 1) % 4:
        parser.error("--points must equal 4*n + 1 and be at least 17")
    vehicle = Vehicle()
    vehicle.battery.initial_state_of_charge = args.soc
    vehicle.battery.max_charge_power_w = args.regen_charge_kw * 1000.0
    vehicle.reset_state()
    vehicle.validate()
    generate(vehicle, sorted(set(args.speeds_mph)), args.points, args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
