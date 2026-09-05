"""Project the 2026 competition DNF and score a capped endurance sweep.

The competition projection uses Maryland's five official lap times and the
matching local competition telemetry for pack energy. The simulation uses the
calibrated current vehicle. The 40 kW pack/motor cap and 40 mph vehicle-speed
cap apply only to endurance. Acceleration and skidpad retain the calibrated
vehicle's default 80 kW/100 mph limits. The uncertainty-adjusted sweep adds
0.10 s to acceleration, scales skidpad time by 1.05, and scales endurance time
and net energy by 1.05. Endurance uses maximum rear regenerative braking only
while pack SOC is below 80%.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.autox_corr.model import LinearRatioModel  # noqa: E402
from analysis.events.acceleration_points import (  # noqa: E402
    run as run_acceleration,
    standard_acceleration_track,
)
from analysis.events.common import coarsen_track  # noqa: E402
from analysis.events.skidpad_points import (  # noqa: E402
    run as run_skidpad,
    standard_skidpad_circle,
)
from lapsim import (  # noqa: E402
    ConstantControlsProfile,
    Controls,
    EnduranceRunConfig,
    EnduranceSimulator,
    FSAE_2026_MI_ACCELERATION_SCORING,
    FSAE_2026_MI6_SCORING,
    FSAE_2026_MI_SKIDPAD_SCORING,
    PathConstraintSolver,
    SkidpadConfig,
    SpatialTrack,
    UniformPeriodicTorqueParameterization,
)
from utils.units import (  # noqa: E402
    miles_per_hour_to_meters_per_second,
    pounds_to_kilograms,
)
from vehicle_model import Vehicle  # noqa: E402


OFFICIAL_RESULTS_URL = (
    "https://www.fsaeonline.com/CompResources/2026/"
    "07af50d8-cbb6-4b9b-aaf8-5ff6a7e44057/FSAE_2026_MI6_results.pdf"
)
COMPETITION_TELEMETRY = (
    ROOT / "analysis/data/telemetry/endurance_selected.csv"
)
TRACK_PATH = ROOT / "analysis/data/track/gnss_imu_endurance_track.csv"
DEFAULT_OUTPUT = ROOT / "outputs/endurance_capped_40kw_40mph"

COMP_LAP_TIMES_S = (68.314, 67.711, 67.858, 65.871, 66.755)
COMP_START_MF4_TIME_S = 465.816454
COMP_REGEN_CREDIT_KWH = 0.5
ACCELERATION_TIME_ADDITION_S = 0.10
SKIDPAD_TIME_SCALE = 1.05
ENDURANCE_TIME_SCALE = 1.05
ENDURANCE_ENERGY_SCALE = 1.05
BASELINE_CHAIN_DRIVE_EFFICIENCY = 0.80
BASELINE_MOTOR_ROTOR_INERTIA_KGM2 = 0.02521
BASELINE_DRAG_COEFFICIENT = 2.4
NOMINAL_WEIGHT_LB = 630.0
EVENT_LAPS = FSAE_2026_MI6_SCORING.event_laps

# Existing 2026 Maryland scores held fixed for the competition what-if.
MARYLAND_STATIC_POINTS = 69.5 + 18.8 + 100.0
MARYLAND_RECORDED_OTHER_DYNAMIC_POINTS = 66.5 + 29.6 + 74.4

# Published score lists are only used to calculate insertion place. Scores are
# in official finishing order; the script does not treat a modeled tie as a win.
OFFICIAL_ENDURANCE_SCORES = (
    275.0, 227.3, 218.1, 216.5, 206.5, 202.3, 202.2, 186.8, 186.2,
    158.1, 154.9, 151.3, 129.2, 112.1, 111.1, 104.1, 81.5, 67.4,
    25.0, 25.0, 25.0, 25.0,
)
OFFICIAL_EFFICIENCY_SCORES = (
    100.0, 89.2, 71.7, 66.1, 58.4, 57.1, 55.8, 52.3, 49.3, 48.6,
    43.0, 42.2, 41.4, 40.1, 38.0, 37.6, 37.0, 34.4, 28.7, 28.2,
)
OFFICIAL_OVERALL_SCORES = (
    895.5, 856.7, 807.2, 756.9, 756.8, 743.9, 717.5, 696.0, 681.3,
    671.3, 625.8, 609.7, 576.1, 562.4, 543.0, 527.8, 508.5, 491.6,
    475.1, 462.0, 429.4, 394.3, 391.5, 370.7, 364.8, 358.5, 324.0,
)

AUTOCROSS_MODEL = LinearRatioModel(
    intercept_ratio=0.1970917639593186,
    acceleration_weight=0.1330940868734024,
    skidpad_weight=0.7431733675405259,
)


@dataclass(frozen=True, slots=True)
class ScoringRun:
    completed_laps: int
    driving_time_s: float
    pack_energy_kwh: float
    failure_reason: str | None = None


def insertion_place(score: float, official_scores: tuple[float, ...]) -> int:
    return 1 + sum(official_score > score + 1e-9 for official_score in official_scores)


def read_competition_power() -> tuple[np.ndarray, np.ndarray]:
    with COMPETITION_TELEMETRY.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    time_s = np.asarray([float(row["mf4_time_s"]) for row in rows])
    power_kw = np.asarray([float(row["battery_power_kw"]) for row in rows])
    valid = np.isfinite(time_s) & np.isfinite(power_kw)
    return time_s[valid], power_kw[valid]


def integrate_power_kwh(
    time_s: np.ndarray,
    power_kw: np.ndarray,
    start_s: float,
    finish_s: float,
    *,
    sign: str = "net",
) -> float:
    if not time_s[0] <= start_s < finish_s <= time_s[-1]:
        raise ValueError("integration interval is outside competition telemetry")
    interior = (time_s > start_s) & (time_s < finish_s)
    sample_time_s = np.concatenate(([start_s], time_s[interior], [finish_s]))
    sample_power_kw = np.interp(sample_time_s, time_s, power_kw)
    if sign == "positive":
        sample_power_kw = np.maximum(sample_power_kw, 0.0)
    elif sign == "negative_magnitude":
        sample_power_kw = -np.minimum(sample_power_kw, 0.0)
    elif sign != "net":
        raise ValueError(f"unknown power integration sign: {sign}")
    return float(np.trapezoid(sample_power_kw, sample_time_s) / 3_600.0)


def competition_projection() -> dict[str, object]:
    time_s, power_kw = read_competition_power()
    five_lap_time_s = sum(COMP_LAP_TIMES_S)
    finish_s = COMP_START_MF4_TIME_S + five_lap_time_s
    measured_net_energy_kwh = integrate_power_kwh(
        time_s, power_kw, COMP_START_MF4_TIME_S, finish_s
    )
    measured_positive_energy_kwh = integrate_power_kwh(
        time_s,
        power_kw,
        COMP_START_MF4_TIME_S,
        finish_s,
        sign="positive",
    )
    measured_recovered_energy_kwh = integrate_power_kwh(
        time_s,
        power_kw,
        COMP_START_MF4_TIME_S,
        finish_s,
        sign="negative_magnitude",
    )
    scale = EVENT_LAPS / len(COMP_LAP_TIMES_S)
    projected_time_s = five_lap_time_s * scale
    projected_before_credit_kwh = measured_net_energy_kwh * scale
    projected_net_energy_kwh = max(
        projected_before_credit_kwh - COMP_REGEN_CREDIT_KWH,
        0.0,
    )
    score = FSAE_2026_MI6_SCORING.score(
        ScoringRun(EVENT_LAPS, projected_time_s, projected_net_energy_kwh)
    )
    projected_overall_points = (
        MARYLAND_STATIC_POINTS
        + MARYLAND_RECORDED_OTHER_DYNAMIC_POINTS
        + score.combined_points
    )
    return {
        "source": {
            "telemetry": str(COMPETITION_TELEMETRY.resolve()),
            "official_results": OFFICIAL_RESULTS_URL,
        },
        "official_lap_times_s": COMP_LAP_TIMES_S,
        "measured_five_lap_time_s": five_lap_time_s,
        "measured_five_lap_net_energy_kwh": measured_net_energy_kwh,
        "measured_five_lap_positive_energy_kwh": measured_positive_energy_kwh,
        "measured_five_lap_recovered_energy_kwh": measured_recovered_energy_kwh,
        "linear_projection_before_regen_credit_kwh": projected_before_credit_kwh,
        "assumed_full_event_regen_credit_kwh": COMP_REGEN_CREDIT_KWH,
        "projected_time_s": projected_time_s,
        "projected_average_lap_time_s": projected_time_s / EVENT_LAPS,
        "projected_net_energy_kwh": projected_net_energy_kwh,
        "endurance_time_points": score.endurance_time_points,
        "endurance_lap_points": score.endurance_lap_points,
        "endurance_points": score.endurance_points,
        "efficiency_points": score.efficiency_points,
        "endurance_plus_efficiency_points": score.combined_points,
        "projected_endurance_place": insertion_place(
            score.endurance_points, OFFICIAL_ENDURANCE_SCORES
        ),
        "projected_efficiency_place": insertion_place(
            score.efficiency_points, OFFICIAL_EFFICIENCY_SCORES
        ),
        "held_static_points": MARYLAND_STATIC_POINTS,
        "held_accel_skidpad_autocross_points": (
            MARYLAND_RECORDED_OTHER_DYNAMIC_POINTS
        ),
        "projected_overall_points": projected_overall_points,
        "projected_overall_place": insertion_place(
            projected_overall_points, OFFICIAL_OVERALL_SCORES
        ),
    }


def configured_vehicle(
    weight_lb: float,
    *,
    endurance_caps: bool,
) -> Vehicle:
    vehicle = Vehicle()
    vehicle.mass_kg = pounds_to_kilograms(weight_lb)
    vehicle.aero.drag_coefficient = BASELINE_DRAG_COEFFICIENT
    vehicle.drivetrain.chain_drive.efficiency = BASELINE_CHAIN_DRIVE_EFFICIENCY
    vehicle.drivetrain.motor.rotor_inertia_kgm2 = (
        BASELINE_MOTOR_ROTOR_INERTIA_KGM2
    )
    vehicle.tire.constant_friction_coefficient = None
    vehicle.cornering_drag_coefficient = 0.036
    vehicle.battery.initial_state_of_charge = 0.9815
    # A high configured sink cap lets pack voltage, motor torque, and 40 kW
    # motor power determine the maximum available regenerative braking.
    vehicle.battery.max_charge_power_w = 80_000.0
    if endurance_caps:
        vehicle.battery.max_discharge_power_w = 40_000.0
        vehicle.drivetrain.motor.peak_power_w = 40_000.0
        vehicle.drivetrain.motor.continuous_power_w = 40_000.0
        vehicle.drivetrain.configured_speed_limit_mps = (
            miles_per_hour_to_meters_per_second(40.0)
        )
    vehicle.validate()
    return vehicle


def simulate_timed_events(weight_lb: float) -> dict[str, float]:
    full_torque = ConstantControlsProfile(Controls(motor_torque_request_nm=230.0))
    acceleration = run_acceleration(
        full_torque,
        standard_acceleration_track(0.5),
        vehicle=configured_vehicle(weight_lb, endurance_caps=False),
    )
    skidpad = run_skidpad(
        full_torque,
        standard_skidpad_circle(0.25),
        vehicle=configured_vehicle(weight_lb, endurance_caps=False),
        config=SkidpadConfig(starting_speed_mps=0.0),
    )
    if not acceleration.completed:
        raise RuntimeError(f"acceleration failed: {acceleration.failure_reason}")
    if not skidpad.completed:
        raise RuntimeError(f"skidpad failed: {skidpad.failure_reason}")
    acceleration_simulated_time_s = float(acceleration.scoring_time_s)
    acceleration_adjusted_time_s = (
        acceleration_simulated_time_s + ACCELERATION_TIME_ADDITION_S
    )
    acceleration_adjusted_score = FSAE_2026_MI_ACCELERATION_SCORING.score(
        acceleration_adjusted_time_s,
        completed=True,
    )
    skidpad_simulated_time_s = float(skidpad.scoring_time_s)
    skidpad_adjusted_time_s = skidpad_simulated_time_s * SKIDPAD_TIME_SCALE
    skidpad_adjusted_score = FSAE_2026_MI_SKIDPAD_SCORING.score(
        skidpad_adjusted_time_s,
        completed=True,
    )
    autocross_points = AUTOCROSS_MODEL.predict(
        acceleration_adjusted_score.points,
        skidpad_adjusted_score.points,
    )
    return {
        "acceleration_simulated_time_s": acceleration_simulated_time_s,
        "acceleration_time_addition_s": ACCELERATION_TIME_ADDITION_S,
        "acceleration_time_s": acceleration_adjusted_time_s,
        "acceleration_points": acceleration_adjusted_score.points,
        "skidpad_simulated_time_s": skidpad_simulated_time_s,
        "skidpad_time_scale": SKIDPAD_TIME_SCALE,
        "skidpad_time_s": skidpad_adjusted_time_s,
        "skidpad_points": skidpad_adjusted_score.points,
        "autocross_points": float(autocross_points),
    }


def telemetry_lap_rows(run) -> list[dict[str, float]]:
    assert run.telemetry is not None
    telemetry = run.telemetry
    times_s = np.asarray(telemetry["vehicle.time_s"])
    lap_index = np.asarray(telemetry["endurance.lap_index"], dtype=int)
    soc = np.asarray(telemetry["battery.state_of_charge"])
    cumulative_net_kwh = (
        np.asarray(telemetry["energy.cumulative_net_j"]) / 3_600_000.0
    )
    regen_power_w = np.asarray(telemetry["vehicle.regenerative_power_w"])
    timestep_s = np.diff(np.concatenate(([0.0], times_s)))
    cumulative_regen_kwh = np.cumsum(regen_power_w * timestep_s) / 3_600_000.0
    rows = []
    previous_time_s = 0.0
    for lap in range(EVENT_LAPS):
        indices = np.flatnonzero(lap_index == lap)
        if not len(indices):
            break
        finish = int(indices[-1])
        rows.append(
            {
                "lap": float(lap + 1),
                "lap_time_s": times_s[finish] - previous_time_s,
                "cumulative_time_s": times_s[finish],
                "end_soc_percent": 100.0 * soc[finish],
                "cumulative_net_energy_kwh": cumulative_net_kwh[finish],
                "cumulative_recovered_energy_kwh": cumulative_regen_kwh[finish],
            }
        )
        previous_time_s = times_s[finish]
    return rows


def simulate_weight(
    track: SpatialTrack,
    weight_lb: float,
    *,
    record_telemetry: bool,
) -> tuple[dict[str, float | str | None], object, object]:
    vehicle = configured_vehicle(weight_lb, endurance_caps=True)
    # At 2 m cells, a conservative 75 psi ceiling-generation pass provides
    # braking-distance buffer; the committed controller can still use 300 psi.
    constraints = PathConstraintSolver(
        maximum_brake_pressure_psi=75.0
    ).solve(track, vehicle)
    profile = UniformPeriodicTorqueParameterization(8).build((1.0,) * 8, track)
    config = EnduranceRunConfig(
        laps=EVENT_LAPS,
        maximum_brake_pressure_psi=300.0,
        path_speed_tolerance_mps=0.2,
        regenerative_braking_soc_threshold=0.8,
    )
    run = EnduranceSimulator().run(
        configured_vehicle(weight_lb, endurance_caps=True),
        constraints,
        profile,
        config,
        record_telemetry=record_telemetry,
    )
    if not run.completed:
        raise RuntimeError(
            f"{weight_lb:.1f} lb endurance failed after {run.completed_laps} laps: "
            f"{run.failure_reason}"
        )
    adjusted_endurance_time_s = run.driving_time_s * ENDURANCE_TIME_SCALE
    adjusted_endurance_energy_kwh = (
        run.pack_energy_kwh * ENDURANCE_ENERGY_SCALE
    )
    adjusted_scoring_run = ScoringRun(
        completed_laps=run.completed_laps,
        driving_time_s=adjusted_endurance_time_s,
        pack_energy_kwh=adjusted_endurance_energy_kwh,
        failure_reason=run.failure_reason,
    )
    score = FSAE_2026_MI6_SCORING.score(adjusted_scoring_run)
    timed = simulate_timed_events(weight_lb)
    dynamic_points = (
        score.combined_points
        + timed["acceleration_points"]
        + timed["skidpad_points"]
        + timed["autocross_points"]
    )
    row: dict[str, float | str | None] = {
        "weight_lb": weight_lb,
        "weight_kg": pounds_to_kilograms(weight_lb),
        "completed_laps": float(run.completed_laps),
        "simulated_endurance_time_s": run.driving_time_s,
        "endurance_time_scale": ENDURANCE_TIME_SCALE,
        "endurance_time_s": adjusted_endurance_time_s,
        "average_lap_time_s": adjusted_endurance_time_s / run.completed_laps,
        "simulated_net_energy_kwh": run.pack_energy_kwh,
        "endurance_energy_scale": ENDURANCE_ENERGY_SCALE,
        "net_energy_kwh": adjusted_endurance_energy_kwh,
        "final_soc_percent": 100.0 * run.final_state_of_charge,
        "endurance_time_points": score.endurance_time_points,
        "endurance_lap_points": score.endurance_lap_points,
        "endurance_points": score.endurance_points,
        "efficiency_points": score.efficiency_points,
        "endurance_plus_efficiency_points": score.combined_points,
        **timed,
        "dynamic_points": dynamic_points,
        "overall_points_with_2026_static_held": (
            dynamic_points + MARYLAND_STATIC_POINTS
        ),
        "illustrative_overall_place_with_2026_static_held": float(
            insertion_place(
                dynamic_points + MARYLAND_STATIC_POINTS,
                OFFICIAL_OVERALL_SCORES,
            )
        ),
    }
    return row, run, constraints


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    known_fields = set(fieldnames)
    for row in rows[1:]:
        for fieldname in row:
            if fieldname not in known_fields:
                fieldnames.append(fieldname)
                known_fields.add(fieldname)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_score_breakdown(path: Path, nominal: dict[str, object]) -> None:
    labels = (
        "Acceleration",
        "Skidpad",
        "Autocross\n(correlation)",
        "Endurance",
        "Efficiency",
    )
    values = (
        float(nominal["acceleration_points"]),
        float(nominal["skidpad_points"]),
        float(nominal["autocross_points"]),
        float(nominal["endurance_points"]),
        float(nominal["efficiency_points"]),
    )
    maxima = (100.0, 75.0, 125.0, 275.0, 100.0)
    figure, axis = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    bars = axis.bar(labels, values, color="#2563eb")
    axis.scatter(range(len(maxima)), maxima, marker="_", s=400, color="#6b7280")
    axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3)
    axis.set_ylabel("Predicted points")
    axis.set_title(
        f"{NOMINAL_WEIGHT_LB:g} lb predicted dynamic-event score\n"
        "Load-sensitive tires; 40 kW / 40 mph cap applied to endurance only; "
        "uncertainty corrections applied\n"
        f"Total: {float(nominal['dynamic_points']):.1f} / 675 points"
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_lap_energy(path: Path, rows: list[dict[str, float]]) -> None:
    lap = np.asarray([row["lap"] for row in rows])
    soc = np.asarray([row["end_soc_percent"] for row in rows])
    net = np.asarray([row["cumulative_net_energy_kwh"] for row in rows])
    recovered = np.asarray(
        [row["cumulative_recovered_energy_kwh"] for row in rows]
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(9.2, 7.0), sharex=True, constrained_layout=True
    )
    axes[0].plot(lap, soc, marker="o", color="#2563eb", label="End-of-lap SOC")
    axes[0].axhline(80.0, color="#dc2626", ls="--", label="Regen enable threshold")
    axes[0].set_ylabel("SOC [%]")
    axes[0].legend(frameon=False)
    axes[1].plot(lap, net, marker="o", color="#ea580c", label="Net pack energy")
    axes[1].plot(
        lap,
        recovered,
        marker="s",
        color="#16a34a",
        label="Recovered energy",
    )
    axes[1].set(xlabel="Completed lap", ylabel="Cumulative energy [kWh]")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{NOMINAL_WEIGHT_LB:g} lb capped-endurance battery trajectory"
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_weight_sweep(path: Path, rows: list[dict[str, object]]) -> None:
    weight = np.asarray([float(row["weight_lb"]) for row in rows])
    dynamic = np.asarray([float(row["dynamic_points"]) for row in rows])
    endurance = np.asarray(
        [float(row["endurance_plus_efficiency_points"]) for row in rows]
    )
    acceleration = np.asarray([float(row["acceleration_points"]) for row in rows])
    skidpad = np.asarray([float(row["skidpad_points"]) for row in rows])
    autocross = np.asarray([float(row["autocross_points"]) for row in rows])
    figure, axes = plt.subplots(
        2, 1, figsize=(9.4, 7.4), sharex=True, constrained_layout=True
    )
    axes[0].plot(weight, dynamic, marker="o", lw=2.2, color="#2563eb")
    axes[0].set_ylabel("Dynamic points / 675")
    axes[0].grid(alpha=0.25)
    axes[1].plot(weight, endurance, marker="o", label="Endurance + efficiency")
    axes[1].plot(weight, acceleration, marker="s", label="Acceleration")
    axes[1].plot(weight, skidpad, marker="^", label="Skidpad")
    axes[1].plot(weight, autocross, marker="d", label="Autocross correlation")
    axes[1].set(xlabel="Total vehicle weight [lb]", ylabel="Event points")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncols=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Predicted score versus vehicle weight\n"
        "Load-sensitive tires; endurance-only caps; +0.10 s acceleration; "
        "×1.05 skidpad/endurance"
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_weight_points_gained(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    weight = np.asarray([float(row["weight_lb"]) for row in rows])
    overall_points = np.asarray(
        [float(row["overall_points_with_2026_static_held"]) for row in rows]
    )
    nominal_index = int(np.argmin(np.abs(weight - NOMINAL_WEIGHT_LB)))
    points_gained = overall_points - overall_points[nominal_index]

    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    axis.plot(weight, points_gained, marker="o", lw=2.4, color="#2563eb")
    axis.axhline(0.0, color="#6b7280", lw=1.0)
    axis.set(
        xlabel="Total vehicle weight [lb]",
        ylabel=f"Overall points gained vs {NOMINAL_WEIGHT_LB:g} lb",
        title="Predicted points gained from vehicle weight reduction",
    )
    axis.annotate(
        f"+{points_gained[0]:.2f} points",
        (weight[0], points_gained[0]),
        xytext=(8, -2),
        textcoords="offset points",
        va="top",
    )
    axis.annotate(
        "baseline",
        (weight[nominal_index], points_gained[nominal_index]),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
    )
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_weight_sweep_performance(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    weight = np.asarray([float(row["weight_lb"]) for row in rows])
    endurance_time_min = np.asarray(
        [float(row["endurance_time_s"]) / 60.0 for row in rows]
    )
    net_energy_kwh = np.asarray(
        [float(row["net_energy_kwh"]) for row in rows]
    )
    final_soc_percent = np.asarray(
        [float(row["final_soc_percent"]) for row in rows]
    )
    figure, axes = plt.subplots(
        3, 1, figsize=(9.4, 8.4), sharex=True, constrained_layout=True
    )
    series = (
        (endurance_time_min, "Endurance time [min]", "#2563eb"),
        (net_energy_kwh, "Net pack energy [kWh]", "#ea580c"),
        (final_soc_percent, "Final SOC [%]", "#16a34a"),
    )
    for axis, (values, ylabel, color) in zip(axes, series, strict=True):
        axis.plot(weight, values, marker="o", lw=2.2, color=color)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Total vehicle weight [lb]")
    figure.suptitle(
        "Load-sensitive capped-endurance performance versus vehicle weight"
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cell-length-m", type=float, default=1.0)
    parser.add_argument("--weight-min-lb", type=float, default=580.0)
    parser.add_argument("--weight-max-lb", type=float, default=630.0)
    parser.add_argument("--weight-step-lb", type=float, default=5.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse non-nominal rows already present in weight_sweep.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cell_length_m <= 0.0 or args.weight_step_lb <= 0.0:
        raise ValueError("cell length and weight step must be positive")
    if args.weight_max_lb < args.weight_min_lb:
        raise ValueError("weight maximum must be at least the minimum")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comp = competition_projection()
    (output_dir / "competition_projection.json").write_text(
        json.dumps(comp, indent=2) + "\n", encoding="utf-8"
    )

    source_track = SpatialTrack.from_csv(TRACK_PATH)
    track = coarsen_track(source_track, args.cell_length_m)
    count = int(round((args.weight_max_lb - args.weight_min_lb) / args.weight_step_lb))
    weights = [
        args.weight_min_lb + index * args.weight_step_lb
        for index in range(count + 1)
    ]
    if not any(abs(weight - NOMINAL_WEIGHT_LB) < 1e-9 for weight in weights):
        weights.append(NOMINAL_WEIGHT_LB)
        weights.sort()

    sweep_rows: list[dict[str, object]] = []
    checkpoint_path = output_dir / "weight_sweep.csv"
    resumed_rows: dict[float, dict[str, object]] = {}
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                resumed_rows[float(row["weight_lb"])] = dict(row)
    nominal_run = None
    nominal_constraints = None
    for weight_lb in weights:
        if (
            abs(weight_lb - NOMINAL_WEIGHT_LB) >= 1e-9
            and weight_lb in resumed_rows
        ):
            print(f"Reusing {weight_lb:.1f} lb checkpoint...", flush=True)
            sweep_rows.append(resumed_rows[weight_lb])
            continue
        print(f"Simulating {weight_lb:.1f} lb...", flush=True)
        row, run, constraints = simulate_weight(
            track,
            weight_lb,
            record_telemetry=abs(weight_lb - NOMINAL_WEIGHT_LB) < 1e-9,
        )
        sweep_rows.append(row)
        write_csv(checkpoint_path, sweep_rows)
        if abs(weight_lb - NOMINAL_WEIGHT_LB) < 1e-9:
            nominal_run = run
            nominal_constraints = constraints

    assert nominal_run is not None and nominal_constraints is not None
    for row in sweep_rows:
        row["recovered_energy_kwh"] = None
        row["gross_energy_without_recovery_kwh"] = None
    nominal = next(
        row for row in sweep_rows
        if row["weight_lb"] == NOMINAL_WEIGHT_LB
    )
    lap_rows = telemetry_lap_rows(nominal_run)
    simulated_recovered_energy_kwh = lap_rows[-1][
        "cumulative_recovered_energy_kwh"
    ]
    recovered_energy_kwh = (
        simulated_recovered_energy_kwh * ENDURANCE_ENERGY_SCALE
    )
    nominal["simulated_recovered_energy_kwh"] = simulated_recovered_energy_kwh
    nominal["recovered_energy_kwh"] = recovered_energy_kwh
    nominal["gross_energy_without_recovery_kwh"] = (
        float(nominal["net_energy_kwh"]) + recovered_energy_kwh
    )

    no_regen_run = EnduranceSimulator().run(
        configured_vehicle(NOMINAL_WEIGHT_LB, endurance_caps=True),
        nominal_constraints,
        UniformPeriodicTorqueParameterization(8).build((1.0,) * 8, track),
        EnduranceRunConfig(
            laps=EVENT_LAPS,
            maximum_brake_pressure_psi=300.0,
            path_speed_tolerance_mps=0.2,
            regenerative_braking_soc_threshold=None,
        ),
        record_telemetry=False,
    )
    adjusted_no_regen = ScoringRun(
        completed_laps=no_regen_run.completed_laps,
        driving_time_s=no_regen_run.driving_time_s * ENDURANCE_TIME_SCALE,
        pack_energy_kwh=no_regen_run.pack_energy_kwh * ENDURANCE_ENERGY_SCALE,
        failure_reason=no_regen_run.failure_reason,
    )
    no_regen_score = FSAE_2026_MI6_SCORING.score(adjusted_no_regen)

    write_csv(output_dir / "weight_sweep.csv", sweep_rows)
    write_csv(output_dir / "nominal_lap_summary.csv", lap_rows)
    summary = {
        "assumptions": {
            "nominal_weight_lb": NOMINAL_WEIGHT_LB,
            "weight_sweep_lb": weights,
            "endurance_pack_and_motor_power_cap_kw": 40.0,
            "endurance_vehicle_speed_cap_mph": 40.0,
            "acceleration_and_skidpad_power_cap_kw": 80.0,
            "acceleration_and_skidpad_speed_cap_mph": 100.0,
            "acceleration_time_addition_s": ACCELERATION_TIME_ADDITION_S,
            "skidpad_time_scale": SKIDPAD_TIME_SCALE,
            "skidpad_fixed_time_addition_s": 0.0,
            "endurance_time_scale": ENDURANCE_TIME_SCALE,
            "endurance_energy_scale": ENDURANCE_ENERGY_SCALE,
            "chain_drive_efficiency": configured_vehicle(
                NOMINAL_WEIGHT_LB, endurance_caps=True
            ).drivetrain.chain_drive.efficiency,
            "chain_drive_efficiency_basis": (
                "80% chain-efficiency baseline"
            ),
            "motor_rotor_inertia_kgm2": BASELINE_MOTOR_ROTOR_INERTIA_KGM2,
            "drag_coefficient": BASELINE_DRAG_COEFFICIENT,
            "tire_model": "load-sensitive lookup table",
            "regen_policy": "maximum rear regen while SOC < 80%; otherwise zero",
            "regen_efficiency": "symmetric chain * motor * inverter efficiency",
            "track": str(TRACK_PATH.resolve()),
            "track_cell_length_m": args.cell_length_m,
            "torque_request_fraction": 1.0,
            "path_constraint_planning_brake_pressure_psi": 75.0,
            "committed_maximum_brake_pressure_psi": 300.0,
            "path_speed_tolerance_mps": 0.2,
            "static_points_held_for_illustrative_overall": MARYLAND_STATIC_POINTS,
        },
        "competition_projection": comp,
        f"nominal_{NOMINAL_WEIGHT_LB:g}_lb": nominal,
        "nominal_no_regen_comparison": {
            "completed_laps": no_regen_run.completed_laps,
            "simulated_time_s": no_regen_run.driving_time_s,
            "time_s": adjusted_no_regen.driving_time_s,
            "simulated_net_energy_kwh": no_regen_run.pack_energy_kwh,
            "net_energy_kwh": adjusted_no_regen.pack_energy_kwh,
            "final_soc_percent": 100.0 * no_regen_run.final_state_of_charge,
            "endurance_plus_efficiency_points": no_regen_score.combined_points,
            "failure_reason": no_regen_run.failure_reason,
        },
        "autocross_model": asdict(AUTOCROSS_MODEL),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    plot_score_breakdown(output_dir / "score_breakdown.png", nominal)
    plot_lap_energy(output_dir / "nominal_energy_soc.png", lap_rows)
    plot_weight_sweep(output_dir / "weight_sweep_points.png", sweep_rows)
    plot_weight_points_gained(
        output_dir / "weight_vs_points_gained.png",
        sweep_rows,
    )
    plot_weight_sweep_performance(
        output_dir / "weight_sweep_performance.png",
        sweep_rows,
    )

    print(
        json.dumps(summary[f"nominal_{NOMINAL_WEIGHT_LB:g}_lb"], indent=2)
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
