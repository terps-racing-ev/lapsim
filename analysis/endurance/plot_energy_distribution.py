"""Plot spatial and lapwise energy distributions for capped endurance."""

from __future__ import annotations

import argparse
import csv
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

from analysis.endurance.project_comp_and_capped_scores import (  # noqa: E402
    NOMINAL_WEIGHT_LB,
    TRACK_PATH,
    SpatialTrack,
    coarsen_track,
    configured_vehicle,
    simulate_weight,
)
from utils.units import miles_per_hour_to_meters_per_second  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs/endurance_capped_40kw_40mph"
JOULES_PER_WATT_HOUR = 3_600.0


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aero_summary(weight_lb: float) -> dict[str, object]:
    vehicle = configured_vehicle(weight_lb, endurance_caps=True)
    speed_mps = miles_per_hour_to_meters_per_second(35.0)
    cases = []
    for lateral_g in (0.0, 0.5, 1.0, 1.5):
        forces = vehicle.aero_forces_n(
            speed_mps,
            lateral_g * vehicle.gravity_mps2,
        )
        cases.append(
            {
                "lateral_acceleration_g": lateral_g,
                "body_roll_deg": np.degrees(forces.body_roll_angle_rad),
                "downforce_multiplier": forces.downforce_multiplier,
                "drag_n": forces.drag_n,
                "downforce_n": forces.downforce_n,
                "front_downforce_n": forces.front_downforce_n,
                "rear_downforce_n": forces.rear_downforce_n,
                "downforce_to_drag_ratio": forces.downforce_n / forces.drag_n,
            }
        )
    baseline_drag_n = float(cases[0]["drag_n"])
    return {
        "weight_lb": weight_lb,
        "speed_mph": 35.0,
        "speed_mps": speed_mps,
        "air_density_kgpm3": vehicle.air_density_kgpm3,
        "frontal_area_m2": vehicle.aero.frontal_area_m2,
        "drag_coefficient": vehicle.aero.drag_coefficient,
        "lift_coefficient": vehicle.aero.lift_coefficient,
        "drag_area_m2": vehicle.aero.drag_area_m2,
        "downforce_area_m2": vehicle.aero.downforce_area_m2,
        "drag_at_cd_2_5_n_for_comparison": (
            baseline_drag_n * 2.5 / vehicle.aero.drag_coefficient
        ),
        "roll_model": (
            "drag is roll-independent; downforce falls linearly to 50% at "
            "1 degree absolute body roll"
        ),
        "cases": cases,
    }


def telemetry_arrays(run) -> dict[str, np.ndarray]:
    if run.telemetry is None:
        raise RuntimeError("endurance telemetry was not recorded")
    return {
        name: np.asarray(values, dtype=float)
        for name, values in run.telemetry.items()
    }


def sample_energy_wh(
    telemetry: dict[str, np.ndarray],
    track: SpatialTrack,
) -> dict[str, np.ndarray]:
    times_s = telemetry["vehicle.time_s"]
    timestep_s = np.diff(np.concatenate(([0.0], times_s)))
    cell_index = telemetry["endurance.cell_index"].astype(int)
    distance_step_m = np.asarray(track.cell_length_m)[cell_index]
    propulsion_power_w = telemetry["inverter.dc_input_power_w"]
    regen_power_w = telemetry["vehicle.regenerative_power_w"]
    return {
        "timestep_s": timestep_s,
        "distance_step_m": distance_step_m,
        "gross_pack_wh": propulsion_power_w * timestep_s / JOULES_PER_WATT_HOUR,
        "recovered_pack_wh": regen_power_w * timestep_s / JOULES_PER_WATT_HOUR,
        "net_pack_wh": (
            telemetry["battery.power_w"] * timestep_s / JOULES_PER_WATT_HOUR
        ),
        "wheel_drive_wh": (
            telemetry["vehicle.drive_force_n"]
            * distance_step_m
            / JOULES_PER_WATT_HOUR
        ),
        "aero_wh": (
            telemetry["aero.drag_n"] * distance_step_m / JOULES_PER_WATT_HOUR
        ),
        "rolling_wh": (
            telemetry["vehicle.rolling_resistance_force_n"]
            * distance_step_m
            / JOULES_PER_WATT_HOUR
        ),
        "cornering_drag_wh": (
            telemetry["vehicle.cornering_drag_force_n"]
            * distance_step_m
            / JOULES_PER_WATT_HOUR
        ),
        "friction_braking_wh": (
            telemetry["vehicle.friction_braking_force_n"]
            * distance_step_m
            / JOULES_PER_WATT_HOUR
        ),
        "regenerative_braking_wheel_wh": (
            telemetry["vehicle.regenerative_braking_force_n"]
            * distance_step_m
            / JOULES_PER_WATT_HOUR
        ),
    }


ENERGY_NAMES = (
    "gross_pack_wh",
    "recovered_pack_wh",
    "net_pack_wh",
    "wheel_drive_wh",
    "aero_wh",
    "rolling_wh",
    "cornering_drag_wh",
    "friction_braking_wh",
    "regenerative_braking_wheel_wh",
)


def lap_rows(
    telemetry: dict[str, np.ndarray],
    energy: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    lap_index = telemetry["endurance.lap_index"].astype(int)
    times_s = telemetry["vehicle.time_s"]
    rows = []
    previous_finish_s = 0.0
    for lap in range(int(np.max(lap_index)) + 1):
        mask = lap_index == lap
        finish_s = float(times_s[np.flatnonzero(mask)[-1]])
        row = {
            "lap": float(lap + 1),
            "lap_time_s": finish_s - previous_finish_s,
            "end_soc_percent": float(
                100.0 * telemetry["battery.state_of_charge"][np.flatnonzero(mask)[-1]]
            ),
        }
        row.update({name: float(np.sum(energy[name][mask])) for name in ENERGY_NAMES})
        rows.append(row)
        previous_finish_s = finish_s
    return rows


def spatial_rows(
    telemetry: dict[str, np.ndarray],
    energy: dict[str, np.ndarray],
    track: SpatialTrack,
    bin_width_m: float,
) -> list[dict[str, float]]:
    lap_index = telemetry["endurance.lap_index"].astype(int)
    station_m = telemetry["endurance.lap_distance_m"]
    bin_count = int(np.ceil(track.length_m / bin_width_m))
    bin_index = np.minimum((station_m / bin_width_m).astype(int), bin_count - 1)
    rows = []
    for lap in range(int(np.max(lap_index)) + 1):
        for spatial_bin in range(bin_count):
            mask = (lap_index == lap) & (bin_index == spatial_bin)
            if not np.any(mask):
                continue
            row = {
                "lap": float(lap + 1),
                "station_start_m": spatial_bin * bin_width_m,
                "station_end_m": min((spatial_bin + 1) * bin_width_m, track.length_m),
                "mean_speed_mps": float(
                    np.average(
                        telemetry["vehicle.speed_mps"][mask],
                        weights=energy["timestep_s"][mask],
                    )
                ),
            }
            row.update(
                {name: float(np.sum(energy[name][mask])) for name in ENERGY_NAMES}
            )
            rows.append(row)
    return rows


def plot_lap_energy(
    path: Path,
    rows: list[dict[str, float]],
    weight_lb: float,
) -> None:
    lap = np.asarray([row["lap"] for row in rows])
    gross = np.asarray([row["gross_pack_wh"] for row in rows])
    recovered = np.asarray([row["recovered_pack_wh"] for row in rows])
    net = np.asarray([row["net_pack_wh"] for row in rows])
    soc = np.asarray([row["end_soc_percent"] for row in rows])
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 8.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    axes[0].plot(lap, gross, marker="o", label="Gross pack discharge")
    axes[0].plot(lap, net, marker="s", label="Net pack energy")
    axes[0].plot(lap, recovered, marker="^", label="Recovered to pack")
    axes[0].set_ylabel("Energy per lap [Wh]")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(lap, soc, marker="o", color="#2563eb")
    axes[1].axhline(80.0, linestyle="--", color="#dc2626", label="Regen threshold")
    axes[1].set(xlabel="Completed lap", ylabel="End-of-lap SOC [%]")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{weight_lb:g} lb capped-endurance energy distribution by lap"
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_aero_35mph(path: Path, summary: dict[str, object]) -> None:
    cases = summary["cases"]
    lateral_g = np.asarray([case["lateral_acceleration_g"] for case in cases])
    downforce_n = np.asarray([case["downforce_n"] for case in cases])
    drag_n = np.asarray([case["drag_n"] for case in cases])
    figure, axis = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
    axis.plot(lateral_g, downforce_n, marker="o", label="Downforce with roll loss")
    axis.plot(lateral_g, drag_n, marker="s", label="Drag (roll-independent)")
    for x_value, y_value in zip(lateral_g, downforce_n, strict=True):
        axis.annotate(
            f"{y_value:.0f} N",
            (x_value, y_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    axis.set(
        xlabel="Lateral acceleration [g]",
        ylabel="Aerodynamic force at 35 mph [N]",
        title=(
            f"35 mph aero forces — Cd {summary['drag_coefficient']:.1f}, "
            f"Cl {summary['lift_coefficient']:.2f}"
        ),
    )
    axis.grid(True, alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _rows_for_lap(
    rows: list[dict[str, float]],
    lap: int,
) -> list[dict[str, float]]:
    return [row for row in rows if int(row["lap"]) == lap]


def plot_spatial_energy(path: Path, rows: list[dict[str, float]]) -> None:
    regen = _rows_for_lap(rows, 22)
    station = np.asarray(
        [0.5 * (row["station_start_m"] + row["station_end_m"]) for row in regen]
    )
    figure, axes = plt.subplots(3, 1, figsize=(13.0, 10.0), sharex=True, constrained_layout=True)

    gross = np.asarray([row["gross_pack_wh"] for row in regen])
    recovered = np.asarray([row["recovered_pack_wh"] for row in regen])
    net = np.asarray([row["net_pack_wh"] for row in regen])
    axes[0].plot(station, gross, label="Gross pack discharge")
    axes[0].plot(station, net, label="Net pack energy")
    axes[0].plot(station, -recovered, linestyle="--", label="Recovered to pack")
    axes[0].axhline(0.0, color="0.45", linewidth=0.8)
    axes[0].set_ylabel("Pack energy [Wh / 10 m]")
    axes[0].legend(frameon=False, ncol=2)

    aero = np.asarray([row["aero_wh"] for row in regen])
    rolling = np.asarray([row["rolling_wh"] for row in regen])
    cornering = np.asarray([row["cornering_drag_wh"] for row in regen])
    axes[1].stackplot(
        station,
        aero,
        rolling,
        cornering,
        labels=("Aero drag", "Rolling resistance", "Cornering drag"),
        alpha=0.8,
    )
    axes[1].set_ylabel("Road-loss work [Wh / 10 m]")
    axes[1].legend(frameon=False, ncol=3)

    friction = np.asarray([row["friction_braking_wh"] for row in regen])
    regenerative = np.asarray(
        [row["regenerative_braking_wheel_wh"] for row in regen]
    )
    axes[2].stackplot(
        station,
        friction,
        regenerative,
        labels=("Friction braking", "Regenerative braking at wheels"),
        alpha=0.8,
    )
    axes[2].set(xlabel="Lap station [m]", ylabel="Braking work [Wh / 10 m]")
    axes[2].legend(frameon=False, ncol=2)

    for axis in axes:
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Energy distribution around the 989 m endurance lap")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weight-lb", type=float, default=NOMINAL_WEIGHT_LB)
    parser.add_argument("--cell-length-m", type=float, default=1.0)
    parser.add_argument("--spatial-bin-m", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cell_length_m <= 0.0 or args.spatial_bin_m <= 0.0:
        raise ValueError("cell length and spatial bin width must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    track = coarsen_track(SpatialTrack.from_csv(TRACK_PATH), args.cell_length_m)
    _, run, _ = simulate_weight(
        track,
        args.weight_lb,
        record_telemetry=True,
    )
    telemetry = telemetry_arrays(run)
    energy = sample_energy_wh(telemetry, track)
    laps = lap_rows(telemetry, energy)
    spatial = spatial_rows(telemetry, energy, track, args.spatial_bin_m)

    aero = aero_summary(args.weight_lb)
    totals = {
        name.replace("_wh", "_kwh"): float(np.sum(energy[name])) / 1_000.0
        for name in ENERGY_NAMES
    }
    summary = {
        "configuration": {
            "weight_lb": args.weight_lb,
            "track_length_m": track.length_m,
            "track_cell_length_m": args.cell_length_m,
            "spatial_bin_m": args.spatial_bin_m,
            "endurance_time_s": run.driving_time_s,
            "completed_laps": run.completed_laps,
        },
        "aero_at_35_mph": aero,
        "event_energy_kwh": totals,
    }
    (output_dir / "energy_distribution_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "energy_distribution_by_lap.csv", laps)
    write_csv(output_dir / "energy_distribution_by_lap_and_station.csv", spatial)
    plot_aero_35mph(output_dir / "aero_35mph.png", aero)
    plot_lap_energy(
        output_dir / "energy_distribution_by_lap.png",
        laps,
        args.weight_lb,
    )
    plot_spatial_energy(output_dir / "energy_distribution_along_lap.png", spatial)
    print(json.dumps(summary, indent=2))
    print(f"Wrote energy distributions to {output_dir}")


if __name__ == "__main__":
    main()
