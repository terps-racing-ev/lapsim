"""Export and plot SOC from the available endurance recording."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis/mf4_to_csv/endurance_selected.csv"
DEFAULT_TRACK_METADATA = ROOT / "analysis/first_endurance_lap/map_derived_track.json"
DEFAULT_OUTPUT = ROOT / "analysis/endurance_soc/output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--track-metadata", type=Path, default=DEFAULT_TRACK_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def main() -> None:
    args = parse_args()
    source = read_numeric_csv(args.input_csv)
    track_metadata = json.loads(args.track_metadata.read_text(encoding="utf-8"))
    track_length_m = float(track_metadata["track_length_m"])
    valid = np.logical_and.reduce(
        tuple(
            np.isfinite(source[name])
            for name in (
                "time_s",
                "gps_speed_mps",
                "battery_soc_percent",
                "battery_power_kw",
                "battery_voltage_v",
                "battery_current_a",
            )
        )
    )
    time_s = source["time_s"][valid]
    time_s = time_s - time_s[0]
    speed_mps = source["gps_speed_mps"][valid]
    distance_m = np.zeros_like(time_s)
    distance_m[1:] = np.cumsum(
        0.5 * (speed_mps[1:] + speed_mps[:-1]) * np.diff(time_s)
    )
    soc_percent = source["battery_soc_percent"][valid]
    power_kw = source["battery_power_kw"][valid]
    voltage_v = source["battery_voltage_v"][valid]
    current_a = source["battery_current_a"][valid]
    cumulative_energy_kwh = np.zeros_like(time_s)
    cumulative_energy_kwh[1:] = np.cumsum(
        0.5 * (power_kw[1:] + power_kw[:-1]) * np.diff(time_s) / 3600.0
    )
    approximate_lap = np.floor(distance_m / track_length_m).astype(int) + 1
    lap_station_m = np.mod(distance_m, track_length_m)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "elapsed_time_s": float(time_s[index]),
            "distance_integrated_from_gnss_speed_m": float(distance_m[index]),
            "approximate_map_lap": int(approximate_lap[index]),
            "approximate_map_lap_station_m": float(lap_station_m[index]),
            "battery_soc_percent": float(soc_percent[index]),
            "soc_drop_from_start_percentage_points": float(
                soc_percent[0] - soc_percent[index]
            ),
            "battery_power_kw": float(power_kw[index]),
            "battery_voltage_v": float(voltage_v[index]),
            "battery_current_a": float(current_a[index]),
            "cumulative_net_pack_energy_kwh": float(cumulative_energy_kwh[index]),
        }
        for index in range(len(time_s))
    ]
    output_csv = args.output_dir / "recorded_endurance_soc.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        layout="constrained",
    )
    axes[0].plot(distance_m / 1_000.0, soc_percent, lw=2.0)
    axes[0].set_ylabel("Recorded HVC SOC [%]")
    axes[1].plot(distance_m / 1_000.0, power_kw, lw=1.0)
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_ylabel("Pack power [kW]")
    axes[2].plot(distance_m / 1_000.0, cumulative_energy_kwh, lw=2.0)
    axes[2].set_ylabel("Net pack energy [kWh]")
    axes[2].set_xlabel("Distance integrated from GNSS speed [km]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Available endurance recording: SOC and energy")
    figure.savefig(args.output_dir / "recorded_endurance_soc.png", dpi=190)
    plt.close(figure)

    metrics = {
        "source_csv": str(args.input_csv.resolve()),
        "sample_count": len(time_s),
        "recorded_duration_s": float(time_s[-1]),
        "distance_integrated_from_gnss_speed_m": float(distance_m[-1]),
        "distance_source": "trapezoidal integration of recorded GNSS speed",
        "map_track_length_m": track_length_m,
        "approximate_map_laps_covered": float(distance_m[-1] / track_length_m),
        "initial_soc_percent": float(soc_percent[0]),
        "final_soc_percent": float(soc_percent[-1]),
        "soc_drop_percentage_points": float(soc_percent[0] - soc_percent[-1]),
        "net_pack_energy_kwh": float(cumulative_energy_kwh[-1]),
        "scope_note": (
            "This is the complete available selected-signal recording, not a "
            "claim that the log contains the full 22 km endurance event."
        ),
    }
    (args.output_dir / "recorded_endurance_soc_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote endurance SOC outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
