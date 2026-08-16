"""Create focused acceleration and motor-torque plots for detected straights."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_samples(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def plot_pair(
    acceleration_axis,
    torque_axis,
    elapsed_time_s: np.ndarray,
    measured_acceleration_mps2: np.ndarray,
    simulated_acceleration_mps2: np.ndarray,
    torque_command_nm: np.ndarray,
    torque_feedback_nm: np.ndarray,
    *,
    title: str,
) -> None:
    acceleration_axis.plot(
        elapsed_time_s,
        measured_acceleration_mps2,
        "o-",
        color="#2A6FBB",
        lw=2.0,
        ms=4.8,
        label="Corrected IMU",
    )
    acceleration_axis.plot(
        elapsed_time_s,
        simulated_acceleration_mps2,
        "s--",
        color="#E45756",
        lw=1.8,
        ms=4.2,
        label="Current simulation",
    )
    acceleration_axis.axhline(0.0, color="#555555", lw=0.8)
    acceleration_axis.set_title(title, loc="left", fontweight="bold")
    acceleration_axis.set_ylabel("Acceleration\n[m/s²]")
    acceleration_axis.legend(frameon=False, ncols=2, loc="best")

    torque_axis.plot(
        elapsed_time_s,
        torque_command_nm,
        "o-",
        color="#F2A541",
        lw=1.8,
        ms=4.4,
        label="Torque command",
    )
    torque_axis.plot(
        elapsed_time_s,
        torque_feedback_nm,
        "o-",
        color="#4C956C",
        lw=2.0,
        ms=4.4,
        label="Torque feedback",
    )
    torque_axis.axhline(0.0, color="#555555", lw=0.8)
    torque_axis.set_ylabel("Motor torque\n[N·m]")
    torque_axis.set_xlabel("Elapsed time within straight [s]")
    torque_axis.legend(frameon=False, ncols=2, loc="best")
    for axis in (acceleration_axis, torque_axis):
        axis.grid(alpha=0.22)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=ROOT / "analysis/acceleration/output/straight_samples.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = (
        args.output_dir or args.samples_csv.parent / "acceleration_torque_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_samples(args.samples_csv)
    required = (
        "straight_number",
        "time_s",
        "distance_from_straight_start_m",
        "imu_longitudinal_accel_mps2",
        "sim_accel_mps2_at_measured_time",
        "torque_command_nm",
        "torque_feedback_nm",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Samples CSV is missing: {', '.join(missing)}")

    straight_numbers = np.unique(data["straight_number"].astype(int))
    fig, overview_axes = plt.subplots(
        2 * len(straight_numbers),
        1,
        figsize=(12, 5.2 * len(straight_numbers)),
        layout="constrained",
    )
    for plot_index, straight_number in enumerate(straight_numbers):
        mask = data["straight_number"].astype(int) == straight_number
        time_s = data["time_s"][mask]
        order = np.argsort(time_s)
        values = {
            name: data[name][mask][order]
            for name in required
            if name != "straight_number"
        }
        elapsed_time_s = values["time_s"] - values["time_s"][0]
        title = (
            f"Straight {straight_number}: acceleration and motor torque · "
            f"{time_s.size} samples · "
            f"{np.min(values['distance_from_straight_start_m']):.1f}–"
            f"{np.max(values['distance_from_straight_start_m']):.1f} m sampled"
        )

        individual, axes = plt.subplots(
            2, 1, figsize=(11, 7.5), sharex=True, layout="constrained"
        )
        plot_pair(
            axes[0],
            axes[1],
            elapsed_time_s,
            values["imu_longitudinal_accel_mps2"],
            values["sim_accel_mps2_at_measured_time"],
            values["torque_command_nm"],
            values["torque_feedback_nm"],
            title=title,
        )
        individual.savefig(
            output_dir / f"straight_{straight_number:02d}_acceleration_torque.png",
            dpi=220,
        )
        plt.close(individual)

        plot_pair(
            overview_axes[2 * plot_index],
            overview_axes[2 * plot_index + 1],
            elapsed_time_s,
            values["imu_longitudinal_accel_mps2"],
            values["sim_accel_mps2_at_measured_time"],
            values["torque_command_nm"],
            values["torque_feedback_nm"],
            title=title,
        )

    fig.suptitle(
        "All detected straights: acceleration and torque",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output_dir / "all_straights_acceleration_torque.png", dpi=200)
    plt.close(fig)
    print(
        f"Wrote {len(straight_numbers)} straight plots and one overview to {output_dir}"
    )


if __name__ == "__main__":
    main()
