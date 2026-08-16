"""Plot corrected IMU Y against simulation track curvature over a distance window."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    data: dict[str, np.ndarray] = {}
    for name in rows[0]:
        try:
            data[name] = np.asarray([float(row[name]) for row in rows], dtype=float)
        except TypeError, ValueError:
            continue
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--track-csv", type=Path, required=True)
    parser.add_argument("--course-map", type=Path, required=True)
    parser.add_argument("--map-alignment", type=Path, required=True)
    parser.add_argument("--start-m", type=float, default=400.0)
    parser.add_argument("--end-m", type=float, default=600.0)
    parser.add_argument(
        "--track-station-offset-m",
        type=float,
        default=2.5,
        help="Track station offset already applied to the physics channels",
    )
    args = parser.parse_args()
    if args.end_m <= args.start_m:
        parser.error("--end-m must be greater than --start-m")

    data = read_csv(args.input_csv)
    track = read_csv(args.track_csv)
    distance_m = data["measured_distance_m"]
    window = (distance_m >= args.start_m) & (distance_m <= args.end_m)
    if np.count_nonzero(window) < 2:
        raise ValueError("Requested distance window contains fewer than two samples")

    distance_m = distance_m[window]
    imu_y_mps2 = data["aligned_imu_lateral_accel_mps2"][window]
    curvature_per_m = data["analysis_map_curvature_per_m"][window]
    recorded_speed_mps = data["analysis_shifted_gnss_speed_mps"][window]
    curvature_acceleration_mps2 = recorded_speed_mps**2 * curvature_per_m

    figure = plt.figure(figsize=(14, 11), layout="constrained")
    grid = figure.add_gridspec(3, 1, height_ratios=(0.45, 1.0, 1.0))
    map_axis = figure.add_subplot(grid[0])
    axes = (figure.add_subplot(grid[1]), figure.add_subplot(grid[2]))

    map_image = plt.imread(args.course_map)
    alignment = json.loads(args.map_alignment.read_text(encoding="utf-8"))
    transform = alignment["effective_affine_transform"]
    matrix = np.asarray(transform["matrix_px_per_m"], dtype=float)
    offset = np.asarray(transform["offset_px"], dtype=float)
    track_xy_m = np.column_stack((track["x_m"], track["y_m"]))
    track_pixels = track_xy_m @ matrix.T + offset
    highlighted = (
        track["distance_m"] >= args.start_m + args.track_station_offset_m
    ) & (track["distance_m"] <= args.end_m + args.track_station_offset_m)
    map_axis.imshow(map_image, origin="upper")
    map_axis.plot(
        track_pixels[:, 0],
        track_pixels[:, 1],
        color="black",
        lw=4.2,
        alpha=0.8,
        label="Complete fused GNSS/IMU track",
    )
    map_axis.plot(
        track_pixels[:, 0],
        track_pixels[:, 1],
        color="#4C78A8",
        lw=2.3,
        label="Track centerline",
    )
    map_axis.plot(
        track_pixels[highlighted, 0],
        track_pixels[highlighted, 1],
        color="#E45756",
        lw=5.0,
        label=f"Highlighted {args.start_m:g}–{args.end_m:g} m",
    )
    highlight_indices = np.flatnonzero(highlighted)
    if highlight_indices.size:
        for index, label in (
            (highlight_indices[0], f"{args.start_m:g} m"),
            (highlight_indices[-1], f"{args.end_m:g} m"),
        ):
            map_axis.scatter(
                track_pixels[index, 0],
                track_pixels[index, 1],
                s=55,
                color="#F2CF5B",
                edgecolor="black",
                zorder=5,
            )
            map_axis.annotate(
                label,
                track_pixels[index],
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
    map_axis.set_xlim(0, map_image.shape[1])
    map_axis.set_ylim(map_image.shape[0], 0)
    map_axis.set_aspect("equal")
    map_axis.axis("off")
    map_axis.legend(loc="lower right", framealpha=0.9, ncol=3)
    map_axis.set_title("Official endurance map with fused simulation track")
    curvature_axis = axes[0].twinx()
    imu_line = axes[0].plot(
        distance_m,
        imu_y_mps2,
        color="#4C78A8",
        lw=1.6,
        label="Corrected IMU Y",
    )[0]
    curvature_line = curvature_axis.plot(
        distance_m,
        curvature_per_m,
        color="#E45756",
        lw=1.6,
        label="Fused-track geometric curvature",
    )[0]
    axes[0].axhline(0.0, color="0.35", lw=0.7)
    axes[0].set_ylabel("Corrected IMU Y [m/s²]", color=imu_line.get_color())
    curvature_axis.set_ylabel(
        "Signed curvature [1/m]", color=curvature_line.get_color()
    )
    axes[0].tick_params(axis="y", colors=imu_line.get_color())
    curvature_axis.tick_params(axis="y", colors=curvature_line.get_color())
    axes[0].legend(
        [imu_line, curvature_line],
        [imu_line.get_label(), curvature_line.get_label()],
        loc="upper right",
        frameon=False,
        ncol=2,
    )

    axes[1].plot(
        distance_m,
        imu_y_mps2,
        color="#4C78A8",
        lw=1.6,
        label="Corrected IMU Y",
    )
    axes[1].plot(
        distance_m,
        curvature_acceleration_mps2,
        color="#E45756",
        lw=1.6,
        label="Track v²κ using recorded speed",
    )
    axes[1].axhline(0.0, color="0.35", lw=0.7)
    axes[1].set(
        xlabel="Lap distance [m]",
        ylabel="Lateral acceleration [m/s²]",
    )
    axes[1].legend(loc="upper right", frameon=False, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(args.start_m, args.end_m)
    curvature_axis.spines["top"].set_visible(False)
    figure.suptitle(
        f"Endurance track and corrected IMU Y vs curvature: "
        f"{args.start_m:g}–{args.end_m:g} m"
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
