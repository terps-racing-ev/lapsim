r"""Post-filter the shared selected-signal CSV to its first endurance lap.

This script never opens an MF4 and does not need ``asammdf``. Run the shared
converter first, then run this script from the ``python_lapsim`` root::

    .\.venv\Scripts\python.exe analysis\first_endurance_lap\generate_first_lap.py
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from math import cos, pi
from pathlib import Path
import csv
import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "analysis" / "mf4_to_csv" / "endurance_selected.csv"
)
DEFAULT_OUTPUT = ANALYSIS_DIR / "first_lap.csv"
DEFAULT_PLOT_OUTPUT = ANALYSIS_DIR / "first_lap_gps_comparison.png"
DEFAULT_OVERLAY_OUTPUT = ANALYSIS_DIR / "first_lap_map_overlay.png"
DEFAULT_MAP_IMAGE = ANALYSIS_DIR / "official_course_map.png"
METERS_PER_DEGREE_LATITUDE = 111_320.0


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_OUTPUT)
    parser.add_argument("--overlay-output", type=Path, default=DEFAULT_OVERLAY_OUTPUT)
    parser.add_argument("--map-image", type=Path, default=DEFAULT_MAP_IMAGE)
    parser.add_argument("--filter-window", type=int, default=7)
    parser.add_argument("--filter-order", type=int, default=2)
    parser.add_argument("--moving-speed-mps", type=float, default=2.0)
    parser.add_argument("--minimum-lap-s", type=float, default=60.0)
    parser.add_argument("--maximum-lap-s", type=float, default=90.0)
    parser.add_argument("--gate-exit-radius-m", type=float, default=50.0)
    return parser.parse_args()


def read_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}\nRun analysis/mf4_to_csv/convert_mf4_to_csv.py first."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("The selected-signal CSV has fewer than two rows")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def write_csv(columns: dict[str, np.ndarray], output: Path) -> Path:
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("All first-lap CSV columns must have equal length")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        names = list(columns)
        writer.writerow(names)
        writer.writerows(zip(*(columns[name] for name in names), strict=True))
    return output.resolve()


def local_xy_m(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_m = (
        (longitude_deg - origin_longitude_deg)
        * METERS_PER_DEGREE_LATITUDE
        * cos(origin_latitude_deg * pi / 180.0)
    )
    y_m = (
        latitude_deg - origin_latitude_deg
    ) * METERS_PER_DEGREE_LATITUDE
    return x_m, y_m


def filtered_gps_path(
    x_m: np.ndarray,
    y_m: np.ndarray,
    window: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth the fixed-rate GNSS positions without changing their timestamps."""
    if window < 3 or window % 2 == 0:
        raise ValueError("--filter-window must be an odd integer of at least 3")
    if order < 1 or order >= window:
        raise ValueError("--filter-order must be positive and less than the window")
    if len(x_m) < window:
        raise ValueError(
            f"The first lap has {len(x_m)} rows, fewer than filter window {window}"
        )
    if not np.all(np.isfinite(x_m)) or not np.all(np.isfinite(y_m)):
        raise ValueError("First-lap GNSS positions must be finite before filtering")
    return (
        savgol_filter(x_m, window_length=window, polyorder=order, mode="interp"),
        savgol_filter(y_m, window_length=window, polyorder=order, mode="interp"),
    )


def fit_gps_to_course_map(
    filtered_x_m: np.ndarray,
    filtered_y_m: np.ndarray,
    map_image: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Find a best-fit affine display transform from GNSS metres to map pixels."""
    rgb = np.asarray(map_image[..., :3], dtype=float)
    if np.nanmax(rgb) <= 1.0:
        rgb *= 255.0
    red = (
        (rgb[..., 0] > 180.0)
        & (rgb[..., 1] < 120.0)
        & (rgb[..., 2] < 120.0)
        & ((rgb[..., 0] - rgb[..., 1]) > 70.0)
    )
    components, component_count = label(red)
    if component_count == 0:
        raise ValueError("No red course line was detected in the official map")
    component_sizes = np.bincount(components.ravel())
    course_mask = components == (int(np.argmax(component_sizes[1:])) + 1)
    course_y, course_x = np.nonzero(course_mask)
    course_width_px = float(np.ptp(course_x))
    course_height_px = float(np.ptp(course_y))
    if course_width_px <= 0.0 or course_height_px <= 0.0:
        raise ValueError("Detected course line has invalid dimensions")

    left_edge_px = float(np.min(course_x))
    near_start = course_x <= left_edge_px + 0.03 * course_width_px
    start_x_px = float(np.median(course_x[near_start]))
    start_y_px = float(np.median(course_y[near_start]))
    distance_to_course_px = distance_transform_edt(~course_mask)

    # Use map-oriented GNSS coordinates: horizontal is south, vertical is east.
    horizontal_m = -filtered_y_m
    vertical_m = filtered_x_m
    horizontal_span_m = float(np.ptp(horizontal_m))
    vertical_span_m = float(np.ptp(vertical_m))
    horizontal_scale = course_width_px / horizontal_span_m
    vertical_scale = course_height_px / vertical_span_m

    def objective(parameters: np.ndarray) -> float:
        x0, scale_x, y0, shear_y, scale_y = parameters
        pixel_x = x0 + scale_x * horizontal_m
        pixel_y = y0 + shear_y * horizontal_m + scale_y * vertical_m
        distances = map_coordinates(
            distance_to_course_px,
            [pixel_y, pixel_x],
            order=1,
            mode="constant",
            cval=100.0,
        )
        endpoint_error = (
            (pixel_x[0] - start_x_px) ** 2
            + (pixel_y[0] - start_y_px) ** 2
            + (pixel_x[-1] - start_x_px) ** 2
            + (pixel_y[-1] - start_y_px) ** 2
        )
        return float(np.mean(np.minimum(distances, 30.0) ** 2) + 0.3 * endpoint_error)

    common_bounds = [
        (
            start_x_px - 0.03 * course_width_px,
            start_x_px + 0.03 * course_width_px,
        ),
        (0.75 * horizontal_scale, 1.25 * horizontal_scale),
        (
            start_y_px - 0.45 * course_height_px,
            start_y_px + 0.45 * course_height_px,
        ),
        (
            -3.5 * course_height_px / horizontal_span_m,
            3.5 * course_height_px / horizontal_span_m,
        ),
    ]
    candidates = []
    for sign in (-1.0, 1.0):
        scale_y_bounds = sorted(
            (sign * 0.25 * vertical_scale, sign * 8.0 * vertical_scale)
        )
        candidates.append(
            differential_evolution(
                objective,
                [*common_bounds, tuple(scale_y_bounds)],
                seed=2026,
                maxiter=250,
                popsize=12,
                polish=True,
                tol=1e-4,
            )
        )
    solution = min(candidates, key=lambda candidate: candidate.fun)
    x0, scale_x, y0, shear_y, scale_y = solution.x
    pixel_x = x0 + scale_x * horizontal_m
    pixel_y = y0 + shear_y * horizontal_m + scale_y * vertical_m
    point_distances = map_coordinates(
        distance_to_course_px,
        [pixel_y, pixel_x],
        order=1,
        mode="constant",
        cval=100.0,
    )
    return {
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "x0_px": float(x0),
        "horizontal_scale_px_per_m": float(scale_x),
        "y0_px": float(y0),
        "horizontal_shear_px_per_m": float(shear_y),
        "vertical_scale_px_per_m": float(scale_y),
        "rms_distance_to_red_line_px": float(np.sqrt(np.mean(point_distances**2))),
    }


def transform_gps_to_map_pixels(
    x_m: np.ndarray,
    y_m: np.ndarray,
    alignment: dict[str, float | np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    horizontal_m = -y_m
    vertical_m = x_m
    pixel_x = (
        float(alignment["x0_px"])
        + float(alignment["horizontal_scale_px_per_m"]) * horizontal_m
    )
    pixel_y = (
        float(alignment["y0_px"])
        + float(alignment["horizontal_shear_px_per_m"]) * horizontal_m
        + float(alignment["vertical_scale_px_per_m"]) * vertical_m
    )
    return pixel_x, pixel_y


def plot_course_overlay(
    map_image: np.ndarray,
    raw_x_m: np.ndarray,
    raw_y_m: np.ndarray,
    alignment: dict[str, float | np.ndarray],
    output_path: Path,
) -> Path:
    raw_pixel_x, raw_pixel_y = transform_gps_to_map_pixels(
        raw_x_m, raw_y_m, alignment
    )
    figure, axis = plt.subplots(figsize=(16, 3.2), layout="constrained")
    axis.imshow(map_image)
    axis.scatter(
        raw_pixel_x,
        raw_pixel_y,
        s=4,
        color="#374151",
        alpha=0.25,
        label="Raw GNSS",
    )
    axis.plot(
        np.asarray(alignment["pixel_x"]),
        np.asarray(alignment["pixel_y"]),
        color="#06b6d4",
        linewidth=1.8,
        label="Filtered GNSS (best-fit affine alignment)",
    )
    axis.set_title("First-lap GNSS overlaid on official course drawing")
    axis.axis("off")
    axis.legend(loc="lower center", ncols=2, fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path.resolve()


def plot_gps_comparison(
    raw_x_m: np.ndarray,
    raw_y_m: np.ndarray,
    filtered_x_m: np.ndarray,
    filtered_y_m: np.ndarray,
    map_image: np.ndarray,
    alignment: dict[str, float | np.ndarray],
    output_path: Path,
) -> Path:
    """Plot the supplied course map beside raw and filtered first-lap GNSS."""
    # Rotate only the displayed coordinates so the long axis has the same
    # left-to-right orientation as the supplied course drawing. The CSV keeps
    # the original east/north coordinates.
    raw_horizontal_m, raw_vertical_m = -raw_y_m, raw_x_m
    filtered_horizontal_m, filtered_vertical_m = -filtered_y_m, filtered_x_m

    figure = plt.figure(figsize=(16, 7.5), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 2.0))

    map_axis = figure.add_subplot(grid[0, :])
    map_axis.imshow(map_image)
    raw_pixel_x, raw_pixel_y = transform_gps_to_map_pixels(
        raw_x_m, raw_y_m, alignment
    )
    map_axis.scatter(
        raw_pixel_x,
        raw_pixel_y,
        s=3,
        color="#374151",
        alpha=0.2,
    )
    map_axis.plot(
        np.asarray(alignment["pixel_x"]),
        np.asarray(alignment["pixel_y"]),
        color="#06b6d4",
        linewidth=1.6,
    )
    map_axis.set_title("Best-fit GNSS overlay on official endurance course map")
    map_axis.axis("off")

    raw_axis = figure.add_subplot(grid[1, 0])
    filtered_axis = figure.add_subplot(grid[1, 1], sharex=raw_axis, sharey=raw_axis)

    raw_axis.plot(
        raw_horizontal_m, raw_vertical_m, color="#6b7280", linewidth=0.8
    )
    raw_axis.scatter(
        raw_horizontal_m,
        raw_vertical_m,
        s=7,
        color="#2563eb",
        alpha=0.75,
        label="Raw 5 Hz GNSS samples",
    )
    raw_axis.set_title("Raw first-lap GNSS path")

    filtered_axis.scatter(
        raw_horizontal_m,
        raw_vertical_m,
        s=6,
        color="#9ca3af",
        alpha=0.35,
        label="Raw samples",
    )
    filtered_axis.plot(
        filtered_horizontal_m,
        filtered_vertical_m,
        color="#dc2626",
        linewidth=2.0,
        label="Savitzky-Golay filtered path",
    )
    filtered_axis.set_title("Filtered first-lap GNSS path")

    for axis in (raw_axis, filtered_axis):
        axis.scatter(
            raw_horizontal_m[0],
            raw_vertical_m[0],
            marker="o",
            s=55,
            color="#16a34a",
            label="Start",
        )
        axis.scatter(
            raw_horizontal_m[-1],
            raw_vertical_m[-1],
            marker="x",
            s=65,
            color="#111827",
            label="Finish",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("South of start (m)")
        axis.set_ylabel("East of start (m)")
        axis.grid(True, linewidth=0.5, alpha=0.3)
        axis.legend(loc="lower right", fontsize=9)

    figure.suptitle(
        "First endurance lap: official map and measured GNSS "
        "(GNSS display rotated to match map orientation)"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path.resolve()


def first_sustained_moving_run(
    time_s: np.ndarray,
    speed_mps: np.ndarray,
    threshold_mps: float,
) -> tuple[int, int]:
    moving_indices = np.flatnonzero(
        np.isfinite(speed_mps) & (speed_mps >= threshold_mps)
    )
    if not len(moving_indices):
        raise ValueError("No GNSS samples meet the moving-speed threshold")
    runs: list[tuple[int, int]] = []
    start = previous = int(moving_indices[0])
    for raw_index in moving_indices[1:]:
        index = int(raw_index)
        if time_s[index] - time_s[previous] > 4.0:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    for start, finish in runs:
        if time_s[finish] - time_s[start] >= 30.0:
            return start, finish
    raise ValueError("No sustained moving GNSS interval was found")


def detect_first_lap(
    data: dict[str, np.ndarray], args: Namespace
) -> tuple[int, int, dict[str, float]]:
    time_s = data["mf4_time_s"]
    latitude_deg = data["gps_latitude_deg"]
    longitude_deg = data["gps_longitude_deg"]
    speed_mps = data["gps_speed_mps"]
    finite = (
        np.isfinite(time_s)
        & np.isfinite(latitude_deg)
        & np.isfinite(longitude_deg)
        & np.isfinite(speed_mps)
    )
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices) < 2:
        raise ValueError("Fewer than two finite GNSS rows are available")
    time_s = time_s[finite]
    latitude_deg = latitude_deg[finite]
    longitude_deg = longitude_deg[finite]
    speed_mps = speed_mps[finite]
    moving_start, moving_finish = first_sustained_moving_run(
        time_s, speed_mps, args.moving_speed_mps
    )
    gate_latitude_deg = float(latitude_deg[moving_start])
    gate_longitude_deg = float(longitude_deg[moving_start])
    x_m, y_m = local_xy_m(
        latitude_deg,
        longitude_deg,
        gate_latitude_deg,
        gate_longitude_deg,
    )
    distance_to_gate_m = np.hypot(x_m, y_m)
    left_gate = np.flatnonzero(
        (np.arange(len(time_s)) > moving_start)
        & (distance_to_gate_m >= args.gate_exit_radius_m)
    )
    if not len(left_gate):
        raise ValueError("The moving run never left the start-gate area")
    candidate_start_s = max(
        time_s[moving_start] + args.minimum_lap_s,
        time_s[int(left_gate[0])],
    )
    candidate_finish_s = min(
        time_s[moving_start] + args.maximum_lap_s,
        time_s[moving_finish] + 8.0,
    )
    candidates = np.flatnonzero(
        (time_s >= candidate_start_s) & (time_s <= candidate_finish_s)
    )
    if not len(candidates):
        raise ValueError("No GNSS rows exist in the expected lap window")
    finish = int(candidates[np.argmin(distance_to_gate_m[candidates])])
    return int(finite_indices[moving_start]), int(finite_indices[finish]), {
        "gate_latitude_deg": gate_latitude_deg,
        "gate_longitude_deg": gate_longitude_deg,
        "finish_gate_error_m": float(distance_to_gate_m[finish]),
    }


def main() -> None:
    args = parse_args()
    data = read_csv(args.input_csv)
    start, finish, gate = detect_first_lap(data, args)
    selection = slice(start, finish + 1)
    selected = {name: values[selection].copy() for name, values in data.items()}
    selected["time_s"] -= selected["time_s"][0]
    selected["gps_distance_trip_m"] -= selected["gps_distance_trip_m"][0]
    gps_x_m, gps_y_m = local_xy_m(
        selected["gps_latitude_deg"],
        selected["gps_longitude_deg"],
        gate["gate_latitude_deg"],
        gate["gate_longitude_deg"],
    )
    gps_x_filtered_m, gps_y_filtered_m = filtered_gps_path(
        gps_x_m,
        gps_y_m,
        args.filter_window,
        args.filter_order,
    )
    columns = {
        "time_s": selected.pop("time_s"),
        "mf4_time_s": selected.pop("mf4_time_s"),
        "gps_latitude_deg": selected.pop("gps_latitude_deg"),
        "gps_longitude_deg": selected.pop("gps_longitude_deg"),
        "gps_x_m": gps_x_m,
        "gps_y_m": gps_y_m,
        "gps_x_filtered_m": gps_x_filtered_m,
        "gps_y_filtered_m": gps_y_filtered_m,
        "gps_speed_mps": selected.pop("gps_speed_mps"),
        "gps_distance_trip_m": selected.pop("gps_distance_trip_m"),
        **selected,
    }
    output = write_csv(columns, args.output)
    if not args.map_image.is_file():
        raise FileNotFoundError(f"Official course map not found: {args.map_image}")
    map_image = mpimg.imread(args.map_image)
    alignment = fit_gps_to_course_map(
        gps_x_filtered_m,
        gps_y_filtered_m,
        map_image,
    )
    overlay_output = plot_course_overlay(
        map_image,
        gps_x_m,
        gps_y_m,
        alignment,
        args.overlay_output,
    )
    plot_output = plot_gps_comparison(
        gps_x_m,
        gps_y_m,
        gps_x_filtered_m,
        gps_y_filtered_m,
        map_image,
        alignment,
        args.plot_output,
    )
    input_metadata_path = args.input_csv.with_suffix(".json")
    input_metadata = (
        json.loads(input_metadata_path.read_text(encoding="utf-8"))
        if input_metadata_path.is_file()
        else {}
    )
    clock_s = columns["mf4_time_s"]
    intervals_s = np.diff(clock_s)
    metadata = {
        "input_csv": str(args.input_csv.resolve()),
        "input_metadata": (
            str(input_metadata_path.resolve()) if input_metadata else None
        ),
        "output_csv": str(output),
        "output_plot": str(plot_output),
        "output_map_overlay": str(overlay_output),
        "official_map_image": str(args.map_image.resolve()),
        "gps_filter": {
            "type": "Savitzky-Golay",
            "window_samples": args.filter_window,
            "polynomial_order": args.filter_order,
        },
        "map_alignment": {
            "type": "best-fit affine display transform",
            "note": "The course drawing is schematic and not georeferenced.",
            **{
                name: value
                for name, value in alignment.items()
                if isinstance(value, float)
            },
        },
        "samples": len(clock_s),
        "median_output_rate_hz": float(1.0 / np.median(intervals_s)),
        "lap_start_mf4_time_s": float(clock_s[0]),
        "lap_finish_mf4_time_s": float(clock_s[-1]),
        "lap_duration_s": float(clock_s[-1] - clock_s[0]),
        "moving_speed_mps": args.moving_speed_mps,
        "minimum_lap_s": args.minimum_lap_s,
        "maximum_lap_s": args.maximum_lap_s,
        "gate_exit_radius_m": args.gate_exit_radius_m,
        **gate,
        "channels": input_metadata.get("channels", {}),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"first lap: {metadata['lap_duration_s']:.3f} s, "
        f"{metadata['samples']} samples at "
        f"{metadata['median_output_rate_hz']:.3f} Hz"
    )
    print(f"csv: {output}")
    print(f"plot: {plot_output}")
    print(f"map overlay: {overlay_output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
