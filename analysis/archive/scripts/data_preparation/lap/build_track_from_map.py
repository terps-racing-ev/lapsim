r"""Trace the official course map into a reusable :class:`SpatialTrack`.

The official red centerline supplies the missing geometric detail. The saved
manual GNSS-to-map alignment supplies the pixel-to-metre transform and driving
direction. Dashed red cone/slalom annotations are pruned from the solid loop.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
import json
from pathlib import Path
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, label


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lapsim.spatial_track import SpatialTrack  # noqa: E402


DEFAULT_MAP = ANALYSIS_DIR / "official_course_map.png"
DEFAULT_ALIGNMENT = ANALYSIS_DIR / "manual_map_alignment.json"
DEFAULT_LAP_CSV = ANALYSIS_DIR / "first_lap.csv"
DEFAULT_OUTPUT = ANALYSIS_DIR / "map_derived_track.csv"
DEFAULT_PREVIEW = ANALYSIS_DIR / "map_derived_track_preview.png"
DEFAULT_SIDE_BY_SIDE = ANALYSIS_DIR / "map_derived_track_side_by_side.png"
DEFAULT_OVERLAY = ANALYSIS_DIR / "map_derived_track_overlay.png"


Pixel = tuple[int, int]  # (row/y, column/x)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--map-image", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--lap-csv", type=Path, default=DEFAULT_LAP_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--side-by-side", type=Path, default=DEFAULT_SIDE_BY_SIDE)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--cell-length-m", type=float, default=1.0)
    parser.add_argument(
        "--smoothing-pixels",
        type=float,
        default=4.0,
        help="Local periodic Gaussian width along the traced map centerline.",
    )
    parser.add_argument(
        "--reference-length-m",
        type=float,
        help="Optionally scale the traced geometry to this length; the default preserves the map size.",
    )
    return parser.parse_args()


def red_course_component(map_image: np.ndarray) -> np.ndarray:
    """Return the largest connected saturated-red feature in the map."""
    rgb = np.asarray(map_image[..., :3], dtype=float)
    if np.nanmax(rgb) <= 1.0:
        rgb *= 255.0
    red = (
        (rgb[..., 0] > 180.0)
        & (rgb[..., 1] < 120.0)
        & (rgb[..., 2] < 120.0)
        & ((rgb[..., 0] - rgb[..., 1]) > 70.0)
    )
    components, count = label(red)
    if count == 0:
        raise ValueError("No saturated-red course line found in the map")
    sizes = np.bincount(components.ravel())
    return components == (int(np.argmax(sizes[1:])) + 1)


def thin_binary(mask: np.ndarray) -> np.ndarray:
    """Reduce a binary line drawing to a one-pixel Zhang-Suen skeleton."""
    skeleton = np.asarray(mask, dtype=np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(skeleton, 1)
            neighbors = (
                padded[:-2, 1:-1],
                padded[:-2, 2:],
                padded[1:-1, 2:],
                padded[2:, 2:],
                padded[2:, 1:-1],
                padded[2:, :-2],
                padded[1:-1, :-2],
                padded[:-2, :-2],
            )
            count = sum(neighbors)
            transitions = sum(
                (neighbors[index] == 0) & (neighbors[(index + 1) % 8] == 1)
                for index in range(8)
            )
            remove = (
                (skeleton == 1)
                & (count >= 2)
                & (count <= 6)
                & (transitions == 1)
            )
            if step == 0:
                remove &= (
                    (neighbors[0] * neighbors[2] * neighbors[4] == 0)
                    & (neighbors[2] * neighbors[4] * neighbors[6] == 0)
                )
            else:
                remove &= (
                    (neighbors[0] * neighbors[2] * neighbors[6] == 0)
                    & (neighbors[0] * neighbors[4] * neighbors[6] == 0)
                )
            if np.any(remove):
                skeleton[remove] = 0
                changed = True
    return skeleton.astype(bool)


def graph_neighbors(pixel: Pixel, pixels: set[Pixel]) -> list[Pixel]:
    """Return 8-connected neighbors without redundant diagonal corner edges."""
    row, column = pixel
    result: list[Pixel] = []
    for row_delta, column_delta in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        candidate = row + row_delta, column + column_delta
        if candidate in pixels:
            result.append(candidate)
    for row_delta, column_delta in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        candidate = row + row_delta, column + column_delta
        if (
            candidate in pixels
            and (row + row_delta, column) not in pixels
            and (row, column + column_delta) not in pixels
        ):
            result.append(candidate)
    return result


def prune_annotation_spurs(
    skeleton: np.ndarray,
    start_pixel_xy: np.ndarray,
) -> tuple[set[Pixel], tuple[Pixel, Pixel]]:
    """Remove dashed annotations connected to the solid course loop."""
    pixels = set(map(tuple, np.argwhere(skeleton)))
    endpoints = [pixel for pixel in pixels if len(graph_neighbors(pixel, pixels)) == 1]
    if len(endpoints) < 2:
        raise ValueError("Course skeleton does not expose a traceable start gap")
    endpoints.sort(
        key=lambda pixel: float(
            np.hypot(pixel[1] - start_pixel_xy[0], pixel[0] - start_pixel_xy[1])
        )
    )
    preserved = set(endpoints[:2])

    while True:
        endpoints = [
            pixel for pixel in pixels if len(graph_neighbors(pixel, pixels)) == 1
        ]
        unwanted = [pixel for pixel in endpoints if pixel not in preserved]
        if not unwanted:
            break
        for endpoint in unwanted:
            if endpoint not in pixels:
                continue
            path = [endpoint]
            previous: Pixel | None = None
            current = endpoint
            while True:
                current_neighbors = graph_neighbors(current, pixels)
                if current != endpoint and len(current_neighbors) != 2:
                    break
                onward = [candidate for candidate in current_neighbors if candidate != previous]
                if not onward:
                    break
                previous, current = current, onward[0]
                path.append(current)
                if len(graph_neighbors(current, pixels)) != 2:
                    break
            for pixel in path[:-1]:
                pixels.discard(pixel)

    final_endpoints = tuple(
        pixel for pixel in pixels if len(graph_neighbors(pixel, pixels)) == 1
    )
    branch_count = sum(
        len(graph_neighbors(pixel, pixels)) > 2 for pixel in pixels
    )
    if len(final_endpoints) != 2 or branch_count:
        raise ValueError(
            "Course skeleton did not reduce to one open loop: "
            f"{len(final_endpoints)} endpoints, {branch_count} branches"
        )
    return pixels, (final_endpoints[0], final_endpoints[1])


def ordered_course_pixels(
    pixels: set[Pixel],
    endpoints: tuple[Pixel, Pixel],
) -> np.ndarray:
    """Trace the pruned one-pixel course line from one start-gap edge to the other."""
    ordered = [endpoints[0]]
    previous: Pixel | None = None
    current = endpoints[0]
    while current != endpoints[1]:
        onward = [
            candidate
            for candidate in graph_neighbors(current, pixels)
            if candidate != previous
        ]
        if len(onward) != 1:
            raise ValueError("Course skeleton is not a single unambiguous path")
        previous, current = current, onward[0]
        ordered.append(current)
        if len(ordered) > len(pixels):
            raise ValueError("Course skeleton traversal did not terminate")
    if len(ordered) != len(pixels):
        raise ValueError("Course traversal did not consume every retained pixel")
    return np.asarray([(column, row) for row, column in ordered], dtype=float)


def load_effective_affine(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    transform = metadata["effective_affine_transform"]
    matrix = np.asarray(transform["matrix_px_per_m"], dtype=float)
    offset = np.asarray(transform["offset_px"], dtype=float)
    if matrix.shape != (2, 2) or offset.shape != (2,):
        raise ValueError("Manual alignment does not contain a 2-D affine transform")
    if abs(float(np.linalg.det(matrix))) <= 1e-12:
        raise ValueError("Manual alignment affine transform is singular")
    return matrix, offset, metadata


def load_lap_xy(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["gps_x_filtered_m"]) for row in rows]),
        np.asarray([float(row["gps_y_filtered_m"]) for row in rows]),
        float(rows[-1]["gps_distance_trip_m"]),
    )


def cumulative_distance(points: np.ndarray, *, closed: bool = False) -> np.ndarray:
    if closed:
        points = np.vstack((points, points[0]))
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


def orient_course_pixels(
    course_pixels: np.ndarray,
    aligned_gnss_pixels: np.ndarray,
) -> tuple[np.ndarray, str, float]:
    """Move the trace start to GNSS start and select the recorded direction."""
    start_index = int(
        np.argmin(np.linalg.norm(course_pixels - aligned_gnss_pixels[0], axis=1))
    )
    rolled = np.roll(course_pixels, -start_index, axis=0)
    gnss_station = cumulative_distance(aligned_gnss_pixels)
    gnss_fraction = gnss_station / gnss_station[-1]

    choices: list[tuple[float, str, np.ndarray]] = []
    for name, candidate in (("forward", rolled), ("reverse", rolled[:0:-1])):
        if name == "reverse":
            candidate = np.vstack((rolled[0], candidate))
        candidate_closed = np.vstack((candidate, candidate[0]))
        station = cumulative_distance(candidate_closed)
        fraction = station / station[-1]
        matched = np.column_stack(
            (
                np.interp(gnss_fraction, fraction, candidate_closed[:, 0]),
                np.interp(gnss_fraction, fraction, candidate_closed[:, 1]),
            )
        )
        error = float(
            np.sqrt(np.mean(np.sum((matched - aligned_gnss_pixels) ** 2, axis=1)))
        )
        choices.append((error, name, candidate_closed))
    error, name, selected = min(choices, key=lambda choice: choice[0])
    return selected, name, error


def periodic_track_from_centerline(
    centerline_m: np.ndarray,
    *,
    target_cell_length_m: float,
    smoothing_pixels: float,
    reference_length_m: float | None,
) -> tuple[SpatialTrack, float, float]:
    """Locally smooth, scale, arc-length-resample, and differentiate a loop."""
    if target_cell_length_m <= 0.0:
        raise ValueError("cell length must be positive")
    if smoothing_pixels < 0.0:
        raise ValueError("smoothing must be nonnegative")
    source = centerline_m[:-1]
    smoothed = np.column_stack(
        (
            gaussian_filter1d(
                source[:, 0], sigma=smoothing_pixels, mode="wrap"
            ),
            gaussian_filter1d(
                source[:, 1], sigma=smoothing_pixels, mode="wrap"
            ),
        )
    )
    smoothed -= smoothed[0]
    smoothed_closed = np.vstack((smoothed, smoothed[0]))
    source_station_m = cumulative_distance(smoothed_closed)
    unscaled_length_m = float(source_station_m[-1])
    if reference_length_m is not None:
        if reference_length_m <= 0.0:
            raise ValueError("reference length must be positive")
        scale = reference_length_m / unscaled_length_m
        smoothed_closed *= scale
        source_station_m *= scale
    else:
        scale = 1.0
    length_m = float(source_station_m[-1])
    cell_count = max(3, int(round(length_m / target_cell_length_m)))
    distance_m = np.linspace(0.0, length_m, cell_count + 1)
    x_spline = CubicSpline(
        source_station_m,
        smoothed_closed[:, 0],
        bc_type="periodic",
    )
    y_spline = CubicSpline(
        source_station_m,
        smoothed_closed[:, 1],
        bc_type="periodic",
    )
    x_m = x_spline(distance_m)
    y_m = y_spline(distance_m)
    x_m -= x_m[0]
    y_m -= y_m[0]
    x_m[-1] = 0.0
    y_m[-1] = 0.0

    centers_m = 0.5 * (distance_m[:-1] + distance_m[1:])
    dx = x_spline(centers_m, 1)
    dy = y_spline(centers_m, 1)
    ddx = x_spline(centers_m, 2)
    ddy = y_spline(centers_m, 2)
    curvature = (dx * ddy - dy * ddx) / np.maximum(
        (dx * dx + dy * dy) ** 1.5,
        1e-12,
    )
    return (
        SpatialTrack(
            distance_m=tuple(map(float, distance_m)),
            x_m=tuple(map(float, x_m)),
            y_m=tuple(map(float, y_m)),
            curvature_per_m=tuple(map(float, curvature)),
            closed=True,
        ),
        unscaled_length_m,
        scale,
    )


def plot_preview(
    map_image: np.ndarray,
    course_pixels: np.ndarray,
    lap_x_m: np.ndarray,
    lap_y_m: np.ndarray,
    track: SpatialTrack,
    output: Path,
) -> Path:
    figure = plt.figure(figsize=(16, 8), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 2.2))
    map_axis = figure.add_subplot(grid[0, :])
    map_axis.imshow(map_image)
    map_axis.plot(
        course_pixels[:, 0],
        course_pixels[:, 1],
        color="#06b6d4",
        linewidth=1.3,
        label="Traced solid centerline",
    )
    map_axis.axis("off")
    map_axis.set_title("Centerline extracted from official map")
    map_axis.legend(loc="lower center", fontsize=9)

    geometry_axis = figure.add_subplot(grid[1, 0])
    geometry_axis.plot(lap_x_m, lap_y_m, color="#9ca3af", label="Filtered GNSS")
    geometry_axis.plot(track.x_m, track.y_m, color="#dc2626", label="Map-derived track")
    geometry_axis.set_aspect("equal", adjustable="box")
    geometry_axis.set_xlabel("East (m)")
    geometry_axis.set_ylabel("North (m)")
    geometry_axis.set_title("Solver geometry")
    geometry_axis.grid(True, alpha=0.3)
    geometry_axis.legend()

    curvature_axis = figure.add_subplot(grid[1, 1])
    curvature_axis.plot(
        track.cell_center_distance_m,
        track.curvature_per_m,
        color="#7c3aed",
    )
    curvature_axis.axhline(0.0, color="#6b7280", linewidth=0.7)
    curvature_axis.set_xlabel("Lap distance (m)")
    curvature_axis.set_ylabel("Curvature (1/m)")
    curvature_axis.set_title("Map-derived curvature")
    curvature_axis.grid(True, alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output.resolve()


def plot_side_by_side(
    map_image: np.ndarray,
    track: SpatialTrack,
    output: Path,
) -> Path:
    """Put the source course drawing directly beside the solver centerline."""
    figure, (map_axis, track_axis) = plt.subplots(
        1,
        2,
        figsize=(18, 3.5),
        gridspec_kw={"width_ratios": (2.0, 1.0)},
        layout="constrained",
    )
    map_axis.imshow(map_image)
    map_axis.set_title("Official course map")
    map_axis.axis("off")

    # Rotate the solver coordinates only for display so start-to-turnaround is
    # left-to-right like the course drawing. Stored east/north data is unchanged.
    display_x_m = -np.asarray(track.y_m)
    display_y_m = np.asarray(track.x_m)
    track_axis.plot(display_x_m, display_y_m, color="#dc2626", linewidth=2.0)
    track_axis.scatter(
        display_x_m[0],
        display_y_m[0],
        color="#16a34a",
        s=55,
        zorder=3,
    )
    track_axis.annotate(
        "Start/finish",
        (display_x_m[0], display_y_m[0]),
        xytext=(10, -18),
        textcoords="offset points",
    )
    track_axis.set_aspect("equal", adjustable="box")
    track_axis.set_xlabel("South of start (m)")
    track_axis.set_ylabel("East of start (m)")
    track_axis.set_title(f"Map-derived solver track ({track.length_m:.1f} m)")
    track_axis.grid(True, alpha=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output.resolve()


def plot_overlay(
    map_image: np.ndarray,
    track: SpatialTrack,
    matrix_px_per_m: np.ndarray,
    offset_px: np.ndarray,
    centerline_origin_m: np.ndarray,
    output: Path,
) -> Path:
    """Overlay the final solver geometry directly on the official drawing."""

    track_m = np.column_stack((track.x_m, track.y_m)) + centerline_origin_m
    track_pixels = track_m @ matrix_px_per_m.T + offset_px
    figure, axis = plt.subplots(figsize=(18, 3.2), layout="constrained")
    axis.imshow(map_image)
    axis.plot(
        track_pixels[:, 0],
        track_pixels[:, 1],
        color="#0284c7",
        linewidth=2.2,
        alpha=0.52,
        label="Map-derived solver centerline (52% opacity)",
    )
    station_500_index = int(np.argmin(np.abs(np.asarray(track.distance_m) - 500.0)))
    axis.scatter(
        track_pixels[station_500_index, 0],
        track_pixels[station_500_index, 1],
        color="#7c3aed",
        edgecolor="white",
        linewidth=0.8,
        s=42,
        zorder=4,
        label="500 m station",
    )
    axis.set_title("Official endurance map with map-derived solver track overlay")
    axis.axis("off")
    axis.legend(loc="lower center", ncols=2, fontsize=9, framealpha=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output.resolve()


def main() -> None:
    args = parse_args()
    map_image = mpimg.imread(args.map_image)
    matrix, offset, alignment_metadata = load_effective_affine(args.alignment)
    lap_x_m, lap_y_m, gnss_trip_length_m = load_lap_xy(args.lap_csv)
    aligned_gnss_pixels = np.column_stack((lap_x_m, lap_y_m)) @ matrix.T + offset

    course_component = red_course_component(map_image)
    skeleton = thin_binary(course_component)
    retained, endpoints = prune_annotation_spurs(skeleton, aligned_gnss_pixels[0])
    traced_pixels = ordered_course_pixels(retained, endpoints)
    oriented_pixels, direction, correspondence_error_px = orient_course_pixels(
        traced_pixels,
        aligned_gnss_pixels,
    )

    inverse_matrix = np.linalg.inv(matrix)
    centerline_m = (oriented_pixels - offset) @ inverse_matrix.T
    centerline_origin_m = centerline_m[0].copy()
    centerline_m -= centerline_origin_m
    # Keep the official map geometry at its traced size by default. The GNSS
    # trip length remains metadata and must not silently resize the course.
    reference_length_m = args.reference_length_m
    track, unscaled_smoothed_length_m, geometry_scale = periodic_track_from_centerline(
        centerline_m,
        target_cell_length_m=args.cell_length_m,
        smoothing_pixels=args.smoothing_pixels,
        reference_length_m=reference_length_m,
    )
    output = track.to_csv(args.output)
    preview = plot_preview(
        map_image,
        oriented_pixels,
        lap_x_m,
        lap_y_m,
        track,
        args.preview,
    )
    side_by_side = plot_side_by_side(map_image, track, args.side_by_side)
    overlay = plot_overlay(
        map_image,
        track,
        matrix,
        offset,
        centerline_origin_m,
        args.overlay,
    )

    curvature = np.abs(np.asarray(track.curvature_per_m))
    signed_curvature = np.asarray(track.curvature_per_m)
    cell_length = np.asarray(track.cell_length_m)
    metadata = {
        "type": "map-derived SpatialTrack",
        "output_csv": str(output),
        "preview": str(preview),
        "side_by_side_comparison": str(side_by_side),
        "overlay_comparison": str(overlay),
        "source_map": str(args.map_image.resolve()),
        "source_manual_alignment": str(args.alignment.resolve()),
        "source_first_lap_csv": str(args.lap_csv.resolve()),
        "manual_alignment_created_utc": alignment_metadata.get("created_utc"),
        "solid_red_component_pixels": int(np.count_nonzero(course_component)),
        "retained_centerline_pixels": len(retained),
        "driving_direction_selection": direction,
        "normalized_progress_correspondence_rmse_px": correspondence_error_px,
        "smoothing": {
            "type": "local periodic Gaussian",
            "sigma_centerline_pixels": args.smoothing_pixels,
        },
        "unscaled_smoothed_length_m": unscaled_smoothed_length_m,
        "reference_length_m": reference_length_m,
        "uniform_geometry_scale": geometry_scale,
        "target_cell_length_m": args.cell_length_m,
        "track_length_m": track.length_m,
        "cell_count": track.cell_count,
        "maximum_absolute_curvature_per_m": float(np.max(curvature)),
        "p95_absolute_curvature_per_m": float(np.percentile(curvature, 95.0)),
        "turning_angle_integral_rad": float(np.sum(signed_curvature * cell_length)),
        "closed_geometry_error_m": float(
            np.hypot(track.x_m[-1] - track.x_m[0], track.y_m[-1] - track.y_m[0])
        ),
        "extraction_assumptions": [
            "largest connected saturated-red component is the course drawing",
            "non-start endpoint branches are dashed cone/slalom annotations",
            "manual affine alignment converts drawing pixels to east/north metres",
        ],
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"track: {track.length_m:.3f} m, {track.cell_count} cells, "
        f"max |curvature| {np.max(curvature):.4f} 1/m"
    )
    print(f"csv: {output}")
    print(f"metadata: {metadata_path}")
    print(f"preview: {preview}")
    print(f"side by side: {side_by_side}")


if __name__ == "__main__":
    main()
