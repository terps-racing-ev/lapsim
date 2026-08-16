r"""Manually fine-tune the first-lap GNSS overlay on the official course map.

Run from the ``python_lapsim`` repository root::

    .\.venv\Scripts\python.exe analysis\first_endurance_lap\align_gps_to_map_gui.py

The controls are display-only: they adjust the automatic map fit without
changing the metre-based GNSS coordinates in ``first_lap.csv``.  Click Save to
write ``manual_map_alignment.json`` and ``manual_map_alignment_preview.png``.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib


EXPORT_REQUESTED = "--export" in sys.argv
matplotlib.use("Agg" if EXPORT_REQUESTED else "TkAgg")

import matplotlib.image as mpimg
from matplotlib.figure import Figure
import numpy as np

if not EXPORT_REQUESTED:
    import tkinter as tk
    from tkinter import messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


ANALYSIS_DIR = Path(__file__).resolve().parent
CSV_PATH = ANALYSIS_DIR / "first_lap.csv"
METADATA_PATH = ANALYSIS_DIR / "first_lap.json"
MAP_PATH = ANALYSIS_DIR / "official_course_map.png"
SAVED_ALIGNMENT_PATH = ANALYSIS_DIR / "manual_map_alignment.json"
PREVIEW_PATH = ANALYSIS_DIR / "manual_map_alignment_preview.png"

CONTROL_DEFAULTS = {
    "translation_x_px": 0.0,
    "translation_y_px": 0.0,
    "rotation_deg": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "x_shear": 0.0,
}


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        action="store_true",
        help="Write a preview without opening the GUI. Uses the saved adjustment if present.",
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=SAVED_ALIGNMENT_PATH,
        help="Saved manual alignment JSON to use with --export.",
    )
    return parser.parse_args()


def read_csv_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the raw and already-filtered local GPS coordinates."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}\nRun generate_first_lap.py before opening the alignment GUI."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = ("gps_x_m", "gps_y_m", "gps_x_filtered_m", "gps_y_filtered_m")
    if not rows or any(name not in rows[0] for name in required):
        raise ValueError(f"{path} must contain: {', '.join(required)}")
    return tuple(
        np.asarray([float(row[name]) for row in rows], dtype=float) for name in required
    )  # type: ignore[return-value]


def load_base_alignment(path: Path) -> dict[str, float]:
    """Load the automatic affine transform produced by generate_first_lap.py."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}\nRun generate_first_lap.py before opening the alignment GUI."
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    alignment = metadata.get("map_alignment")
    required = (
        "x0_px",
        "horizontal_scale_px_per_m",
        "y0_px",
        "horizontal_shear_px_per_m",
        "vertical_scale_px_per_m",
    )
    if not isinstance(alignment, dict) or any(name not in alignment for name in required):
        raise ValueError(f"{path} does not contain a complete map_alignment section")
    return {name: float(alignment[name]) for name in required}


def transform_to_automatic_pixels(
    x_m: np.ndarray,
    y_m: np.ndarray,
    base_alignment: dict[str, float],
) -> np.ndarray:
    """Apply the original automatic affine metre-to-pixel transform."""
    horizontal_m = -y_m
    vertical_m = x_m
    return np.column_stack(
        (
            base_alignment["x0_px"]
            + base_alignment["horizontal_scale_px_per_m"] * horizontal_m,
            base_alignment["y0_px"]
            + base_alignment["horizontal_shear_px_per_m"] * horizontal_m
            + base_alignment["vertical_scale_px_per_m"] * vertical_m,
        )
    )


def manual_matrix(controls: dict[str, float]) -> np.ndarray:
    """Return the scale/shear/rotation part of the manual pixel adjustment."""
    radians = np.deg2rad(controls["rotation_deg"])
    rotation = np.array(
        (
            (np.cos(radians), -np.sin(radians)),
            (np.sin(radians), np.cos(radians)),
        )
    )
    scale_and_shear = np.array(
        (
            (controls["scale_x"], controls["x_shear"]),
            (0.0, controls["scale_y"]),
        )
    )
    return rotation @ scale_and_shear


def apply_manual_adjustment(
    automatic_pixels: np.ndarray,
    controls: dict[str, float],
    map_width_px: int,
    map_height_px: int,
) -> np.ndarray:
    """Adjust pixels about the centre of the map so resize/rotation stay intuitive."""
    pivot = np.array((map_width_px / 2.0, map_height_px / 2.0))
    translation = np.array(
        (controls["translation_x_px"], controls["translation_y_px"])
    )
    return pivot + (automatic_pixels - pivot) @ manual_matrix(controls).T + translation


def effective_affine_transform(
    base_alignment: dict[str, float],
    controls: dict[str, float],
    map_width_px: int,
    map_height_px: int,
) -> dict[str, list[float]]:
    """Express the combined transform as pixel = matrix @ [east, north] + offset."""
    # Automatic pixels are B @ [east, north] + b.  The manual transform is
    # M @ (automatic_pixels - pivot) + pivot + translation.
    base_matrix = np.array(
        (
            (0.0, -base_alignment["horizontal_scale_px_per_m"]),
            (
                base_alignment["vertical_scale_px_per_m"],
                -base_alignment["horizontal_shear_px_per_m"],
            ),
        )
    )
    base_offset = np.array((base_alignment["x0_px"], base_alignment["y0_px"]))
    matrix = manual_matrix(controls)
    pivot = np.array((map_width_px / 2.0, map_height_px / 2.0))
    translation = np.array(
        (controls["translation_x_px"], controls["translation_y_px"])
    )
    return {
        "matrix_px_per_m": (matrix @ base_matrix).tolist(),
        "offset_px": (matrix @ (base_offset - pivot) + pivot + translation).tolist(),
    }


def make_figure(
    map_image: np.ndarray,
    raw_pixels: np.ndarray,
    filtered_pixels: np.ndarray,
) -> tuple[Figure, Any, Any, Any]:
    """Create the map figure and return its mutable raw/filtered artists."""
    height, width = map_image.shape[:2]
    figure = Figure(figsize=(14, max(3.3, 14 * height / width)), layout="constrained")
    axis = figure.add_subplot()
    axis.imshow(map_image, origin="upper")
    raw_artist = axis.scatter(
        raw_pixels[:, 0],
        raw_pixels[:, 1],
        s=7,
        color="#1f2937",
        alpha=0.32,
        label="Raw GNSS",
        zorder=2,
    )
    (filtered_artist,) = axis.plot(
        filtered_pixels[:, 0],
        filtered_pixels[:, 1],
        color="#06b6d4",
        linewidth=2.0,
        label="Filtered GNSS",
        zorder=3,
    )
    axis.scatter(
        filtered_pixels[0, 0], filtered_pixels[0, 1], s=38, color="#16a34a", label="Start", zorder=4
    )
    axis.scatter(
        filtered_pixels[-1, 0], filtered_pixels[-1, 1], s=44, marker="x", color="#a21caf", label="Finish", zorder=4
    )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("First-lap GNSS overlay — manual display adjustment")
    axis.axis("off")
    axis.legend(loc="lower center", ncols=4, fontsize=8)
    return figure, axis, raw_artist, filtered_artist


def save_alignment(
    base_alignment: dict[str, float],
    controls: dict[str, float],
    map_image: np.ndarray,
    raw_auto: np.ndarray,
    filtered_auto: np.ndarray,
    json_path: Path = SAVED_ALIGNMENT_PATH,
    preview_path: Path = PREVIEW_PATH,
) -> tuple[Path, Path]:
    """Write a portable JSON record and a PNG showing exactly what was saved."""
    height, width = map_image.shape[:2]
    raw_pixels = apply_manual_adjustment(raw_auto, controls, width, height)
    filtered_pixels = apply_manual_adjustment(filtered_auto, controls, width, height)
    figure, _, _, _ = make_figure(map_image, raw_pixels, filtered_pixels)
    figure.savefig(preview_path, dpi=220, bbox_inches="tight")
    metadata: dict[str, Any] = {
        "type": "manual GNSS-to-course-map display alignment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Display-only; first_lap.csv east/north coordinates are unchanged.",
        "source_files": {
            "first_lap_csv": CSV_PATH.name,
            "automatic_alignment": METADATA_PATH.name,
            "course_map": MAP_PATH.name,
            "preview": preview_path.name,
        },
        "map_image_size_px": {"width": int(width), "height": int(height)},
        "base_automatic_alignment": base_alignment,
        "manual_adjustment": {
            **controls,
            "rotation_convention": "positive values rotate clockwise in image pixel coordinates",
            "x_shear_convention": "horizontal pixels added per vertical pixel before rotation",
            "pivot_px": [width / 2.0, height / 2.0],
        },
        "effective_affine_transform": effective_affine_transform(
            base_alignment, controls, width, height
        ),
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return json_path.resolve(), preview_path.resolve()


def load_saved_controls(path: Path) -> dict[str, float]:
    """Load only known manual controls and tolerate older/incomplete save files."""
    if not path.is_file():
        return CONTROL_DEFAULTS.copy()
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        adjustment = saved.get("manual_adjustment", {})
        return {
            name: float(adjustment.get(name, default))
            for name, default in CONTROL_DEFAULTS.items()
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return CONTROL_DEFAULTS.copy()


class AlignmentApp:
    """Small Tk interface for iterating on a map overlay."""

    CONTROL_SPECS = (
        ("translation_x_px", "Offset X", "px", -500.0, 500.0, 0.5),
        ("translation_y_px", "Offset Y", "px", -250.0, 250.0, 0.5),
        ("rotation_deg", "Rotation", "deg", -45.0, 45.0, 0.1),
        ("scale_x", "Horizontal scale", "×", 0.25, 2.0, 0.005),
        ("scale_y", "Vertical scale", "×", 0.25, 2.0, 0.005),
        ("x_shear", "Horizontal shear", "px/px", -1.0, 1.0, 0.005),
    )

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("First-lap GNSS map alignment")
        self.root.minsize(900, 420)
        self.root.geometry("1500x650")
        self.map_image = mpimg.imread(MAP_PATH)
        self.height, self.width = self.map_image.shape[:2]
        self.base_alignment = load_base_alignment(METADATA_PATH)
        raw_x, raw_y, filtered_x, filtered_y = read_csv_columns(CSV_PATH)
        self.raw_auto = transform_to_automatic_pixels(raw_x, raw_y, self.base_alignment)
        self.filtered_auto = transform_to_automatic_pixels(filtered_x, filtered_y, self.base_alignment)
        initial_values = load_saved_controls(SAVED_ALIGNMENT_PATH)
        self.variables = {
            name: tk.DoubleVar(value=initial_values[name]) for name in CONTROL_DEFAULTS
        }
        self._updating = False

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        controls_frame = ttk.Frame(outer, padding=(0, 0, 12, 0))
        controls_frame.pack(side=tk.LEFT, fill=tk.Y)
        plot_frame = ttk.Frame(outer)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._build_controls(controls_frame)
        if SAVED_ALIGNMENT_PATH.is_file():
            self.status.configure(text=f"Loaded {SAVED_ALIGNMENT_PATH.name}")

        initial_controls = self.controls()
        raw_pixels = apply_manual_adjustment(self.raw_auto, initial_controls, self.width, self.height)
        filtered_pixels = apply_manual_adjustment(self.filtered_auto, initial_controls, self.width, self.height)
        self.figure, self.axis, self.raw_artist, self.filtered_artist = make_figure(
            self.map_image, raw_pixels, filtered_pixels
        )
        self.start_artist = self.axis.collections[-2]
        self.finish_artist = self.axis.collections[-1]
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        for variable in self.variables.values():
            variable.trace_add("write", self._on_control_change)

    def _build_controls(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Manual adjustment", font=("TkDefaultFont", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text="Adjust the automatic fit.\nUse the map toolbar to zoom/pan.",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 10))
        for name, label, unit, minimum, maximum, increment in self.CONTROL_SPECS:
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
            spinbox = ttk.Spinbox(
                row,
                from_=minimum,
                to=maximum,
                increment=increment,
                textvariable=self.variables[name],
                width=10,
            )
            spinbox.pack(side=tk.LEFT)
            spinbox.bind("<KeyRelease>", self._on_control_change)
            spinbox.bind("<FocusOut>", self._on_control_change)
            ttk.Label(row, text=unit, width=6).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Separator(parent).pack(fill=tk.X, pady=12)
        ttk.Button(parent, text="Reset to automatic fit", command=self.reset).pack(fill=tk.X, pady=2)
        ttk.Button(parent, text="Save alignment + preview", command=self.save).pack(fill=tk.X, pady=2)
        self.status = ttk.Label(parent, text="Unsaved", justify=tk.LEFT, wraplength=250)
        self.status.pack(fill=tk.X, pady=(10, 0))

    def controls(self) -> dict[str, float]:
        try:
            return {name: float(variable.get()) for name, variable in self.variables.items()}
        except tk.TclError:
            return CONTROL_DEFAULTS.copy()

    def _on_control_change(self, *_: object) -> None:
        if not self._updating:
            self.redraw()

    def redraw(self) -> None:
        controls = self.controls()
        raw_pixels = apply_manual_adjustment(self.raw_auto, controls, self.width, self.height)
        filtered_pixels = apply_manual_adjustment(self.filtered_auto, controls, self.width, self.height)
        self.raw_artist.set_offsets(raw_pixels)
        self.filtered_artist.set_data(filtered_pixels[:, 0], filtered_pixels[:, 1])
        self.start_artist.set_offsets(filtered_pixels[0])
        self.finish_artist.set_offsets(filtered_pixels[-1])
        self.status.configure(text="Unsaved — live preview updated")
        self.canvas.draw_idle()

    def reset(self) -> None:
        # Re-read the generated metadata so Reset always means the current
        # automatic fit, even if generate_first_lap.py was re-run while this
        # window was open.
        self.base_alignment = load_base_alignment(METADATA_PATH)
        raw_x, raw_y, filtered_x, filtered_y = read_csv_columns(CSV_PATH)
        self.raw_auto = transform_to_automatic_pixels(raw_x, raw_y, self.base_alignment)
        self.filtered_auto = transform_to_automatic_pixels(
            filtered_x, filtered_y, self.base_alignment
        )
        self._updating = True
        for name, default in CONTROL_DEFAULTS.items():
            self.variables[name].set(default)
        self._updating = False
        self.redraw()
        self.status.configure(text="Reset to automatic fit — unsaved")

    def save(self) -> None:
        json_path, preview_path = save_alignment(
            self.base_alignment,
            self.controls(),
            self.map_image,
            self.raw_auto,
            self.filtered_auto,
        )
        self.status.configure(text=f"Saved:\n{json_path.name}\n{preview_path.name}")
        messagebox.showinfo("Alignment saved", f"Saved:\n{json_path}\n\nPreview:\n{preview_path}")


def export_preview(alignment_path: Path) -> tuple[Path, Path]:
    """Non-interactive mode used for repeatable preview/export checks."""
    map_image = mpimg.imread(MAP_PATH)
    base_alignment = load_base_alignment(METADATA_PATH)
    raw_x, raw_y, filtered_x, filtered_y = read_csv_columns(CSV_PATH)
    raw_auto = transform_to_automatic_pixels(raw_x, raw_y, base_alignment)
    filtered_auto = transform_to_automatic_pixels(filtered_x, filtered_y, base_alignment)
    controls = load_saved_controls(alignment_path)
    return save_alignment(base_alignment, controls, map_image, raw_auto, filtered_auto)


def main() -> None:
    args = parse_args()
    if args.export:
        json_path, preview_path = export_preview(args.alignment)
        print(f"alignment: {json_path}")
        print(f"preview: {preview_path}")
        return
    root = tk.Tk()
    AlignmentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
