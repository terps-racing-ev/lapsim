r"""Stitch measured, unpowered coast segments into a 35-to-0 mph trace.

Run from the repository root::

    .venv\Scripts\python.exe analysis\testing\coastdown\stitch_35_to_zero.py

The join points are matched by VCU speed. Time gaps between the original
recordings are removed; the result is a composite test, not one continuous
physical run. The CSV retains the source recording and source time for every
sample. No drag model or extrapolated zero-speed point is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_35mph_drag import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, read_control_summary
from plot_vcu_speed import ROOT, VCU_SPEED_CHANNEL, read_vcu_speed


@dataclass(frozen=True)
class Source:
    path: Path
    search_window_s: tuple[float, float]
    start_mph: float
    finish_mph: float


SOURCES = (
    Source(DEFAULT_INPUT, (579.1, 584.8), 35.0, 24.0),
    Source(DEFAULT_INPUT, (364.5, 384.6), 24.0, 1.0),
    Source(ROOT / "logs" / "2026-09-06_23-54-22_14m.mf4", (329.0, 331.3), 1.0, 0.0),
)


def descending_crossing(
    time_s: np.ndarray, speed_mph: np.ndarray, target_mph: float, *, last: bool = False
) -> float:
    """Linearly locate a downward crossing of a target speed."""

    crossings = np.flatnonzero(
        (speed_mph[:-1] > target_mph) & (speed_mph[1:] <= target_mph)
    )
    if not len(crossings):
        raise ValueError(f"No downward crossing of {target_mph:g} mph in search window")
    i = int(crossings[-1] if last else crossings[0])
    fraction = (speed_mph[i] - target_mph) / (speed_mph[i] - speed_mph[i + 1])
    return float(time_s[i] + fraction * (time_s[i + 1] - time_s[i]))


def measured_segment(source: Source, time_s: np.ndarray, speed_mph: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start, finish = source.search_window_s
    selected = (time_s >= start) & (time_s <= finish)
    if np.count_nonzero(selected) < 2:
        raise ValueError(f"Too few VCU samples in {source.path.name}: {start:g}-{finish:g} s")
    search_time = time_s[selected]
    search_speed = speed_mph[selected]
    # Launch transients can cross the join speed several times. Start at the
    # last downward crossing so the stitched trace follows the settled coast.
    first = descending_crossing(search_time, search_speed, source.start_mph, last=True)
    after_first = search_time >= first
    last = descending_crossing(
        search_time[after_first], search_speed[after_first], source.finish_mph
    )
    if last <= first:
        raise ValueError(f"Invalid coast segment in {source.path.name}")
    interior = (time_s > first) & (time_s < last)
    segment_time = np.r_[first, time_s[interior], last]
    segment_speed = np.r_[source.start_mph, speed_mph[interior], source.finish_mph]
    return segment_time, segment_speed


def build_composite() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    recordings: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    elapsed_offset = 0.0

    for number, source in enumerate(SOURCES, start=1):
        if source.path not in recordings:
            time_s, speed_mph, _, _ = read_vcu_speed(source.path)
            recordings[source.path] = (time_s, speed_mph)
        time_s, speed_mph = recordings[source.path]
        source_time, source_speed = measured_segment(source, time_s, speed_mph)
        controls = read_control_summary(
            source.path, ((float(source_time[0]), float(source_time[-1])),)
        )[0]
        composite_time = elapsed_offset + source_time - source_time[0]
        sources.append(
            {
                "segment": number,
                "input": str(source.path.resolve()),
                "search_window_s": source.search_window_s,
                "source_window_s": [float(source_time[0]), float(source_time[-1])],
                "composite_window_s": [float(composite_time[0]), float(composite_time[-1])],
                "speed_window_mph": [source.start_mph, source.finish_mph],
                "samples": int(len(source_time)),
                "controls": controls,
            }
        )
        # Each join has the same speed and composite time on both sides.
        # Keep one copy of the boundary sample in the CSV.
        first_row = 0 if number == 1 else 1
        for i in range(first_row, len(source_time)):
            rows.append(
                {
                    "composite_time_s": float(composite_time[i]),
                    "speed_mph": float(source_speed[i]),
                    "segment": number,
                    "source_file": source.path.name,
                    "source_time_s": float(source_time[i]),
                    "interpolated_boundary": int(i == 0 or i == len(source_time) - 1),
                }
            )
        elapsed_offset = float(composite_time[-1])

    if not np.isclose(rows[0]["speed_mph"], 35.0) or not np.isclose(rows[-1]["speed_mph"], 0.0):
        raise ValueError("Composite did not span 35 to 0 mph")
    if not np.all(np.diff([row["composite_time_s"] for row in rows]) > 0):
        raise ValueError("Composite time must increase strictly")
    return rows, sources


def save_outputs(output_dir: Path, rows: list[dict[str, object]], sources: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vcu_coastdown_35_to_0_composite.csv"
    plot_path = output_dir / "vcu_coastdown_35_to_0_composite.png"
    report_path = output_dir / "vcu_coastdown_35_to_0_composite.json"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(11, 5.5), layout="constrained")
    colors = ("#1f77b4", "#e6862d", "#2a9d8f")
    for source in sources:
        number = int(source["segment"])
        selected = [row for row in rows if row["segment"] == number]
        axis.plot(
            [row["composite_time_s"] for row in selected],
            [row["speed_mph"] for row in selected],
            color=colors[(number - 1) % len(colors)],
            linewidth=1.6,
            label=f"Segment {number}: {source['speed_window_mph'][0]:g}–{source['speed_window_mph'][1]:g} mph",
        )
        if number > 1:
            axis.axvline(source["composite_window_s"][0], color="0.55", linestyle="--", linewidth=0.9)
    axis.set(xlabel="Composite elapsed time [s]", ylabel="VCU speed [mph]", title="Measured composite coastdown: 35 to 0 mph")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(plot_path, dpi=200, facecolor="white")
    plt.close(figure)

    report = {
        "description": "Measured coast segments joined at matching VCU speeds; time gaps removed. This is a composite, not a single physical run.",
        "speed_channel": VCU_SPEED_CHANNEL,
        "join_speeds_mph": [24.0, 1.0],
        "duration_s": rows[-1]["composite_time_s"],
        "samples": len(rows),
        "sources": sources,
        "csv": str(csv_path.resolve()),
        "plot": str(plot_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Composite: 35 to 0 mph in {report['duration_s']:.2f} s ({len(rows)} samples)")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Plot: {plot_path.resolve()}")
    print(f"Report: {report_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    rows, sources = build_composite()
    save_outputs(args.output_dir, rows, sources)


if __name__ == "__main__":
    main()
