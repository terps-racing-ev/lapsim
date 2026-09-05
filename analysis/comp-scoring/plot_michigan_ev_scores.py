"""Plot recent FSAE Michigan EV 9th- and 10th-place score cutoffs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "scores.csv"
NINTH_COLOR = "#0072B2"
TENTH_COLOR = "#D55E00"
MARYLAND_COLOR = "#009E73"
PROJECTED_COLOR = "#CC79A7"
MEDIAN_COLOR = "#0072B2"
GRID_COLOR = "#B8B8B8"
YEARS = (2023, 2024, 2025, 2026)
METRICS = (
    ("cost", "Cost"),
    ("presentation", "Presentation"),
    ("design", "Design"),
    ("acceleration", "Acceleration"),
    ("skidpad", "Skid pad"),
    ("autocross", "Autocross"),
    ("endurance_efficiency", "Endurance + efficiency"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument(
        "--maryland-projected-endurance-efficiency",
        type=float,
        default=254.4,
        help="Projected combined endurance and efficiency points.",
    )
    return parser.parse_args()


def load_scores(path: Path) -> list[dict[str, str | float | int]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric_fields = {
        "total",
        "cost",
        "presentation",
        "design",
        "acceleration",
        "skidpad",
        "autocross",
        "endurance",
        "efficiency",
    }
    parsed: list[dict[str, str | float | int]] = []
    for row in rows:
        item: dict[str, str | float | int] = dict(row)
        item["year"] = int(row["year"])
        for field in numeric_fields:
            item[field] = float(row[field])
        item["endurance_efficiency"] = float(row["endurance"]) + float(
            row["efficiency"]
        )
        parsed.append(item)
    return parsed


def series_rows(
    rows: list[dict[str, str | float | int]], series: str
) -> list[dict[str, str | float | int]]:
    return sorted(
        (row for row in rows if row["series"] == series),
        key=lambda row: int(row["year"]),
    )


def median_rows(
    rows: list[dict[str, str | float | int]],
) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    for year in YEARS:
        finishers = [
            row
            for row in rows
            if int(row["year"]) == year
            and row["series"] in {"5th", "6th", "7th", "8th", "9th", "10th"}
        ]
        if len(finishers) != 6:
            raise ValueError(f"Expected places 5–10 for {year}, got {len(finishers)}")
        item: dict[str, float | int] = {"year": year}
        for metric, _ in METRICS:
            item[metric] = median(float(row[metric]) for row in finishers)
        result.append(item)
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(2022.75, 2026.35)
    axis.set_xticks(YEARS)
    axis.set_xlabel("Competition year")
    axis.set_ylabel("Score (points)")
    axis.grid(axis="y", color=GRID_COLOR, alpha=0.35, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_cutoff_series(
    axis: plt.Axes,
    ninth: list[dict[str, str | float | int]],
    tenth: list[dict[str, str | float | int]],
    metric: str,
) -> None:
    for rows, color, label in (
        (ninth, NINTH_COLOR, "9th place"),
        (tenth, TENTH_COLOR, "10th place"),
    ):
        x = [int(row["year"]) for row in rows]
        y = [float(row[metric]) for row in rows]
        axis.plot(x, y, marker="o", linewidth=2.2, color=color, label=label)

    for ninth_row, tenth_row in zip(ninth, tenth, strict=True):
        year = int(ninth_row["year"])
        ninth_value = float(ninth_row[metric])
        tenth_value = float(tenth_row[metric])
        ninth_offset, tenth_offset = (
            (9, -14) if ninth_value >= tenth_value else (-14, 9)
        )
        for value, color, label_offset in (
            (ninth_value, NINTH_COLOR, ninth_offset),
            (tenth_value, TENTH_COLOR, tenth_offset),
        ):
            axis.annotate(
                f"{value:.1f}",
                (year, value),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if label_offset > 0 else "top",
                fontsize=8.5,
                color=color,
            )


def add_reference(
    axis: plt.Axes,
    value: float,
    *,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    axis.axhline(value, color=color, linestyle=linestyle, linewidth=1.8, label=label)
    axis.annotate(
        f"{value:.1f}",
        (2026.28, value),
        xytext=(0, 9),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=color,
    )


def plot_cumulative(
    ninth: list[dict[str, str | float | int]],
    tenth: list[dict[str, str | float | int]],
    maryland: dict[str, str | float | int],
    projected_endurance_efficiency: float,
    output_path: Path,
    *,
    include_actual: bool = True,
    include_projected: bool = True,
) -> None:
    projected_total = (
        float(maryland["total"])
        - float(maryland["endurance_efficiency"])
        + projected_endurance_efficiency
    )
    figure, axis = plt.subplots(figsize=(11.5, 6.3), constrained_layout=True)
    plot_cutoff_series(axis, ninth, tenth, "total")
    if include_actual:
        add_reference(
            axis,
            float(maryland["total"]),
            color=MARYLAND_COLOR,
            linestyle=":",
            label="Maryland 2026 actual",
        )
    if include_projected:
        add_reference(
            axis,
            projected_total,
            color=PROJECTED_COLOR,
            linestyle="-.",
            label="Maryland projected",
        )
    style_axis(axis)
    axis.margins(y=0.12)
    axis.set_title("FSAE Michigan EV cumulative score cutoff (2023–2026)")
    axis.legend(loc="best", frameon=False, ncols=2)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_event_group(
    event_medians: list[dict[str, float | int]],
    maryland: dict[str, str | float | int],
    projected_endurance_efficiency: float,
    output_path: Path,
    metrics: tuple[tuple[str, str], ...],
    title: str,
    *,
    include_maryland: bool = True,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(6.5 * len(metrics), 5.4),
        constrained_layout=False,
        squeeze=False,
    )
    for axis, (metric, event_title) in zip(axes.ravel(), metrics, strict=True):
        x = [int(row["year"]) for row in event_medians]
        y = [float(row[metric]) for row in event_medians]
        axis.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            color=MEDIAN_COLOR,
            label="Median of 5th–10th",
        )
        for year, value in zip(x, y, strict=True):
            axis.annotate(
                f"{value:.1f}",
                (year, value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=MEDIAN_COLOR,
            )
        if include_maryland:
            add_reference(
                axis,
                float(maryland[metric]),
                color=MARYLAND_COLOR,
                linestyle=":",
                label="Maryland 2026 actual",
            )
            if metric == "endurance_efficiency":
                add_reference(
                    axis,
                    projected_endurance_efficiency,
                    color=PROJECTED_COLOR,
                    linestyle="-.",
                    label="Maryland projected",
                )
        style_axis(axis)
        axis.margins(y=0.16)
        axis.set_title(event_title)
    legend_handles: list[Line2D] = [
        Line2D(
            [],
            [],
            color=MEDIAN_COLOR,
            marker="o",
            linewidth=2.2,
            label="Median of 5th–10th",
        ),
    ]
    if include_maryland:
        legend_handles.append(
            Line2D(
                [],
                [],
                color=MARYLAND_COLOR,
                linestyle=":",
                linewidth=1.8,
                label="Maryland 2026 actual",
            )
        )
        if any(metric == "endurance_efficiency" for metric, _ in metrics):
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color=PROJECTED_COLOR,
                    linestyle="-.",
                    linewidth=1.8,
                    label="Maryland projected",
                )
            )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.22,
        top=0.78,
        wspace=0.28,
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncols=len(legend_handles),
    )
    figure.suptitle(title, fontsize=16, y=0.96)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = load_scores(args.data)
    ninth = series_rows(rows, "9th")
    tenth = series_rows(rows, "10th")
    event_medians = median_rows(rows)
    maryland_rows = series_rows(rows, "Maryland")
    if len(ninth) != 4 or len(tenth) != 4 or len(maryland_rows) != 1:
        raise ValueError("Expected four cutoff rows per place and one Maryland row")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cumulative_path = args.output_dir / "cumulative_scores.png"
    cumulative_no_maryland_path = (
        args.output_dir / "cumulative_scores_no_maryland.png"
    )
    cumulative_projected_only_path = (
        args.output_dir / "cumulative_scores_projected_only.png"
    )
    event_groups = (
        (
            (("cost", "Cost"), ("presentation", "Presentation"), ("design", "Design")),
            "FSAE Michigan EV median static-event scores for 5th–10th (2023–2026)",
            "static_event_scores.png",
        ),
        (
            (("acceleration", "Acceleration"), ("skidpad", "Skid pad")),
            "FSAE Michigan EV median acceleration and skid-pad scores for 5th–10th (2023–2026)",
            "acceleration_skidpad_scores.png",
        ),
        (
            (("autocross", "Autocross"), ("endurance_efficiency", "Endurance + efficiency")),
            "FSAE Michigan EV median dynamic-event scores for 5th–10th (2023–2026)",
            "autocross_endurance_efficiency_scores.png",
        ),
    )
    plot_cumulative(
        ninth,
        tenth,
        maryland_rows[0],
        args.maryland_projected_endurance_efficiency,
        cumulative_path,
    )
    plot_cumulative(
        ninth,
        tenth,
        maryland_rows[0],
        args.maryland_projected_endurance_efficiency,
        cumulative_no_maryland_path,
        include_actual=False,
        include_projected=False,
    )
    plot_cumulative(
        ninth,
        tenth,
        maryland_rows[0],
        args.maryland_projected_endurance_efficiency,
        cumulative_projected_only_path,
        include_actual=False,
    )
    event_paths: list[Path] = []
    for metrics, title, filename in event_groups:
        output_path = args.output_dir / filename
        no_maryland_path = args.output_dir / filename.replace(
            ".png", "_no_maryland.png"
        )
        plot_event_group(
            event_medians,
            maryland_rows[0],
            args.maryland_projected_endurance_efficiency,
            output_path,
            metrics,
            title,
        )
        plot_event_group(
            event_medians,
            maryland_rows[0],
            args.maryland_projected_endurance_efficiency,
            no_maryland_path,
            metrics,
            title,
            include_maryland=False,
        )
        event_paths.extend((output_path, no_maryland_path))
    print(cumulative_path.resolve())
    print(cumulative_no_maryland_path.resolve())
    print(cumulative_projected_only_path.resolve())
    for path in event_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
