"""Generic, uniformly discretized racing-line geometry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, copysign, cos, isfinite, sin
from os import PathLike
from pathlib import Path
import csv

from .track import Curve, Straight, Track


@dataclass(frozen=True, slots=True)
class SpatialTrack:
    """Cell-based racing line independent of any vehicle or scoring rules.

    Point channels contain ``cell_count + 1`` entries. Cell curvature contains
    one entry per interval. A closed track repeats its start point at the end;
    distance remains strictly increasing through the complete lap.
    """

    distance_m: tuple[float, ...]
    x_m: tuple[float, ...]
    y_m: tuple[float, ...]
    curvature_per_m: tuple[float, ...]
    closed: bool = True

    def __post_init__(self) -> None:
        point_count = len(self.distance_m)
        if point_count < 2:
            raise ValueError("A spatial track needs at least two points")
        if len(self.x_m) != point_count or len(self.y_m) != point_count:
            raise ValueError("x_m and y_m must match distance_m")
        if len(self.curvature_per_m) != point_count - 1:
            raise ValueError("curvature_per_m must contain one value per cell")
        channels = (
            self.distance_m,
            self.x_m,
            self.y_m,
            self.curvature_per_m,
        )
        if any(not all(isfinite(value) for value in channel) for channel in channels):
            raise ValueError("Spatial-track channels must be finite")
        if abs(self.distance_m[0]) > 1e-9:
            raise ValueError("distance_m must start at zero")
        if any(
            upper <= lower for lower, upper in zip(self.distance_m, self.distance_m[1:])
        ):
            raise ValueError("distance_m must be strictly increasing")

    @property
    def cell_count(self) -> int:
        return len(self.curvature_per_m)

    @property
    def length_m(self) -> float:
        return self.distance_m[-1]

    @property
    def cell_length_m(self) -> tuple[float, ...]:
        return tuple(
            upper - lower for lower, upper in zip(self.distance_m, self.distance_m[1:])
        )

    @property
    def cell_center_distance_m(self) -> tuple[float, ...]:
        return tuple(
            0.5 * (lower + upper)
            for lower, upper in zip(self.distance_m, self.distance_m[1:])
        )

    def wrap_distance_m(self, distance_m: float) -> float:
        """Wrap a station onto a closed lap, preserving an exact zero."""

        if not isfinite(distance_m):
            raise ValueError("distance_m must be finite")
        if not self.closed:
            if not 0.0 <= distance_m <= self.length_m:
                raise ValueError("distance is outside the open track")
            return distance_m
        return distance_m % self.length_m

    @classmethod
    def from_csv(
        cls,
        path: str | PathLike[str],
        *,
        closed: bool = True,
    ) -> "SpatialTrack":
        """Load point geometry and per-cell curvature from a portable CSV.

        Each row contains ``distance_m``, ``x_m``, and ``y_m``. The
        ``curvature_per_m`` value belongs to the cell beginning at that row;
        it must therefore be blank on the final endpoint row.
        """

        input_path = Path(path)
        with input_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"distance_m", "x_m", "y_m", "curvature_per_m"}
        if not rows:
            raise ValueError("Spatial-track CSV contains no rows")
        if not required.issubset(rows[0]):
            missing = ", ".join(sorted(required.difference(rows[0])))
            raise ValueError(f"Spatial-track CSV is missing columns: {missing}")
        if rows[-1]["curvature_per_m"].strip():
            raise ValueError("Final spatial-track CSV curvature must be blank")

        return cls(
            distance_m=tuple(float(row["distance_m"]) for row in rows),
            x_m=tuple(float(row["x_m"]) for row in rows),
            y_m=tuple(float(row["y_m"]) for row in rows),
            curvature_per_m=tuple(
                float(row["curvature_per_m"]) for row in rows[:-1]
            ),
            closed=closed,
        )

    def to_csv(self, path: str | PathLike[str]) -> Path:
        """Write the generic point/cell representation consumed by `from_csv`."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("distance_m", "x_m", "y_m", "curvature_per_m"))
            for index, (distance_m, x_m, y_m) in enumerate(
                zip(self.distance_m, self.x_m, self.y_m, strict=True)
            ):
                curvature = (
                    self.curvature_per_m[index]
                    if index < self.cell_count
                    else ""
                )
                writer.writerow((distance_m, x_m, y_m, curvature))
        return output_path.resolve()

    def to_track(self) -> Track:
        """Convert cells to the legacy straight/constant-curvature segments."""

        segments: list[Straight | Curve] = []
        for length_m, curvature in zip(
            self.cell_length_m,
            self.curvature_per_m,
            strict=True,
        ):
            if abs(curvature) <= 1e-15:
                segments.append(Straight(length_m))
            else:
                segments.append(
                    Curve(
                        radius_m=1.0 / abs(curvature),
                        span_rad=curvature * length_m,
                    )
                )
        return Track.from_segments(segments)

    @classmethod
    def from_track(
        cls,
        track: Track,
        *,
        maximum_cell_length_m: float = 2.0,
        closed: bool = True,
        close_geometry: bool = True,
    ) -> "SpatialTrack":
        """Discretize any straight/constant-curvature :class:`Track`.

        ``close_geometry`` distributes a small source-centerline closure error
        over the plotted x/y coordinates. It does not change distance or
        curvature and therefore does not change the physics.
        """

        if maximum_cell_length_m <= 0:
            raise ValueError("maximum_cell_length_m must be positive")

        cell_lengths_m: list[float] = []
        curvatures_per_m: list[float] = []
        for segment in track.segments:
            if segment.length_m <= 0:
                raise ValueError("Track segment lengths must be positive")
            cell_count = ceil(segment.length_m / maximum_cell_length_m)
            cell_length_m = segment.length_m / cell_count
            if isinstance(segment, Straight):
                curvature_per_m = 0.0
            elif isinstance(segment, Curve):
                if segment.radius_m <= 0 or segment.span_rad == 0:
                    raise ValueError("Curve radius and span must be nonzero")
                curvature_per_m = copysign(1.0 / segment.radius_m, segment.span_rad)
            else:
                raise TypeError(f"Unsupported segment type: {type(segment).__name__}")
            cell_lengths_m.extend([cell_length_m] * cell_count)
            curvatures_per_m.extend([curvature_per_m] * cell_count)

        distance_m = [0.0]
        x_m = [0.0]
        y_m = [0.0]
        heading_rad = 0.0
        for cell_length_m, curvature_per_m in zip(
            cell_lengths_m, curvatures_per_m, strict=True
        ):
            distance_m.append(distance_m[-1] + cell_length_m)
            if abs(curvature_per_m) <= 1e-15:
                x_m.append(x_m[-1] + cell_length_m * cos(heading_rad))
                y_m.append(y_m[-1] + cell_length_m * sin(heading_rad))
                continue
            next_heading_rad = heading_rad + curvature_per_m * cell_length_m
            x_m.append(
                x_m[-1] + (sin(next_heading_rad) - sin(heading_rad)) / curvature_per_m
            )
            y_m.append(
                y_m[-1] - (cos(next_heading_rad) - cos(heading_rad)) / curvature_per_m
            )
            heading_rad = next_heading_rad

        if closed and close_geometry:
            end_x_m = x_m[-1] - x_m[0]
            end_y_m = y_m[-1] - y_m[0]
            total_length_m = distance_m[-1]
            for index, station_m in enumerate(distance_m):
                fraction = station_m / total_length_m
                x_m[index] -= fraction * end_x_m
                y_m[index] -= fraction * end_y_m

        return cls(
            distance_m=tuple(distance_m),
            x_m=tuple(x_m),
            y_m=tuple(y_m),
            curvature_per_m=tuple(curvatures_per_m),
            closed=closed,
        )

    @classmethod
    def from_cells(
        cls,
        *,
        cell_length_m: Sequence[float],
        curvature_per_m: Sequence[float],
        closed: bool = True,
    ) -> "SpatialTrack":
        """Build generic geometry directly from cell lengths and curvature."""

        if len(cell_length_m) != len(curvature_per_m) or not cell_length_m:
            raise ValueError(
                "Cell lengths and curvature must have equal nonzero length"
            )
        segments: list[Straight | Curve] = []
        for length_m, curvature in zip(cell_length_m, curvature_per_m, strict=True):
            if length_m <= 0:
                raise ValueError("Cell lengths must be positive")
            if abs(curvature) <= 1e-15:
                segments.append(Straight(float(length_m)))
            else:
                segments.append(
                    Curve(
                        radius_m=1.0 / abs(float(curvature)),
                        span_rad=float(curvature) * float(length_m),
                    )
                )
        return cls.from_track(
            Track.from_segments(segments),
            maximum_cell_length_m=max(float(value) for value in cell_length_m),
            closed=closed,
        )


__all__ = ["SpatialTrack"]
