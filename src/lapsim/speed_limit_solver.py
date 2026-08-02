"""Local physical speed-limit calculation around a track."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, copysign, cos, sin

from .environment import STANDARD_AIR_DENSITY_KGPM3, STANDARD_GRAVITY_MPS2
from .track import Curve, Straight, Track
from .utils.units import meters_per_second_to_miles_per_hour
from .vehicle import Vehicle


DEFAULT_MAX_STEP_M = 0.5
DEFAULT_SPEED_MAP_LINE_WIDTH = 4.0
START_MARKER_SIZE = 70.0


@dataclass(frozen=True, slots=True)
class SpeedLimitMap:
    """Local speed ceilings and discretized geometry for one track lap."""

    distance_m: tuple[float, ...]
    x_m: tuple[float, ...]
    y_m: tuple[float, ...]
    speed_limit_mps: tuple[float, ...]
    cell_length_m: tuple[float, ...]
    curvature_per_m: tuple[float, ...]

    def plot_speed_map(
        self,
        speeds_mps: Sequence[float] | None = None,
        *,
        title: str = "Local speed-limit map",
    ):
        """Plot local limits or another point-centered speed channel."""

        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
        import matplotlib.pyplot as plt

        plotted_speeds_mps = (
            tuple(speeds_mps)
            if speeds_mps is not None
            else self.speed_limit_mps
        )
        if len(plotted_speeds_mps) != len(self.x_m):
            raise ValueError(
                "speeds_mps must contain one speed for each map coordinate"
            )

        line_segments = [
            [
                (self.x_m[index], self.y_m[index]),
                (self.x_m[index + 1], self.y_m[index + 1]),
            ]
            for index in range(len(self.x_m) - 1)
        ]
        segment_speeds_mph = [
            meters_per_second_to_miles_per_hour(
                0.5 * (plotted_speeds_mps[index] + plotted_speeds_mps[index + 1])
            )
            for index in range(len(plotted_speeds_mps) - 1)
        ]
        normalization = Normalize(
            vmin=min(segment_speeds_mph),
            vmax=max(segment_speeds_mph),
        )
        colored_track = LineCollection(
            line_segments,
            cmap="turbo",
            norm=normalization,
            linewidth=DEFAULT_SPEED_MAP_LINE_WIDTH,
        )
        colored_track.set_array(segment_speeds_mph)

        figure, axes = plt.subplots(figsize=(8, 10), facecolor="white")
        axes.add_collection(colored_track)
        axes.scatter(
            self.x_m[0],
            self.y_m[0],
            s=START_MARKER_SIZE,
            color="black",
            label="Start / finish",
            zorder=3,
        )
        axes.autoscale()
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        axes.set_title(title)
        axes.grid(True, color="0.85")
        axes.legend()
        figure.colorbar(colored_track, ax=axes, label="Speed [mph]")
        figure.tight_layout()
        return figure, axes


@dataclass(frozen=True, slots=True)
class _Cell:
    length_m: float
    curvature_per_m: float


class SpeedLimitSolver:
    """Calculate independent local cornering and vehicle speed ceilings."""

    def __init__(
        self,
        vehicle: Vehicle,
        max_step_m: float = DEFAULT_MAX_STEP_M,
        gravity_mps2: float = STANDARD_GRAVITY_MPS2,
        air_density_kgpm3: float = STANDARD_AIR_DENSITY_KGPM3,
    ) -> None:
        if max_step_m <= 0:
            raise ValueError("max_step_m must be positive")
        if gravity_mps2 <= 0:
            raise ValueError("gravity_mps2 must be positive")
        if air_density_kgpm3 <= 0:
            raise ValueError("air_density_kgpm3 must be positive")
        self.vehicle = vehicle
        self.max_step_m = max_step_m
        self.gravity_mps2 = gravity_mps2
        self.air_density_kgpm3 = air_density_kgpm3

    def solve(self, track: Track) -> SpeedLimitMap:
        """Calculate local limits without acceleration or braking passes."""

        self.vehicle.validate()
        cells = self._build_cells(track)
        cell_limits_mps = [
            self._steady_state_speed_limit(cell.curvature_per_m)
            for cell in cells
        ]
        return self._make_map(cells, cell_limits_mps)

    def _build_cells(self, track: Track) -> list[_Cell]:
        cells: list[_Cell] = []
        for segment in track.segments:
            if segment.length_m <= 0:
                raise ValueError("Track segment lengths must be positive")
            cell_count = ceil(segment.length_m / self.max_step_m)
            cell_length_m = segment.length_m / cell_count

            if isinstance(segment, Straight):
                curvature_per_m = 0.0
            elif isinstance(segment, Curve):
                if segment.radius_m <= 0:
                    raise ValueError("Curve radii must be positive")
                if segment.span_rad == 0:
                    raise ValueError("Curve spans cannot be zero")
                curvature_per_m = copysign(
                    1.0 / segment.radius_m,
                    segment.span_rad,
                )
            else:
                raise TypeError(
                    f"Unsupported segment type: {type(segment).__name__}"
                )

            cells.extend(
                _Cell(cell_length_m, curvature_per_m)
                for _ in range(cell_count)
            )

        if not cells:
            raise ValueError("Track must contain at least one segment")
        return cells

    def _steady_state_speed_limit(self, curvature_per_m: float) -> float:
        vehicle_speed_limit_mps = (
            self.vehicle.drivetrain.vehicle_speed_limit_mps
        )
        absolute_curvature_per_m = abs(curvature_per_m)
        if absolute_curvature_per_m == 0:
            return vehicle_speed_limit_mps

        lower_speed_mps = 0.0
        upper_speed_mps = vehicle_speed_limit_mps
        for _ in range(50):
            candidate_speed_mps = 0.5 * (
                lower_speed_mps + upper_speed_mps
            )
            aero_forces = self.vehicle.aero.forces_n(
                candidate_speed_mps,
                self.air_density_kgpm3,
            )
            tire_normal_loads = self.vehicle.chassis.tire_normal_loads_n(
                self.vehicle.mass_kg,
                self.gravity_mps2,
                aero_forces,
            )
            available_lateral_force_n = sum(
                self.vehicle.tire.lateral_force_capacity_n(normal_load_n)
                for normal_load_n in tire_normal_loads.all_n
            )
            required_lateral_force_n = (
                self.vehicle.mass_kg
                * candidate_speed_mps**2
                * absolute_curvature_per_m
            )
            if required_lateral_force_n <= available_lateral_force_n:
                lower_speed_mps = candidate_speed_mps
            else:
                upper_speed_mps = candidate_speed_mps
        return lower_speed_mps

    @staticmethod
    def _make_map(
        cells: list[_Cell],
        cell_limits_mps: list[float],
    ) -> SpeedLimitMap:
        distance_m = [0.0]
        x_m = [0.0]
        y_m = [0.0]
        heading_rad = 0.0

        for cell in cells:
            distance_m.append(distance_m[-1] + cell.length_m)
            if cell.curvature_per_m == 0:
                next_x_m = x_m[-1] + cell.length_m * cos(heading_rad)
                next_y_m = y_m[-1] + cell.length_m * sin(heading_rad)
            else:
                turn_angle_rad = cell.curvature_per_m * cell.length_m
                turn_direction = copysign(1.0, cell.curvature_per_m)
                radius_m = 1.0 / abs(cell.curvature_per_m)
                next_heading_rad = heading_rad + turn_angle_rad
                next_x_m = x_m[-1] + turn_direction * radius_m * (
                    sin(next_heading_rad) - sin(heading_rad)
                )
                next_y_m = y_m[-1] - turn_direction * radius_m * (
                    cos(next_heading_rad) - cos(heading_rad)
                )
                heading_rad = next_heading_rad
            x_m.append(next_x_m)
            y_m.append(next_y_m)

        return SpeedLimitMap(
            distance_m=tuple(distance_m),
            x_m=tuple(x_m),
            y_m=tuple(y_m),
            speed_limit_mps=tuple(cell_limits_mps + [cell_limits_mps[0]]),
            cell_length_m=tuple(cell.length_m for cell in cells),
            curvature_per_m=tuple(cell.curvature_per_m for cell in cells),
        )
