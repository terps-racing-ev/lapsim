"""Vehicle-dependent speed constraints on a prescribed spatial track."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import atan

from vehicle_model.environment import (
    STANDARD_AIR_DENSITY_KGPM3,
    STANDARD_GRAVITY_MPS2,
)
from vehicle_model.vehicle import Vehicle

from .controls import Controls
from .spatial_track import SpatialTrack


class PathConstraintViolation(RuntimeError):
    """Raised when the vehicle cannot follow the prescribed path curvature."""


@dataclass(frozen=True, slots=True)
class PathSpeedConstraints:
    """Local corner limits and cyclic maximum-braking speed ceiling."""

    track: SpatialTrack
    local_corner_speed_mps: tuple[float, ...]
    braking_speed_ceiling_mps: tuple[float, ...]
    passes: int

    def __post_init__(self) -> None:
        expected = self.track.cell_count
        if len(self.local_corner_speed_mps) != expected:
            raise ValueError("Local limits must contain one value per track cell")
        if len(self.braking_speed_ceiling_mps) != expected:
            raise ValueError("Braking ceilings must contain one value per track cell")


class PathConstraintSolver:
    """Precompute torque-profile-independent path speed ceilings."""

    def __init__(
        self,
        *,
        convergence_tolerance_mps: float = 1e-5,
        maximum_passes: int = 500,
        gravity_mps2: float = STANDARD_GRAVITY_MPS2,
        air_density_kgpm3: float = STANDARD_AIR_DENSITY_KGPM3,
    ) -> None:
        if convergence_tolerance_mps <= 0:
            raise ValueError("convergence_tolerance_mps must be positive")
        if maximum_passes <= 0:
            raise ValueError("maximum_passes must be positive")
        self.convergence_tolerance_mps = convergence_tolerance_mps
        self.maximum_passes = maximum_passes
        self.gravity_mps2 = gravity_mps2
        self.air_density_kgpm3 = air_density_kgpm3

    def solve(self, track: SpatialTrack, vehicle: Vehicle) -> PathSpeedConstraints:
        """Calculate local lateral limits then a cyclic backward brake pass."""

        if not track.closed:
            raise ValueError(
                "The endurance path-constraint solver requires a closed track"
            )
        vehicle.validate()
        local_limits = [
            self._steady_state_speed_limit(vehicle, curvature_per_m)
            for curvature_per_m in track.curvature_per_m
        ]
        ceilings = local_limits.copy()
        cell_lengths = track.cell_length_m

        for pass_number in range(1, self.maximum_passes + 1):
            largest_change_mps = 0.0
            for cell_index in range(track.cell_count - 1, -1, -1):
                next_index = (cell_index + 1) % track.cell_count
                reachable_speed_mps = self._maximum_entry_speed_mps(
                    vehicle=vehicle,
                    next_speed_mps=ceilings[next_index],
                    local_speed_limit_mps=local_limits[cell_index],
                    curvature_per_m=track.curvature_per_m[cell_index],
                    cell_length_m=cell_lengths[cell_index],
                )
                new_speed_mps = min(
                    ceilings[cell_index],
                    local_limits[cell_index],
                    reachable_speed_mps,
                )
                largest_change_mps = max(
                    largest_change_mps,
                    ceilings[cell_index] - new_speed_mps,
                )
                ceilings[cell_index] = new_speed_mps
            if largest_change_mps < self.convergence_tolerance_mps:
                return PathSpeedConstraints(
                    track=track,
                    local_corner_speed_mps=tuple(local_limits),
                    braking_speed_ceiling_mps=tuple(ceilings),
                    passes=pass_number,
                )

        raise RuntimeError(
            f"Cyclic braking ceiling did not converge after {self.maximum_passes} passes"
        )

    def _maximum_entry_speed_mps(
        self,
        *,
        vehicle: Vehicle,
        next_speed_mps: float,
        local_speed_limit_mps: float,
        curvature_per_m: float,
        cell_length_m: float,
    ) -> float:
        """Solve the variable-deceleration distance equation conservatively."""

        def residual(entry_speed_mps: float) -> float:
            candidate = deepcopy(vehicle)
            candidate.speed_mps = entry_speed_mps
            candidate.longitudinal_acceleration_mps2 = 0.0
            candidate.lateral_acceleration_mps2 = entry_speed_mps**2 * curvature_per_m
            maximum_braking = Controls(
                front_brake_pressure_psi=1.0e9,
                rear_brake_pressure_psi=1.0e9,
                steering_angle_rad=atan(
                    curvature_per_m * candidate.chassis.wheelbase_m
                ),
            )
            try:
                candidate.update_state(maximum_braking, cell_length_m)
            except ValueError as error:
                if "stops before the end of the cell" in str(error):
                    return -(next_speed_mps**2)
                raise
            return candidate.speed_mps**2 - next_speed_mps**2

        lower_speed_mps = min(next_speed_mps, local_speed_limit_mps)
        upper_speed_mps = local_speed_limit_mps
        if residual(upper_speed_mps) <= 0.0:
            return upper_speed_mps
        for _ in range(40):
            candidate_speed_mps = 0.5 * (lower_speed_mps + upper_speed_mps)
            if residual(candidate_speed_mps) <= 0.0:
                lower_speed_mps = candidate_speed_mps
            else:
                upper_speed_mps = candidate_speed_mps
        return lower_speed_mps

    def _steady_state_speed_limit(
        self, vehicle: Vehicle, curvature_per_m: float
    ) -> float:
        vehicle_speed_limit_mps = vehicle.drivetrain.vehicle_speed_limit_mps
        absolute_curvature_per_m = abs(curvature_per_m)
        if absolute_curvature_per_m <= 1e-15:
            return vehicle_speed_limit_mps

        lower_speed_mps = 0.0
        upper_speed_mps = vehicle_speed_limit_mps
        for _ in range(50):
            candidate_speed_mps = 0.5 * (lower_speed_mps + upper_speed_mps)
            aero_forces = vehicle.aero.forces_n(
                candidate_speed_mps, self.air_density_kgpm3
            )
            tire_normal_loads = vehicle.suspension.tire_normal_loads_n(
                vehicle.mass_kg,
                self.gravity_mps2,
                aero_forces,
                vehicle.chassis,
                lateral_acceleration_mps2=(
                    candidate_speed_mps**2 * absolute_curvature_per_m
                ),
            )
            available_lateral_force_n = sum(
                vehicle.tire.lateral_force_capacity_n(normal_load_n)
                for normal_load_n in tire_normal_loads.all_n
            )
            required_lateral_force_n = (
                vehicle.mass_kg * candidate_speed_mps**2 * absolute_curvature_per_m
            )
            if required_lateral_force_n <= available_lateral_force_n:
                lower_speed_mps = candidate_speed_mps
            else:
                upper_speed_mps = candidate_speed_mps
        return lower_speed_mps


__all__ = [
    "PathConstraintSolver",
    "PathConstraintViolation",
    "PathSpeedConstraints",
]
