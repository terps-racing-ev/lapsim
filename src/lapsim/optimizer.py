"""Derivative-free optimization of reusable endurance torque profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from vehicle_model.vehicle import Vehicle

from .endurance import EnduranceRunConfig, EnduranceRunResult, EnduranceSimulator
from .path_constraints import PathConstraintSolver, PathSpeedConstraints
from .scoring import ScoreBreakdown, ScoringModel
from .spatial_track import SpatialTrack
from .torque_profile import (
    TorqueProfile,
    TorqueProfileParameterization,
    UniformPeriodicTorqueParameterization,
)


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Numerical budget independent of track, vehicle, and scoring model."""

    maximum_iterations: int = 20
    population_size_multiplier: int = 6
    seed: int = 7
    polish: bool = True
    tolerance: float = 1e-3

    def __post_init__(self) -> None:
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if self.population_size_multiplier <= 0:
            raise ValueError("population_size_multiplier must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True, slots=True)
class LocalPolishConfig:
    """Budget for derivative-free refinement of an existing profile."""

    maximum_evaluations: int = 80
    final_trust_region_radius: float = 1e-3

    def __post_init__(self) -> None:
        if self.maximum_evaluations <= 0:
            raise ValueError("maximum_evaluations must be positive")
        if self.final_trust_region_radius <= 0:
            raise ValueError("final_trust_region_radius must be positive")


@dataclass(frozen=True, slots=True)
class EnduranceOptimizationResult:
    variables: tuple[float, ...]
    profile: TorqueProfile
    run: EnduranceRunResult
    score: ScoreBreakdown
    baseline_run: EnduranceRunResult
    baseline_score: ScoreBreakdown
    constraints: PathSpeedConstraints
    objective_evaluations: int
    optimizer_iterations: int
    optimizer_message: str
    optimizer_success: bool


class EnduranceTorqueOptimizer:
    """Optimize a periodic normalized torque profile for an injected setup.

    ``vehicle_factory`` is deliberately required instead of copying a Vehicle:
    replaceable component models may be stateful or non-copyable, and every
    candidate must begin from a genuinely fresh state. This also makes vehicle
    parameter sweeps a simple matter of supplying another factory.
    """

    def __init__(
        self,
        *,
        vehicle_factory: Callable[[], Vehicle],
        track: SpatialTrack,
        scoring_model: ScoringModel,
        run_config: EnduranceRunConfig,
        parameterization: TorqueProfileParameterization | None = None,
        simulator: EnduranceSimulator | None = None,
        constraint_solver: PathConstraintSolver | None = None,
    ) -> None:
        self.vehicle_factory = vehicle_factory
        self.track = track
        self.scoring_model = scoring_model
        self.run_config = run_config
        self.parameterization = (
            parameterization
            if parameterization is not None
            else UniformPeriodicTorqueParameterization()
        )
        self.simulator = simulator if simulator is not None else EnduranceSimulator()
        self.constraint_solver = (
            constraint_solver
            if constraint_solver is not None
            else PathConstraintSolver()
        )
        if not track.closed:
            raise ValueError("Endurance torque optimization requires a closed track")
        reference_vehicle = self._fresh_vehicle()
        self.constraints = self.constraint_solver.solve(track, reference_vehicle)
        self._objective_evaluations = 0

    def _fresh_vehicle(self) -> Vehicle:
        vehicle = self.vehicle_factory()
        if not isinstance(vehicle, Vehicle):
            raise TypeError("vehicle_factory must return a Vehicle")
        vehicle.validate()
        return vehicle

    def evaluate(
        self,
        variables,
        *,
        record_telemetry: bool = False,
    ) -> tuple[TorqueProfile, EnduranceRunResult, ScoreBreakdown]:
        profile = self.parameterization.build(variables, self.track)
        run = self.simulator.run(
            self._fresh_vehicle(),
            self.constraints,
            profile,
            self.run_config,
            record_telemetry=record_telemetry,
        )
        return profile, run, self.scoring_model.score(run)

    def optimize(
        self, config: OptimizationConfig | None = None
    ) -> EnduranceOptimizationResult:
        optimization = config if config is not None else OptimizationConfig()
        variable_count = self.parameterization.variable_count
        reference_vehicle = self._fresh_vehicle()
        bounds = self.parameterization.bounds(reference_vehicle)
        if len(bounds) != variable_count:
            raise ValueError("Parameterization bounds do not match variable_count")

        baseline_variables = np.ones(variable_count, dtype=float)
        _, baseline_run, baseline_score = self.evaluate(baseline_variables)
        rng = np.random.default_rng(optimization.seed)
        population_count = max(
            5,
            optimization.population_size_multiplier * variable_count,
        )
        initial_population = rng.uniform(
            low=np.asarray([bound[0] for bound in bounds]),
            high=np.asarray([bound[1] for bound in bounds]),
            size=(population_count, variable_count),
        )
        preferred_seed_levels = (
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.60,
            0.75,
        )
        seed_levels = preferred_seed_levels[: max(0, population_count - 1)]
        for row_index, level in enumerate(seed_levels):
            initial_population[row_index, :] = level
        initial_population[-1, :] = baseline_variables

        self._objective_evaluations = 0

        def objective(variables: np.ndarray) -> float:
            self._objective_evaluations += 1
            return self._objective_value(variables)

        scipy_result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=optimization.maximum_iterations,
            popsize=optimization.population_size_multiplier,
            tol=optimization.tolerance,
            seed=optimization.seed,
            polish=optimization.polish,
            init=initial_population,
            workers=1,
            updating="immediate",
        )
        variables = tuple(float(value) for value in scipy_result.x)
        profile, run, score = self.evaluate(variables, record_telemetry=True)
        return EnduranceOptimizationResult(
            variables=variables,
            profile=profile,
            run=run,
            score=score,
            baseline_run=baseline_run,
            baseline_score=baseline_score,
            constraints=self.constraints,
            objective_evaluations=self._objective_evaluations,
            optimizer_iterations=int(scipy_result.nit),
            optimizer_message=str(scipy_result.message),
            optimizer_success=bool(scipy_result.success),
        )

    def polish_from(
        self,
        initial_variables,
        config: LocalPolishConfig | None = None,
    ) -> EnduranceOptimizationResult:
        """Refine a saved profile on this optimizer's track and vehicle setup.

        This is intended for the fine-resolution stage of a coarse-to-fine
        workflow. COBYQA is bounded and derivative-free, which is appropriate
        for the simulator's clipping and automatic-braking active sets.
        """

        polish = config if config is not None else LocalPolishConfig()
        variable_count = self.parameterization.variable_count
        variables = np.asarray(tuple(float(value) for value in initial_variables))
        if variables.shape != (variable_count,):
            raise ValueError(
                f"Expected {variable_count} initial variables, got {variables.size}"
            )
        bounds = self.parameterization.bounds(self._fresh_vehicle())
        for value, (lower, upper) in zip(variables, bounds, strict=True):
            if not lower <= value <= upper:
                raise ValueError("Initial variables must lie within their bounds")

        _, baseline_run, baseline_score = self.evaluate(np.ones(variable_count))
        self._objective_evaluations = 0

        def objective(candidate: np.ndarray) -> float:
            self._objective_evaluations += 1
            return self._objective_value(candidate)

        scipy_result = minimize(
            objective,
            variables,
            method="COBYQA",
            bounds=bounds,
            options={
                "maxfev": polish.maximum_evaluations,
                "final_tr_radius": polish.final_trust_region_radius,
            },
        )
        best_variables = tuple(float(value) for value in scipy_result.x)
        profile, run, score = self.evaluate(best_variables, record_telemetry=True)
        return EnduranceOptimizationResult(
            variables=best_variables,
            profile=profile,
            run=run,
            score=score,
            baseline_run=baseline_run,
            baseline_score=baseline_score,
            constraints=self.constraints,
            objective_evaluations=self._objective_evaluations,
            optimizer_iterations=int(getattr(scipy_result, "nit", 0)),
            optimizer_message=str(scipy_result.message),
            optimizer_success=bool(scipy_result.success),
        )

    def _objective_value(self, variables) -> float:
        _, run, score = self.evaluate(variables)
        missing_laps = self.run_config.laps - run.completed_laps
        if missing_laps > 0:
            # Completion dominates all point tradeoffs. Within incomplete
            # candidates, prefer more completed distance, then points.
            return 10_000.0 + 1_000.0 * missing_laps - score.combined_points
        return -score.combined_points


__all__ = [
    "EnduranceOptimizationResult",
    "EnduranceTorqueOptimizer",
    "LocalPolishConfig",
    "OptimizationConfig",
]
