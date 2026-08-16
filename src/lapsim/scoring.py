"""Pluggable endurance scoring models and event-data presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class EnduranceRunSummary(Protocol):
    completed_laps: int
    driving_time_s: float
    pack_energy_kwh: float
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Transparent score plus every intermediate eligibility quantity."""

    endurance_time_points: float
    endurance_lap_points: float
    endurance_points: float
    efficiency_points: float
    combined_points: float
    average_lap_time_s: float | None
    adjusted_energy_per_lap_kg: float | None
    efficiency_factor: float | None
    endurance_time_eligible: bool
    efficiency_time_eligible: bool
    efficiency_energy_eligible: bool
    efficiency_completion_eligible: bool


@runtime_checkable
class ScoringModel(Protocol):
    def score(self, run: EnduranceRunSummary) -> ScoreBreakdown: ...


@dataclass(frozen=True, slots=True)
class FSAEEnduranceEfficiencyScoring:
    """Data-driven Formula SAE endurance plus efficiency scoring.

    No competition constants are hidden in the solver. Construct another
    instance for another event, or supply an entirely different
    :class:`ScoringModel` implementation.
    """

    event_laps: int
    endurance_minimum_time_s: float
    endurance_maximum_time_s: float
    fastest_average_lap_time_s: float
    minimum_adjusted_energy_per_lap_kg: float
    maximum_adjusted_energy_per_lap_kg: float
    efficiency_factor_minimum: float
    efficiency_factor_maximum: float
    electric_energy_to_adjusted_kg_per_kwh: float = 0.65
    maximum_endurance_time_points: float = 250.0
    maximum_efficiency_points: float = 100.0
    driver_change_lap: int = 11

    def __post_init__(self) -> None:
        positive = (
            self.event_laps,
            self.endurance_minimum_time_s,
            self.endurance_maximum_time_s,
            self.fastest_average_lap_time_s,
            self.minimum_adjusted_energy_per_lap_kg,
            self.maximum_adjusted_energy_per_lap_kg,
            self.electric_energy_to_adjusted_kg_per_kwh,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Scoring reference values must be positive")
        if self.endurance_maximum_time_s <= self.endurance_minimum_time_s:
            raise ValueError("Endurance maximum time must exceed minimum time")
        if (
            self.maximum_adjusted_energy_per_lap_kg
            <= self.minimum_adjusted_energy_per_lap_kg
        ):
            raise ValueError("Maximum adjusted energy must exceed minimum")
        if self.efficiency_factor_maximum <= self.efficiency_factor_minimum:
            raise ValueError("Efficiency-factor maximum must exceed minimum")
        if not 1 <= self.driver_change_lap <= self.event_laps:
            raise ValueError("driver_change_lap must lie within the event")

    def endurance_lap_points(self, completed_laps: int) -> float:
        laps = min(max(int(completed_laps), 0), self.event_laps)
        points = laps + 1
        if laps >= self.driver_change_lap:
            points += 2
        return float(min(points, 25))

    def score(self, run: EnduranceRunSummary) -> ScoreBreakdown:
        completed_laps = min(max(int(run.completed_laps), 0), self.event_laps)
        lap_points = self.endurance_lap_points(completed_laps)
        average_lap_time_s = (
            run.driving_time_s / completed_laps if completed_laps > 0 else None
        )
        adjusted_energy_per_lap_kg = (
            run.pack_energy_kwh
            * self.electric_energy_to_adjusted_kg_per_kwh
            / completed_laps
            if completed_laps > 0
            else None
        )

        endurance_time_eligible = (
            completed_laps == self.event_laps
            and run.failure_reason is None
            and run.driving_time_s <= self.endurance_maximum_time_s
        )
        endurance_time_points = 0.0
        if endurance_time_eligible:
            numerator = self.endurance_maximum_time_s / run.driving_time_s - 1.0
            denominator = (
                self.endurance_maximum_time_s / self.endurance_minimum_time_s - 1.0
            )
            endurance_time_points = self.maximum_endurance_time_points * min(
                max(numerator / denominator, 0.0), 1.0
            )

        # The rules explicitly permit partial-completion efficiency scoring
        # once the vehicle crosses the start line after driver change.
        efficiency_completion_eligible = completed_laps >= self.driver_change_lap
        maximum_average_lap_time_s = 1.45 * self.fastest_average_lap_time_s
        efficiency_time_eligible = (
            average_lap_time_s is not None
            and average_lap_time_s <= maximum_average_lap_time_s
        )
        efficiency_energy_eligible = (
            adjusted_energy_per_lap_kg is not None
            and adjusted_energy_per_lap_kg <= self.maximum_adjusted_energy_per_lap_kg
        )

        efficiency_factor: float | None = None
        efficiency_points = 0.0
        if (
            efficiency_completion_eligible
            and efficiency_time_eligible
            and efficiency_energy_eligible
            and adjusted_energy_per_lap_kg is not None
            and adjusted_energy_per_lap_kg > 0.0
            and average_lap_time_s is not None
        ):
            efficiency_factor = (
                self.fastest_average_lap_time_s / average_lap_time_s
            ) * (self.minimum_adjusted_energy_per_lap_kg / adjusted_energy_per_lap_kg)
            normalized = (efficiency_factor - self.efficiency_factor_minimum) / (
                self.efficiency_factor_maximum - self.efficiency_factor_minimum
            )
            efficiency_points = self.maximum_efficiency_points * min(
                max(normalized, 0.0), 1.0
            )

        endurance_points = endurance_time_points + lap_points
        return ScoreBreakdown(
            endurance_time_points=endurance_time_points,
            endurance_lap_points=lap_points,
            endurance_points=endurance_points,
            efficiency_points=efficiency_points,
            combined_points=endurance_points + efficiency_points,
            average_lap_time_s=average_lap_time_s,
            adjusted_energy_per_lap_kg=adjusted_energy_per_lap_kg,
            efficiency_factor=efficiency_factor,
            endurance_time_eligible=endurance_time_eligible,
            efficiency_time_eligible=efficiency_time_eligible,
            efficiency_energy_eligible=efficiency_energy_eligible,
            efficiency_completion_eligible=efficiency_completion_eligible,
        )


FSAE_2026_MI6_SCORING = FSAEEnduranceEfficiencyScoring(
    event_laps=22,
    endurance_minimum_time_s=1312.281,
    endurance_maximum_time_s=1902.808,
    fastest_average_lap_time_s=59.649,
    minimum_adjusted_energy_per_lap_kg=0.0840,
    maximum_adjusted_energy_per_lap_kg=0.2002,
    efficiency_factor_minimum=0.289,
    efficiency_factor_maximum=0.797,
)


__all__ = [
    "EnduranceRunSummary",
    "FSAEEnduranceEfficiencyScoring",
    "FSAE_2026_MI6_SCORING",
    "ScoreBreakdown",
    "ScoringModel",
]
