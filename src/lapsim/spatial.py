"""Distance-indexed channels for prescribed-path replay and validation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

DISTANCE_JITTER_TOLERANCE_M = 0.02


@dataclass(frozen=True, slots=True)
class SpatialCoordinate:
    """A finite, non-decreasing distance coordinate with duplicate points removed.

    GNSS distance is occasionally repeated or moves backwards by a few
    centimetres.  Repeated samples and reversals no larger than
    ``DISTANCE_JITTER_TOLERANCE_M`` are collapsed; a material reversal is
    rejected rather than silently changing the recorded path.
    """

    distance_m: np.ndarray
    source_indices: np.ndarray

    @classmethod
    def from_samples(cls, distance_m: Sequence[float]) -> "SpatialCoordinate":
        raw_distance_m = np.asarray(distance_m, dtype=float)
        if raw_distance_m.ndim != 1 or raw_distance_m.size < 2:
            raise ValueError("Spatial distance requires at least two samples")
        if not np.all(np.isfinite(raw_distance_m)):
            raise ValueError("Spatial distance must be finite")

        normalized_distance_m = raw_distance_m - raw_distance_m[0]
        if np.any(np.diff(normalized_distance_m) < -DISTANCE_JITTER_TOLERANCE_M):
            raise ValueError("Spatial distance materially decreases")
        monotonic_distance_m = np.maximum.accumulate(normalized_distance_m)
        change_indices = np.flatnonzero(np.diff(monotonic_distance_m) > 1e-9)
        # Each group ends immediately before the next change.  Keeping that
        # final sample is appropriate for a zero-order-held CAN channel.
        source_indices = np.concatenate(
            (change_indices, np.asarray([len(monotonic_distance_m) - 1]))
        )
        compact_distance_m = monotonic_distance_m[source_indices]
        if compact_distance_m.size < 2:
            raise ValueError("Spatial distance does not advance")
        return cls(
            distance_m=compact_distance_m,
            source_indices=source_indices,
        )

    def values(self, samples: Sequence[float]) -> np.ndarray:
        """Return samples corresponding to the unique spatial coordinate."""

        values = np.asarray(samples, dtype=float)
        if values.ndim != 1 or values.size <= int(self.source_indices[-1]):
            raise ValueError("Spatial channel does not match its source samples")
        selected = values[self.source_indices]
        if not np.all(np.isfinite(selected)):
            raise ValueError("Spatial channel must be finite")
        return selected

    def _query_distance(self, distance_m: float | np.ndarray) -> np.ndarray:
        query_distance_m = np.asarray(distance_m, dtype=float)
        if not np.all(np.isfinite(query_distance_m)):
            raise ValueError("Spatial query distance must be finite")
        lower_bound_m = self.distance_m[0]
        upper_bound_m = self.distance_m[-1]
        if np.any(query_distance_m < lower_bound_m) or np.any(
            query_distance_m > upper_bound_m
        ):
            raise ValueError("Spatial query is outside the recorded domain")
        return query_distance_m

    def interpolate(
        self,
        samples: Sequence[float],
        distance_m: float | np.ndarray,
    ) -> np.ndarray:
        """Linearly interpolate a smooth channel along the racing line."""

        return np.interp(
            self._query_distance(distance_m),
            self.distance_m,
            self.values(samples),
        )

    def zero_order_hold(
        self,
        samples: Sequence[float],
        distance_m: float | np.ndarray,
    ) -> np.ndarray:
        """Sample a CAN-like command without interpolating between updates."""

        query_distance_m = self._query_distance(distance_m)
        indices = (
            np.searchsorted(
                self.distance_m,
                query_distance_m,
                side="right",
            )
            - 1
        )
        indices = np.clip(indices, 0, len(self.distance_m) - 1)
        return self.values(samples)[indices]
