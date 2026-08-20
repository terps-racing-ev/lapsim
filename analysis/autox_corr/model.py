"""Fit and evaluate a two-input linear ratio model for autocross score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LinearRatioModel:
    """Linear model in normalized event-score ratios.

    The model is fit in the form::

        autocross_ratio = intercept_ratio
                       + w_acceleration * acceleration_ratio
                       + w_skidpad * skidpad_ratio

    where each ratio is a score divided by that event's maximum score.  The
    public ``predict`` method optionally clips the result to the legal
    autocross score range.
    """

    acceleration_weight: float
    skidpad_weight: float
    intercept_ratio: float = 0.0
    acceleration_max_score: float = 100.0
    skidpad_max_score: float = 75.0
    autocross_max_score: float = 125.0

    def predict_raw(
        self,
        acceleration_score: float | np.ndarray,
        skidpad_score: float | np.ndarray,
    ) -> float | np.ndarray:
        """Return the unbounded predicted autocross score."""

        acceleration = np.asarray(acceleration_score, dtype=float)
        skidpad = np.asarray(skidpad_score, dtype=float)
        prediction = self.autocross_max_score * (
            self.intercept_ratio
            + self.acceleration_weight * acceleration / self.acceleration_max_score
            + self.skidpad_weight * skidpad / self.skidpad_max_score
        )
        return float(prediction) if prediction.ndim == 0 else prediction

    def predict(
        self,
        acceleration_score: float | np.ndarray,
        skidpad_score: float | np.ndarray,
        *,
        clip: bool = True,
    ) -> float | np.ndarray:
        """Return a predicted autocross score, clipped by default to 0-125."""

        prediction = np.asarray(
            self.predict_raw(acceleration_score, skidpad_score), dtype=float
        )
        if clip:
            prediction = np.clip(prediction, 0.0, self.autocross_max_score)
        return float(prediction) if prediction.ndim == 0 else prediction


def fit_linear_ratio_model(
    acceleration_score: np.ndarray,
    skidpad_score: np.ndarray,
    autocross_score: np.ndarray,
    *,
    acceleration_max_score: float = 100.0,
    skidpad_max_score: float = 75.0,
    autocross_max_score: float = 125.0,
    include_intercept: bool = True,
) -> LinearRatioModel:
    """Fit a least-squares model in normalized score ratios."""

    acceleration = np.asarray(acceleration_score, dtype=float)
    skidpad = np.asarray(skidpad_score, dtype=float)
    autocross = np.asarray(autocross_score, dtype=float)
    if not (acceleration.ndim == skidpad.ndim == autocross.ndim == 1):
        raise ValueError("all score inputs must be one-dimensional")
    if not (len(acceleration) == len(skidpad) == len(autocross)):
        raise ValueError("all score inputs must have the same length")
    if len(autocross) < 2:
        raise ValueError("at least two complete teams are required")

    predictors = np.column_stack(
        (
            acceleration / acceleration_max_score,
            skidpad / skidpad_max_score,
        )
    )
    design = (
        np.column_stack((np.ones(len(predictors)), predictors))
        if include_intercept
        else predictors
    )
    weights, *_ = np.linalg.lstsq(
        design,
        autocross / autocross_max_score,
        rcond=None,
    )
    return LinearRatioModel(
        acceleration_weight=float(weights[-2]),
        skidpad_weight=float(weights[-1]),
        intercept_ratio=float(weights[0]) if include_intercept else 0.0,
        acceleration_max_score=acceleration_max_score,
        skidpad_max_score=skidpad_max_score,
        autocross_max_score=autocross_max_score,
    )
