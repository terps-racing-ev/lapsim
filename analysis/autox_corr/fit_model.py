"""Fit the EV 2026 autocross correlation model and write diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:  # Supports both ``python -m analysis.autox_corr.fit_model`` and direct use.
    from .model import fit_linear_ratio_model
except ImportError:  # pragma: no cover - only used for direct script execution.
    from model import fit_linear_ratio_model


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.csv"
OUTPUT_PATH = ROOT / "outputs"


def load_data(path: Path = DATA_PATH) -> list[dict[str, str]]:
    """Load the complete-score rows copied from the PDF overall results."""

    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    total_sum_squares = np.sum((actual - np.mean(actual)) ** 2)
    return {
        "r2": float(1.0 - np.sum(residual**2) / total_sum_squares),
        "rmse_points": float(np.sqrt(np.mean(residual**2))),
        "mae_points": float(np.mean(np.abs(residual))),
    }


def main() -> None:
    rows = load_data()
    acceleration = np.array([float(row["acceleration_score"]) for row in rows])
    skidpad = np.array([float(row["skidpad_score"]) for row in rows])
    actual = np.array([float(row["autocross_score"]) for row in rows])

    model = fit_linear_ratio_model(acceleration, skidpad, actual)
    raw_prediction = np.asarray(model.predict(acceleration, skidpad, clip=False))
    clipped_prediction = np.asarray(model.predict(acceleration, skidpad, clip=True))

    OUTPUT_PATH.mkdir(exist_ok=True)
    predictions_path = OUTPUT_PATH / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "overall_place",
                "car_number",
                "team_name",
                "acceleration_score",
                "skidpad_score",
                "actual_autocross_score",
                "predicted_autocross_score",
                "predicted_autocross_score_clipped",
                "residual_actual_minus_raw",
            ]
        )
        for row, raw, clipped in zip(rows, raw_prediction, clipped_prediction):
            writer.writerow(
                [
                    row["overall_place"],
                    row["car_number"],
                    row["team_name"],
                    row["acceleration_score"],
                    row["skidpad_score"],
                    row["autocross_score"],
                    f"{raw:.6f}",
                    f"{clipped:.6f}",
                    f"{float(row['autocross_score']) - raw:.6f}",
                ]
            )

    summary = {
        "source": {
            "file": "FSAE_2026_MI6_results.pdf",
            "table": "Formula SAE Electric 2026 Overall Results",
            "pages": "1-4",
            "complete_team_count": len(rows),
            "zero_autocross_score_count": int(np.count_nonzero(actual == 0.0)),
        },
        "model": {
            "type": "linear_ratio_with_intercept",
            "equation": (
                "clip(125 * (intercept_ratio + "
                "w_accel * acceleration_score / 100 + "
                "w_skidpad * skidpad_score / 75), 0, 125)"
            ),
            "intercept_ratio": model.intercept_ratio,
            "intercept_points": model.intercept_ratio * model.autocross_max_score,
            "acceleration_weight": model.acceleration_weight,
            "skidpad_weight": model.skidpad_weight,
            "acceleration_max_score": model.acceleration_max_score,
            "skidpad_max_score": model.skidpad_max_score,
            "autocross_max_score": model.autocross_max_score,
        },
        "fit_metrics_raw_prediction": _metric_summary(actual, raw_prediction),
        "fit_metrics_clipped_prediction": _metric_summary(
            actual, clipped_prediction
        ),
    }
    summary_path = OUTPUT_PATH / "model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    top10_indices = np.argsort(actual)[-10:]
    top10_mask = np.zeros(len(rows), dtype=bool)
    top10_mask[top10_indices] = True
    maryland_mask = np.array(["Maryland" in row["team_name"] for row in rows])
    other_mask = ~(top10_mask | maryland_mask)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.scatter(
        actual[other_mask],
        clipped_prediction[other_mask],
        color="#9e9e9e",
        alpha=0.75,
        edgecolor="white",
        label="Other teams",
    )
    axis.scatter(
        actual[top10_mask & ~maryland_mask],
        clipped_prediction[top10_mask & ~maryland_mask],
        color="#1565c0",
        alpha=0.9,
        edgecolor="white",
        label="Top 10 observed autocross scores",
    )
    axis.scatter(
        actual[maryland_mask],
        clipped_prediction[maryland_mask],
        color="#d84315",
        marker="*",
        s=150,
        edgecolor="black",
        linewidth=0.7,
        label="Univ. of Maryland",
        zorder=4,
    )
    for observed, predicted in zip(
        actual[maryland_mask], clipped_prediction[maryland_mask]
    ):
        axis.annotate(
            "Maryland",
            (observed, predicted),
            xytext=(7, 7),
            textcoords="offset points",
            color="#8d2b0e",
            fontsize=9,
        )
    axis.plot(
        [0.0, 125.0],
        [0.0, 125.0],
        color="#555555",
        linestyle="--",
        linewidth=1.0,
    )
    axis.set(
        title="EV 2026 autocross score: observed vs predicted",
        xlabel="Observed autocross score",
        ylabel="Predicted autocross score",
        xlim=(0.0, 125.0),
        ylim=(0.0, 125.0),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH / "autocross_prediction.png", dpi=160)
    plt.close(figure)

    print(f"Teams used: {len(rows)}")
    print(
        "Intercept: "
        f"{model.intercept_ratio * model.autocross_max_score:.6f} autocross points"
    )
    print(f"Acceleration weight: {model.acceleration_weight:.6f}")
    print(f"Skidpad weight: {model.skidpad_weight:.6f}")
    print(
        "Raw fit: "
        f"R^2={summary['fit_metrics_raw_prediction']['r2']:.4f}, "
        f"RMSE={summary['fit_metrics_raw_prediction']['rmse_points']:.2f} points, "
        f"MAE={summary['fit_metrics_raw_prediction']['mae_points']:.2f} points"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {predictions_path}")


if __name__ == "__main__":
    main()
