"""Save separate longitudinal and lateral tire-friction line plots."""

from __future__ import annotations

import argparse
from math import isfinite, radians
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_model.environment import STANDARD_GRAVITY_MPS2  # noqa: E402
from vehicle_model.mech.chassis import (  # noqa: E402
    DEFAULT_STATIC_FRONT_WEIGHT_FRACTION,
)
from vehicle_model.mech.pacejka import (  # noqa: E402
    Pacejka52UpcR20LateralModel,
    Pacejka52UpcR20LongitudinalModel,
)
from vehicle_model.vehicle import DEFAULT_MASS_KG  # noqa: E402


DEFAULT_REAR_TIRE_LOAD_N = (
    DEFAULT_MASS_KG
    * STANDARD_GRAVITY_MPS2
    * (1.0 - DEFAULT_STATIC_FRONT_WEIGHT_FRACTION)
    / 2.0
)
PSI_TO_PA = 6_894.757293168
CAMBER_DEG = np.linspace(-4.0, 4.0, 81)
SLIP_RATIO_PERCENT = np.linspace(-20.0, 20.0, 301)
SLIP_ANGLE_DEG = np.linspace(-15.0, 15.0, 301)


def friction_grid(
    model: Pacejka52UpcR20LongitudinalModel | Pacejka52UpcR20LateralModel,
    *,
    normal_load_n: float,
    pressure_psi: float,
    slip_values: np.ndarray,
    slip_to_model_units: float,
) -> np.ndarray:
    """Return signed Fx/Fz or Fy/Fz, with camber as rows and slip as columns."""
    return np.array(
        [
            [
                model.force_n(
                    normal_load_n,
                    float(slip) * slip_to_model_units,
                    camber_angle_rad=radians(float(camber)),
                    inflation_pressure_pa=pressure_psi * PSI_TO_PA,
                )
                / normal_load_n
                for slip in slip_values
            ]
            for camber in CAMBER_DEG
        ]
    )


def new_figure(title: str, normal_load_n: float, pressure_psi: float):
    figure, axis = plt.subplots(figsize=(9, 6))
    figure.suptitle(
        f"{title}\nFz = {normal_load_n:.0f} N per tire, "
        f"pressure = {pressure_psi:g} psi"
    )
    axis.grid(True, alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return figure, axis


def save_figure(figure, path: Path, note: str) -> None:
    figure.text(0.5, 0.015, note, ha="center", fontsize=8)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(path.resolve())


def plot_slip_curves(
    *,
    slip_values: np.ndarray,
    mu: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    note: str,
    normal_load_n: float,
    pressure_psi: float,
    output_path: Path,
) -> None:
    figure, axis = new_figure(title, normal_load_n, pressure_psi)
    selected_rows = np.linspace(0, len(CAMBER_DEG) - 1, 5, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected_rows)))
    for row, color in zip(selected_rows, colors, strict=True):
        axis.plot(
            slip_values,
            mu[row],
            color=color,
            linewidth=2,
            label=f"{CAMBER_DEG[row]:+g}°",
        )
    axis.axhline(0, color="0.4", linewidth=0.8)
    axis.axvline(0, color="0.4", linewidth=0.8)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.legend(frameon=False, title="Camber")
    save_figure(figure, output_path, note)


def plot_camber_peaks(
    *,
    mu: np.ndarray,
    title: str,
    positive_label: str,
    negative_label: str,
    note: str,
    normal_load_n: float,
    pressure_psi: float,
    output_path: Path,
) -> None:
    figure, axis = new_figure(title, normal_load_n, pressure_psi)
    positive_peaks = np.max(mu, axis=1)
    negative_peaks = -np.min(mu, axis=1)
    axis.plot(
        CAMBER_DEG,
        positive_peaks,
        color="#c44e52",
        linewidth=2.2,
        label=positive_label,
    )
    axis.plot(
        CAMBER_DEG,
        negative_peaks,
        color="#4c72b0",
        linewidth=2.2,
        label=negative_label,
    )
    lower = float(min(np.min(positive_peaks), np.min(negative_peaks)))
    upper = float(max(np.max(positive_peaks), np.max(negative_peaks)))
    padding = max(0.03, 0.15 * (upper - lower))
    axis.set(
        xlabel="Camber [deg]",
        ylabel="Peak |μ|",
        ylim=(lower - padding, upper + padding),
    )
    axis.legend(frameon=False)
    save_figure(figure, output_path, note)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normal-load-n",
        type=float,
        default=DEFAULT_REAR_TIRE_LOAD_N,
        help="Load on one tire [N]; default is the vehicle's static rear-tire load",
    )
    parser.add_argument("--pressure-psi", type=float, default=10.0)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/tires"
    )
    args = parser.parse_args()
    if not isfinite(args.normal_load_n) or not 0 < args.normal_load_n <= 1_800:
        parser.error("--normal-load-n must be within (0, 1800] N")
    if not isfinite(args.pressure_psi) or args.pressure_psi <= 0:
        parser.error("--pressure-psi must be positive and finite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    longitudinal_mu = friction_grid(
        Pacejka52UpcR20LongitudinalModel(),
        normal_load_n=args.normal_load_n,
        pressure_psi=args.pressure_psi,
        slip_values=SLIP_RATIO_PERCENT,
        slip_to_model_units=0.01,
    )
    lateral_mu = friction_grid(
        Pacejka52UpcR20LateralModel(),
        normal_load_n=args.normal_load_n,
        pressure_psi=args.pressure_psi,
        slip_values=SLIP_ANGLE_DEG,
        slip_to_model_units=np.pi / 180.0,
    )

    plot_slip_curves(
        slip_values=SLIP_RATIO_PERCENT,
        mu=longitudinal_mu,
        title="Longitudinal μ versus slip ratio",
        xlabel="Slip ratio [%]",
        ylabel="Signed μx = Fx / Fz",
        note="UPC pure-longitudinal fit; its coefficients are an engineering estimate for the R20.",
        normal_load_n=args.normal_load_n,
        pressure_psi=args.pressure_psi,
        output_path=args.output_dir / "longitudinal_mu_vs_slip_ratio.png",
    )
    plot_camber_peaks(
        mu=longitudinal_mu,
        title="Peak longitudinal μ versus camber",
        positive_label="Drive (+Fx)",
        negative_label="Brake (−Fx magnitude)",
        note="Peak over slip ratios from −20% to +20%; UPC longitudinal coefficients are an R20 estimate.",
        normal_load_n=args.normal_load_n,
        pressure_psi=args.pressure_psi,
        output_path=args.output_dir / "longitudinal_mu_vs_camber.png",
    )
    plot_slip_curves(
        slip_values=SLIP_ANGLE_DEG,
        mu=lateral_mu,
        title="Lateral μ versus slip angle",
        xlabel="Slip angle [deg]",
        ylabel="Signed μy = Fy / Fz",
        note="UPC pure-lateral R20 fit; positive slip angle produces restoring (negative) Fy.",
        normal_load_n=args.normal_load_n,
        pressure_psi=args.pressure_psi,
        output_path=args.output_dir / "lateral_mu_vs_slip_angle.png",
    )
    plot_camber_peaks(
        mu=lateral_mu,
        title="Peak lateral μ versus camber",
        positive_label="+Fy magnitude",
        negative_label="−Fy magnitude",
        note="Directional peak magnitudes over slip angles from −15° to +15°; UPC pure-lateral R20 fit.",
        normal_load_n=args.normal_load_n,
        pressure_psi=args.pressure_psi,
        output_path=args.output_dir / "lateral_mu_vs_camber.png",
    )


if __name__ == "__main__":
    main()
