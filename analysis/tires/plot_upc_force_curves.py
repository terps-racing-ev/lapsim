"""Generate static UPC Hoosier R20 pure-slip force analysis plots."""

from __future__ import annotations

import argparse
from math import radians
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vehicle_model import (  # noqa: E402
    Pacejka52UpcR20LateralModel,
    Pacejka52UpcR20LongitudinalModel,
)


PSI_TO_PA = 6_894.757293168
PRESSURES_PSI = np.array([6.0, 8.0, 10.0, 12.0, 14.0])
CAMBERS_DEG = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
ACTIVE_PRESSURE_PSI = 10.0
ACTIVE_CAMBER_DEG = -1.0
SLIP_ANGLES_DEG = np.linspace(-15.0, 15.0, 241)
SLIP_RATIOS = np.linspace(-0.20, 0.20, 241)


def _style_axis(axis: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.axvline(0.0, color="0.35", linewidth=0.8)
    axis.grid(True, alpha=0.28)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _pressure_label(pressure_psi: float) -> str:
    suffix = " (active)" if pressure_psi == ACTIVE_PRESSURE_PSI else ""
    return f"{pressure_psi:g} psi{suffix}"


def _camber_label(camber_deg: float) -> str:
    suffix = " (active)" if camber_deg == ACTIVE_CAMBER_DEG else ""
    return f"{camber_deg:+g} deg{suffix}"


def _finalize_figure(figure: plt.Figure, output_path: Path, note: str) -> None:
    figure.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_lateral(
    model: Pacejka52UpcR20LateralModel,
    *,
    normal_load_n: float,
    output_path: Path,
) -> None:
    """Plot Fy against slip angle and its pressure/camber peak sweeps."""

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    pressure_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(PRESSURES_PSI)))
    camber_colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(CAMBERS_DEG)))

    for color, pressure_psi in zip(pressure_colors, PRESSURES_PSI, strict=True):
        force_n = np.array(
            [
                model.force_n(
                    normal_load_n,
                    radians(angle_deg),
                    camber_angle_rad=radians(ACTIVE_CAMBER_DEG),
                    inflation_pressure_pa=pressure_psi * PSI_TO_PA,
                )
                for angle_deg in SLIP_ANGLES_DEG
            ]
        )
        axes[0, 0].plot(
            SLIP_ANGLES_DEG,
            force_n,
            color=color,
            linewidth=2.4 if pressure_psi == ACTIVE_PRESSURE_PSI else 1.5,
            label=_pressure_label(pressure_psi),
        )

    for color, camber_deg in zip(camber_colors, CAMBERS_DEG, strict=True):
        force_n = np.array(
            [
                model.force_n(
                    normal_load_n,
                    radians(angle_deg),
                    camber_angle_rad=radians(camber_deg),
                    inflation_pressure_pa=ACTIVE_PRESSURE_PSI * PSI_TO_PA,
                )
                for angle_deg in SLIP_ANGLES_DEG
            ]
        )
        axes[0, 1].plot(
            SLIP_ANGLES_DEG,
            force_n,
            color=color,
            linewidth=2.4 if camber_deg == ACTIVE_CAMBER_DEG else 1.5,
            label=_camber_label(camber_deg),
        )

    pressure_peaks_n = []
    for pressure_psi in PRESSURES_PSI:
        forces_n = np.array(
            [
                model.force_n(
                    normal_load_n,
                    radians(angle_deg),
                    camber_angle_rad=radians(ACTIVE_CAMBER_DEG),
                    inflation_pressure_pa=pressure_psi * PSI_TO_PA,
                )
                for angle_deg in SLIP_ANGLES_DEG
            ]
        )
        pressure_peaks_n.append(np.max(np.abs(forces_n)))
    axes[1, 0].plot(
        PRESSURES_PSI,
        pressure_peaks_n,
        "o-",
        color="#1f77b4",
        linewidth=1.8,
        label=f"camber {ACTIVE_CAMBER_DEG:+g} deg",
    )
    axes[1, 0].scatter(
        [ACTIVE_PRESSURE_PSI],
        [pressure_peaks_n[2]],
        color="#d62728",
        zorder=3,
        label="active point",
    )

    camber_peaks_n = []
    for camber_deg in CAMBERS_DEG:
        forces_n = np.array(
            [
                model.force_n(
                    normal_load_n,
                    radians(angle_deg),
                    camber_angle_rad=radians(camber_deg),
                    inflation_pressure_pa=ACTIVE_PRESSURE_PSI * PSI_TO_PA,
                )
                for angle_deg in SLIP_ANGLES_DEG
            ]
        )
        camber_peaks_n.append(np.max(np.abs(forces_n)))
    axes[1, 1].plot(
        CAMBERS_DEG,
        camber_peaks_n,
        "o-",
        color="#9467bd",
        linewidth=1.8,
        label=f"pressure {ACTIVE_PRESSURE_PSI:g} psi",
    )
    axes[1, 1].scatter(
        [ACTIVE_CAMBER_DEG],
        [camber_peaks_n[1]],
        color="#d62728",
        zorder=3,
        label="active point",
    )

    _style_axis(
        axes[0, 0],
        "Slip-angle curve: pressure sweep",
        "Slip angle [deg]",
        "Lateral force Fy [N]",
    )
    _style_axis(
        axes[0, 1],
        "Slip-angle curve: camber sweep",
        "Slip angle [deg]",
        "Lateral force Fy [N]",
    )
    _style_axis(
        axes[1, 0],
        "Peak lateral force: pressure sweep",
        "Inflation pressure [psi]",
        "Peak |Fy| [N]",
    )
    _style_axis(
        axes[1, 1],
        "Peak lateral force: camber sweep",
        "Camber [deg]",
        "Peak |Fy| [N]",
    )
    for axis in axes.flat:
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle(
        f"Hoosier 16x7.5-10 R20 UPC lateral force sweeps (Fz = {normal_load_n:g} N)",
        fontsize=14,
    )
    _finalize_figure(
        figure,
        output_path,
        "Pure lateral MF-Tyre model; active setup is 10 psi and -1 deg camber.",
    )


def plot_longitudinal(
    model: Pacejka52UpcR20LongitudinalModel,
    *,
    normal_load_n: float,
    output_path: Path,
) -> None:
    """Plot Fx against slip ratio and its pressure/camber peak sweeps."""

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    pressure_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(PRESSURES_PSI)))
    camber_colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(CAMBERS_DEG)))

    for color, pressure_psi in zip(pressure_colors, PRESSURES_PSI, strict=True):
        force_n = np.array(
            [
                model.force_n(
                    normal_load_n,
                    slip_ratio,
                    camber_angle_rad=radians(ACTIVE_CAMBER_DEG),
                    inflation_pressure_pa=pressure_psi * PSI_TO_PA,
                )
                for slip_ratio in SLIP_RATIOS
            ]
        )
        axes[0, 0].plot(
            100.0 * SLIP_RATIOS,
            force_n,
            color=color,
            linewidth=2.4 if pressure_psi == ACTIVE_PRESSURE_PSI else 1.5,
            label=_pressure_label(pressure_psi),
        )

    for color, camber_deg in zip(camber_colors, CAMBERS_DEG, strict=True):
        force_n = np.array(
            [
                model.force_n(
                    normal_load_n,
                    slip_ratio,
                    camber_angle_rad=radians(camber_deg),
                    inflation_pressure_pa=ACTIVE_PRESSURE_PSI * PSI_TO_PA,
                )
                for slip_ratio in SLIP_RATIOS
            ]
        )
        axes[0, 1].plot(
            100.0 * SLIP_RATIOS,
            force_n,
            color=color,
            linewidth=2.4 if camber_deg == ACTIVE_CAMBER_DEG else 1.5,
            label=_camber_label(camber_deg),
        )

    pressure_peaks_n = []
    for pressure_psi in PRESSURES_PSI:
        forces_n = np.array(
            [
                model.force_n(
                    normal_load_n,
                    slip_ratio,
                    camber_angle_rad=radians(ACTIVE_CAMBER_DEG),
                    inflation_pressure_pa=pressure_psi * PSI_TO_PA,
                )
                for slip_ratio in SLIP_RATIOS
            ]
        )
        pressure_peaks_n.append(np.max(np.abs(forces_n)))
    axes[1, 0].plot(
        PRESSURES_PSI,
        pressure_peaks_n,
        "o-",
        color="#1f77b4",
        linewidth=1.8,
        label=f"camber {ACTIVE_CAMBER_DEG:+g} deg",
    )
    axes[1, 0].scatter(
        [ACTIVE_PRESSURE_PSI],
        [pressure_peaks_n[2]],
        color="#d62728",
        zorder=3,
        label="active point",
    )

    camber_peaks_n = []
    for camber_deg in CAMBERS_DEG:
        forces_n = np.array(
            [
                model.force_n(
                    normal_load_n,
                    slip_ratio,
                    camber_angle_rad=radians(camber_deg),
                    inflation_pressure_pa=ACTIVE_PRESSURE_PSI * PSI_TO_PA,
                )
                for slip_ratio in SLIP_RATIOS
            ]
        )
        camber_peaks_n.append(np.max(np.abs(forces_n)))
    axes[1, 1].plot(
        CAMBERS_DEG,
        camber_peaks_n,
        "o-",
        color="#9467bd",
        linewidth=1.8,
        label=f"pressure {ACTIVE_PRESSURE_PSI:g} psi",
    )
    axes[1, 1].scatter(
        [ACTIVE_CAMBER_DEG],
        [camber_peaks_n[1]],
        color="#d62728",
        zorder=3,
        label="active point",
    )

    _style_axis(
        axes[0, 0],
        "Slip-ratio curve: pressure sweep",
        "Slip ratio [%]",
        "Longitudinal force Fx [N]",
    )
    _style_axis(
        axes[0, 1],
        "Slip-ratio curve: camber sweep",
        "Slip ratio [%]",
        "Longitudinal force Fx [N]",
    )
    _style_axis(
        axes[1, 0],
        "Peak longitudinal force: pressure sweep",
        "Inflation pressure [psi]",
        "Peak |Fx| [N]",
    )
    _style_axis(
        axes[1, 1],
        "Peak longitudinal force: camber sweep",
        "Camber [deg]",
        "Peak |Fx| [N]",
    )
    for axis in axes.flat:
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle(
        "Hoosier 16x7.5-10 R20 UPC longitudinal force sweeps "
        f"(Fz = {normal_load_n:g} N)",
        fontsize=14,
    )
    _finalize_figure(
        figure,
        output_path,
        "Pure longitudinal MF-Tyre model; active setup is 10 psi and -1 deg camber.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-load-n", type=float, default=800.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/tires",
    )
    args = parser.parse_args()
    if args.normal_load_n <= 0.0:
        parser.error("--normal-load-n must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lateral_path = args.output_dir / "upc_r20_lateral_force_sweeps.png"
    longitudinal_path = args.output_dir / "upc_r20_longitudinal_force_sweeps.png"
    plot_lateral(
        Pacejka52UpcR20LateralModel(),
        normal_load_n=args.normal_load_n,
        output_path=lateral_path,
    )
    plot_longitudinal(
        Pacejka52UpcR20LongitudinalModel(),
        normal_load_n=args.normal_load_n,
        output_path=longitudinal_path,
    )
    print(lateral_path.resolve())
    print(longitudinal_path.resolve())


if __name__ == "__main__":
    main()
