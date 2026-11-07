"""Plot signed lateral tire friction versus camber and slip angle."""

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
from vehicle_model.mech.pacejka import Pacejka52UpcR20LateralModel  # noqa: E402
from vehicle_model.vehicle import DEFAULT_MASS_KG  # noqa: E402


DEFAULT_REAR_TIRE_LOAD_N = (
    DEFAULT_MASS_KG
    * STANDARD_GRAVITY_MPS2
    * (1.0 - DEFAULT_STATIC_FRONT_WEIGHT_FRACTION)
    / 2.0
)
DEFAULT_OUTPUT = ROOT / "outputs/tires/upc_r20_camber_slip_angle_mu_3d.png"
PSI_TO_PA = 6_894.757293168


def friction_surface(
    model: Pacejka52UpcR20LateralModel,
    *,
    normal_load_n: float,
    pressure_psi: float,
    camber_limit_deg: float,
    slip_angle_limit_deg: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camber [deg], slip angle [deg], and signed mu_y = Fy/Fz grids."""
    camber_deg = np.linspace(-camber_limit_deg, camber_limit_deg, grid_size)
    slip_angle_deg = np.linspace(
        -slip_angle_limit_deg, slip_angle_limit_deg, grid_size
    )
    cambers, slip_angles = np.meshgrid(camber_deg, slip_angle_deg)
    mu = np.fromiter(
        (
            model.force_n(
                normal_load_n,
                radians(slip_angle),
                camber_angle_rad=radians(camber),
                inflation_pressure_pa=pressure_psi * PSI_TO_PA,
            )
            / normal_load_n
            for camber, slip_angle in zip(
                cambers.flat, slip_angles.flat, strict=True
            )
        ),
        dtype=float,
        count=cambers.size,
    ).reshape(cambers.shape)
    return cambers, slip_angles, mu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normal-load-n",
        type=float,
        default=DEFAULT_REAR_TIRE_LOAD_N,
        help="Load on one tire [N]; default is the vehicle's static rear-tire load",
    )
    parser.add_argument("--pressure-psi", type=float, default=10.0)
    parser.add_argument("--camber-limit-deg", type=float, default=4.0)
    parser.add_argument("--slip-angle-limit-deg", type=float, default=15.0)
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not isfinite(args.normal_load_n) or not 0 < args.normal_load_n <= 1_800:
        parser.error("--normal-load-n must be within (0, 1800] N")
    if not isfinite(args.pressure_psi) or args.pressure_psi <= 0:
        parser.error("--pressure-psi must be positive and finite")
    if not isfinite(args.camber_limit_deg) or not 0 < args.camber_limit_deg <= 6:
        parser.error("--camber-limit-deg must be within (0, 6] degrees")
    if (
        not isfinite(args.slip_angle_limit_deg)
        or not 0 < args.slip_angle_limit_deg <= 55
    ):
        parser.error("--slip-angle-limit-deg must be within (0, 55] degrees")
    if args.grid_size < 3:
        parser.error("--grid-size must be at least 3")

    camber, slip_angle, mu = friction_surface(
        Pacejka52UpcR20LateralModel(),
        normal_load_n=args.normal_load_n,
        pressure_psi=args.pressure_psi,
        camber_limit_deg=args.camber_limit_deg,
        slip_angle_limit_deg=args.slip_angle_limit_deg,
        grid_size=args.grid_size,
    )

    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    limit = float(np.max(np.abs(mu)))
    surface = axis.plot_surface(
        camber,
        slip_angle,
        mu,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        linewidth=0,
        antialiased=True,
    )
    axis.set(
        xlabel="Camber [deg]",
        ylabel="Slip angle [deg]",
        zlabel="Signed lateral μy = Fy / Fz",
        zlim=(-1.1 * limit, 1.1 * limit),
    )
    axis.view_init(elev=27, azim=-60)
    figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.12, label="Signed μy")
    figure.suptitle(
        "Hoosier R20: lateral μ versus camber and slip angle\n"
        f"Fz = {args.normal_load_n:.0f} N per tire, pressure = {args.pressure_psi:g} psi"
    )
    figure.text(
        0.5,
        0.025,
        "Pure lateral UPC MF-Tyre fit; positive slip angle gives restoring "
        "(negative) Fy. R20 lateral coefficients originate from TTC tests.",
        ha="center",
        fontsize=8,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
