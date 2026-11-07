"""MF-Tyre coefficient models for the Hoosier 16x7.5-10 R20 tire.

These classes retain the source coefficient-space inclination convention.
Vehicle-local camber (negative inward) is converted to its opposite sign by
the Tire adapter; direct calls here accept raw coefficient-space inclination.
"""

from dataclasses import dataclass
from math import atan, isfinite, sin


@dataclass(frozen=True, slots=True)
class Pacejka52UpcR20LateralModel:
    """Pure-lateral MF-Tyre 5.2 fit for the Hoosier 16x7.5-10 R20.

    These coefficients and scaling values come from the public ``.tir``
    appendix to Marc Miret Pros's 2024 UPC thesis. The published vehicle
    model is a hybrid: the 16-inch TTC data supplied pure lateral and
    aligning-moment data, while longitudinal and combined-slip terms were
    carried over from another tire model. This class exposes the published
    pure-lateral block only.

    Angles are radians, pressure is absolute pascals, and normal load is the
    positive force carried by one tire in newtons.
    Outputs default to 80% of the published fit; use
    ``correlation_factor=1.0`` for the original force curve and capacity.
    """

    # [VERTICAL] and [OPERATING_CONDITIONS]
    nominal_load_n: float = 1_000.0  # FNOMIN
    nominal_pressure_pa: float = 82_737.0  # NOMPRES
    # Scale the complete published force curve, including fitted offsets.
    correlation_factor: float = 0.80

    # [LATERAL_COEFFICIENTS]
    pcy1: float = 1.477553032924075
    pdy1: float = 2.419662634889429
    pdy2: float = -0.096991237613422
    pdy3: float = 9.699367924763585
    pey1: float = -0.001014423462658
    pey2: float = -0.001567587511733
    pey3: float = -26.252507850904447
    pey4: float = 4.418491888020646e3
    pky1: float = -48.096691177591670
    pky2: float = 1.79989978802
    pky3: float = 0.776633692944650
    phy1: float = 9.632469743033963e-05
    phy2: float = 4.485102635543676e-04
    phy3: float = 0.044112063116
    pvy1: float = -0.005996641445337
    pvy2: float = 0.031420448279870
    pvy3: float = -0.840635268453471
    pvy4: float = -0.482847397434584
    ppy1: float = 0.006522910697878
    ppy2: float = 0.732182121539935
    ppy3: float = -0.248004155399173
    ppy4: float = 0.344037334268957

    # [SCALING_COEFFICIENTS]
    nominal_load_scale: float = 0.8  # LFZO
    shape_scale: float = 1.22  # LCY
    friction_scale: float = 0.805  # LMUY
    curvature_scale: float = 1.0  # LEY
    stiffness_scale: float = 0.88  # LKY
    camber_stiffness_scale: float = 1.07  # LKYG
    horizontal_shift_scale: float = 0.0  # LHY
    vertical_shift_scale: float = 1.0  # LVY
    camber_scale: float = 1.0  # LGAY

    def validate(self) -> None:
        """Validate the reference values and MF scaling factors."""

        if not isfinite(self.nominal_load_n) or self.nominal_load_n <= 0.0:
            raise ValueError("Pacejka nominal_load_n must be finite and positive")
        if not isfinite(self.nominal_pressure_pa) or self.nominal_pressure_pa <= 0.0:
            raise ValueError("Pacejka nominal_pressure_pa must be finite and positive")
        if (
            not isfinite(self.correlation_factor)
            or not 0.0 < self.correlation_factor <= 1.0
        ):
            raise ValueError("Pacejka correlation_factor must be in (0, 1]")
        positive_scales = (
            self.nominal_load_scale,
            self.shape_scale,
            self.friction_scale,
            self.curvature_scale,
            self.stiffness_scale,
            self.camber_stiffness_scale,
            self.vertical_shift_scale,
            self.camber_scale,
        )
        if any(not isfinite(scale) or scale <= 0.0 for scale in positive_scales):
            raise ValueError("Pacejka scale factors must be finite and positive")
        if (
            not isfinite(self.horizontal_shift_scale)
            or self.horizontal_shift_scale < 0.0
        ):
            raise ValueError(
                "Pacejka horizontal_shift_scale must be finite and nonnegative"
            )

    def force_n(
        self,
        normal_load_n: float,
        slip_angle_rad: float,
        *,
        camber_angle_rad: float = 0.0,
        inflation_pressure_pa: float | None = None,
    ) -> float:
        """Return one tire's MF-Tyre 5.2 pure-slip lateral force."""

        if not isfinite(normal_load_n):
            raise ValueError("normal_load_n must be finite")
        if not isfinite(slip_angle_rad):
            raise ValueError("slip_angle_rad must be finite")
        if not isfinite(camber_angle_rad):
            raise ValueError("camber_angle_rad must be finite")
        if normal_load_n <= 0.0:
            return 0.0

        terms = self._terms(
            normal_load_n,
            camber_angle_rad,
            inflation_pressure_pa,
        )
        # The UPC file uses USE_MODE=14, so MF-Tyre uses the non-star
        # side-slip and inclination variables (alpha and gamma directly).
        alpha_y = slip_angle_rad + terms.horizontal_shift
        alpha_sign = 0.0
        if alpha_y > 0.0:
            alpha_sign = 1.0
        elif alpha_y < 0.0:
            alpha_sign = -1.0
        curvature = (
            (self.pey1 + self.pey2 * terms.normalized_load_change)
            * (1.0 - (self.pey3 + self.pey4 * terms.scaled_camber) * alpha_sign)
            * self.curvature_scale
        )
        # MF-Tyre constrains the lateral curvature factor to E_y <= 1.
        curvature = min(curvature, 1.0)
        bx_alpha = (
            terms.cornering_stiffness_n_per_rad
            * alpha_y
            / (terms.shape_factor * terms.peak_factor_n + 1e-12)
        )
        phase = terms.shape_factor * atan(
            bx_alpha - curvature * (bx_alpha - atan(bx_alpha))
        )
        return self.correlation_factor * (
            terms.peak_factor_n * sin(phase) + terms.vertical_shift_n
        )

    def peak_force_n(
        self,
        normal_load_n: float,
        *,
        camber_angle_rad: float = 0.0,
        inflation_pressure_pa: float | None = None,
    ) -> float:
        """Return the fitted pure-lateral force magnitude at one load."""

        if not isfinite(normal_load_n):
            raise ValueError("normal_load_n must be finite")
        if not isfinite(camber_angle_rad):
            raise ValueError("camber_angle_rad must be finite")
        if normal_load_n <= 0.0:
            return 0.0
        terms = self._terms(
            normal_load_n,
            camber_angle_rad,
            inflation_pressure_pa,
        )
        return self.correlation_factor * (
            abs(terms.peak_factor_n) + abs(terms.vertical_shift_n)
        )

    def _terms(
        self,
        normal_load_n: float,
        camber_angle_rad: float,
        inflation_pressure_pa: float | None,
    ) -> "_UpcLateralTerms":
        pressure_pa = (
            self.nominal_pressure_pa
            if inflation_pressure_pa is None
            else inflation_pressure_pa
        )
        if not isfinite(pressure_pa) or pressure_pa <= 0.0:
            raise ValueError("inflation_pressure_pa must be finite and positive")

        nominal_load_n = self.nominal_load_n * self.nominal_load_scale
        dfz = (normal_load_n - nominal_load_n) / nominal_load_n
        dpi = (pressure_pa - self.nominal_pressure_pa) / self.nominal_pressure_pa
        scaled_camber = camber_angle_rad * self.camber_scale

        shape_factor = self.pcy1 * self.shape_scale
        friction_coefficient = (
            (self.pdy1 + self.pdy2 * dfz)
            * (1.0 + self.ppy3 * dpi + self.ppy4 * dpi**2)
            * (1.0 - self.pdy3 * scaled_camber**2)
            * self.friction_scale
        )
        peak_factor_n = friction_coefficient * normal_load_n

        pressure_adjusted_load_factor = self.pky2 * (1.0 + self.ppy2 * dpi)
        stiffness_at_zero_camber_n_per_rad = (
            self.pky1
            * nominal_load_n
            * (1.0 + self.ppy1 * dpi)
            * sin(
                2.0
                * atan(normal_load_n / (pressure_adjusted_load_factor * nominal_load_n))
            )
            * self.stiffness_scale
        )
        cornering_stiffness_n_per_rad = stiffness_at_zero_camber_n_per_rad * (
            1.0 - self.pky3 * abs(scaled_camber)
        )

        horizontal_shift = (
            (self.phy1 + self.phy2 * dfz) * self.horizontal_shift_scale
            + self.phy3 * scaled_camber * self.camber_stiffness_scale
        )
        vertical_shift_n = (
            normal_load_n
            * (
                (self.pvy1 + self.pvy2 * dfz) * self.vertical_shift_scale
                + (self.pvy3 + self.pvy4 * dfz)
                * scaled_camber
                * self.camber_stiffness_scale
            )
            * self.friction_scale
        )

        return _UpcLateralTerms(
            normalized_load_change=dfz,
            scaled_camber=scaled_camber,
            shape_factor=shape_factor,
            peak_factor_n=peak_factor_n,
            cornering_stiffness_n_per_rad=cornering_stiffness_n_per_rad,
            horizontal_shift=horizontal_shift,
            vertical_shift_n=vertical_shift_n,
        )


@dataclass(frozen=True, slots=True)
class _UpcLateralTerms:
    normalized_load_change: float
    scaled_camber: float
    shape_factor: float
    peak_factor_n: float
    cornering_stiffness_n_per_rad: float
    horizontal_shift: float
    vertical_shift_n: float
