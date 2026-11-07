"""MF-Tyre coefficient models for the Hoosier 16x7.5-10 R20 tire.

These classes retain the source coefficient-space inclination convention.
Vehicle-local camber (negative inward) is converted to its opposite sign by
the Tire adapter; direct calls here accept raw coefficient-space inclination.
"""

from dataclasses import dataclass
from math import atan, exp, isfinite, sin


@dataclass(frozen=True, slots=True)
class Pacejka52UpcR20LongitudinalModel:
    """Pure-longitudinal MF-Tyre 5.2 fit from the UPC Hoosier R20 file.

    The coefficients are copied from the public UPC ``.tir`` appendix for the
    Hoosier 16x7.5-10 R20.  The UPC report notes that this longitudinal block
    was carried over from another tire model, so it is useful as a published
    engineering estimate rather than a validated 16-inch longitudinal test
    result.

    Slip ratio is dimensionless, pressure is absolute pascals, camber is
    radians, and normal load is the positive force carried by one tire in
    newtons.
    Outputs default to 80% of the published fit; use
    ``correlation_factor=1.0`` for the original force curve and capacity.
    """

    # [VERTICAL] and [OPERATING_CONDITIONS]
    nominal_load_n: float = 1_000.0  # FNOMIN
    nominal_pressure_pa: float = 82_737.0  # NOMPRES
    # Scale the complete published force curve, including fitted offsets.
    correlation_factor: float = 0.80

    # [LONGITUDINAL_COEFFICIENTS]
    pcx1: float = 1.56182385630461
    pdx1: float = 2.85110165444411
    pdx2: float = -0.407371062621199
    pdx3: float = 19.2614164371048
    pex1: float = 0.422733521935518
    pex2: float = 0.0201014651759272
    pex3: float = 0.00229620366835657
    pex4: float = 0.00999560321822648
    pkx1: float = 52.2555801076937
    pkx2: float = -0.00614387636030559
    pkx3: float = -0.155693449933703
    phx1: float = 0.000340072159079639
    phx2: float = -0.00161506337261505
    pvx1: float = -0.00331590453741636
    pvx2: float = 0.0788908435936460
    ppx1: float = -0.705319191164713
    ppx2: float = -0.589490385312056
    ppx3: float = -0.242820291164767
    ppx4: float = 0.205179565555030

    # [SCALING_COEFFICIENTS]
    nominal_load_scale: float = 0.8  # LFZO
    shape_scale: float = 0.995  # LCX
    friction_scale: float = 0.78  # LMUX
    curvature_scale: float = 1.0  # LEX
    stiffness_scale: float = 0.99  # LKX
    horizontal_shift_scale: float = 0.95  # LHX
    vertical_shift_scale: float = 1.0  # LVX
    camber_scale: float = 1.0  # LGAX

    def validate(self) -> None:
        """Validate the dimensional reference values and scale factors."""

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
            self.horizontal_shift_scale,
            self.vertical_shift_scale,
            self.camber_scale,
        )
        if any(not isfinite(scale) or scale <= 0.0 for scale in positive_scales):
            raise ValueError("Pacejka scale factors must be finite and positive")

    def force_n(
        self,
        normal_load_n: float,
        slip_ratio: float,
        *,
        camber_angle_rad: float = 0.0,
        inflation_pressure_pa: float | None = None,
    ) -> float:
        """Return one tire's pure-longitudinal MF-Tyre force."""

        if not isfinite(normal_load_n):
            raise ValueError("normal_load_n must be finite")
        if not isfinite(slip_ratio):
            raise ValueError("slip_ratio must be finite")
        if not isfinite(camber_angle_rad):
            raise ValueError("camber_angle_rad must be finite")
        if normal_load_n <= 0.0:
            return 0.0

        terms = self._terms(
            normal_load_n,
            slip_ratio,
            camber_angle_rad,
            inflation_pressure_pa,
        )
        kappa_x = slip_ratio + terms.horizontal_shift
        kappa_sign = 0.0
        if kappa_x > 0.0:
            kappa_sign = 1.0
        elif kappa_x < 0.0:
            kappa_sign = -1.0
        curvature = (
            (
                self.pex1
                + self.pex2 * terms.normalized_load_change
                + self.pex3 * terms.normalized_load_change**2
            )
            * (1.0 - self.pex4 * kappa_sign)
            * self.curvature_scale
        )
        bx_kappa = terms.stiffness_factor * kappa_x
        phase = (
            self.pcx1
            * self.shape_scale
            * atan(bx_kappa - curvature * (bx_kappa - atan(bx_kappa)))
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
        """Return the fitted pure-longitudinal force capacity.

        The Magic Formula sine term is bounded by ``+/-D_x``.  The capacity
        therefore uses the larger directional bound after the fitted vertical
        shift, matching the capacity treatment of the lateral model.
        """

        if not isfinite(normal_load_n):
            raise ValueError("normal_load_n must be finite")
        if not isfinite(camber_angle_rad):
            raise ValueError("camber_angle_rad must be finite")
        if normal_load_n <= 0.0:
            return 0.0
        terms = self._terms(
            normal_load_n,
            0.0,
            camber_angle_rad,
            inflation_pressure_pa,
        )
        return self.correlation_factor * (
            abs(terms.peak_factor_n) + abs(terms.vertical_shift_n)
        )

    def _terms(
        self,
        normal_load_n: float,
        slip_ratio: float,
        camber_angle_rad: float,
        inflation_pressure_pa: float | None,
    ) -> "_UpcLongitudinalTerms":
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

        shape_factor = self.pcx1 * self.shape_scale
        friction_coefficient = (
            (self.pdx1 + self.pdx2 * dfz)
            * (1.0 + self.ppx3 * dpi + self.ppx4 * dpi**2)
            * (1.0 - self.pdx3 * scaled_camber**2)
            * self.friction_scale
        )
        peak_factor_n = friction_coefficient * normal_load_n
        stiffness_n = (
            normal_load_n
            * (self.pkx1 + self.pkx2 * dfz)
            * exp(self.pkx3 * dfz)
            * (1.0 + self.ppx1 * dpi + self.ppx2 * dpi**2)
            * self.stiffness_scale
        )
        shape_peak_product = shape_factor * peak_factor_n
        stiffness_factor = (
            stiffness_n / shape_peak_product
            if abs(shape_peak_product) >= 1e-12
            else 0.0
        )
        horizontal_shift = (self.phx1 + self.phx2 * dfz) * self.horizontal_shift_scale
        vertical_shift_n = (
            normal_load_n * (self.pvx1 + self.pvx2 * dfz) * self.vertical_shift_scale
        )
        return _UpcLongitudinalTerms(
            normalized_load_change=dfz,
            peak_factor_n=peak_factor_n,
            stiffness_factor=stiffness_factor,
            horizontal_shift=horizontal_shift,
            vertical_shift_n=vertical_shift_n,
        )


@dataclass(frozen=True, slots=True)
class _UpcLongitudinalTerms:
    normalized_load_change: float
    peak_factor_n: float
    stiffness_factor: float
    horizontal_shift: float
    vertical_shift_n: float
