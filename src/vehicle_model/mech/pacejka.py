"""MF-Tyre models for the Hoosier 16x7.5-10 R20 tire."""

from dataclasses import dataclass
from math import atan, exp, isfinite, sin, tan


@dataclass(frozen=True, slots=True)
class Pacejka61LateralModel:
    """Pure-side-slip subset of an MF-Tyre 6.1 parameter set.

    The defaults were extracted from ``16x7.5-10 R20 Pacejka
    Coefficients.mat``. Angles supplied to :meth:`force_n` are radians,
    pressure is absolute pressure in pascals, and normal load is the positive
    force carried by one tire in newtons.
    """

    nominal_load_n: float = 1080.0  # FNOMIN
    nominal_pressure_pa: float = 98_000.0  # NOMPRES

    # Lateral force coefficients from the MATLAB ``mfparams`` structure.
    pcy1: float = 1.6
    pdy1: float = 2.3
    pdy2: float = -0.29534367955293434
    pdy3: float = 2.996147054970238
    pey1: float = 0.3697191883502767
    pey2: float = -0.06664224082194274
    pey3: float = 1.7242879351718403
    pey4: float = -59.413341508269205
    pey5: float = -147.0812300386499
    pky1: float = -29.144083838755247
    pky2: float = 1.49155875601304
    pky3: float = -0.9241086249725864
    pky4: float = 1.9989302942568425
    pky5: float = 11.030081974309828
    pky6: float = -3.652040589388632
    pky7: float = -2.336912095954469
    phy1: float = 0.0002765973165278696
    phy2: float = 0.003
    pvy1: float = 0.09
    pvy2: float = 0.12
    pvy3: float = -1.234101956586106
    pvy4: float = -1.2172361879271507
    ppy1: float = -0.3984835802395689
    ppy2: float = 0.4284617979987908
    ppy3: float = -0.1524351935406058
    ppy4: float = 0.09604233237160076
    ppy5: float = -1.2850817229022373

    # Scaling coefficients in the supplied model are all one.
    nominal_load_scale: float = 1.0  # LFZO
    shape_scale: float = 1.0  # LCY
    friction_scale: float = 1.0  # LMUY
    curvature_scale: float = 1.0  # LEY
    stiffness_scale: float = 1.0  # LKY
    camber_stiffness_scale: float = 1.0  # LKYC
    horizontal_shift_scale: float = 1.0  # LHY
    vertical_shift_scale: float = 1.0  # LVY
    camber_scale: float = 1.0  # LGAY

    def validate(self) -> None:
        """Validate the dimensional reference values and scale factors."""

        if not isfinite(self.nominal_load_n) or self.nominal_load_n <= 0.0:
            raise ValueError("Pacejka nominal_load_n must be finite and positive")
        if not isfinite(self.nominal_pressure_pa) or self.nominal_pressure_pa <= 0.0:
            raise ValueError(
                "Pacejka nominal_pressure_pa must be finite and positive"
            )
        positive_scales = (
            self.nominal_load_scale,
            self.shape_scale,
            self.friction_scale,
            self.curvature_scale,
            self.stiffness_scale,
            self.camber_stiffness_scale,
            self.horizontal_shift_scale,
            self.vertical_shift_scale,
            self.camber_scale,
        )
        if any(not isfinite(scale) or scale <= 0.0 for scale in positive_scales):
            raise ValueError("Pacejka scale factors must be finite and positive")

    def force_n(
        self,
        normal_load_n: float,
        slip_angle_rad: float,
        *,
        camber_angle_rad: float = 0.0,
        inflation_pressure_pa: float | None = None,
    ) -> float:
        """Return one tire's pure-slip lateral force.

        Positive normal load means compression. The coefficient set has a
        negative cornering stiffness, so positive slip angle produces the
        conventional restoring (negative) lateral force.
        """

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
        alpha_y = tan(slip_angle_rad) + terms.horizontal_shift
        alpha_sign = 0.0
        if alpha_y > 0.0:
            alpha_sign = 1.0
        elif alpha_y < 0.0:
            alpha_sign = -1.0
        curvature = (
            (self.pey1 + self.pey2 * terms.normalized_load_change)
            * (
                1.0
                + self.pey5 * terms.scaled_camber**2
                - (self.pey3 + self.pey4 * terms.scaled_camber) * alpha_sign
            )
            * self.curvature_scale
        )
        # MF-Tyre constrains the lateral curvature factor to E_y <= 1.
        curvature = min(curvature, 1.0)
        bx_alpha = terms.stiffness_factor * alpha_y
        phase = terms.shape_factor * atan(
            bx_alpha - curvature * (bx_alpha - atan(bx_alpha))
        )
        return terms.peak_factor_n * sin(phase) + terms.vertical_shift_n

    def peak_force_n(
        self,
        normal_load_n: float,
        *,
        camber_angle_rad: float = 0.0,
        inflation_pressure_pa: float | None = None,
    ) -> float:
        """Return the maximum lateral-force magnitude for one tire.

        The pure-slip Magic Formula sine term is bounded by ``+/-D_y`` and
        reaches those extrema for this coefficient set. Including the fitted
        vertical shift makes the larger directional peak ``D_y + |SV_y|``.
        """

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
        return abs(terms.peak_factor_n) + abs(terms.vertical_shift_n)

    def _terms(
        self,
        normal_load_n: float,
        camber_angle_rad: float,
        inflation_pressure_pa: float | None,
    ) -> "_LateralTerms":
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
        scaled_camber = sin(camber_angle_rad) * self.camber_scale

        shape_factor = self.pcy1 * self.shape_scale
        friction_coefficient = (
            (self.pdy1 + self.pdy2 * dfz)
            * (1.0 + self.ppy3 * dpi + self.ppy4 * dpi**2)
            * (1.0 - self.pdy3 * scaled_camber**2)
            * self.friction_scale
        )
        peak_factor_n = friction_coefficient * normal_load_n

        camber_vertical_shift_n = (
            normal_load_n
            * (self.pvy3 + self.pvy4 * dfz)
            * scaled_camber
            * self.camber_stiffness_scale
            * self.friction_scale
        )
        vertical_shift_n = (
            normal_load_n
            * (self.pvy1 + self.pvy2 * dfz)
            * self.vertical_shift_scale
            * self.friction_scale
            + camber_vertical_shift_n
        )

        stiffness_at_zero_camber_n_per_rad = (
            self.pky1
            * nominal_load_n
            * (1.0 + self.ppy1 * dpi)
            * sin(
                self.pky4
                * atan(
                    normal_load_n
                    / (
                        (self.pky2 + self.pky5 * scaled_camber**2)
                        * (1.0 + self.ppy2 * dpi)
                        * nominal_load_n
                    )
                )
            )
            * self.stiffness_scale
        )
        cornering_stiffness_n_per_rad = stiffness_at_zero_camber_n_per_rad * (
            1.0 - self.pky3 * abs(scaled_camber)
        )
        camber_stiffness_n_per_rad = (
            normal_load_n
            * (self.pky6 + self.pky7 * dfz)
            * (1.0 + self.ppy5 * dpi)
            * self.camber_stiffness_scale
        )

        stiffness_epsilon = 1e-12
        protected_cornering_stiffness = cornering_stiffness_n_per_rad
        if abs(protected_cornering_stiffness) < stiffness_epsilon:
            protected_cornering_stiffness = (
                stiffness_epsilon
                if protected_cornering_stiffness >= 0.0
                else -stiffness_epsilon
            )
        horizontal_shift = (
            (self.phy1 + self.phy2 * dfz) * self.horizontal_shift_scale
            + (
                camber_stiffness_n_per_rad * scaled_camber
                - camber_vertical_shift_n
            )
            / protected_cornering_stiffness
        )

        shape_peak_product = shape_factor * peak_factor_n
        stiffness_factor = (
            cornering_stiffness_n_per_rad / shape_peak_product
            if abs(shape_peak_product) >= stiffness_epsilon
            else 0.0
        )
        return _LateralTerms(
            normalized_load_change=dfz,
            scaled_camber=scaled_camber,
            shape_factor=shape_factor,
            peak_factor_n=peak_factor_n,
            stiffness_factor=stiffness_factor,
            horizontal_shift=horizontal_shift,
            vertical_shift_n=vertical_shift_n,
        )


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
    """

    # [VERTICAL] and [OPERATING_CONDITIONS]
    nominal_load_n: float = 1_000.0  # FNOMIN
    nominal_pressure_pa: float = 82_737.0  # NOMPRES

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
            raise ValueError(
                "Pacejka nominal_pressure_pa must be finite and positive"
            )
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
            self.pey1 + self.pey2 * terms.normalized_load_change
        ) * (
            1.0
            - (self.pey3 + self.pey4 * terms.scaled_camber) * alpha_sign
        ) * self.curvature_scale
        # MF-Tyre constrains the lateral curvature factor to E_y <= 1.
        curvature = min(curvature, 1.0)
        bx_alpha = terms.cornering_stiffness_n_per_rad * alpha_y / (
            terms.shape_factor * terms.peak_factor_n + 1e-12
        )
        phase = terms.shape_factor * atan(
            bx_alpha - curvature * (bx_alpha - atan(bx_alpha))
        )
        return terms.peak_factor_n * sin(phase) + terms.vertical_shift_n

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
        return abs(terms.peak_factor_n) + abs(terms.vertical_shift_n)

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
                * atan(
                    normal_load_n
                    / (pressure_adjusted_load_factor * nominal_load_n)
                )
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
        vertical_shift_n = normal_load_n * (
            (self.pvy1 + self.pvy2 * dfz) * self.vertical_shift_scale
            + (self.pvy3 + self.pvy4 * dfz)
            * scaled_camber
            * self.camber_stiffness_scale
        ) * self.friction_scale

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
    """

    # [VERTICAL] and [OPERATING_CONDITIONS]
    nominal_load_n: float = 1_000.0  # FNOMIN
    nominal_pressure_pa: float = 82_737.0  # NOMPRES

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
            raise ValueError(
                "Pacejka nominal_pressure_pa must be finite and positive"
            )
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
            self.pex1
            + self.pex2 * terms.normalized_load_change
            + self.pex3 * terms.normalized_load_change**2
        ) * (1.0 - self.pex4 * kappa_sign) * self.curvature_scale
        bx_kappa = terms.stiffness_factor * kappa_x
        phase = self.pcx1 * self.shape_scale * atan(
            bx_kappa - curvature * (bx_kappa - atan(bx_kappa))
        )
        return terms.peak_factor_n * sin(phase) + terms.vertical_shift_n

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
        return abs(terms.peak_factor_n) + abs(terms.vertical_shift_n)

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
        horizontal_shift = (
            self.phx1 + self.phx2 * dfz
        ) * self.horizontal_shift_scale
        vertical_shift_n = (
            normal_load_n
            * (self.pvx1 + self.pvx2 * dfz)
            * self.vertical_shift_scale
        )
        return _UpcLongitudinalTerms(
            normalized_load_change=dfz,
            peak_factor_n=peak_factor_n,
            stiffness_factor=stiffness_factor,
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


@dataclass(frozen=True, slots=True)
class _UpcLongitudinalTerms:
    normalized_load_change: float
    peak_factor_n: float
    stiffness_factor: float
    horizontal_shift: float
    vertical_shift_n: float


@dataclass(frozen=True, slots=True)
class _LateralTerms:
    normalized_load_change: float
    scaled_camber: float
    shape_factor: float
    peak_factor_n: float
    stiffness_factor: float
    horizontal_shift: float
    vertical_shift_n: float
