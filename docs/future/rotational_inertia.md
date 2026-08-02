# Rotational Inertia

The simulator includes a simple effective-mass model for longitudinal
rotational inertia. The current inputs come from
`Emrax228_Motor_Parameters.xlsx`:

- Motor rotor inertia: `0.01215 kg*m^2`, at motor speed
- Final-drive input inertia: `0.00005 kg*m^2`, at motor speed
- Final-drive output inertia: `0.003 kg*m^2`, at wheel speed

The workbook labels all three values as placeholders. Ryder's simulator only
contains a yaw-inertia parameter and does not provide longitudinal rotating
component inertias.

## Current model

Input-side components rotate at the motor-to-wheel speed ratio, so their
inertias are reflected to wheel speed:

\[
I_\mathrm{wheel} =
\left(I_\mathrm{motor}+I_\mathrm{input}\right)G^2+I_\mathrm{output}
\]

The wheel-referenced inertia is then converted to equivalent translating mass:

\[
m_\mathrm{rotating}=\frac{I_\mathrm{wheel}}{r^2}
\]

where `G` is the final-drive ratio and `r` is the loaded tire rolling radius.
At the current `3.7` ratio and `0.2032 m` radius, the workbook values produce:

- Wheel-referenced inertia: approximately `0.1700 kg*m^2`
- Equivalent rotating mass: approximately `4.12 kg`
- Total effective longitudinal mass: approximately `298.95 kg`

The simulator uses effective mass only when converting net longitudinal force
to acceleration or braking deceleration. Physical vehicle mass remains in
weight, tire normal load, aerodynamic load distribution, lateral-force demand,
and longitudinal load-transfer calculations.

Rotational inertia is stored energy, not a dissipative efficiency loss. During
acceleration it reduces acceleration for a given applied torque. During braking
the rotating components must also be decelerated. The current friction-braking
model assumes that energy is dissipated rather than recovered.

## Missing inputs and future improvements

The workbook does not provide inertias for:

- Wheel and tire assemblies
- Sprockets and chain
- Differential and axles beyond the lumped final-drive placeholders

These should be added as separately mutable component values once measured or
calculated. A more detailed model could track rotational kinetic energy at each
shaft explicitly, distinguish forward and back-drive efficiencies, and expose
the stored energy to a regenerative-braking model.

[Terps Racing EV inertia analysis](https://umd0.sharepoint.com/:x:/s/TeamsTerpsRacingEV/IQCKI-4UHm9KTIB_JhJ99tA3Afz67L0oSUZD23Ufqr302e4?e=SgHW4U)
