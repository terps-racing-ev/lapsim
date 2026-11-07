# G-G capacity bounds

These curves are optimistic tire-capacity bounds, not achievable vehicle
performance envelopes. Do not turn their lateral intercepts directly into
a skidpad prediction: compatible wheel slip angles, steering, body sideslip,
force projection, and yaw balance can impose substantially lower limits.

```powershell
.\.venv\Scripts\python.exe analysis/performance/plot_gg_envelope.py
```

Writes `outputs/performance/gg/gg_envelopes.png` (and SVG),
`gg_regen_comparison.png`, the signed boundary samples in `gg_boundary.csv`,
axis/extreme limits in `gg_limits.csv`, and configuration in `gg_summary.json`.
Positive longitudinal acceleration is propulsion; negative is braking.
Speeds default to 10, 20, 30, 40, 50, 60, and 70 mph. Zero speed has no
cornering motion, so it is excluded.

Uses the current `Vehicle()` components, including the default lateral
Pacejka model and longitudinal load-sensitive coefficient table, 800 psi
front/rear brake cap, rear drive, aero loss with roll, camber, longitudinal
and lateral load transfer, resistance, and equivalent rotating mass.
No vehicle source defaults are changed.

With the corrected 800 psi hardware limit, the sampled regen and friction-only
envelopes coincide: friction braking already saturates the available rear tire
grip. Regen can replace rear friction braking to recover energy, but does not
add rear contact-patch capacity. Straight braking ranges from about 1.39 g at
10 mph to 2.31 g at 70 mph under these model assumptions.

The comparison assumes 80% SOC, zero initial battery polarization, and an
80 kW DC charging cap, matching the regen-enabled setup in
`analysis/endurance/project_comp_and_capped_scores.py`. The actual charging
limit also respects pack voltage and resistance. The unmodified vehicle's
charging cap is zero, so simply requesting regen on `Vehicle()` would produce
the same curve as friction-only braking. These are scenario assumptions,
not a validated charge capability or regen calibration.

Customize with `--soc 0.6 --regen-charge-kw 30 --speeds-mph 20 40 70`.
`--regen-charge-kw 0` checks the disabled-regen case. `--points` defaults to
361 (must be 4*n+1); `--output-dir` changes the export directory.

## Calculation and interpretation

Each acceleration pair sets the load transfer, aero, roll and camber.
Lateral force has the yaw-balanced front/rear split used by the existing
capacity solvers (`static_front_weight_fraction`); within each axle it is
distributed in proportion to each tire's lateral capacity. The remaining
longitudinal force follows each tire's friction ellipse. Rear drive uses
the model's ideal limited-slip capacity allocation. Brake requests split
equally left/right, and friction plus regen share the rear tire capacities.
Axle pressure may be independently reduced from the maximum, and regen is
blended with rear friction braking to use the available grip.

Feasibility requires both lateral axle demands to fit their capacities and
`effective_mass * ax + resistance` to fit between maximum braking and maximum
rear propulsion. Rays from the coasting point trace the first boundary of
the connected feasible region; every saved point is checked for force
feasibility. The plots show signed acceleration in g. Peak lateral magnitude
in the summary is the largest boundary value, which can occur while braking;
it is not necessarily a constant-speed cornering limit. The separate
`zero_longitudinal_lateral_g` column holds the lateral limit at ax=0.

These are quasi-static peak-grip capacity estimates. They do not solve
individual slip angles, full steering/yaw equilibrium, steering travel limits,
wheel-lock dynamics, or transient stability. In particular, tight low-speed
turns can be steering-limited before reaching this theoretical tire envelope.
The curves describe the modeled hardware limits and ideal modulation, not a
measured G-G diagram or proof that a driver/controller can sustain every point.

## Skidpad cross-check

At the repository's 9.125 m path radius, interpolating the 20 and 30 mph
ax=0 capacities gives 4.375 s at about 1.92 g. This overstates the model's
sustained circle performance. A separate direct four-wheel force/yaw balance
check using `analysis/skidpad/optimize_static_camber.py::solve_balance`, with
unchanged zero static camber and 80% SOC, finds a local equilibrium limit of
about 1.649 g left and 1.646 g right, or 4.719 and 4.724 s per circle.
The force residual threshold was 1e-5 m/s^2; g was bisected over 1.6 to 1.7.
Results and wheel states are saved in `outputs/performance/gg/skidpad_audit.json`.

The existing nested `solve_skidpad_operating_point` returns an earlier limit
near 1.477 g / 4.986 s. Direct balance finds feasible points above it, so that
nested result is sensitive to numerical convergence and should not be treated
as a proven physical ceiling. The direct search is also a local equilibrium
estimate, not a global maximum or a validation against measured skidpad data;
it does not establish dynamic stability.

Validation: `python -m unittest discover -s tests -p test_gg_envelope.py` checks an analytic
constant-mu braking ellipse, shared tire saturation, disabled charging, and
that enabling regen preserves the friction-only feasible region.
