# Skidpad setup analysis

Run `python analysis/skidpad/optimize_static_camber.py` to optimize static
front/rear camber at a 9.125 m radius and 1.5 g. Output is written to
`outputs/skidpad/static_camber/`: `summary.json`, `report.md`,
`camber_sweep.csv`, and `camber_sweep.png`.

The analysis uses the default vehicle and MF-Tyre 6.1 lateral fit, including
the measured wheel-travel camber gain, 7.3 degree front caster, full
anti-Ackermann, body roll, load transfer, and aero. Toe defaults to zero;
`--front-toe-deg 1` evaluates one degree toe-in per front wheel.
`--tire-model upc` selects the alternate UPC lateral fit. Use a separate
`--output-dir` when comparing configurations.

At fixed radius and acceleration, speed is prescribed by `sqrt(ay * radius)`.
The objective maximizes each axle's weaker left/right sum of directional tire
peak forces over 0 to 15 degrees restoring slip. Static settings are symmetric
across an axle. Steering is iterated against a separate four-wheel balance
solve for drive force, steering and body slip, holding speed and curvature
fixed and checking longitudinal force, lateral force and yaw moment residuals.

The individual peak sum is an envelope bound, since the tires may reach their
peaks at different slip angles. This does not predict maximum skidpad speed.
Camber is vehicle-local: negative is inward. The `Tire` adapter converts to
coefficient inclination with `gamma = -camber`; left/right slip and force
mirroring is separate. The optimum depends on the tire fit and slip bound.
No vehicle setup defaults are edited. The JSON records boundary optima,
convergence, operating point feasibility, wheel loads/travel/camber, and
comparison with zero camber.

## Comparing VD parameters

Run `python analysis/skidpad/sweep_vd_parameters.py` to sweep 24 alignment,
suspension, mass/CG, geometry, tire-pressure and aero parameters independently.
Baseline camber is -1.98 degrees front and -2.56 degrees rear, with the other
current vehicle defaults. Each range has 21 samples plus the exact baseline.
Ranges are screening assumptions and appear explicitly in the output.

`outputs/skidpad/vd_sweep/vd_sensitivity_all.png` contains every curve and a
ranking in one image. `summary.json`, `all_sweep_points.json`,
`sweep_points.csv` and `report.md` retain the underlying results. Use
`--replot` to redraw the saved results without repeating the physics.

Both turns are balanced at a 9.125 m radius and 1.5 g. Paired axle envelopes
keep each axle's actual inter-wheel slip split, so toe and Ackermann affect
the comparison. The vehicle envelope is constrained by the weaker axle/turn,
using static weight fraction as the approximate yaw demand split. All plots
share a percentage scale, and the ranking uses the full-range effect.
Best possible gains are also reported. These are changes in a grip envelope
at fixed conditions, not maximum-g or lap-time predictions. Other parameters
and baseline camber are held fixed; effects cannot simply be added together.
