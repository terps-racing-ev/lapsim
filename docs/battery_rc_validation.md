# One-RC endurance battery validation

`scripts/fit_endurance_battery_rc.py` fits a first-order Thevenin battery to
the synchronized HVC-only trace produced by
`validate_endurance_battery_voltage.py`. It does not decode or use GPS,
inverter, motor, driver-control, or vehicle-dynamics signals.

The active endurance interval is the continuous window from the first to the
last original discharge-fit sample: 465.654--803.568 seconds in the MF4. All
3,345 valid battery samples in that interval are used so the fit sees both
loaded operation and voltage recovery. HVC SOC drives the firmware OCV table
directly. No SOC offset, SOC scale, terminal-voltage offset, or time shift is
fitted.

The model is:

```text
V_terminal = OCV(HVC_SOC) - I*R0 - V_p
dV_p/dt = (I*R1 - V_p)/(R1*C1)
```

A soft-L1 loss with a 2 V transition limits the influence of isolated CAN or
sensor outliers. The fitted full-window parameters are:

| Parameter | Result |
|---|---:|
| `R0` | 0.121473 Ohm |
| `R1` | 0.431981 Ohm |
| `C1` | 10.479 F |
| `R1*C1` | 4.527 s |
| initial `V_p` at window start | 0.550 V |

The initial polarization voltage is a fitted initial condition for this
window, not a reusable pack constant. The simulated default is zero for a
rested pack.

## Accuracy

| Evaluation | Static R RMSE | One-RC RMSE |
|---|---:|---:|
| All active samples | 9.198 V | 2.338 V |
| Discharge current at least 10 A | 4.666 V | 1.956 V |
| Chronological final 35% holdout | 9.844 V | 2.669 V |

For the holdout check, only the first 65% of the window is fitted. Its state
is then propagated chronologically through the final 35% without refitting.
This is a stronger check than scoring the same data used to estimate the
parameters.

The fitted polarization state ranges up to about 26.5 V in this log. It is
therefore a major part of the measured loaded voltage rather than a small
correction to the ohmic drop.

## Outputs

Run:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\fit_endurance_battery_rc.py
```

Outputs are written to
`outputs/endurance_validation/battery_rc_fit/`:

- `battery_rc_fit_metrics.json`: fitted parameters and comparison metrics.
- `battery_rc_fit_trace.csv`: measured and predicted voltage plus RC state.
- `rc_model_overview.png`: full-event voltage, current, and residuals.
- `rc_voltage_components.png`: ohmic and polarization voltage decomposition.
- `rc_residual_diagnostics.png`: histograms, parity, and residual trends.
- `rc_transient_zooms.png`: largest current steps and voltage recovery.

## Limits

The parameters come from one log and its temperature range. One RC branch
cannot represent multiple relaxation time scales, OCV hysteresis, temperature
dependence, or aging. Residual structure at low current and around some sharp
transitions is evidence for those omitted effects. A second RC branch should
only be added after validating against a different run; otherwise it can
mostly absorb timestamp skew and noise.
