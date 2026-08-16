# Current-model first-lap replay

## Purpose

This script shows what the current `Vehicle()` model predicts when it receives
the recorded first-lap motor-torque commands and GNSS-derived track curvature.
It does not fit parameters, load the old propulsion calibration, or modify any
vehicle defaults.

The simulation advances in time, but looks up controls and curvature at its
current distance around the recorded racing line. Measured speed, motor RPM,
pack power, voltage, current, SOC, and accelerations are comparison outputs;
they are not fed back after initialization.

## Inputs and initialization

- Motor torque command: zero-order held by racing-line distance.
- Curvature: derived from the smoothed GNSS racing line and linearly
  interpolated by distance.
- Initial speed and HVC SOC: copied from the first recorded sample.
- Initial battery RC polarization: initialized from the first measured pack
  voltage/current and the battery model's OCV relationship.
- Vehicle and component parameters: current source defaults from `Vehicle()`.

No timing offset is identified or applied. The station coordinate is the raw
GNSS trip distance normalized at the start.

The recording contains rear-brake pressure, while `Controls` requires brake
force. Because the current model has no hydraulic/caliper pressure-to-force
model, braking is unapplied by default. This missing input adapter is reported
in the metrics. A deliberate external assumption can be tested with
`--brake-force-per-psi-n`, but it is not a fitted default.

## Current result

With current defaults and no pressure-to-force adapter:

| Metric | Measured | Simulation |
|---|---:|---:|
| Distance coverage | 988.89 m | 100% |
| Lap time | 69.49 s | 39.07 s |
| Final SOC | 94.500% | 95.932% |
| Net pack energy | 0.2311 kWh | 0.1434 kWh |
| Speed RMSE | - | 13.28 m/s |
| Motor-speed RMSE | - | 2087 RPM |
| Pack-power RMSE | - | 11.36 kW |
| Pack-voltage RMSE | - | 4.95 V |
| Pack-current RMSE | - | 27.09 A |

This is intentionally an uncorrected model result. The very early simulated
lap is not evidence that the car can run that time: recorded braking is absent,
and the current no-slip point-mass model lacks yaw/wheel dynamics and can fail
to enforce the prescribed racing line at the requested curvature.

## Reproduce

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe scripts\validate_first_endurance_lap.py
```

Outputs are written to
`outputs/endurance_validation/first_lap_soc_open_loop/`:

- `first_lap_actual_vs_sim.png`: SOC, speed, motor RPM, pack power, voltage,
  current, and longitudinal/lateral acceleration.
- `first_lap_soc_trace.csv`: measured and simulated channels on one distance
  coordinate.
- `first_lap_sim_full_telemetry.csv`: every component-owned simulation
  telemetry channel at every physics step.
- `first_lap_soc_metrics.json`: inputs, assumptions, errors, and coverage.
