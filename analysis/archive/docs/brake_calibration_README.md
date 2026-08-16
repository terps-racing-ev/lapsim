# Endurance brake-model calibration

## Finding

The previous replay could not identify braking correctly because it applied one
VCU BSE pressure to both axles. The raw log shows:

- `CAN1.VCU_BSE.VCU_BSE_PSI` is the front circuit. It matches
  `VCU_Regen_Front_PSI` with correlation 0.99997 during active samples and is
  logged at 100 Hz.
- `MOBO_Power_Telemetry.MOBO_BSE_PSI_Rear` is the independent rear circuit. It
  is only logged at about 5 Hz, so the converter interpolates it before the
  samples are projected onto track station.
- Regen was not commanded on this lap: the logged VCU regen request and balance
  torque remain zero. Negative inverter torque feedback is retained as rear
  motor/backdrive braking.

Controls are projected onto the fused-track station and replayed by distance.
The simulation therefore applies the recorded driver action at the same track
location even when simulated speed differs. Elapsed time and pressure-pulse
duration are consequences of simulated cell speed rather than recorded time.

## Pressure-to-force models

`linear-hardware-gains` retains the supplied values exactly once:

- Front: 10.12849472 lbf-in/psi
- Rear: 5.390972994 lbf-in/psi

This model remains available and unchanged. Raising tire mu does not materially
increase its pressure-limited stops.

`firmware-force-map` is an opt-in analysis hypothesis based on the force terms
in TREV4-Controls' Ryder brake-balance equation:

```text
front = 183 + 8.7696*p - 0.0141145*p^2 + 0.0000068383*p^3
rear  = 5.6077*p
```

Treating those terms as newtons gives the best current lap validation. That unit
interpretation still needs confirmation from the original equation derivation
or a brake-dyno test; it is deliberately not a default vehicle parameter.

## Best current replay

```powershell
.venv\Scripts\python.exe analysis\full_lap_comparison\analyze_full_lap.py `
  --brake-pressure-model firmware-force-map `
  --brake-deadband-psi 5 `
  --brake-gain-count-per-axle 1 `
  --negative-torque-policy rear-brake `
  --cornering-drag-coefficient 0.036 `
  --constant-tire-mu 1.8 `
  --drag-coefficient 2.5 `
  --motor-to-wheel-efficiency 0.86 `
  --output-dir analysis/full_lap_comparison/braking_corrected_distance_native
```

Current results, with path limiting disabled:

- speed RMSE: 1.793 m/s
- longitudinal acceleration RMSE versus GNSS kinematics: 1.640 m/s^2
- braking acceleration bias versus GNSS kinematics: -0.619 m/s^2
- simulated distance: 1019.81 m (exact requested spatial domain)
- internally calculated simulated time: 76.94 s
- recorded time: 69.50 s

The remaining uncertainty is dominated by the 5 Hz rear-pressure channel, the
unconfirmed firmware force-map units, road grade, and the single-coefficient
tire-scrub model. A synchronized 100 Hz rear-pressure log plus a static
pressure-to-wheel-force test would make the brake calibration identifiable
without relying on the firmware-map inference.
