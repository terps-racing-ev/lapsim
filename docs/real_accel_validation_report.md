# Real acceleration log validation

> Status note: the explicit-slip experiment documented below was subsequently
> removed from the simulator. The current chronological model is again the
> no-slip point-mass baseline; these results are retained for comparison only.

## Post-slip-model rerun

The selected launch was rerun with the recorded launch-torque schedule and the
70 kW motor-mechanical power cap. The measured 3.760 s target is still the time
inferred from rear-wheel rotation, not an independent timing-gate measurement.

| Variant | Rear-wheel-equivalent 75 m time | Motor-speed RMSE |
|---|---:|---:|
| Pre-slip point-mass baseline | 3.844 s | 1,254 RPM |
| Default explicit-slip model | 3.902 s | 1,178 RPM |
| Tire scale selected only to match crossing time | 3.760 s | 1,186 RPM |
| Slip parameters fitted to motor RPM | 3.962 s | 1,043 RPM |

The default explicit-slip model reduces motor-speed RMSE by about 6% relative
to the point-mass baseline. An RPM-only fit reduces it by about 17%, but drives
the wheel inertia and slip-stiffness parameters to their fitting bounds
(0.15 kg m^2 and 0.40 slip ratio) and still misses the prominent logged RPM
surge and recovery. This is evidence of remaining model-form error rather than
a reason to adopt those fitted values as physical parameters. Likely missing
dynamics include wheel hop/suspension vertical motion and a tire force curve
with post-peak force falloff.

Scaling tire grip by 1.3978 makes the simulated rear-wheel-equivalent crossing
time exactly 3.760 s, but does not improve the motor-speed trace. It therefore
is not a defensible tire calibration by itself.

See [the post-slip overlay](../outputs/real_accel_validation/run13_slip_validation.png)
and [its numerical summary](../outputs/real_accel_validation/run13_slip_validation_summary.csv).

## Executive summary

The supplied `6.24_accel.MF4` is readable and decodes successfully with the supplied VCU, inverter, and HVC DBC files. It is a 56.38-minute acceleration-test session containing 18 launch-control activations rather than one endurance run.

The most important result is that the real car did **not** use one static torque-versus-RPM curve. Its launch controller combined:

- a time-based launch torque schedule;
- a 220 N·m maximum command;
- a 70 kW motor-mechanical power cap; and
- no active traction control during these runs.

Adding the recorded launch schedule and the 70 kW mechanical cap to the existing simulation changes its 75 m result from 3.818 s to 3.844 s. Scaling longitudinal tire grip by 1.0543 makes the simulated scalar crossing time equal the 3.760 s value inferred from rear-wheel rotation.

That last number is **not an independently measured vehicle time**. Both front wheel-speed channels are invalid and zero throughout the log, while the reported VCU speed is derived from motor/rear-wheel speed. The motor trace shows a large speed surge and drop consistent with wheelspin, wheel hop, or another driven-wheel oscillation. Consequently, the complete RPM trace cannot be calibrated as vehicle speed with the current no-slip point-mass model.

No simulator source files were changed during the original analysis. All model variations in the tables below were applied at runtime. The temporary calibration variants were discarded; only this report and the analysis outputs remain.

## Source data and decoding

| Item | Result |
|---|---:|
| File | [`logs/6.24_accel.MF4`](../logs/6.24_accel.MF4) |
| File size | 166,073,098 bytes |
| MDF version | 4.11 |
| Recording start | 2026-06-24 04:54:48 UTC |
| Recording duration | 3,382.853 s (56.38 min) |
| Raw MDF groups | 73 |
| Decoded groups | 126 |
| Decoded channel names | 1,222 |
| Launch-control activations | 18 |

The decoder found the expected VCU, inverter, and HVC signals, including accelerator position, VCU torque command, inverter torque command and feedback, motor RPM, DC bus current and voltage, pack power, pack voltage, launch-controller state, and launch configuration.

## Selected comparison run

Launch 13 was used for the detailed comparison because it is the first activation in the session that reaches approximately 75 m when rear-wheel rotation is integrated. Its recorded configuration was:

| Parameter | Recorded value |
|---|---:|
| Maximum torque | 220 N·m |
| Power limit enabled | Yes |
| Power cap | 70 kW |
| Off-the-line torque | 150 N·m |
| Initial ramp torque | 205 N·m |
| Final ramp torque | 220 N·m |
| Off-the-line interval | 80 ms |
| Ramp duration | 1,000 ms |
| Traction control enabled | No |
| Front-left speed valid | No |
| Front-right speed valid | No |

The configured 3.7 final-drive ratio and 16-inch tire geometry are consistent with motor RPM and the VCU-reported speed. The median ratio inferred from motor RPM and VCU speed is 3.72; the small difference is consistent with effective rolling radius and signal quantization.

## What the real torque curve is doing

The low-speed torque is governed by elapsed launch time, so torque is not a single-valued function of RPM during the first part of the run. Once the power limit becomes active, the envelope follows

\[
T_{cap}(n) = \frac{P}{\omega}
           = \frac{70{,}000}{2\pi n/60}.
\]

This predicts 191 N·m at 3,500 RPM, 167 N·m at 4,000 RPM, and 149 N·m at 4,500 RPM. The logged medians are approximately 188, 166, and 153 N·m, respectively. Therefore, the high-RPM section is explained well by a 70 kW motor-mechanical cap applied below the existing Ryder torque ceiling.

During the selected run, median motor mechanical power in the high-power region is about 72 kW, median pack power is about 74 kW, and peak pack power is about 76 kW. This is consistent with a 70 kW mechanical target plus electrical and drivetrain losses. Modeling the setting as a 70 kW battery-terminal cap would double-count those losses and underpredict motor torque.

See [the observed torque table](../outputs/real_accel_validation/run13_observed_torque_curve.csv) and [the torque comparison plot](../outputs/real_accel_validation/run13_torque_curve_comparison.png).

## Simulation experiments

All variants used the current 650 lb vehicle, 3.7 ratio, 16-inch tire, rotational inertias, aero model, rolling resistance, and load-sensitive tire model unless stated otherwise.

The comparison target below is 3.760 s, obtained by converting logged motor RPM through the simulator's 3.7 ratio and 0.2032 m radius and then integrating rear-wheel-equivalent speed through 0.3 m rollout plus 75 m. It is a useful consistency target, but it is not an official timing-gate measurement.

| Variant | Simulated time | Difference from rear-wheel target | Interpretation |
|---|---:|---:|---|
| Current 80 kW, full-torque baseline | 3.818 s | +0.058 s | Existing model |
| 70 kW mechanical cap, full torque | 3.883 s | +0.123 s | Power cap only |
| Direct replay of logged torque command | 3.728 s | -0.032 s | Command already contains launch and power limiting |
| Recreated launch schedule + 70 kW mechanical cap | 3.844 s | +0.084 s | Best reusable control-law representation |
| Same launch model, tire coefficients × 1.0543 | 3.760 s | 0.000 s | Exact scalar time match |

The 5.43% tire adjustment is small enough to be plausible as a tire-model calibration, but the target itself contains rear-wheel slip. It must not be treated as a validated friction-coefficient measurement.

## Why the complete traces do not match

The selected real motor-speed trace rises to roughly 4,200 RPM near 1.0 s, falls to roughly 2,700 RPM near 2.2 s while torque remains positive, and then rises again. The current simulator cannot produce this shape because it imposes

\[
\omega_{motor} = \frac{v}{r}G,
\]

so motor speed and vehicle speed are always kinematically locked. Excess motor torque is clipped at the tire-force limit rather than accelerating a slipping driven wheel.

The data needed to distinguish wheel motion from vehicle motion are missing in this recording:

- both front wheel-speed validity flags are always false;
- both decoded front wheel-speed values are zero;
- traction control is disabled and inactive; and
- VCU speed follows motor RPM through the configured ratio and tire radius, so it is not an independent reference.

Direct torque replay therefore gives a similar integrated 75 m crossing while still missing the motor-RPM trace by about 1,447 RPM RMS. Recreating the launch control law misses by about 1,254 RPM RMS.

Allowing an optimizer to change mass, tire grip, and drag did not solve the shape error. A whole-trace fit moved to 361.5 kg, 1.36× tire grip, and the 3.0 m² drag-area upper bound, yet still had 1,232 RPM RMS error. Fitting only after 1 s moved to 381.5 kg, 2.44× tire grip, and the same drag bound, with about 1,061 RPM RMS error over that region. Those boundary-seeking, implausible values show that this is a missing-state problem, not a normal scalar-parameter calibration problem.

See [the full run-13 comparison](../outputs/real_accel_validation/run13_sim_comparison.png).

## Recommended next model step

To match this type of real launch trace physically, the model needs a separate driven-wheel state:

1. Integrate rear wheel angular speed from motor torque, final-drive inertia, and tire reaction torque.
2. Calculate longitudinal slip from wheel circumferential speed and vehicle speed.
3. Replace the present instantaneous grip clip with a longitudinal force-versus-slip model.
4. Apply the recorded time-based launch request and 70 kW motor-mechanical cap.
5. Validate vehicle speed against an undriven wheel, GPS, IMU integration, or timing-gate data.

Until an independent vehicle-speed source is available, the current point-mass model can be calibrated to total time or rear-wheel rotation, but a torque/grip calibration is not uniquely identifiable.

## Retained outputs

- [Run-13 simulator comparison](../outputs/real_accel_validation/run13_sim_comparison.png)
- [Observed torque versus drivetrain limits](../outputs/real_accel_validation/run13_torque_curve_comparison.png)
- [Binned observed torque data](../outputs/real_accel_validation/run13_observed_torque_curve.csv)
