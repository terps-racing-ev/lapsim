# Straight acceleration comparison

`analyze_straight_acceleration.py` isolates low-curvature sections from the
combined GNSS/IMU track and starts a fresh replay at the first recorded sample
on each straight. Every comparison uses distance from that matched start point
as its x-axis. Per-straight plots cover speed, corrected longitudinal and
lateral acceleration, motor torque, friction-braking force, and HVC pack power. The output also
includes the isolated geometry, distance-aligned
samples, metrics, a track overview, and an assumptions-rich JSON report.
The speed panel also shows vehicle speed inferred from measured motor RPM using
the 3.455 reduction ratio and 0.2032 m rolling radius. This is a kinematic
no-slip reference, so disagreement with GNSS can include driven-wheel slip as
well as timing or effective-radius error.
The longitudinal-acceleration panel includes the time derivative of that
RPM-derived speed. It assumes the driven-wheel acceleration is entirely
longitudinal and therefore does not correct for wheel slip.

## Interactive tuning GUI

Double-click `launch_tuning_gui.cmd`, or run this from the repository root:

```powershell
.venv\Scripts\python.exe analysis\straight_acceleration\tuning_gui.py
```

The local browser GUI provides drag-coefficient and motor-to-wheel-efficiency
controls. Pressing **Run replay** reruns all five distance-indexed straight
comparisons and refreshes the plots. GUI-generated artifacts are kept in
`analysis/straight_acceleration/tuning_gui_output` so the primary analysis
outputs are not overwritten. Close the terminal window or press Ctrl+C to stop
the local server.
Three location views are written: the fused track with highlighted straights,
the official course map by itself, and the fused straights overlaid on the
official map using the saved display alignment.

Run from the repository root after producing `first_lap.csv`, the combined
GNSS/IMU track, and `analysis/corrected_imu/first_lap_corrected_imu.csv`:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe analysis\straight_acceleration\analyze_straight_acceleration.py `
  --motor-to-wheel-efficiency 0.8093392555676976 `
  --negative-torque-policy clip
```

Brake pressure defaults to the vehicle model's combined front-plus-rear axle
torque gain divided by its tire rolling radius. Use
`--brake-force-per-psi-n` only to override that model-derived value.

To analyze only detected straights 1, 2, and 5:

```powershell
.venv\Scripts\python.exe analysis\straight_acceleration\analyze_straight_acceleration.py `
  --straight-numbers 1 2 5 `
  --brake-force-per-psi-n 0 `
  --negative-torque-policy clip `
  --output-dir analysis\straight_acceleration\straights_1_2_5
```

The zero brake scale and torque clipping above reproduce the earlier plot-only
comparison explicitly. They are not a physical brake or regen model.

After writing straight samples, infer motor-shaft-to-wheel efficiency with the
updated vehicle force balance and generate calibration plots:

```powershell
.venv\Scripts\python.exe analysis\straight_acceleration\infer_motor_to_wheel_efficiency.py `
  --samples-csv analysis\straight_acceleration\aero_updated_all\straight_samples.csv
```

The fit retains positive-acceleration, positive-torque, brake-free samples and
solves `F_required = efficiency * T_motor * ratio / tire_radius` through the
origin. `F_required` includes effective longitudinal mass, drag, and rolling
resistance. The output report explicitly treats this as an in-sample
calibration rather than independent validation.

The corrected-IMU CSV must contain `time_s` or `mf4_time_s`, plus these clear
columns (the documented names are preferred):

- `corrected_longitudinal_accel_mps2`
- `corrected_lateral_accel_mps2`
- `corrected_vertical_accel_mps2`

The in-repository correction script currently writes the equivalent
`accel_x_gravity_removed_mps2`, `accel_y_gravity_removed_mps2`, and
`accel_z_gravity_removed_mps2`; they are accepted automatically. Future
producers should prefer the clearer `corrected_*` names above.

Paths are explicit CLI options, so the tool can consume a separately named
output without moving files:

```powershell
.venv\Scripts\python.exe analysis\straight_acceleration\analyze_straight_acceleration.py `
  --corrected-imu-csv path\to\corrected_imu.csv `
  --output-dir analysis\straight_acceleration\output
```

## Alignment and classification

GNSS speed and position are shifted *backward* by `0.3072 s`: the GNSS values
logged at `t + 0.3072` are compared at `t`. Shifting position as well as speed
keeps the distance axis aligned with inverter and IMU signals. The default
comes from a recent RPM/GNSS alignment using motor-speed vehicle conversion
`ratio = 3.455` and rolling radius `0.2032 m`. Override it with
`--gnss-lag-s` only when a new alignment is available.

HVC pack power is shifted backward by `0.09 s` to align its transitions with
motor mechanical power. If the input CSV metadata says that the MF4 converter
already applied this correction, the analysis applies no additional shift.
Override the desired total correction with `--hvc-power-lag-s`.

The braking-force panel converts measured brake pressure using the supplied
`--brake-force-per-psi-n` scale and compares it with the friction-braking force
actually applied by the simulation after its grip limit. This panel excludes
regenerative braking because signed motor torque is not yet supported.
The motor-torque panel also shows friction braking as a negative motor-shaft-
equivalent torque, calculated as
`-F_brake * tire_radius / (ratio * motor_to_wheel_efficiency)`. This is a
force-equivalent comparison scale; the physical friction brakes remain at the
wheels and do not transmit torque through the chain drive.

The recorded GNSS positions are projected to
`analysis/gnss_imu_track/output/gnss_imu_endurance_track.csv`; closed-lap
stations are unwrapped in driving order.
Straight sections are contiguous fused-track cells with `abs(curvature) <= 0.015 1/m`
and at least `18 m` length. Both thresholds are CLI options and written to the
JSON report.

Each replay starts with that straight's shifted measured GNSS speed, corrected
IMU longitudinal/lateral acceleration, and measured HVC SOC. It uses zero path
curvature and torque command by default. Command, feedback, and simulated
delivered torque are retained in every graph; select feedback as the replay
input with `--torque-source feedback`.

Torque and brake pressure are spatial control profiles. During replay they are
interpolated using the simulation's current distance rather than selected by a
recorded timestamp. `--simulation-spatial-step-m` controls how far the model
moves before controls are sampled again. `Vehicle` derives the elapsed time for
each cell internally. Consequently, a recorded torque change at 20 m is applied
at 20 m even when simulated speed differs from recorded speed.

Each replay also seeds the vehicle's public longitudinal-acceleration state with
the corrected IMU value at straight entry. This gives the model the requested
same starting acceleration and supplies the first load-transfer iteration; the
next value is recomputed from force balance because the point-mass model has no
acceleration-memory dynamics. If selected samples contain brake pressure, the
script now refuses to run until `--brake-force-per-psi-n` is supplied. Negative
torque is rejected unless `--negative-torque-policy clip` is selected. These
checks prevent plots from silently presenting unmodeled controls as a physical
replay.
