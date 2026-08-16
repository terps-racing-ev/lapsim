# Corrected IMU data

`generate_corrected_imu.py` turns the first-lap raw IMU accelerometer axes into
fixed, level vehicle axes and removes the measured stationary gravity vector.
It does not estimate dynamic pitch, roll, or yaw, and it does not smooth the
IMU data.

Run it from the `python_lapsim` repository root:

```powershell
.\.venv\Scripts\python.exe analysis\corrected_imu\generate_corrected_imu.py
```

The default correction target is
`analysis/first_endurance_lap/first_lap.csv`. That file starts while the car is
moving, so the default calibration source is instead the complete recording,
`analysis/mf4_to_csv/endurance_selected.csv`. The default MF4-time calibration
window is 450--465 s: a pre-lap stationary plateau. Within that explicit window
the script only accepts samples at or below 0.30 m/s GNSS speed and at or below
5 deg/s 3-axis angular-rate magnitude. It calculates a component-wise median
static gravity vector from the accepted samples, then derives the correction.

For the current recording the plateau is `[2.875, -0.750, 9.000] m/s²`; this
is selected from the data rather than hard-coded as an angle.

## Outputs

- `first_lap_corrected_imu.csv`: all source first-lap columns plus the six
  corrected acceleration columns below.
- `first_lap_corrected_imu.json`: full input/output provenance, stationary
  selection, gravity estimate, derived and applied angles, rotation matrix,
  sign conventions, and residual stationary validation.
- `first_lap_raw_vs_corrected_axes.png`: raw, leveled, and gravity-removed
  axes over the first lap.
- `stationary_gravity_validation.png`: the source plateau before and after
  correction; gravity-removed stationary axes should be near zero.

The output CSV additions are:

- `accel_x_leveled_mps2`, `accel_y_leveled_mps2`, `accel_z_leveled_mps2`:
  fixed level axes that still include +gravity in Z.
- `accel_x_gravity_removed_mps2`, `accel_y_gravity_removed_mps2`,
  `accel_z_gravity_removed_mps2`: fixed level axes after subtracting the
  measured stationary gravity magnitude from Z. Use X and Y directly for
  longitudinal and lateral acceleration comparisons.
- `corrected_longitudinal_accel_mps2`, `corrected_lateral_accel_mps2`, and
  `corrected_vertical_accel_mps2`: semantic aliases of those gravity-removed
  X/Y/Z channels for downstream analysis interfaces.

## Axis and rotation convention

The assumed vehicle/sensor mounting convention is right-handed: `+X` forward
(longitudinal), `+Y` left (lateral), and `+Z` up. Stationary raw acceleration
is therefore expected to point approximately toward `+Z`; the current sensor
has gravity leaking into `+X` and `-Y`.

The script applies an active rotation to each acceleration vector: first
`Rx(roll)`, then `Ry(pitch)`, such that static gravity maps to `[0, 0, |g|]`.
The documented pitch and roll are *sensor-to-level correction angles*, not a
vehicle attitude measurement. No yaw correction is made. If a future mounting
check finds a different sign convention, do not reinterpret the current CSV;
re-run the stage with the correct convention and record it in the metadata.

## Overrides

Use a different stationary range or explicit correction angles when justified:

```powershell
.\.venv\Scripts\python.exe analysis\corrected_imu\generate_corrected_imu.py `
  --calibration-start-mf4-time-s 450 `
  --calibration-end-mf4-time-s 465 `
  --pitch-deg -17.6 --roll-deg -4.8
```

`--calibration-input`, positional `input_csv`, `--output`, `--metadata`, and
the two plot paths allow the stage to be reused for another recording without
editing the script.
