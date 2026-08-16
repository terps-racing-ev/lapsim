# Full-lap comparison

`analyze_full_lap.py` replays the first endurance lap entirely by projected
track station. Front VCU pressure, rear MOBO pressure, motor torque, and track
curvature are sampled at the same distance coordinate. `Vehicle.update_state`
advances one spatial cell and derives elapsed time internally for battery,
drivetrain, brake, and slip state. Logged negative motor torque is reflected to
rear-axle backdrive braking.

The analysis also writes two lateral-validation plots on recorded lap station:

- `imu_y_vs_map_curvature_distance.png` compares corrected IMU Y with the
  already-derived map track curvature.
- `imu_y_vs_gnss_curvature_distance.png` compares corrected IMU Y with signed
  curvature calculated from the already-filtered GNSS path. GNSS geometry is
  resampled every 0.5 m and differentiated with an 11 m periodic
  Savitzky-Golay window. This retains the rapid direction changes in the
  300 m chicane while limiting position-quantization noise.
- `imu_y_vs_map_v2_curvature_distance.png` and
  `imu_y_vs_gnss_v2_curvature_distance.png` convert each curvature to expected
  lateral acceleration with `v²κ`. Here `v` is specifically the post-offset
  GNSS speed after the configured GNSS lag correction.
- `imu_y_vs_normalized_v2_curvature_distance.png` shows both `v²κ` channels in
  aligned panels after matching each channel's full-lap mean and standard
  deviation to corrected IMU Y. This changes amplitude and offset only; it does
  not remove curvature spikes or shift either channel in distance.
- `gnss_curvature_smoothing_on_official_map.png` overlays signed GNSS-derived
  curvature on the manually aligned official map in three panels: no additional
  curvature smoothing, a 5 m window, and a 10 m window. A fourth panel overlays
  IMU-equivalent curvature (`a_y/v²`, outer band) with the selected 11 m GNSS
  curvature (inner band). All panels use post-offset speed and one symmetric
  color scale with 98th-percentile clipping for readability.
- `imu_y_vs_gnss_smoothing_windows_distance.png` overlays corrected IMU Y with
  normalized post-offset GNSS `v²κ` using no additional smoothing and 5 m,
  10 m, and 11 m windows.

The distance-domain figure compares both traces at the same recorded lap
station. Duplicate projected stations are collapsed while retaining the latest
zero-order-held control. `first_lap_all_data.csv` includes every numeric log
channel, aligned corrected IMU, track curvature, and all model telemetry and
active-limit channels.

Run from the repository root:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe analysis\full_lap_comparison\analyze_full_lap.py `
  --negative-torque-policy rear-brake `
  --brake-pressure-model firmware-force-map `
  --brake-deadband-psi 5 `
  --brake-gain-count-per-axle 1 `
  --constant-tire-mu 1.8 `
  --cornering-drag-coefficient 0.036 `
  --drag-coefficient 2.5 `
  --motor-to-wheel-efficiency 0.86
```

The replay can use the configured front and rear torque-per-pressure hardware
gains directly. The optional firmware force map
uses the front polynomial and rear linear terms found in TREV4-Controls' Ryder
brake-balance calculation; interpreting those terms as newtons is an explicit
analysis hypothesis, not a replacement for brake-dyno validation. The fitted
cornering-drag term represents tire scrub; it is separate from mechanical
braking and does not force the simulation onto the recorded speed path.

`braking_accuracy_distance.png` is the compact validation figure for speed,
longitudinal acceleration, both pressure circuits, front/rear applied braking,
and rear regenerative braking.
