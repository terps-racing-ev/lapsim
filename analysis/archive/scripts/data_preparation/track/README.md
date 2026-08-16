# GNSS + corrected-IMU endurance track

This analysis reconstructs the first endurance-lap centerline without asking
slow, noisy GNSS samples to describe every corner. GNSS provides the
long-wavelength position and heading. The corrected lateral accelerometer
channel provides local signed curvature:

```text
curvature = corrected_lateral_accel_mps2 / gps_speed_mps^2
```

The script first shifts GNSS speed by the recorded 0.3072 s timing offset. It
then integrates curvature into heading and position and applies only
low-frequency GNSS heading and position corrections. This preserves the IMU's
shorter corner features while preventing inertial drift. Samples below 4 m/s
are excluded from the division and spatially interpolated because curvature
becomes ill-conditioned as speed approaches zero.

By default, the curvature scale is calibrated so the integrated heading change
is one complete turn over this closed lap. Without this correction the current
IMU/speed combination produces only about 60% of a turn, systematically making
all reconstructed corners too shallow. Use `--curvature-gain VALUE` to replace
the automatic closed-lap calibration for an open course or a different winding
number.

The final geometry also uses the already extracted official centerline as an
85% constraint. This is intentional: telemetry reconstructs the driven racing
line, which can shortcut or entirely miss a bend in the schematic course
centerline. The constraint makes the delivered track curve-complete while GNSS
retains its placement/scale role and corrected IMU Y retains measured corner
direction and sharpness. Set `--reference-geometry-weight 0` for a strictly
telemetry-only reconstruction, or adjust the weight between 0 and 1.

Run from the `python_lapsim` repository root:

```powershell
.\.venv\Scripts\python.exe analysis\gnss_imu_track\generate_gnss_imu_track.py
```

Default outputs in `analysis/gnss_imu_track/output` are:

- `gnss_imu_endurance_track.csv`: uniform 0.5 m fused geometry, corrected IMU
  Y, curvature, heading, and the low-frequency GNSS correction.
- `gnss_imu_endurance_track.json`: inputs, parameters, method, quality metrics,
  and limitations.
- `gnss_imu_track_on_official_map.png`: fused product and filtered GNSS over
  the official endurance map, plus the corrected-IMU curvature diagnostic.

The plot uses the existing manual affine alignment only for display. It does
not warp the output track coordinates. The official course drawing is
schematic rather than georeferenced, so exact line-on-line agreement is not a
survey-accuracy test.
