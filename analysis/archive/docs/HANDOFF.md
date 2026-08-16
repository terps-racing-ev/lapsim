# Recorded-lap analysis handoff

## Pipeline

1. `mf4_to_csv/convert_mf4_to_csv.py` extracts selected signals at the requested
   frequency.
2. `first_endurance_lap/generate_first_lap.py` selects the first lap.
3. `first_endurance_lap/build_track_from_map.py` produces the map-derived
   `SpatialTrack`.
4. `corrected_imu/generate_corrected_imu.py` estimates the stationary sensor
   orientation and writes gravity-corrected acceleration.
5. `straight_acceleration/analyze_straight_acceleration.py` resets the vehicle at
   each detected straight entry and compares local longitudinal behavior.
6. `full_lap_comparison/analyze_full_lap.py` projects all controls onto station
   and performs one distance-domain replay across the complete recorded lap.

Shared CSV loading, alignment, track projection, curvature lookup, and recorded
control conversion live in `analysis/replay_common.py`.

## Important control contract

The log contains brake pressure, while `Controls` consumes friction-brake force.
The full-lap analysis supports independent front/rear pressure using either the
supplied linear hardware gains or the opt-in firmware force map. The firmware
map's force units still require brake-dyno confirmation. Straight-only analysis
retains the explicit scalar `--brake-force-per-psi-n` adapter.

Negative recorded motor torque is rejected by default. Full-lap analysis can
map it to rear motor/backdrive braking with `--negative-torque-policy rear-brake`;
straight-only analysis can explicitly clip it.

The current distance-native full-lap result is in
`full_lap_comparison/braking_corrected_distance_native/`.

## Current alignment assumptions

- GNSS speed is shifted backward by 0.3072 seconds.
- IMU pitch/roll correction comes from the stationary window documented in
  `corrected_imu/first_lap_corrected_imu_metadata.json`.
- Map curvature is converted to an equivalent bicycle-model road-wheel angle.
- Torque feedback, not command, drives the full-lap replay.

## Recommended next task

Log rear brake pressure at 100 Hz and validate front/rear pressure-to-wheel-force
on a brake dyno. Those measurements would confirm or replace the current
firmware-map inference without changing the distance-domain vehicle contract.
