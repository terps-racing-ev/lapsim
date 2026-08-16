# Canonical analysis data

- `telemetry/`: selected endurance telemetry extracted from the source log.
- `lap/`: first-lap channels, official map, alignment, and map-derived track.
- `imu/`: gravity-corrected first-lap IMU channels.
- `track/`: current fused GNSS/IMU endurance track.
- `brakes/`: high-rate front/rear brake and control channels.

The scripts that originally generated these artifacts are retained under
`analysis/archive/scripts/data_preparation`.
