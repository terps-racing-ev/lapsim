# Analysis

Only the current analysis workflows live here:

- `events/`: common points-producing endurance, acceleration, and skidpad
  entry points. Each accepts a controls profile plus `SpatialTrack` and exports
  a JSON summary and complete telemetry CSV.
- `endurance/`: full-lap distance replay and drivetrain/acceleration plots.
- `accel/`: straight-line distance replay and calibration utilities.
- `data/`: canonical recorded inputs, corrected IMU, brake channels, maps, and
  fused GNSS/IMU track.
- `common.py`: shared recorded-data alignment and control conversion.
- `archive/`: superseded scripts, parameter sweeps, and historical outputs.

Both active simulations use distance as their independent coordinate.
`Vehicle` derives elapsed time internally for stateful components.

See the workflow-specific READMEs for runnable commands and
`docs/event_simulation.md` for the shared event API.
