# Analysis

Only the current analysis workflows live here:

- `events/`: common points-producing endurance, acceleration, and skidpad
  entry points. Each accepts a controls profile plus `SpatialTrack` and exports
  a JSON summary and complete telemetry CSV.
- `endurance/`: full-lap distance replay and drivetrain/acceleration plots.
- `accel/`: straight-line distance replay and calibration utilities.
- `aero/active_aero.py`: baseline-versus-active endurance/efficiency comparison;
  reports time, pack energy, endurance points, efficiency points, and the
  combined points delta.
- `tires/plot_camber_slip_mu_3d.py`: 3D pure-longitudinal friction surface for
  the UPC Hoosier R20 model. Run `python analysis/tires/plot_camber_slip_mu_3d.py`;
  the PNG is saved to `outputs/tires/`. The default 743 N load is the static
  rear-wheel load from the vehicle's mass and weight distribution; use
  `--normal-load-n` to inspect another load.
- `tires/plot_camber_slip_angle_mu_3d.py`: matching signed lateral `Fy/Fz`
  surface versus camber and slip angle. Run
  `python analysis/tires/plot_camber_slip_angle_mu_3d.py` to save its PNG in
  `outputs/tires/`.
- `tires/plot_mu_2d.py`: four separate longitudinal/lateral friction line plots
  versus slip and camber. Run `python analysis/tires/plot_mu_2d.py` to save all
  four PNGs in `outputs/tires/`. Camber plots show the peak directional `|μ|`
  over each plotted slip range.
- `data/`: canonical recorded inputs, corrected IMU, brake channels, maps, and
  fused GNSS/IMU track.
- `common.py`: shared recorded-data alignment and control conversion.
- `archive/`: superseded scripts, parameter sweeps, and historical outputs.

The vehicle model uses time as its independent coordinate. Spatial analysis
workflows adapt their track cells to time steps at the solver boundary.

See the workflow-specific READMEs for runnable commands and
`docs/event_simulation.md` for the shared event API.
