# Endurance analysis

`analyze_endurance.py` projects recorded controls onto the fused GNSS/IMU
track and replays the complete lap by distance. Its default output is
`analysis/endurance/output`.

Run the current calibrated scenario from the repository root:

```powershell
.venv\Scripts\python.exe analysis\endurance\analyze_endurance.py `
  --negative-torque-policy rear-brake `
  --brake-pressure-model firmware-force-map `
  --brake-deadband-psi 5 `
  --brake-gain-count-per-axle 1 `
  --constant-tire-mu 1.8 `
  --cornering-drag-coefficient 0.036 `
  --drag-coefficient 2.5 `
  --motor-to-wheel-efficiency 0.86
```

Important outputs:

- `first_lap_comparison_distance.png`: speed, longitudinal acceleration,
  motor speed, pack power, torque, and brake pressure.
- `drivetrain_signals_vs_distance.png`: drivetrain comparison dashboard.
- `acceleration_xy_vs_distance.png`: corrected IMU X/Y against simulation.
- `front_rear_braking_vs_distance.png`: independent axle braking.
- `wheel_slip_vs_distance.png`: simple wheel-slip model comparison.
- `first_lap_metrics.json`: assumptions and fit metrics.

Generate the automatic corner-by-corner accuracy report after the endurance
replay:

```powershell
.venv\Scripts\python.exe analysis\endurance\plot_corner_accuracy.py
```

This writes a selected-corner map, RMSE summary, detailed speed/longitudinal/
lateral traces, and machine-readable metrics to
`analysis/endurance/output/corner_accuracy`.

`plot_drivetrain.py` rebuilds the drivetrain dashboard from
`first_lap_all_data.csv`. `plot_curvature_window.py` creates focused corrected
IMU Y versus track-curvature/map windows.
