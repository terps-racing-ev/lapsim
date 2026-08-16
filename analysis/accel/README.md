# Acceleration analysis

`analyze_acceleration.py` identifies low-curvature sections of the fused track
and replays each straight by distance from its matched entry point. Its default
output is `analysis/accel/output`.

Run from the repository root:

```powershell
.venv\Scripts\python.exe analysis\accel\analyze_acceleration.py `
  --motor-to-wheel-efficiency 0.86 `
  --negative-torque-policy clip
```

Supporting tools:

- `fit_efficiency.py`: infer motor-shaft-to-wheel efficiency from the current
  `straight_samples.csv`.
- `fit_brakes.py`: fit equivalent brake torque from pressure and corrected IMU.
- `tune.py` / `launch_tuning.cmd`: local tuning UI.

The canonical recorded lap, corrected IMU, course map, alignment, and fused
track are loaded from `analysis/data` by default.
