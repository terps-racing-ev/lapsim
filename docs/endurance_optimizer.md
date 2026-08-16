# Endurance torque-profile optimizer

The endurance optimizer searches for a periodic motor-torque request that
maximizes an injected scoring model. The Michigan 2026 script is one preset;
the solver itself does not contain Michigan track geometry, event constants,
or a fixed vehicle.

## Architecture

```text
Track or custom geometry
        |
        v
SpatialTrack ---- fresh Vehicle factory
        |                 |
        +-------> PathConstraintSolver
                         |
                         v
                corner/braking ceilings
                         |
torque variables -> periodic normalized torque profile
                         |
                         v
                 EnduranceSimulator
                         |
           time, energy, SOC, telemetry
                         |
                         v
                    ScoringModel
                         |
                         v
              derivative-free optimizer
```

`SpatialTrack` is the solver-facing racing line. It stores cell lengths,
curvature, and plotting coordinates, and can be built from any existing
`Track`. `PathConstraintSolver` calculates the local lateral limit and a
cyclic backward braking ceiling once for each track/vehicle combination.

The optimization variables are normalized motor-torque requests from zero to
one at evenly spaced distance knots. Linear periodic interpolation produces a
request at every path cell. At runtime the request is multiplied by the
motor's RPM-dependent torque limit, then the drivetrain, battery, and tire
models apply their physical limits. This keeps a profile parameterization
valid when a sweep changes motor data, gearing, battery limits, mass, or tires.

`EnduranceSimulator` follows the prescribed curvature and preserves component
state for the entire run. The torque profile controls propulsion. A
deterministic path controller reduces drive torque, coasts, or uses friction
braking when needed to stay below the path ceiling. Therefore braking is not
an extra optimizer variable, and a profile that stays on throttle too long is
penalized through wasted energy.

Every objective evaluation receives a fresh `Vehicle` from a factory. This is
required because the battery and future thermal models are stateful. Reusing a
mutated vehicle would make candidates depend on evaluation order.

## Michigan 2026 benchmark

Run the included benchmark from the repository root:

```powershell
$env:PYTHONPATH=(Resolve-Path src)
.\.venv\Scripts\python.exe scripts\optimize_mi6_endurance.py
```

The benchmark uses `EV_MI_Endur.xlsx`, 22 laps, and the 2026 Michigan Electric
endurance and efficiency constants stored in `FSAE_2026_MI6_SCORING`. It writes
the optimized torque map, full telemetry, metrics, and plots to
`outputs/endurance_optimization/mi6_2026/`.

The coarse differential-evolution search was refined with COBYQA on 1 m
spatial cells. That bounded search found 296.46 modeled combined points:
253.03 endurance and 43.43 efficiency points. It completes 22 laps in
1349.08 s using 5.426 kWh. The best constant normalized torque request in the
same 1 m comparison sweep scored 288.84 points, while full torque depleted the
modeled pack after 16 laps. The search stopped at its configured evaluation
budget, so this is the best profile found, not proof of a global maximum.

These saved results predate the corrected 3.455 chain ratio, tire-owned 8 in
rolling radius, and explicit 0.65800 m^2 / Cd 1.0 / Cl -2.82 aero inputs. They
used a 3.7 ratio, 0.21622 m fitted radius, CdA 1.803 m^2, and downforce-area
magnitude 3.43 m^2. They are retained as historical artifacts only. The torque
profile and score must be re-optimized before they are treated as results for
the corrected vehicle.

Re-evaluate the saved profile at multiple spatial resolutions with:

```powershell
$env:PYTHONPATH=(Resolve-Path src)
.\.venv\Scripts\python.exe scripts\verify_endurance_profile.py
```

The final saved profile scores 295.83 points when replayed at 0.5 m and 296.46
at 1 m, a 0.63-point numerical spread. Coarser cells remain useful for global
search but should not be used for the final reported score.

To reproduce that final sensitivity plot, pass
`--result-dir outputs\endurance_optimization\mi6_2026_verified` and
`--cell-lengths-m 0.5 0.75 1 1.5 2` to the verification script.

For a coarse-to-fine run, use each stage's metrics to seed a bounded COBYQA
polish on smaller spatial cells:

```powershell
.\.venv\Scripts\python.exe scripts\optimize_mi6_endurance.py `
  --cell-length-m 3 `
  --control-points 8 `
  --initial-profile-metrics outputs\endurance_optimization\mi6_2026\optimization_metrics.json `
  --local-polish-evaluations 80 `
  --output-dir outputs\endurance_optimization\mi6_2026_fine

.\.venv\Scripts\python.exe scripts\optimize_mi6_endurance.py `
  --cell-length-m 1 `
  --control-points 8 `
  --initial-profile-metrics outputs\endurance_optimization\mi6_2026_fine\optimization_metrics.json `
  --local-polish-evaluations 60 `
  --output-dir outputs\endurance_optimization\mi6_2026_verified
```

## Different tracks

The Michigan command accepts another compatible track spreadsheet directly:

```powershell
.\.venv\Scripts\python.exe scripts\optimize_mi6_endurance.py `
  --track path\to\another_track.xlsx `
  --output-dir outputs\endurance_optimization\another_track
```

That command still uses Michigan scoring. For a different event, construct a
new `FSAEEnduranceEfficiencyScoring` preset or implement the small
`ScoringModel` protocol.

For geometry that is not a straight/constant-radius spreadsheet, construct a
`SpatialTrack` directly from spatial cells. A future GNSS/spline importer can
produce the same generic type without changing the simulator or optimizer.

## Vehicle and multi-parameter sweeps

Parameter sweeps create a closure that returns a new vehicle for every
candidate:

```python
from lapsim import (
    EnduranceRunConfig,
    EnduranceTorqueOptimizer,
    FSAE_2026_MI6_SCORING,
    SpatialTrack,
)
from lapsim.track import Track
from vehicle_model import Vehicle

track = SpatialTrack.from_track(
    Track("EV_MI_Endur.xlsx"),
    maximum_cell_length_m=4.0,
)

for mass_kg in (285.0, 295.0, 305.0):
    for motor_power_w in (60_000.0, 70_000.0, 80_000.0):
        def factory(mass_kg=mass_kg, motor_power_w=motor_power_w):
            vehicle = Vehicle()
            vehicle.mass_kg = mass_kg
            vehicle.drivetrain.motor.peak_power_w = motor_power_w
            vehicle.validate()
            return vehicle

        optimizer = EnduranceTorqueOptimizer(
            vehicle_factory=factory,
            track=track,
            scoring_model=FSAE_2026_MI6_SCORING,
            run_config=EnduranceRunConfig(laps=22),
        )
        result = optimizer.optimize()
```

The path constraints are precomputed once when each optimizer is constructed,
so changes to mass, tires, brakes, aero, or other physics correctly produce a
new ceiling. A sweep that only changes scoring can reuse the same track and
vehicle setup. Each sweep result should retain its vehicle parameters, track
source, numerical resolution, seed, and optimizer budget for reproducibility.

## Current interpretation and limitations

The output is model-optimal, not a predicted competition score. The default
vehicle was calibrated using the same first competition lap, so this event is
not independent validation. The current model omits wheel slip, yaw dynamics,
road grade, thermal derating, auxiliary power, and regenerative braking. The
same torque profile repeats every lap, although electrical state evolves
continuously. Driver-change stopped time and battery relaxation are not yet
modeled.

Before using a profile on the car, validate held-out laps, add the missing
limits that can bind in endurance, and run finer spatial and optimizer-budget
sensitivity checks.
