# Points-producing event simulation

The supported analysis boundary is now the same for endurance, acceleration,
and skidpad:

```text
Vehicle + SpatialTrack + ControlsProfile
                    |
                    v
             simulate_<event>()
                    |
                    v
 EventResult(points, time, energy, status, telemetry)
```

Use `ControlsProfile` for any object that implements
`controls_at(distance_m) -> Controls`. `ConstantControlsProfile` and
`PiecewiseLinearControlsProfile` cover common analysis inputs. Track curvature
is authoritative: event simulators supply path-following steering while the
profile supplies motor torque, hydraulic brake pressure, and regen requests.

Every call returns an `EventResult`. Its `telemetry` is always a `Telemetry`
mapping with aligned vehicle and component channels, even when a run fails.
The common summary fields are:

- `completed`, `failure_reason`, `completed_laps`, and `lap_times_s`
- `elapsed_time_s`, `scoring_time_s`, `distance_m`, and `energy_kwh`
- `estimated_points`, `maximum_points`, and `point_breakdown`
- `telemetry`

## Programmatic use

```python
from lapsim import (
    ConstantControlsProfile,
    Controls,
    SpatialTrack,
    simulate_acceleration,
)
from vehicle_model import Vehicle

track = SpatialTrack.from_cells(
    cell_length_m=(0.5,) * 150,
    curvature_per_m=(0.0,) * 150,
    closed=False,
)
profile = ConstantControlsProfile(
    Controls(motor_torque_request_nm=230.0)
)
result = simulate_acceleration(Vehicle(), track, profile)

print(result.estimated_points)
print(result.telemetry["battery.terminal_voltage_v"])
```

The same profile/track contract is accepted by `simulate_skidpad` and
`simulate_endurance`. Endurance additionally accepts a normalized
`TorqueProfile` for compatibility with the torque optimizer; that path uses
the automatic braking controller.

The standard skidpad helper uses a 9.125 m vehicle-path radius: the centerline
between the 15.25 m inner circle and 21.25 m outer circle.

## Analysis entry points

Each event has a small script that can also be imported by sweeps:

```powershell
$env:PYTHONPATH=(Resolve-Path src)
python analysis/events/acceleration_points.py
python analysis/events/skidpad_points.py
python analysis/events/endurance_points.py
```

Each script writes `<event>_summary.json` and `<event>_telemetry.csv`. Their
`run(profile, track, ...)` functions return the same `EventResult` directly.

## Scoring references

Acceleration and skidpad use explicit `TimedEventScoring` objects. The default
Michigan 2026 Electric references are taken from the official final results:

- Acceleration: 3.697 s minimum, 5.546 s maximum, 100 maximum points, 4.5
  minimum completion points, first-power time ratio.
- Skidpad: 4.782 s minimum, 5.978 s maximum, 75 maximum points, 3.5 minimum
  completion points, squared time ratio.

Endurance continues to use `FSAE_2026_MI6_SCORING`, which exposes endurance
and efficiency points separately in `point_breakdown`.

Competition-dependent reference times must remain injected scoring data. Use a
new `TimedEventScoring` instance when estimating points for another event.
