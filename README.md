# Python Lap Simulator

Initial goal: calculate the minimum lap time around a fixed racing line.

The simulator will be developed incrementally by the user. The assistant may
review edits, correct small mechanical mistakes, and ask questions about bugs
or modeling assumptions, but should not implement the simulation logic.

## Basic use

```python
from lapsim import LapTimeSolver, SpeedLimitSolver
from lapsim.track import Track
from lapsim.vehicle import Vehicle

vehicle = Vehicle()
track = Track("EV_MI_Endur.xlsx")

speed_limits = SpeedLimitSolver(vehicle).solve(track)
lap = LapTimeSolver(vehicle).solve(
    speed_limits,
    starting_speed_mps=10.0,
)

print(lap.starting_speed_mps)
print(lap.lap_time_s)
print(lap.telemetry.total_energy_kwh)
```

`SpeedLimitSolver` calculates unconstrained local speed ceilings, using the
vehicle's maximum speed wherever cornering does not impose a lower limit.
`LapTimeSolver` then performs the acceleration and braking passes from the
specified starting speed, integrates the resulting profile, and creates
telemetry.

## Chronological simulation

`Vehicle.update_state()` uses a no-slip point-mass model. Motor speed is locked
kinematically to vehicle speed, and requested drive force is clipped by the
rear tires' combined longitudinal/lateral force capacity. Rotational inertia is
retained as equivalent longitudinal mass.
