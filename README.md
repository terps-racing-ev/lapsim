# Formula SAE Event Simulator

Simulate endurance, acceleration, and skidpad from a distance-indexed controls
profile and track. Every event returns estimated points, timing, energy,
completion status, and aligned component telemetry through one result type.

The vehicle is composed from independently replaceable motor, inverter,
chain-drive, battery, aero, chassis, suspension, tire, and brake models. The
included implementations are intentionally simple baselines; their interfaces
are stable seams for higher-fidelity models.

`vehicle_model/vehicle.py` is the top-level coordinator. Component files are
organized by the `electrical`, `aero`, `powertrain`, and `mech` subteam
packages, while the root `vehicle_model` package continues to re-export the
main classes for concise imports.

## Documentation

- [Architecture](docs/architecture.md): package boundaries, data flow, state,
  and parameter ownership.
- [Event simulation](docs/event_simulation.md): the shared profile/track API,
  points models, telemetry contract, and runnable analysis entry points.
- [Extending models](docs/extending_models.md): replace a component, add state,
  and run parameter sweeps.
- [Baseline parameters](docs/model_parameters.md): defaults and the power-flow
  convention.
- [Battery model](docs/battery_model.md): OCV lookup, SOC update, one-RC
  polarization, and voltage limits.
- [Battery voltage validation](docs/battery_voltage_validation.md):
  battery-only endurance fit and residuals.
- [One-RC battery validation](docs/battery_rc_validation.md): dynamic fit,
  chronological holdout, and transient plots.
- [First endurance-lap validation](docs/first_endurance_lap_validation.md):
  current-default recorded-control replay, graphs, and limitations.
- [Open-loop first-lap SOC validation](docs/first_lap_soc_validation.md):
  distance-indexed controls-plus-curvature replay and comparison with HVC SOC.
- [Endurance analysis](analysis/endurance/README.md): full-lap distance replay,
  drivetrain signals, acceleration, braking, and wheel slip.
- [Straight acceleration comparison](analysis/accel/README.md):
  corrected-IMU versus recorded-control replay on map-defined straights.
- [Endurance torque-profile optimizer](docs/endurance_optimizer.md): reusable
  track/vehicle architecture, Michigan 2026 scoring, sweeps, and limitations.

## Basic use

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
print(result.scoring_time_s)
print(result.telemetry["motor.speed_rpm"])
```

Swap in `simulate_skidpad` or `simulate_endurance` without changing the
profile, track, result, or telemetry concepts. The older speed-limit,
minimum-lap-time, and recorded-replay APIs remain available as lower-level
physics and validation tools.

Parameters live on the component that owns them:

```python
vehicle.drivetrain.motor.peak_power_w = 60_000.0
vehicle.drivetrain.inverter.efficiency = 0.96
vehicle.drivetrain.chain_drive.ratio = 4.1
vehicle.aero.frontal_area_m2 = 0.658
vehicle.aero.drag_coefficient = 1.0
vehicle.aero.lift_coefficient = -2.82
vehicle.validate()
```

To add a detailed model, supply an object implementing the corresponding
protocol in `vehicle_model.interfaces`. No solver changes are required while
the interface still describes the needed physics.

## Inferring delivered torque from an MF4 log

`scripts/infer_delivered_torque.py` decodes GNSS ground speed, fits a local
quadratic to calculate acceleration, and applies the inverse longitudinal
vehicle model:

```text
wheel force = effective mass * acceleration + drag + rolling resistance
motor torque = wheel force * tire radius / (ratio * differential efficiency)
```

Install the MF4 extra with Python 3.11-3.13, then supply the CAN database:

```powershell
python -m pip install -e ".[mf4]"
python scripts/infer_delivered_torque.py logs/6.24_accel.MF4 `
  --dbc path/to/can9-database.dbc `
  --feedback-dbc path/to/inverter.dbc `
  --hvc-dbc path/to/hvc.dbc `
  --start-s 1759.04515 --end-s 1763.04675 `
  --motor-power-limit-kw 70 `
  --output outputs/real_accel_validation/run13_inferred_torque.csv `
  --plot outputs/real_accel_validation/run13_inferred_torque.png `
  --motor-speed-derivative-plot `
    outputs/real_accel_validation/run13_motor_speed_derivative.png `
  --slip-comparison-plot `
    outputs/real_accel_validation/run13_slip_aware_torque_comparison.png
```

The CSV retains signed inverse-model torque and a propulsion-only version. A
negative signed value describes the wheel force required by the measured
deceleration; it is not proof that regenerative braking was active. It also
reports both wheel-equivalent torque reflected through the gear ratio and the
higher motor-shaft torque after correcting for post-motor differential loss.
The inference uses GNSS ground-speed acceleration and the simulator's no-slip
equivalent rotating mass. During wheelspin it does not include torque that goes
into accelerating the driven wheels relative to the vehicle, so it can
underestimate delivered motor torque.

When an HVC DBC is supplied, the slip-aware comparison also back-calculates
battery-terminal power from measured motor speed and inferred motor-shaft
torque. It divides motor mechanical power by the configured motor and inverter
efficiencies; differential loss is not applied again because it is already in
the motor-shaft torque inference. Measured battery power is calculated as
`HVC_Current_High_A * HVC_Batt_Voltage_V` so high-power operation does not
saturate the current measurement.

`SpeedLimitSolver` calculates unconstrained local speed ceilings, using the
vehicle's maximum speed wherever cornering does not impose a lower limit.
`LapTimeSolver` then performs the acceleration and braking passes from the
specified starting speed, integrates the resulting profile, and creates
telemetry.

## Distance-domain simulation

`Vehicle.update_state(controls, distance_step_m)` advances exactly one positive
spatial cell. It solves `v_next^2 = v^2 + 2*a*distance_step`, derives elapsed
time internally from cell-average speed, and passes that time to the battery,
drivetrain, brake, suspension, tire, and aero states. The tire model resolves
normal load, lateral force, drive/brake force, combined capacity, and slip at
each contact patch. Motor speed follows the average driven-tire surface speed,
and rotational inertia is retained as equivalent longitudinal mass.

Explicit control sequences can be run with the public `replay_controls()` API.
`RecordedLap` loads synchronized track and control data for validation.

Replay telemetry is a namespaced mapping populated by the vehicle and each
component:

```python
telemetry = replay_controls(vehicle, controls, distance_step_m=0.25)
speed_mps = telemetry["vehicle.speed_mps"]
traction_limited = telemetry["limits.traction_active"]
all_channels = telemetry.as_dict()
```

Higher-fidelity component models add signals through `update_telemetry()`;
the replay and recorder do not require a central schema change.

## First endurance-lap validation

Extract the first completed lap from the competition MF4, then replay its
distance-indexed controls through the current vehicle model:

```powershell
python scripts/extract_first_endurance_lap.py
$env:PYTHONPATH='src'
python scripts/validate_first_endurance_lap.py
```

The validation performs no parameter fitting and does not alter vehicle
defaults. It writes plots, telemetry, and metrics under
`outputs/endurance_validation/first_lap_soc_open_loop/`. The replay assumptions
are documented in `docs/first_lap_soc_validation.md`.

## Endurance torque optimization

The endurance solver optimizes a periodic normalized motor-torque request on
an injected spatial track, fresh vehicle factory, and scoring model. Run the
Michigan 2026 benchmark and its spatial-resolution verification with:

```powershell
$env:PYTHONPATH=(Resolve-Path src)
.\.venv\Scripts\python.exe scripts\optimize_mi6_endurance.py
.\.venv\Scripts\python.exe scripts\verify_endurance_profile.py
```

The implementation is grouped under `lapsim.courses`, `lapsim.solvers`,
`lapsim.events`, and `lapsim.optimization`. The root `lapsim` package is the
stable public facade; Michigan data remains confined to scoring presets and
analysis inputs.

## 75 m acceleration simulation

Run the standing-start, full-torque acceleration model in 0.5 m spatial
segments and export its complete telemetry, metrics, and dashboard with:

```powershell
.\.venv\Scripts\python.exe scripts\simulate_75m_acceleration.py
```

Use `--distance-m`, `--segment-length-m`, and `--output-dir` to change the
experiment without editing the script.
