# Endurance state-update latency and real-time feasibility

## Executive result

The endurance simulation now uses hydraulic brake pressure at the control
boundary. The brake component converts front/rear PSI to axle force once per
vehicle update, and the endurance path controller no longer performs repeated
deep-copy/probe simulations.

On the benchmark laptop, both the vehicle update and the complete endurance
cell are comfortably inside a 10 ms soft-real-time budget at p95:

| Work measured | Mean | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|
| Committed `Vehicle.update_state()` | 0.295 ms | 0.480 ms | 1.582 ms | 3.833 ms |
| Complete endurance cell, no telemetry | 0.446 ms | 0.702 ms | 2.045 ms | 4.143 ms |
| Complete endurance cell, telemetry enabled | 0.528 ms | 0.852 ms | 2.331 ms | 50.941 ms |

The complete no-telemetry solver ran **62.9 times faster than simulated vehicle
time**. None of the 5,934 measured cells took longer to compute than its own
simulated physical timestep. No no-telemetry sample exceeded 10 ms; one
telemetry-enabled sample did, corresponding to a 0.0169% observed miss rate.
Windows and ordinary CPython remain soft-real-time rather than hard-real-time
environments.

## Brake-control architecture

`Controls` now contains:

- `front_brake_pressure_psi`
- `rear_brake_pressure_psi`
- `rear_regenerative_brake_force_request_n`

Regenerative braking stays a force request because it is not hydraulic.
Friction-brake force is calculated exclusively by the brake component:

\[
P_{f,eff}=\max(P_f-P_{deadband},0)
\]

\[
F_{f,request}=\frac{P_{f,eff}K_f}{r_{tire}}
\]

with the equivalent calculation for the rear axle. The vehicle then limits
each requested axle force using current normal load, lateral-force usage, tire
capacity, and optional slip relaxation.

The brake component owns both the default linear hardware-gain map and the
existing firmware force map. Recorded replay passes measured pressures through
unchanged and configures the selected map on the vehicle; it no longer performs
an independent per-sample pressure-to-force calculation.

The offline endurance driver still starts with the force needed to reach the
next precomputed speed ceiling. It allocates that request across the axles,
uses the brake component's inverse map to produce a PSI command, and commits
one vehicle update. A 1% configurable force-command margin absorbs the small
difference between the controller's load-transfer estimate and the committed
vehicle solve without an iterative state probe.

## Benchmark definition

The benchmark used the fused endurance track at
`analysis/data/track/gnss_imu_endurance_track.csv`: 989.0 m split into 1,978
cells of 0.5 m. It ran a constant 0.5 normalized torque request for one warm-up
lap followed by three measured laps, giving 5,934 latency samples per telemetry
mode. Garbage collection remained enabled.

The test machine was an Intel Core Ultra 7 256V with eight logical CPUs,
Windows 11, and 64-bit CPython 3.14.6. Timing used `perf_counter_ns`. No CPU
affinity, real-time priority, or background-load isolation was applied.

One complete-cell sample begins at a torque-profile request and ends at the
next cell's request. It includes path control, pressure calculation, the
committed state update, limit checks, energy accounting, and optional
telemetry. The simulated physical timestep averaged 28.05 ms per cell, with a
minimum of 14.94 ms and p95 of 44.88 ms.

## Current cost profile

The unprofiled three-lap direct timers show:

| Logical stage | Mean per cell | Share of wall time |
|---|---:|---:|
| Committed `Vehicle.update_state()` | 0.295 ms | 66.1% |
| Initial path drive/load-transfer calculation | 0.0555 ms | 12.4% |
| Axle brake-request calculation when active | 0.0069 ms/call | 0.39% |
| Pressure mapping, control construction, checks, and timing overhead | 0.0940 ms | 21.1% |

The old brake closure accounted for 88.7% of wall time. It has been removed:
there are now zero temporary vehicle probe updates per endurance cell.

A separate one-lap deterministic profile is slower than normal execution and
is used only for attribution. Its self-time proportions were:

| Function group | Self-time proportion |
|---|---:|
| Python/SciPy/runtime overhead | 38.19% |
| Tire force capacity | 27.46% |
| Brake force and slip | 12.59% |
| Vehicle force balance/spatial integration | 8.95% |
| Powertrain and wheel slip | 5.19% |
| Endurance path controller | 3.29% |
| Suspension/load transfer | 2.13% |
| Battery/electrical | 1.90% |
| Aerodynamics | 0.23% |
| Torque-profile interpolation | 0.08% |

## On-car recommendation

The pressure-driven `Vehicle.update_state()` path is suitable for a 100 Hz
soft-real-time prototype on hardware with performance comparable to the
benchmark laptop. Before deploying it as a deadline-critical process:

1. Repeat the benchmark on the actual onboard computer.
2. Include sensor acquisition, CAN parsing, logging, and competing processes.
3. Test under sustained thermal load and collect a much longer tail-latency
   distribution.
4. Keep path-constraint generation offline and cached; its measured startup
   cost was 67.94 s.

The isolated workstation measurements do not establish a hard-real-time
guarantee, but the p95 compute budget has substantial headroom.

## Reproduction and raw metrics

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path src)
.\.venv\Scripts\python.exe analysis\performance\benchmark_endurance_update.py
```

The benchmark implementation is
`analysis/performance/benchmark_endurance_update.py`. Machine-readable results
are in `analysis/performance/output/endurance_update_metrics.json`.
