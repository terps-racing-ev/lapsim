# Baseline model parameters

These are the default model parameters, not claims that every value has been
validated against current hardware. Each constant is defined next to the model
that consumes it so its source can be replaced without editing a solver. The
default vehicle battery is `RCTheveninBattery`; see `battery_model.md` for its
source data and sign convention.

| Path | Default | Meaning |
|---|---:|---|
| `vehicle.mass_kg` | 294.84 kg | total vehicle mass (650 lb) |
| `vehicle.rolling_resistance_coefficient` | 0.012 | whole-vehicle rolling resistance |
| `tire.rolling_radius_m` | 0.2032 m | 8 in tire rolling radius |
| `tire.inflation_pressure_pa` | 98,000 Pa | nominal pressure from the 16x7.5-10 R20 MF-Tyre fit |
| `tire.camber_angle_rad` | 0 rad | fixed camber used by the point-mass lap solver |
| `wheel_slip.peak_longitudinal_slip_ratio` | 0.10 | quasi-static driven-wheel slip at the available rear-tire force limit |
| `brakes.front_torque_per_pressure_lbfin_per_psi` | 10.12849472 lbf in/psi | total front-axle brake-torque gain |
| `brakes.rear_torque_per_pressure_lbfin_per_psi` | 5.390972994 lbf in/psi | total rear-axle brake-torque gain |
| `drivetrain.driven_wheel_inertia_kgm2` | 0.75 kg m^2 | combined driven-wheel inertia |
| `motor.peak_power_w` | 80 kW | peak shaft-power ceiling |
| `motor.continuous_power_w` | 50 kW | continuous shaft-power reference |
| `motor.max_speed_rpm` | 7,000 rpm | mechanical speed ceiling |
| `motor.efficiency` | 0.95964 | constant conversion efficiency fitted from first 2025 endurance lap |
| `motor.rotor_inertia_kgm2` | 0.01215 kg m^2 | motor rotor inertia |
| `inverter.efficiency` | 0.97 | constant baseline DC conversion efficiency |
| `chain_drive.ratio` | 3.455 | motor speed / wheel speed |
| `chain_drive.efficiency` | 0.776853 | motor-shaft to wheel efficiency inferred from 25 brake-free positive-acceleration samples across five straights of the first endurance lap |
| `battery.max_discharge_power_w` | 80 kW | DC pack-terminal discharge ceiling |
| `battery.max_charge_power_w` | 0 kW | DC pack-terminal regen ceiling |
| `battery.series_cells / parallel_cells` | 108 / 3 | default OCV pack configuration |
| `battery.pack_capacity_ah` | 15.4 Ah | nominal pack capacity used for SOC Coulomb counting |
| `battery.internal_resistance_ohm` | 0.121472656 ohm | instantaneous pack resistance R0 |
| `battery.polarization_resistance_ohm` | 0.431980951 ohm | first-order polarization resistance R1 |
| `battery.polarization_capacitance_f` | 10.478534 F | pack-equivalent polarization capacitance C1 |
| `battery.polarization_time_constant_s` | 4.526527 s | R1*C1 relaxation time |
| `aero.frontal_area_m2` | 0.65800 m^2 | 1019.902 in^2 reference frontal area |
| `aero.drag_coefficient` | 1.22878 | drag coefficient Cd, derived from 3.62 downforce coefficient / 2.946 L/D |
| `aero.lift_coefficient` | -3.62 | signed lift coefficient Cl; supplied positive downforce coefficient converted to SAE convention |
| `aero.front_downforce_fraction` | 0.526929 | supplied fraction of downforce on the front axle |
| `chassis.wheelbase_m` | 1.5494 m | axle-to-axle distance (61 in) |
| `chassis.cg_height_m` | 0.2794 m | CG height (11 in) |
| `chassis.static_front_weight_fraction` | 0.475 | static front axle load fraction |

The motor peak and continuous torque tables live in
`vehicle_model/powertrain/motor.py`. Lateral tire force uses the pure-side-slip
MF-Tyre 6.1 equations and coefficients extracted from
`16x7.5-10 R20 Pacejka Coefficients.mat`; the implementation lives in
`vehicle_model/mech/pacejka.py`. Since the point-mass lap solver does not solve
individual wheel slip angles, it uses the model's peak pure-lateral force at
each tire's normal load as its lateral capacity. The longitudinal side remains
the previous interpolated load-sensitive friction approximation in
`vehicle_model/mech/tire.py`.
The tire also owns the 8 in rolling radius used by drivetrain speed, torque,
force, and reflected-inertia conversions. `drivetrain.rolling_radius_m`
remains a compatibility alias to that same value.

The baseline propulsion energy path is:

```text
DC pack terminal -> inverter -> motor shaft -> chain drive -> driven wheels
```

`Drivetrain.efficiency` is a convenient product of the three constant
baseline efficiencies. Actual power calculations call each component's
conversion methods independently, allowing a map-based replacement to vary
with operating point.

The rolling radius, motor efficiency, and chain-drive efficiency are currently
promoted calibration values rather than independent hardware measurements.
They were fitted using the same endurance lap used by the validation replay,
so that replay measures consistency with the calibration data, not
out-of-sample prediction quality.
