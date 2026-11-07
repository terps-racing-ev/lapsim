# Baseline model parameters

These are the default model parameters, not claims that every value has been
validated against current hardware. Each constant is defined next to the model
that consumes it so its source can be replaced without editing a solver. The
default vehicle battery is `RCTheveninBattery`; see `battery_model.md` for its
source data and sign convention.

| Path | Default | Meaning |
|---|---:|---|
| `vehicle.mass_kg` | 285.76 kg | total vehicle mass (630 lb) |
| `vehicle.rolling_resistance_coefficient` | 0.012 | whole-vehicle rolling resistance |
| `vehicle.cornering_drag_coefficient` | 0.036 | fitted tire-scrub loss coefficient in `F_drag = coefficient * Fy^2 / Fz` |
| `tire.rolling_radius_m` | 0.2032 m | 8 in tire rolling radius |
| `tire.inflation_pressure_pa` | 98,000 Pa | default operating pressure; the UPC fit's nominal reference is 82,737 Pa |
| `tire.pacejka_lateral` | `Pacejka52UpcR20LateralModel` | default load-sensitive pure-lateral UPC MF-Tyre 5.2 model |
| `tire.pacejka_longitudinal` | `Pacejka52UpcR20LongitudinalModel` | default load-sensitive pure-longitudinal UPC MF-Tyre 5.2 model |
| `tire.pacejka_lateral.correlation_factor / pacejka_longitudinal.correlation_factor` | 0.80 | scales each complete published force curve and peak capacity to 80%, including fitted offsets |
| `tire.lateral_coefficients` | 1.625 to 1.425 | fallback lateral-capacity table when Pacejka is disabled |
| `alignment.anti_ackermann_fraction` | 1.0 | parallel steer at 0; ideal anti-Ackermann at 1, with the outside front wheel taking the larger angle |
| `alignment.front_toe_in_angle_rad / rear_toe_in_angle_rad` | 0 rad | symmetric per-wheel toe; positive means toe-in |
| `alignment.front_static_camber_angle_rad / rear_static_camber_angle_rad` | 0 rad | tire-local static camber; negative leans the tire top toward the vehicle centerline |
| `alignment.front_caster_angle_rad` | 7.3 deg | measured front caster used to calculate steer-induced camber |
| `alignment.*_camber_gain_curve` | measured at 6.35 mm intervals over +/-25.4 mm | per-corner piecewise-linear camber change versus suspension travel, clamped beyond the measured range |
| `tire.peak_lateral_slip_angle_rad` | 10 deg | slip angle corresponding to fallback table capacity when Pacejka is disabled |
| `tire.peak_longitudinal_slip_ratio` | 0.10 | quasi-static longitudinal slip at each tire's combined-force limit |
| `tire.longitudinal_slip_relaxation_length_m` | 0 m | optional per-tire transient force/slip buildup; zero retains the endurance-validated quasi-static baseline |
| `brakes.front_torque_per_pressure_lbfin_per_psi` | 10.12849472 lbf in/psi | total front-axle brake-torque gain |
| `brakes.rear_torque_per_pressure_lbfin_per_psi` | 5.390972994 lbf in/psi | total rear-axle brake-torque gain |
| `brakes.maximum_pressure_psi` | 800 psi | independent front/rear hydraulic pressure ceiling applied before force conversion |
| `drivetrain.driven_wheel_inertia_kgm2` | 0.75 kg m^2 | combined driven-wheel inertia |
| `motor.peak_power_w` | 80 kW | peak shaft-power ceiling |
| `motor.continuous_power_w` | 50 kW | continuous shaft-power reference |
| `motor.max_speed_rpm` | 7,000 rpm | mechanical speed ceiling |
| `motor.efficiency_map` | EMRAX 228 CC digitized bands | RPM/torque-dependent motoring efficiency; bilinear interpolation of band lower edges |
| `motor.efficiency` | 0.95964 | fallback for missing map data; legacy constant regen assumption |
| `motor.rotor_inertia_kgm2` | 0.02521 kg m^2 | EMRAX 228 MV datasheet rotor inertia |
| `inverter.efficiency` | 0.97 | constant baseline DC conversion efficiency |
| `chain_drive.ratio` | 3.455 | motor speed / wheel speed |
| `chain_drive.efficiency` | 0.80 | current motor-shaft to wheel chain-efficiency baseline |
| `battery.max_discharge_power_w` | 80 kW | DC pack-terminal discharge ceiling |
| `battery.max_charge_power_w` | 0 kW | DC pack-terminal regen ceiling |
| `battery.series_cells / parallel_cells` | 108 / 3 | default OCV pack configuration |
| `battery.pack_capacity_ah` | 15.4 Ah | nominal pack capacity used for SOC Coulomb counting |
| `battery.internal_resistance_ohm` | 0.121472656 ohm | instantaneous pack resistance R0 |
| `battery.polarization_resistance_ohm` | 0.431980951 ohm | first-order polarization resistance R1 |
| `battery.polarization_capacitance_f` | 10.478534 F | pack-equivalent polarization capacitance C1 |
| `battery.polarization_time_constant_s` | 4.526527 s | R1*C1 relaxation time |
| `aero.frontal_area_m2` | 0.983996414 m^2 | real car aero-package reference area |
| `aero.drag_coefficient` | 1.60488 | current endurance-validation Cd, rescaled from 2.4 at the legacy 0.65800 m^2 reference area to preserve CdA |
| `aero.lift_coefficient` | -2.42070 | signed Cl, rescaled from -3.62 at the legacy 0.65800 m^2 reference area to preserve ClA |
| `aero.front_downforce_fraction` | 0.526929 | supplied fraction of downforce on the front axle |
| `aero.roll_limit_rad` | 0.017453 rad | 1 degree absolute body-roll breakpoint |
| `aero.downforce_retention_at_roll_limit` | 0.50 | downforce falls linearly from 100% at zero roll to 50% at the breakpoint and remains at 50% above it |
| `ActiveAero.drag_reduction_fraction` | 0.30 | fractional drag reduction in the low-drag configuration |
| `ActiveAero.downforce_reduction_fraction` | 0.30 | fractional downforce reduction in the low-drag configuration |
| `ActiveAero.straight_curvature_threshold_per_m` | 0.005 1/m | automatic low-drag deployment threshold; equivalent to a 200 m minimum path radius |
| active-aero installed mass allowance | 3.0 lb | add explicitly to `vehicle.mass_kg` when modeling the installed hardware |
| `chassis.wheelbase_m` | 1.5494 m | axle-to-axle distance (61 in) |
| `chassis.cg_height_m` | 0.2794 m | CG height (11 in) |
| `chassis.front_track_width_m` | 1.2192 m | front track width (48 in) |
| `chassis.rear_track_width_m` | 1.1684 m | measured rear track width (46 in) |
| `chassis.front_axle_height_m` | 0.2032 m | measured front axle/unsprung reference height (8 in) |
| `chassis.rear_axle_height_m` | 0.2032 m | measured rear axle/unsprung reference height (8 in) |
| `chassis.front_roll_axis_height_m` | 0.0530 m | measured front roll-axis height (2.088 in) |
| `chassis.rear_roll_axis_height_m` | 0.0733 m | measured rear roll-axis height (2.884 in) |
| `chassis.static_front_weight_fraction` | 0.47 | static front axle load fraction |
| `suspension.total_roll_stiffness_nm_per_rad` | 73,814 N m/rad | 950.2 lbf ft/deg total elastic roll stiffness |
| `suspension.front_lateral_load_transfer_fraction` | 0.55 | front share of elastic roll moment/TLLTD |

## Motor efficiency

`Motor()` loads `data/motor_data/emrax_228_cc_efficiency_bands.csv` once and
shares the immutable map between vehicle instances and solver previews. Only
the positive motoring region is mapped. The published chart contains bands,
so the scalar simulation uses their lower edges (86%, 90%, 94%, 95%, or 96%)
and bilinear interpolation between available grid nodes. That interpolation is
a modeling assumption, not a measured smooth efficiency surface or guaranteed
physical bound. A numeric node with no upper band value remains usable at its
published lower edge; blank lower values never become zero efficiency.

Out-of-range, boundary-uncertain, and missing interpolation corners use
`motor.efficiency`, the explicitly configured fallback. No extrapolation or
nearest-valid filling is performed. This includes zero speed/torque and speeds
above the chart's 5,500 rpm extent; the mechanical speed limit stays 7,000 rpm.
The map does not change the peak/continuous torque envelope or shaft-power cap.

Electrical demand uses shaft power divided by operating-point efficiency.
Available driving torque solves the inverse against pack power after inverter
loss, so the same map constrains propulsion as well as SOC/energy accounting.
`MotorModel.mechanical_power_limit_w` accepts optional `motor_speed_rpm` for
this operating-point limit. Calls without RPM retain nominal/fallback power
reporting; the driving envelope always supplies RPM.

Telemetry includes `motor.efficiency`, `motor.efficiency_map_valid`,
`motor.efficiency_fallback_active`, `motor.electrical_input_power_w`, and
`motor.loss_power_w`. Regen retains the existing constant-efficiency assumption
because the supplied source has no negative-torque/generating map. It is not
inferred by mirroring the motoring data.

For a deliberate constant-efficiency comparison:

```python
vehicle.drivetrain.motor.efficiency_map = None
vehicle.drivetrain.motor.efficiency = 0.95963664851588
```

The motor peak and continuous torque tables live in
`vehicle_model/powertrain/motor.py`. The baseline tire uses
`Pacejka52UpcR20LateralModel` and `Pacejka52UpcR20LongitudinalModel`, imported
from the public UPC `.tir` appendix for the Hoosier 16x7.5-10 R20, with both
complete force curves and peak capacities scaled to 80%. This output scaling
preserves the fitted slip positions and load, pressure, and camber dependence.
Their implementations and coefficients live in `mech/pacejka52_lateral.py`
and `mech/pacejka52_longitudinal.py`, respectively. Set `correlation_factor=1.0`
on either model to recover that published pure-slip curve.
The interpolated coefficient tables remain fallbacks when the corresponding
Pacejka model is set to `None`. The older MATLAB MF-Tyre 6.1 lateral fit
remains available explicitly as `Pacejka61LateralModel` in
`vehicle_model/mech/pacejka61.py`. All three models are exposed through
`vehicle_model/mech/pacejka.py`.

`WheelAlignment`, `WheelKinematics`, and all `Tire` force/capacity methods use
vehicle-local camber: negative means the tire top leans inward on either side.
The `Tire` adapter converts this to the fitted coefficient-space inclination
with `gamma = -camber`, for both forces and capacities. Left/right mirroring
separately transforms slip angle and lateral force. Direct calls to the
`Pacejka*Model` classes retain raw source inclination, so MATLAB reference
values and the published coefficients keep their original convention.

For example:

```python
from math import radians

from vehicle_model import (
    Pacejka52UpcR20LateralModel,
    Pacejka52UpcR20LongitudinalModel,
    Tire,
    Vehicle,
    WheelAlignment,
)

vehicle = Vehicle(
    tire=Tire(
        pacejka_lateral=Pacejka52UpcR20LateralModel(),
        pacejka_longitudinal=Pacejka52UpcR20LongitudinalModel(),
        inflation_pressure_pa=10.0 * 6_894.757293168,
    ),
    alignment=WheelAlignment(
        front_static_camber_angle_rad=radians(-1.0),
        rear_static_camber_angle_rad=radians(-1.0),
    ),
)
```

The complete published parameter file, including longitudinal, combined-slip,
and aligning-moment coefficients, is preserved at
`data/tires/hoosier_16x7p5-10_r20_upc.tir`. The source is the [UPC thesis
record](https://upcommons.upc.edu/entities/publication/ea016aa5-38ad-4bf1-a7f1-c473733153bb).

The UPC file is a hybrid vehicle-model fit: its 16-inch test data supplied
pure lateral and aligning-moment data, while longitudinal and combined-slip
terms were carried over from another tire model. The imported lateral and
longitudinal classes expose those published pure-slip blocks independently,
with the default 0.80 correlation factors applied to their outputs. The
`analysis/events/acceleration_points.py` entry point uses both UPC blocks by
default with 10 psi and -1 degree camber on every wheel.
The tire owns the 8 in rolling radius used by drivetrain speed, torque, force,
and reflected-inertia conversions.

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

The baseline suspension applies quasi-static lateral load transfer using the
measured front/rear roll-axis heights and track widths. It solves the elastic
roll-stiffness split so total front load transfer is 55%; with the default
geometry this requires about 61.1% front elastic roll stiffness. The 950.2 lbf
ft/deg total stiffness produces roughly 0.47 degrees of elastic body roll per g
for the default vehicle. The supplied 8-inch axle heights are retained in the
geometry and telemetry, but unsprung-mass transfer cannot be calculated until
front and rear unsprung masses are available. Roll-center migration,
compliance, and transient roll damping remain outside the model.

Camber gain uses the measured curves in `camber gain.xlsx`, zeroed at each
curve's static value so static camber remains an independent setup parameter.
The measured front-left curve is used for both front wheels because no
front-right measurement was supplied. The rear-left and rear-right curves are
kept separate. Positive suspension travel is bump; interpolation is linear
between measurements and values are clamped outside the +/-25.4 mm sweep.
