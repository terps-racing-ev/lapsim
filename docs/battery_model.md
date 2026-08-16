# Battery models

`RCTheveninBattery` is the default, stateful `Vehicle` battery. It implements
the same `BatteryModel` protocol as the static `OCVPackBattery` and ideal
constant-power `Battery`, so all three remain directly swappable.

The default pack is 108 series groups by 3 parallel cells (108s3p). Its
nominal capacity is 15.4 Ah at the pack terminals: 5.133333 Ah per cell times
three parallel cells. SOC is a fraction from 0 to 1 and is Coulomb-counted
from the pack current. There is no telemetry-derived capacity estimate and no
SOC offset in this model.

## Data and units

The cell OCV lookup is the 390-point table in Terps Racing EV's
[`ocv_lookup_table.c`](https://github.com/terps-racing-ev/HVC-Firmware/blob/hardware-testing/Core/Src/config/ocv_lookup_table.c).
The firmware stores cell OCV in mV and SOC in 0--10000 units; this model stores
the same values as V and 0--1 respectively, in ascending SOC order.

The matching
[`cells.h`](https://github.com/terps-racing-ev/HVC-Firmware/blob/hardware-testing/Core/Inc/Config/cells.h)
specifies a 2.8 V minimum, a 4.2 V maximum, and three physical cells per
parallel group. Its nominal 15 mOhm cell-resistance constant is retained as a
reference only; it is not the default simulator resistance.

## Static OCV-plus-resistance model

`OCVPackBattery` uses a 0.348617379 Ohm pack resistance, supplied by the
original discharge-only calibration using HVC high-range current samples at
or above 10 A and trusted HVC SOC. The calibration uses no SOC offset. Its
equivalent cell value preserves the explicit 108s3p scaling:

```text
R_cell = 0.348617379 ohm * 3 / 108 = 0.009684 ohm
R_pack = 108 * R_cell / 3 = 0.348617379 ohm
```

At each state update, positive current is discharge and negative current is
charge:

```text
Vout = Voc - I * R_pack
P_terminal = Vout * I
```

The model solves the low-current root of this quadratic for terminal-power
requests, updates SOC using the 15.4 Ah pack capacity, and enforces the OCV
lookup's cell-voltage bounds. `max_discharge_power_w` and
`max_charge_power_w` remain independent configurable terminal-power caps.
The stored terminal voltage and `operating_open_circuit_voltage_v` describe
the operating point used for the most recent state update; the
`open_circuit_voltage_v` property reflects the updated SOC.

## Default one-RC Thevenin model

`RCTheveninBattery` retains the same OCV and Coulomb-counted SOC but separates
the immediate voltage drop from slower polarization:

```text
V_terminal = V_OCV(SOC) - I * R0 - V_p
dV_p/dt = (I * R1 - V_p) / (R1 * C1)
```

The default battery-only endurance calibration is:

| Parameter | Pack value | Equivalent cell value |
|---|---:|---:|
| `R0` | 0.121472656 Ohm | 3.37424 mOhm |
| `R1` | 0.431980951 Ohm | 11.99947 mOhm |
| `C1` | 10.478534 F | 377.227236 F |
| `R1*C1` | 4.526527 s | 4.526527 s |

The RC state uses the exact zero-order-hold update for each timestep:

```text
a = exp(-dt / (R1*C1))
V_p_next = a * V_p + R1 * (1-a) * I
```

This avoids changing the fitted time constant when the simulation timestep
changes. `initial_polarization_voltage_v` defaults to zero for a rested pack.
For replay starting in the middle of a run, initialize it from prior current
history or a measured operating state; it is not a fixed cell parameter.

The RC model uses `R0` when solving terminal current from requested terminal
power. The existing polarization state reduces the effective source voltage,
so it also affects the instantaneous voltage-based charge and discharge
limits.

For a different pack, supply the cell OCV table, capacity, series/parallel
count, resistances, capacitance, and voltage bounds as constructor parameters.
Do not replace a validated HVC SOC trace with a fitted offset in this
component.
