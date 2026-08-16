# Endurance battery-voltage validation

`scripts/validate_endurance_battery_voltage.py` is a standalone validation
tool for the 108s3p accumulator. It decodes only the HVC DBC and uses only
HVC current, pack-terminal voltage, SOC, and the HVC-reported pack-power
channel. No vehicle-motion, driver-command, inverter, GPS, or IMU channel is
decoded or used.

The tool downloads and parses the firmware OCV table at
<https://github.com/terps-racing-ev/HVC-Firmware/blob/hardware-testing/Core/Src/config/ocv_lookup_table.c>.
The firmware table's second field is on a 0–10,000 scale, so HVC SOC in percent
is multiplied by 100 before interpolation. The table voltage is per-cell mV;
the pack OCV is 108 times the interpolated cell OCV. Logged HVC SOC is treated
as ground truth; it is not fitted, shifted, or scaled. The approximately 15.4
Ah pack capacity is recorded as pack context, but is not required because the
fit uses the logged SOC directly rather than coulomb-counting it.

The fitted model is:

```text
V_terminal = V_OCV(SOC) - I_discharge * R_pack
```

The high-range HVC current is selected. Its polarity is checked against the
HVC `HVC_Pack_Power_kW` signal; a positive raw current is treated as discharge
when that agrees with positive reported pack power. The power signal is not an
input to the resistance fit. Samples with invalid SOC (including initial zero
telemetry), all charge/negative-current samples, and discharge currents under
10 A are excluded from the default fit. The latter avoids allowing small
current-sensor quantization and OCV-table offsets to dominate a one-parameter
resistance fit. Residual metrics are separately reported for fitted discharge,
negative-current/charge, low-current, and all-valid samples. Resistance is the only
fitted parameter: neither a SOC offset/scale nor a terminal-voltage offset is
introduced.

This is deliberately a static validation, not an electrochemical model. Its
residuals therefore include polarization and hysteresis, temperature effects
absent from the one-dimensional firmware OCV table, and any timing skew between
HVC broadcasts. Those effects should be considered before interpreting the
single fitted resistance outside this log's operating range.

Run it with the MF4 extra installed:

```powershell
..\.tmp_mf4\venv\Scripts\python.exe scripts\validate_endurance_battery_voltage.py
```

Outputs are written to `outputs/endurance_validation/battery_voltage_fit/`:

- `battery_voltage_fit_metrics.json` — resistance, assumptions, sign-check, and residual metrics.
- `battery_voltage_fit_trace.csv` — synchronized battery-only trace and fit residuals.
- `ocv_lookup_table.csv` — the exact parsed firmware OCV table used.
- Six PNG files — overview, voltage/current comparison, residual diagnostics,
  an automatically selected propulsion-event zoom, predicted-vs-measured parity,
  and the OCV-table plot.

Use `--ocv-table path\to\ocv_lookup_table.c` to reproduce a prior run from a
saved firmware source file without network access; the JSON also records the
source SHA-256.

The synchronized trace from this static fit is also the input to
`scripts/fit_endurance_battery_rc.py`. See `battery_rc_validation.md` for the
dynamic one-RC fit, chronological holdout, and transient comparison plots.
