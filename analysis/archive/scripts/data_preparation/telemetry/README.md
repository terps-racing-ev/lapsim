# Selected MF4-to-CSV conversion

`convert_mf4_to_csv.py` is shared by recorded-data analyses. Its default
endurance profile exports only selected GNSS, IMU, control, motor, and battery
signals for the recording. It does not decode every MF4 channel.

The default output is `endurance_selected.csv` beside the script. Its rows use
the native GNSS-speed clock (approximately 5 Hz). Use `--frequency-hz` for a
uniform output rate, `--start-s`/`--finish-s` for a smaller window, or repeated
`--signal` and `--hold-signal` arguments to replace the default signal profile.

The JSON written beside each CSV records every selected source channel and its
native median sample frequency.

The default endurance profile automatically shifts `battery_power_kw` backward
by `0.09 s`. In other words, HVC power logged at `t + 0.09 s` is written on the
row for `t`. This correction comes from aligning HVC pack power with motor
mechanical power (`torque feedback * motor speed`) on the endurance recording.
The per-channel `backward_time_shift_s` metadata records the correction so
downstream analyses can avoid applying it twice. Custom `--signal` channels do
not receive this default correction.

Analysis-specific signal lists and window detection belong in their analysis
folders. For an example, see `analysis/first_endurance_lap`.
