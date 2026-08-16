# Analysis archive

This folder preserves superseded work without exposing it as a current entry
point:

- `scripts/data_preparation/`: original telemetry, lap, IMU, and track builders.
- `scripts/legacy/`: one-off analyses replaced by the current workflows.
- `results/endurance/`: historical brake fits, station sweeps, and lap variants.
- `results/acceleration/`: historical straight-replay and tuning variants.
- `results/data_preparation/`: diagnostic plots from the original data build.
- `docs/`: prior workflow notes retained for provenance.

Archived scripts keep their historical paths and may require explicit input
arguments if rerun. New work should use `analysis/endurance` or
`analysis/accel`.
