# First endurance lap

This folder contains the scripts and generated files for analyzing the first
lap of the recorded competition endurance run.

## Generate the first-lap CSV

From the `python_lapsim` repository root:

First generate the shared selected-signal CSV for the complete recording:

```powershell
..\.tmp_mf4\venv\Scripts\python.exe analysis\mf4_to_csv\convert_mf4_to_csv.py
```

Then post-filter that CSV to the first lap:

```powershell
.\.venv\Scripts\python.exe analysis\first_endurance_lap\generate_first_lap.py
```

`generate_first_lap.py` then reads that CSV, detects the lap, and post-filters
the rows. It never opens or decodes the MF4. It writes:

- `first_lap.csv`: decoded measurements on the native approximately 5 Hz GNSS
  speed clock, including raw and filtered local GNSS coordinates.
- `first_lap.json`: channel sources, each source's native frequency, sampling
  rules, lap-detection settings, GNSS filter settings, timing, and gate closure
  error.
- `first_lap_gps_comparison.png`: the official course drawing, raw first-lap
  GNSS samples, and the Savitzky-Golay filtered GNSS path. The plotted GNSS is
  rotated for visual comparison with the course drawing; CSV coordinates remain
  east/north.
- `first_lap_map_overlay.png`: raw and filtered GNSS drawn directly over the
  official course map using an automatically fitted affine display transform.

The official map is a schematic drawing rather than georeferenced survey data,
so a single affine transform cannot make every corner coincide exactly. The
alignment is for visual comparison only and is recorded in `first_lap.json`; it
does not alter the GNSS coordinates used by later analysis.

## Manually adjust the map overlay

After generating the first lap, open the local alignment window from the
`python_lapsim` repository root:

```powershell
.\.venv\Scripts\python.exe analysis\first_endurance_lap\align_gps_to_map_gui.py
```

The map updates live while changing pixel offsets, rotation, independent
horizontal/vertical scale, or horizontal shear. `Reset to automatic fit`
restores the transform recorded in `first_lap.json`. The Matplotlib toolbar can
be used to zoom and pan without affecting the transform.

Click `Save alignment + preview` when the path lines up. It writes two files in
this folder that can be shared back for use in later plotting:

- `manual_map_alignment.json`: the manual controls, original automatic fit, and
  a complete effective affine transform from CSV east/north metres to map
  pixels.
- `manual_map_alignment_preview.png`: the official map with the exact raw and
  filtered paths from that saved transform.

For a non-interactive re-export of the saved alignment, use:

```powershell
.\.venv\Scripts\python.exe analysis\first_endurance_lap\align_gps_to_map_gui.py --export
```

## Build the solver track from the aligned map

After saving the manual alignment, trace the solid red course loop and build a
uniformly sampled `SpatialTrack` with:

```powershell
.\.venv\Scripts\python.exe analysis\first_endurance_lap\build_track_from_map.py
```

This ignores the dashed red cone/slalom annotations, removes their branches
from the extracted image skeleton, inverts the saved affine alignment into
east/north metres, and uses the aligned GNSS sequence to select the start and
driving direction. It writes:

- `map_derived_track.csv`: portable point geometry and per-cell curvature,
  loadable with `SpatialTrack.from_csv(...)`.
- `map_derived_track.json`: provenance, extraction settings, length, curvature,
  closure, and geometry checks.
- `map_derived_track_preview.png`: extracted map centerline, comparison with the
  filtered GNSS path, and curvature versus lap distance.
- `map_derived_track_side_by_side.png`: the official drawing directly beside
  the reconstructed solver geometry in the same left-to-right orientation.

The default local periodic Gaussian filter removes one-pixel stair-stepping
without globally pulling the map's curves flat. After filtering, one uniform
scale sets the total length to the recorded GNSS trip distance; curve removal is
not used to force a length match. The resulting `SpatialTrack` can be converted
for older segment-based solvers with `spatial_track.to_track()`.

Shared MF4 decoding, signal selection, and frequency metadata live in
`analysis/mf4_to_csv`. This analysis owns only first-lap gate logic and derived
local coordinates.

The only smoothing is the documented GNSS position filter used for the two
filtered coordinate columns. No calibration, parameter fitting, or simulation
is performed. Later first-lap scripts should read `first_lap.csv` rather than
decode the MF4 independently.
